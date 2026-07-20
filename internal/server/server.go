// Package server is the inbound edge of the router (ADR-0003, ADR-0016). It
// terminates HTTP, authenticates consumers (ADR-0009), decodes each supported
// consumer protocol into the canonical request model with the inbound adapters,
// builds the matching response sink, and drives a request through
// internal/router. All protocol translation lives here on the inbound edge and
// in the response sinks — never in internal/router.
//
// It may import internal/model, internal/router, and internal/observability
// plus the standard library (ADR-0003).
package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"time"

	"github.com/mattbucci/simple-llm-router/internal/model"
	"github.com/mattbucci/simple-llm-router/internal/observability"
	"github.com/mattbucci/simple-llm-router/internal/router"
)

// Router is the application-layer entry point the server drives (ADR-0003). The
// server owns this interface so it depends on a behavior, not a concrete type:
// *router.Router satisfies it structurally and a fake satisfies it in tests.
type Router interface {
	// Route resolves the request, runs the matching strategy, writes the response
	// to sink, and returns the Outcome for the per-request log line (ADR-0011).
	Route(ctx context.Context, req *model.ChatRequest, sink router.ResponseSink) (*router.Outcome, error)
}

// Server is the HTTP front end. It is stateless per request: everything needed
// to serve a request is derived from the request plus the health snapshot
// (ADR-0006, ADR-0015).
type Server struct {
	router      Router
	health      router.HealthView
	metrics     *observability.Metrics
	auth        Authenticator
	maxBodySize int64
	logger      *slog.Logger

	// Audio gateway proxy (ADR-0022); nil when unconfigured, in which case the
	// audio routes are not registered and their paths 404.
	audioProxy *audioProxy
}

// New builds a Server. maxBodySize caps the inbound request body
// (ADR-0008: large multimodal payloads, generous default set in config). The
// audio config wires the optional audio gateway proxy (ADR-0022); an empty
// BaseURL leaves it nil and its routes are not served. A nil logger falls back to
// slog.Default.
func New(rt Router, health router.HealthView, metrics *observability.Metrics, auth Authenticator, maxBodySize int64, audio AudioConfig, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	s := &Server{
		router:      rt,
		health:      health,
		metrics:     metrics,
		auth:        auth,
		maxBodySize: maxBodySize,
		logger:      logger,
	}
	// Build the gateway proxy when configured. Config has already validated the
	// base URL (ADR-0010), so a build error here is defensive; log and skip.
	if audio.Gateway.Configured() {
		if p, err := newAudioProxy(audio.Gateway, audio.Connect, logger); err != nil {
			logger.Error("audio gateway proxy disabled", slog.String("error", err.Error()))
		} else {
			s.audioProxy = p
		}
	}
	return s
}

// Handler returns the router's HTTP handler. The consumer request endpoints
// under /v1/* are wrapped by the auth middleware (ADR-0009); the operational
// endpoints (/healthz, /readyz, /metrics) are unauthenticated (ADR-0011).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Consumer endpoints — protocol is decided by the endpoint hit (ADR-0016).
	mux.Handle("POST /v1/chat/completions", s.authed(http.HandlerFunc(s.handleOpenAIChat)))
	mux.Handle("POST /v1/messages", s.authed(http.HandlerFunc(s.handleAnthropicMessages)))
	mux.Handle("GET /v1/models", s.authed(http.HandlerFunc(s.handleModels)))

	// Audio gateway endpoints — registered only when the gateway is configured, so
	// otherwise these paths 404 (ADR-0022). They reuse the inbound auth middleware
	// (ADR-0009) but bypass internal/router entirely; the gateway fans each one out
	// to its local or cloud engine. The /v1/voices subtree is registered
	// method-less so list/register/delete (and any /{id} child) all pass through.
	if s.audioProxy != nil {
		h := s.authed(s.audioHandler(s.audioProxy))
		mux.Handle("POST /v1/audio/speech", h)
		mux.Handle("POST /v1/audio/transcriptions", h)
		mux.Handle("POST /v1/audio/isolation", h)
		mux.Handle("POST /v1/sound-effects", h)
		mux.Handle("POST /v1/music", h)
		mux.Handle("/v1/voices", h)
		mux.Handle("/v1/voices/", h)
	}

	// Operational endpoints — never authenticated (ADR-0011).
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /readyz", s.handleReadyz)
	mux.HandleFunc("GET /metrics", s.handleMetrics)

	return mux
}

func (s *Server) handleOpenAIChat(w http.ResponseWriter, r *http.Request) {
	s.handleChat(w, r, model.ProtocolOpenAI)
}

func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	s.handleChat(w, r, model.ProtocolAnthropic)
}

// handleChat is the shared request path for both consumer protocols. It buffers
// the body, decodes it via the inbound adapter, constructs the translating sink,
// and routes. The body is buffered rather than streamed (ADR-0008's SHOULD-stream
// is deliberately traded away) because failover (ADR-0006) and cross-protocol
// translation (ADR-0016) both require the request bytes to be replayable.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request, consumer model.Protocol) {
	start := time.Now()
	ctx := observability.WithRequestID(r.Context(), observability.NewRequestID())

	s.metrics.AddInFlight(1)
	defer s.metrics.AddInFlight(-1)

	var sink router.ResponseSink
	status := http.StatusOK
	rec := observability.RequestRecord{ConsumerProtocol: string(consumer)}

	// (A) Registered first, runs last: exactly one log line + one metrics
	// record per request (ADR-0011), with whatever status we ended up at.
	defer func() {
		rec.Status = status
		rec.Latency = time.Since(start)
		rec.Emit(ctx, s.logger)
		s.metrics.RequestDone(rec.Backend, rec.UpstreamModel, status, rec.Latency)
	}()

	// (B) Registered last, runs first: the request path must never crash the
	// process (ADR-0015). If anything below panics, surface a 500 if we still
	// own the response, then let (A) log it.
	defer func() {
		if p := recover(); p != nil {
			status = http.StatusInternalServerError
			if sink == nil || !sink.Wrote() {
				writeAPIErrorFor(w, consumer, &model.APIError{
					Status:  http.StatusInternalServerError,
					Code:    "internal_error",
					Message: "internal server error",
				})
			}
		}
	}()

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, s.maxBodySize))
	if err != nil {
		apiErr := bodyReadError(err)
		status = apiErr.Status
		writeAPIErrorFor(w, consumer, apiErr)
		return
	}

	req, err := decodeRequest(consumer, body)
	if err != nil {
		var apiErr *model.APIError
		if !errors.As(err, &apiErr) {
			apiErr = model.ErrBadRequest(err.Error())
		}
		status = apiErr.Status
		writeAPIErrorFor(w, consumer, apiErr)
		return
	}
	rec.ModelAlias = req.Model

	sink = newSink(consumer, w)

	outcome, rerr := s.router.Route(ctx, req, sink)
	if outcome != nil {
		rec.Backend = outcome.Backend
		rec.UpstreamModel = outcome.UpstreamModel
		rec.ProviderProtocol = string(outcome.ProviderProtocol)
		rec.Failovers = outcome.Failovers
		rec.PromptTokens = outcome.Usage.PromptTokens
		rec.CompletionTokens = outcome.Usage.CompletionTokens
		rec.ReasoningTokens = outcome.Usage.ReasoningTokens
		if outcome.Status != 0 {
			status = outcome.Status
		}
	}
	if rerr != nil {
		// An error response (or a status override) is only possible while no
		// response byte has reached the client yet. Once the sink has committed —
		// e.g. a stream emitted bytes — the HTTP status is fixed and the failure
		// is surfaced inline by the strategy, so we keep the committed status and
		// just log it (ADR-0006, ADR-0007).
		if !sink.Wrote() {
			var apiErr *model.APIError
			if !errors.As(rerr, &apiErr) {
				// A non-APIError escaping Route means an internal failure; only
				// the router's own APIErrors are part of the contract.
				apiErr = model.ErrUpstreamUnavailable()
			}
			status = apiErr.Status
			writeAPIErrorFor(w, consumer, apiErr)
		}
	}
}

// handleModels aggregates the upstream model ids discovered across the fleet into
// the OpenAI list shape (ADR-0011, ADR-0016).
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	snap := s.health.Snapshot()
	set := map[string]struct{}{}
	for _, b := range snap.Backends {
		for id := range b.Models {
			set[id] = struct{}{}
		}
	}
	ids := make([]string, 0, len(set))
	for id := range set {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	type modelObject struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}
	list := struct {
		Object string        `json:"object"`
		Data   []modelObject `json:"data"`
	}{Object: "list", Data: make([]modelObject, 0, len(ids))}
	for _, id := range ids {
		list.Data = append(list.Data, modelObject{ID: id, Object: "model", OwnedBy: "simple-llm-router"})
	}
	writeJSON(w, http.StatusOK, list)
}

// handleHealthz reports process liveness; it is 200 whenever the server is
// running (ADR-0011).
func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, "ok\n")
}

// handleReadyz reports readiness: 200 only when at least one backend is healthy,
// else 503 (ADR-0011).
func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	if s.health.Snapshot().Healthy() {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "ready\n")
		return
	}
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = io.WriteString(w, "no healthy backend\n")
}

// handleMetrics refreshes the per-backend health gauges from the current
// snapshot (so the backend layer need not import observability, ADR-0003) and
// then renders the Prometheus text exposition (ADR-0011).
func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	for name, b := range s.health.Snapshot().Backends {
		s.metrics.SetHealth(name, b.Healthy)
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, s.metrics.Render())
}

// decodeRequest dispatches to the inbound adapter for the consumer protocol,
// producing a canonical, OpenAI-shaped *model.ChatRequest (ADR-0016).
func decodeRequest(consumer model.Protocol, body []byte) (*model.ChatRequest, error) {
	switch consumer {
	case model.ProtocolAnthropic:
		return parseAnthropicRequest(body)
	default:
		return parseOpenAIRequest(body)
	}
}

// newSink builds the response sink for the consumer protocol: a verbatim relay
// for OpenAI consumers, a translating sink for Anthropic consumers (ADR-0016).
func newSink(consumer model.Protocol, w http.ResponseWriter) router.ResponseSink {
	switch consumer {
	case model.ProtocolAnthropic:
		return newAnthropicSink(w)
	default:
		return newOpenAISink(w)
	}
}

// bodyReadError maps an inbound body read failure to an APIError. Exceeding the
// configured cap is 413; anything else is a malformed/aborted request (400).
func bodyReadError(err error) *model.APIError {
	var maxErr *http.MaxBytesError
	if errors.As(err, &maxErr) {
		return &model.APIError{
			Status:  http.StatusRequestEntityTooLarge,
			Code:    "request_too_large",
			Message: "request body exceeds the configured maximum size",
		}
	}
	return model.ErrBadRequest("could not read request body")
}

// writeAPIError writes an OpenAI-shaped error document (ADR-0011). Prefer
// writeAPIErrorFor on a request path that knows its consumer protocol.
func writeAPIError(w http.ResponseWriter, e *model.APIError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Status)
	_, _ = w.Write(e.Body())
}

// writeAPIErrorFor writes e in the envelope the consumer's protocol expects.
// Errors raised before routing — auth, malformed JSON, unknown model — come from
// the router itself rather than an upstream, so without this an Anthropic client
// receives an OpenAI-shaped document from /v1/messages that its SDK cannot parse
// and whose wording client-side retry logic matches on (ADR-0019).
func writeAPIErrorFor(w http.ResponseWriter, consumer model.Protocol, e *model.APIError) {
	if consumer != model.ProtocolAnthropic {
		writeAPIError(w, e)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.Status)
	_, _ = w.Write(anthropicErrorBody(e.Status, e.Body()))
}

// writeJSON marshals v and writes it with a JSON content type.
func writeJSON(w http.ResponseWriter, status int, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(b)
}

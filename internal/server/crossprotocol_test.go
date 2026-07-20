package server_test

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mattbucci/simple-llm-router/internal/model"
	"github.com/mattbucci/simple-llm-router/internal/observability"
	"github.com/mattbucci/simple-llm-router/internal/router"
	"github.com/mattbucci/simple-llm-router/internal/server"
)

// streamResponse stands in for a backend that answers a streaming request with an
// OpenAI SSE body: the router splices its records and the sink frames/translates
// each one (ADR-0007). Content-Type is text/event-stream so the path mirrors a
// real provider stream.
func streamResponse(sse string) func([]byte, bool) (*model.UpstreamResponse, error) {
	return func([]byte, bool) (*model.UpstreamResponse, error) {
		return &model.UpstreamResponse{
			Status: http.StatusOK,
			Header: http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:   io.NopCloser(strings.NewReader(sse)),
		}, nil
	}
}

// TestAnthropicConsumerStreamTranslatesOpenAIBackend covers the ADR-0007 SHOULD:
// an Anthropic consumer streaming (stream:true) against an OpenAI backend gets the
// OpenAI delta chunks translated on the fly into the Anthropic SSE event sequence —
// message_start, content_block_start, content_block_delta(text_delta),
// content_block_stop, message_delta, message_stop — with NO OpenAI [DONE]
// terminator leaking through the translating sink, and in the right order.
func TestAnthropicConsumerStreamTranslatesOpenAIBackend(t *testing.T) {
	// An OpenAI provider stream: two text deltas, then a final chunk with
	// finish_reason + usage, then the OpenAI [DONE] sentinel.
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"up-oai","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"up-oai","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","object":"chat.completion.chunk","model":"up-oai","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2}}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")

	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: streamResponse(sse)}
	srv := newServer(t,
		map[string]router.Backend{"b1": be},
		map[string]*router.Alias{"alias-oai": proxyAlias("alias-oai", "up-oai", "b1")},
		snapshot(model.BackendState{Name: "b1", Healthy: true}),
		nil, io.Discard,
	)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"alias-oai","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	body := rec.Body.String()

	// The cross-protocol sink must have synthesized the full Anthropic event suite.
	for _, want := range []string{
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		`"type":"text_delta"`,
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("translated stream missing %q:\n%s", want, body)
		}
	}

	// The OpenAI text deltas must survive into the Anthropic text_delta events.
	for _, want := range []string{"Hello", " world"} {
		if !strings.Contains(body, want) {
			t.Fatalf("translated stream dropped delta %q:\n%s", want, body)
		}
	}

	// finish_reason "stop" maps to the Anthropic end_turn stop reason.
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("message_delta missing stop_reason end_turn:\n%s", body)
	}

	// The OpenAI terminator must never reach an Anthropic consumer.
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("translated Anthropic stream must not carry the OpenAI [DONE] terminator:\n%s", body)
	}

	// Event ordering: start opens the message before any delta; the terminators
	// close it after every delta (ADR-0007 stateful translation).
	mustOrder(t, body,
		"event: message_start",
		"event: content_block_start",
		"event: content_block_delta",
		"event: content_block_stop",
		"event: message_delta",
		"event: message_stop",
	)
}

// TestAnthropicToOpenAIDropsNativeOnlyFields covers the ADR-0018 SHOULD
// (complementing the same-protocol survive test): when an Anthropic consumer is
// routed cross-protocol to an OpenAI backend, the canonical translation carries
// only fields with an OpenAI meaning. tools and tool_choice now have one and are
// translated (TestAnthropicToolsTranslateToOpenAI); top_k, metadata, and
// cache_control still have none, so they remain ABSENT from the forwarded
// upstream body — those survive only on the same-protocol native relay.
func TestAnthropicToOpenAIDropsNativeOnlyFields(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"chatcmpl-1","model":"up-oai","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)}
	srv := newServer(t,
		map[string]router.Backend{"b1": be},
		map[string]*router.Alias{"alias-oai": proxyAlias("alias-oai", "up-oai", "b1")},
		snapshot(model.BackendState{Name: "b1", Healthy: true}),
		nil, io.Discard,
	)

	reqBody := `{"model":"alias-oai","max_tokens":128,"top_k":40,"metadata":{"user_id":"u1"},"tool_choice":{"type":"auto"},"tools":[{"name":"get_weather","description":"w","input_schema":{"type":"object"}}],"system":[{"type":"text","text":"be brief","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"weather?"}]}`
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(reqBody))
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var sent map[string]json.RawMessage
	if err := json.Unmarshal(be.lastBody, &sent); err != nil {
		t.Fatalf("forwarded body: %v", err)
	}

	// The Anthropic-native fields must not have crossed the canonical boundary.
	for _, k := range []string{"top_k", "metadata"} {
		if _, ok := sent[k]; ok {
			t.Fatalf("native-only field %q forwarded to OpenAI backend: %s", k, be.lastBody)
		}
	}
	// cache_control lives nested inside the system blocks; the system array is
	// flattened to a plain system-role message, so the directive must not appear
	// anywhere in the forwarded bytes.
	if strings.Contains(string(be.lastBody), "cache_control") {
		t.Fatalf("cache_control leaked into forwarded body: %s", be.lastBody)
	}

	// Sanity: the cross-protocol canonical path actually ran — model rewritten to
	// the upstream id, max_tokens carried over, and the system block hoisted into a
	// leading system-role message (the very flattening that drops cache_control).
	var fwdModel string
	_ = json.Unmarshal(sent["model"], &fwdModel)
	if fwdModel != "up-oai" {
		t.Fatalf("forwarded model = %q, want up-oai", fwdModel)
	}
	if _, ok := sent["max_tokens"]; !ok {
		t.Fatalf("max_tokens dropped from forwarded body: %s", be.lastBody)
	}
	var msgs []struct {
		Role string `json:"role"`
	}
	if err := json.Unmarshal(sent["messages"], &msgs); err != nil {
		t.Fatalf("forwarded messages: %v", err)
	}
	if len(msgs) != 2 || msgs[0].Role != "system" || msgs[1].Role != "user" {
		t.Fatalf("canonical messages = %s, want [system,user]", sent["messages"])
	}
}

// ----- fake Router: unit-test a handler without a real router.New --------------

// fakeRouter implements the server-owned server.Router interface (ADR-0003) so a
// handler can be exercised without wiring a real *router.Router. It records the
// canonical request the handler decoded and delegates the response to a closure.
type fakeRouter struct {
	gotReq *model.ChatRequest
	fn     func(ctx context.Context, req *model.ChatRequest, sink router.ResponseSink) (*router.Outcome, error)
}

func (f *fakeRouter) Route(ctx context.Context, req *model.ChatRequest, sink router.ResponseSink) (*router.Outcome, error) {
	f.gotReq = req
	return f.fn(ctx, req, sink)
}

// newServerWithRouter builds a Server over an arbitrary server.Router, bypassing
// router.New so a handler can be tested in isolation against a fake.
func newServerWithRouter(t *testing.T, rt server.Router) *server.Server {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	metrics := observability.New(ctx)
	health := fakeHealth{model.EmptySnapshot()}
	logger := slog.New(slog.NewJSONHandler(io.Discard, nil))
	return server.New(rt, health, metrics, server.NewStaticTokenAuth(nil), 100<<20, server.AudioConfig{}, logger)
}

// TestHandlerDrivesFakeRouter shows the handler decoding the consumer request into
// the canonical model and relaying whatever the router writes to the sink — proven
// against a fake Router (the server-owned interface), with no real router
// constructed.
func TestHandlerDrivesFakeRouter(t *testing.T) {
	t.Run("decodes request and relays router response", func(t *testing.T) {
		const replyBody = `{"id":"chatcmpl-z","choices":[{"message":{"content":"pong"}}]}`
		fr := &fakeRouter{
			fn: func(_ context.Context, _ *model.ChatRequest, sink router.ResponseSink) (*router.Outcome, error) {
				if werr := sink.WriteResponse(http.StatusOK,
					http.Header{"Content-Type": []string{"application/json"}}, []byte(replyBody)); werr != nil {
					return nil, werr
				}
				return &router.Outcome{Status: http.StatusOK, Backend: "fake-be", UpstreamModel: "fake-up"}, nil
			},
		}
		srv := newServerWithRouter(t, fr)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"my-alias","stream":false,"messages":[{"role":"user","content":"ping"}]}`))
		srv.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
		}
		if rec.Body.String() != replyBody {
			t.Fatalf("body = %q, want %q", rec.Body.String(), replyBody)
		}
		// The handler must have decoded the OpenAI consumer request before routing.
		if fr.gotReq == nil {
			t.Fatalf("router was never called")
		}
		if fr.gotReq.Consumer != model.ProtocolOpenAI {
			t.Fatalf("req.Consumer = %q, want openai", fr.gotReq.Consumer)
		}
		if fr.gotReq.Model != "my-alias" {
			t.Fatalf("req.Model = %q, want my-alias", fr.gotReq.Model)
		}
		if fr.gotReq.Stream {
			t.Fatalf("req.Stream = true, want false")
		}
	})

	t.Run("surfaces a router APIError when no byte was committed", func(t *testing.T) {
		fr := &fakeRouter{
			fn: func(_ context.Context, _ *model.ChatRequest, _ router.ResponseSink) (*router.Outcome, error) {
				err := model.ErrModelNotFound("ghost")
				return &router.Outcome{Status: err.Status}, err
			},
		}
		srv := newServerWithRouter(t, fr)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			strings.NewReader(`{"model":"ghost","messages":[]}`))
		srv.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusNotFound {
			t.Fatalf("status = %d, want 404 (%s)", rec.Code, rec.Body.String())
		}
		var e struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
			t.Fatalf("error body: %v", err)
		}
		if e.Error.Code != "model_not_found" {
			t.Fatalf("error code = %q, want model_not_found", e.Error.Code)
		}
	})

	t.Run("anthropic endpoint decodes anthropic consumer", func(t *testing.T) {
		fr := &fakeRouter{
			fn: func(_ context.Context, _ *model.ChatRequest, sink router.ResponseSink) (*router.Outcome, error) {
				// An OpenAI-canonical completion the Anthropic sink will translate.
				_ = sink.WriteResponse(http.StatusOK,
					http.Header{"Content-Type": []string{"application/json"}},
					[]byte(`{"id":"chatcmpl-a","model":"up","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
				return &router.Outcome{Status: http.StatusOK}, nil
			},
		}
		srv := newServerWithRouter(t, fr)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/v1/messages",
			strings.NewReader(`{"model":"claude-alias","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`))
		srv.Handler().ServeHTTP(rec, req)

		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
		}
		if fr.gotReq == nil || fr.gotReq.Consumer != model.ProtocolAnthropic {
			t.Fatalf("req.Consumer = %v, want anthropic", fr.gotReq)
		}
		// The translating sink must have produced Anthropic shape from the canonical
		// completion the fake router wrote.
		var out struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("response not JSON: %v (%s)", err, rec.Body.String())
		}
		if out.Type != "message" {
			t.Fatalf("response type = %q, want message (Anthropic-shaped): %s", out.Type, rec.Body.String())
		}
	})
}

// mustOrder asserts the given substrings appear in body in the given order.
func mustOrder(t *testing.T, body string, parts ...string) {
	t.Helper()
	prev := 0
	for i, p := range parts {
		idx := strings.Index(body[prev:], p)
		if idx < 0 {
			t.Fatalf("ordering: %q not found after part %d:\n%s", p, i, body)
		}
		prev += idx + len(p)
	}
}

package server

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/mattbucci/simple-llm-router/internal/model"
)

// Authenticator decides whether an inbound request is allowed. It is the single
// pluggable seam for consumer-side auth so a real IdP (JWT/OAuth) can replace
// static tokens later without touching handlers (ADR-0009).
type Authenticator interface {
	// Authenticate reports whether the request presents acceptable credentials.
	Authenticate(r *http.Request) bool
}

// staticTokenAuth accepts a fixed allowlist of bearer tokens.
type staticTokenAuth struct {
	tokens [][]byte
}

// NewStaticTokenAuth returns an Authenticator backed by a static token
// allowlist (ADR-0009). An empty list trusts the LAN and accepts every request.
// A non-empty list requires a matching token presented either as
// "Authorization: Bearer <token>" (OpenAI style) or "x-api-key: <token>"
// (Anthropic style); comparison is constant-time to avoid timing oracles.
func NewStaticTokenAuth(tokens []string) Authenticator {
	a := &staticTokenAuth{}
	for _, t := range tokens {
		if t == "" {
			continue
		}
		a.tokens = append(a.tokens, []byte(t))
	}
	return a
}

// Authenticate implements Authenticator.
func (a *staticTokenAuth) Authenticate(r *http.Request) bool {
	if len(a.tokens) == 0 {
		return true // empty allowlist: trust the LAN (ADR-0009).
	}
	presented := []byte(extractToken(r))
	if len(presented) == 0 {
		return false
	}
	// Compare against every configured token without an early return so a match
	// at the first vs last entry is not observable via timing.
	matched := 0
	for _, t := range a.tokens {
		matched |= subtle.ConstantTimeCompare(presented, t)
	}
	return matched == 1
}

// extractToken pulls the presented token from either accepted header style
// (ADR-0009). Authorization: Bearer takes precedence; x-api-key is the fallback.
func extractToken(r *http.Request) string {
	if h := r.Header.Get("Authorization"); h != "" {
		const prefix = "Bearer "
		if len(h) >= len(prefix) && strings.EqualFold(h[:len(prefix)], prefix) {
			return strings.TrimSpace(h[len(prefix):])
		}
	}
	if k := r.Header.Get("X-Api-Key"); k != "" {
		return strings.TrimSpace(k)
	}
	return ""
}

// authed wraps a handler with the inbound auth check and strips the consumer
// credential at the trust boundary so it is never forwarded upstream
// (ADR-0009): backends only ever see the router's own injected credential.
// The rejection is written in the consumer protocol's error envelope — the
// endpoint decides the protocol (ADR-0016), so the caller passes it in
// (ADR-0019).
func (s *Server) authed(consumer model.Protocol, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !s.auth.Authenticate(r) {
			writeAPIErrorFor(w, consumer, &model.APIError{
				Status:  http.StatusUnauthorized,
				Code:    "unauthorized",
				Message: "missing or invalid credentials",
			})
			return
		}
		// Strip inbound credentials regardless of whether auth was required, so
		// nothing downstream can accidentally relay them.
		r.Header.Del("Authorization")
		r.Header.Del("X-Api-Key")
		next.ServeHTTP(w, r)
	})
}

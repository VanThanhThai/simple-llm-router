package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mattbucci/simple-llm-router/internal/model"
)

// TestStaticTokenAuth covers ADR-0009: when tokens are configured a matching
// token is required (Bearer or x-api-key); an empty allowlist trusts the LAN.
func TestStaticTokenAuth(t *testing.T) {
	cases := []struct {
		name      string
		tokens    []string
		authHdr   string
		apiKeyHdr string
		want      bool
	}{
		{name: "empty allowlist accepts all", tokens: nil, want: true},
		{name: "bearer match", tokens: []string{"sek"}, authHdr: "Bearer sek", want: true},
		{name: "bearer mismatch", tokens: []string{"sek"}, authHdr: "Bearer nope", want: false},
		{name: "x-api-key match", tokens: []string{"sek"}, apiKeyHdr: "sek", want: true},
		{name: "missing credential rejected", tokens: []string{"sek"}, want: false},
		{name: "second token in allowlist", tokens: []string{"a", "b"}, authHdr: "Bearer b", want: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := NewStaticTokenAuth(tc.tokens)
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			if tc.authHdr != "" {
				r.Header.Set("Authorization", tc.authHdr)
			}
			if tc.apiKeyHdr != "" {
				r.Header.Set("X-Api-Key", tc.apiKeyHdr)
			}
			if got := a.Authenticate(r); got != tc.want {
				t.Fatalf("Authenticate = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestAuthedStripsInboundCredential covers ADR-0009: the inbound consumer
// credential is removed at the trust boundary so it can never reach a backend.
func TestAuthedStripsInboundCredential(t *testing.T) {
	s := &Server{auth: NewStaticTokenAuth(nil)} // empty allowlist accepts all

	var sawAuth, sawAPIKey string
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuth = r.Header.Get("Authorization")
		sawAPIKey = r.Header.Get("X-Api-Key")
		w.WriteHeader(http.StatusOK)
	})

	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	r.Header.Set("Authorization", "Bearer consumer-secret")
	r.Header.Set("X-Api-Key", "consumer-key")
	rec := httptest.NewRecorder()

	s.authed(model.ProtocolOpenAI, next).ServeHTTP(rec, r)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if sawAuth != "" {
		t.Fatalf("downstream saw Authorization = %q, want stripped", sawAuth)
	}
	if sawAPIKey != "" {
		t.Fatalf("downstream saw X-Api-Key = %q, want stripped", sawAPIKey)
	}
}

// TestAuthRejects401 covers ADR-0009: a configured allowlist rejects an
// unauthenticated request with 401 and an OpenAI-shaped error body.
func TestAuthRejects401(t *testing.T) {
	s := &Server{auth: NewStaticTokenAuth([]string{"sek"})}
	called := false
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { called = true })

	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	s.authed(model.ProtocolOpenAI, next).ServeHTTP(rec, r)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	if called {
		t.Fatalf("next handler must not run on a rejected request")
	}
}

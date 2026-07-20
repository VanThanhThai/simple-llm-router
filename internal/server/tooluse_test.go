package server_test

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/mattbucci/simple-llm-router/internal/model"
	"github.com/mattbucci/simple-llm-router/internal/router"
)

// toolServer wires an Anthropic consumer to a single OpenAI backend — the
// cross-protocol path where tool translation has to happen.
func toolServer(t *testing.T, be *fakeBackend) *serverUnderTest {
	t.Helper()
	srv := newServer(t,
		map[string]router.Backend{"b1": be},
		map[string]*router.Alias{"alias-oai": proxyAlias("alias-oai", "up-oai", "b1")},
		snapshot(model.BackendState{Name: "b1", Healthy: true}),
		nil, io.Discard,
	)
	return &serverUnderTest{t: t, handler: srv.Handler()}
}

type serverUnderTest struct {
	t       *testing.T
	handler http.Handler
}

func (s *serverUnderTest) post(body string) *httptest.ResponseRecorder {
	s.t.Helper()
	rec := httptest.NewRecorder()
	s.handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body)))
	return rec
}

// TestAnthropicToolsTranslateToOpenAI covers the request half of tool use: an
// Anthropic consumer's tool definitions and tool_choice must reach an OpenAI
// backend in OpenAI's own shape, or the model is never told the tools exist and
// can never call one.
func TestAnthropicToolsTranslateToOpenAI(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"chatcmpl-1","model":"up-oai","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","max_tokens":64,` +
		`"tools":[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],` +
		`"tool_choice":{"type":"any","disable_parallel_tool_use":true},` +
		`"messages":[{"role":"user","content":"weather in Paris?"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var sent struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string         `json:"name"`
				Description string         `json:"description"`
				Parameters  map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice        json.RawMessage `json:"tool_choice"`
		ParallelToolCalls *bool           `json:"parallel_tool_calls"`
	}
	if err := json.Unmarshal(be.lastBody, &sent); err != nil {
		t.Fatalf("forwarded body: %v", err)
	}

	if len(sent.Tools) != 1 {
		t.Fatalf("forwarded tools = %s, want exactly one", be.lastBody)
	}
	tool := sent.Tools[0]
	if tool.Type != "function" || tool.Function.Name != "get_weather" {
		t.Fatalf("tool not translated to an OpenAI function tool: %s", be.lastBody)
	}
	if tool.Function.Description != "Get weather" {
		t.Fatalf("tool description lost: %s", be.lastBody)
	}
	// input_schema must land as "parameters" — the schema is what constrains the
	// model's arguments, so dropping it silently degrades every call.
	if tool.Function.Parameters == nil || tool.Function.Parameters["properties"] == nil {
		t.Fatalf("input_schema did not become function.parameters: %s", be.lastBody)
	}
	// Anthropic "any" (must call something) is OpenAI "required".
	if string(sent.ToolChoice) != `"required"` {
		t.Fatalf("tool_choice = %s, want \"required\"", sent.ToolChoice)
	}
	// disable_parallel_tool_use is a flag on Anthropic's tool_choice but a
	// top-level parameter on OpenAI, with inverted sense.
	if sent.ParallelToolCalls == nil || *sent.ParallelToolCalls {
		t.Fatalf("parallel_tool_calls = %v, want false: %s", sent.ParallelToolCalls, be.lastBody)
	}
}

// TestAnthropicToolChoiceNamedTool covers the remaining tool_choice forms.
func TestAnthropicToolChoiceNamedTool(t *testing.T) {
	for _, tc := range []struct {
		name  string
		given string
		want  string
	}{
		{"auto", `{"type":"auto"}`, `"auto"`},
		{"none", `{"type":"none"}`, `"none"`},
		{"named", `{"type":"tool","name":"get_weather"}`, `{"type":"function","function":{"name":"get_weather"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}]}`)}
			srv := toolServer(t, be)
			rec := srv.post(`{"model":"alias-oai","max_tokens":16,"tool_choice":` + tc.given +
				`,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`)
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
			}
			var sent struct {
				ToolChoice json.RawMessage `json:"tool_choice"`
			}
			if err := json.Unmarshal(be.lastBody, &sent); err != nil {
				t.Fatalf("forwarded body: %v", err)
			}
			// Compare decoded values: encoding/json sorts object keys, so the
			// byte order of an equivalent document is not stable.
			var got, want any
			if err := json.Unmarshal(sent.ToolChoice, &got); err != nil {
				t.Fatalf("tool_choice not valid JSON: %s", sent.ToolChoice)
			}
			if err := json.Unmarshal([]byte(tc.want), &want); err != nil {
				t.Fatalf("bad want fixture: %s", tc.want)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("tool_choice = %s, want %s", sent.ToolChoice, tc.want)
			}
		})
	}
}

// TestAnthropicToolResultFansOutToOpenAIMessages covers the structural half of
// the translation. Anthropic carries a tool result as a tool_result block inside
// a user turn; OpenAI requires a separate role:"tool" message per result. A turn
// with two results must therefore become two messages, and the assistant's
// tool_use blocks must become tool_calls on the assistant message — otherwise a
// multi-turn agent loop cannot be replayed to the backend at all.
func TestAnthropicToolResultFansOutToOpenAIMessages(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[{"index":0,"message":{"content":"done"},"finish_reason":"stop"}]}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","max_tokens":64,"messages":[` +
		`{"role":"user","content":"weather in Paris and Rome?"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}},{"type":"tool_use","id":"toolu_2","name":"get_weather","input":{"city":"Rome"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"18C"},{"type":"tool_result","tool_use_id":"toolu_2","content":[{"type":"text","text":"24C"}]},{"type":"text","text":"thanks"}]}` +
		`]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var sent struct {
		Messages []struct {
			Role      string          `json:"role"`
			Content   json.RawMessage `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
			ToolCallID string `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(be.lastBody, &sent); err != nil {
		t.Fatalf("forwarded body: %v", err)
	}

	// user, assistant(2 tool_calls), tool, tool, user("thanks")
	if len(sent.Messages) != 5 {
		t.Fatalf("forwarded %d messages, want 5: %s", len(sent.Messages), be.lastBody)
	}

	assistant := sent.Messages[1]
	if assistant.Role != "assistant" || len(assistant.ToolCalls) != 2 {
		t.Fatalf("assistant turn did not carry two tool_calls: %s", be.lastBody)
	}
	if assistant.ToolCalls[0].ID != "toolu_1" || assistant.ToolCalls[0].Function.Name != "get_weather" {
		t.Fatalf("tool_use did not become an OpenAI tool call: %s", be.lastBody)
	}
	// Anthropic's structured "input" becomes OpenAI's JSON-string "arguments".
	var args map[string]any
	if err := json.Unmarshal([]byte(assistant.ToolCalls[0].Function.Arguments), &args); err != nil {
		t.Fatalf("arguments is not a JSON string of the input object: %q", assistant.ToolCalls[0].Function.Arguments)
	}
	if args["city"] != "Paris" {
		t.Fatalf("tool input lost in translation: %v", args)
	}

	// Each tool_result becomes its own role:"tool" message, keyed by tool_use_id.
	for i, wantID := range []string{"toolu_1", "toolu_2"} {
		m := sent.Messages[2+i]
		if m.Role != "tool" || m.ToolCallID != wantID {
			t.Fatalf("message %d = role %q id %q, want tool/%s: %s", 2+i, m.Role, m.ToolCallID, wantID, be.lastBody)
		}
	}
	// A block-form tool_result flattens to its text.
	if !strings.Contains(string(sent.Messages[3].Content), "24C") {
		t.Fatalf("block-form tool_result content lost: %s", be.lastBody)
	}
	// Prose accompanying the results follows them, as OpenAI ordering requires.
	if sent.Messages[4].Role != "user" || !strings.Contains(string(sent.Messages[4].Content), "thanks") {
		t.Fatalf("trailing user text not preserved after tool results: %s", be.lastBody)
	}
}

// TestOpenAIToolCallsBecomeAnthropicToolUse covers the response half: an OpenAI
// backend's tool_calls must surface to the Anthropic consumer as tool_use content
// blocks. Previously the sink emitted stop_reason "tool_use" with a text-only
// body — a response that violates Anthropic's contract and that no SDK can act on.
func TestOpenAIToolCallsBecomeAnthropicToolUse(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(
		`{"id":"chatcmpl-1","model":"up-oai","choices":[{"index":0,"message":{"role":"assistant","content":null,` +
			`"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},` +
			`"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":5}}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","max_tokens":64,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"weather?"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var got struct {
		Content []struct {
			Type  string         `json:"type"`
			ID    string         `json:"id"`
			Name  string         `json:"name"`
			Input map[string]any `json:"input"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("response body: %v", err)
	}

	if got.StopReason != "tool_use" {
		t.Fatalf("stop_reason = %q, want tool_use", got.StopReason)
	}
	var found bool
	for _, b := range got.Content {
		if b.Type != "tool_use" {
			continue
		}
		found = true
		if b.ID != "call_1" || b.Name != "get_weather" {
			t.Fatalf("tool_use block = %+v, want id call_1 name get_weather", b)
		}
		// OpenAI's JSON-string arguments must be parsed back into an object.
		if b.Input["city"] != "Paris" {
			t.Fatalf("tool_use input = %v, want city=Paris", b.Input)
		}
	}
	if !found {
		t.Fatalf("no tool_use block in the translated response: %s", rec.Body.String())
	}
}

// TestAnthropicStreamEmitsToolUseBlocks covers streaming tool calls, the path an
// agent actually runs on. OpenAI streams a call as an id/name fragment followed
// by argument fragments; the Anthropic consumer needs a tool_use
// content_block_start and input_json_delta events to reassemble it.
func TestAnthropicStreamEmitsToolUseBlocks(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{"role":"assistant","content":"Checking"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":6}}`,
		"",
		"data: [DONE]",
		"",
		"",
	}, "\n")

	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: streamResponse(sse)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","stream":true,"max_tokens":64,"tools":[{"name":"get_weather","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"weather?"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()

	for _, want := range []string{
		`"type":"tool_use"`,
		`"name":"get_weather"`,
		`"id":"call_1"`,
		`"type":"input_json_delta"`,
		`"stop_reason":"tool_use"`,
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("translated tool stream missing %q:\n%s", want, body)
		}
	}

	// The argument fragments must stream through verbatim so the consumer can
	// concatenate them back into the original JSON object.
	for _, want := range []string{`{\"city\":`, `\"Paris\"}`} {
		if !strings.Contains(body, want) {
			t.Fatalf("argument fragment %q missing:\n%s", want, body)
		}
	}

	// The text block opened first must be closed before the tool block opens —
	// Anthropic streams one open content block at a time.
	mustOrder(t, body,
		"event: message_start",
		`"type":"text_delta"`,
		"event: content_block_stop",
		`"type":"tool_use"`,
		`"type":"input_json_delta"`,
		"event: message_delta",
		"event: message_stop",
	)

	// Blocks must carry distinct indices, or the consumer merges them.
	if !strings.Contains(body, `"index":1`) {
		t.Fatalf("tool block did not get its own content-block index:\n%s", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("OpenAI terminator leaked into the Anthropic stream:\n%s", body)
	}
}

// TestAnthropicPreRoutingErrorUsesAnthropicEnvelope covers ADR-0019 on the
// Anthropic edge: an error raised before routing (here, an unknown model) must
// use the Anthropic error envelope. An OpenAI-shaped document is unparseable by
// an Anthropic SDK, and clients match retry behavior on the error type.
func TestAnthropicPreRoutingErrorUsesAnthropicEnvelope(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[]}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"no-such-model","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (%s)", rec.Code, rec.Body.String())
	}

	var got struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("error body: %v", err)
	}
	if got.Type != "error" {
		t.Fatalf("envelope type = %q, want \"error\": %s", got.Type, rec.Body.String())
	}
	if got.Error.Type != "not_found_error" {
		t.Fatalf("error type = %q, want not_found_error: %s", got.Error.Type, rec.Body.String())
	}
	if got.Error.Message == "" {
		t.Fatalf("error message dropped: %s", rec.Body.String())
	}
}

// TestAnthropicAuthErrorUsesAnthropicEnvelope covers the earliest pre-routing
// failure of all: rejection by the auth middleware. It runs before handleChat, so
// it must pick the envelope from the endpoint rather than a parsed request — a
// misconfigured Anthropic client's very first error is this one, and an
// OpenAI-shaped 401 is unparseable by its SDK (ADR-0009, ADR-0019).
func TestAnthropicAuthErrorUsesAnthropicEnvelope(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[]}`)}
	srv := newServer(t,
		map[string]router.Backend{"b1": be},
		map[string]*router.Alias{"alias-oai": proxyAlias("alias-oai", "up-oai", "b1")},
		snapshot(model.BackendState{Name: "b1", Healthy: true}),
		[]string{"sk-router-test"}, io.Discard,
	)

	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages",
		strings.NewReader(`{"model":"alias-oai","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401 (%s)", rec.Code, rec.Body.String())
	}
	var got struct {
		Type  string `json:"type"`
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("error body: %v", err)
	}
	if got.Type != "error" || got.Error.Type != "authentication_error" {
		t.Fatalf("401 on /v1/messages = %s, want Anthropic envelope with authentication_error", rec.Body.String())
	}

	// The OpenAI endpoint must keep the OpenAI error shape.
	rec = httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"alias-oai","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401 (%s)", rec.Code, rec.Body.String())
	}
	var oai struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &oai); err != nil || oai.Error.Code != "unauthorized" {
		t.Fatalf("401 on /v1/chat/completions = %s, want OpenAI envelope with code unauthorized", rec.Body.String())
	}
}

// TestToolCallsFinishWithoutCallsDowngradesStopReason pins the response contract
// the hard way: a degenerate upstream that reports finish_reason "tool_calls"
// without sending any tool call must NOT surface stop_reason "tool_use" — that
// stop reason obliges the router to also deliver the tool_use blocks it refers
// to, and there are none.
func TestToolCallsFinishWithoutCallsDowngradesStopReason(t *testing.T) {
	t.Run("unary", func(t *testing.T) {
		be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(
			`{"id":"c","model":"up-oai","choices":[{"index":0,"message":{"role":"assistant","content":"half an answer"},"finish_reason":"tool_calls"}]}`)}
		srv := toolServer(t, be)
		rec := srv.post(`{"model":"alias-oai","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
		}
		var got struct {
			StopReason string `json:"stop_reason"`
		}
		if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
			t.Fatalf("response body: %v", err)
		}
		if got.StopReason != "end_turn" {
			t.Fatalf("stop_reason = %q, want end_turn (no tool_use block was emitted): %s", got.StopReason, rec.Body.String())
		}
	})

	t.Run("stream", func(t *testing.T) {
		sse := strings.Join([]string{
			`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{"role":"assistant","content":"half"},"finish_reason":null}]}`,
			"",
			`data: {"id":"chatcmpl-x","model":"up-oai","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
			"",
			"data: [DONE]",
			"",
			"",
		}, "\n")
		be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: streamResponse(sse)}
		srv := toolServer(t, be)
		rec := srv.post(`{"model":"alias-oai","stream":true,"max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
		}
		body := rec.Body.String()
		if strings.Contains(body, `"stop_reason":"tool_use"`) {
			t.Fatalf("stream reported stop_reason tool_use without any tool_use block:\n%s", body)
		}
		if !strings.Contains(body, `"stop_reason":"end_turn"`) {
			t.Fatalf("stream did not downgrade to end_turn:\n%s", body)
		}
	})
}

// TestAnthropicImageToolResultSurvives is the regression guard for the bug that
// made every vision call fabricate. OpenAI's role:"tool" content is text-only, so
// an image returned by a tool used to be dropped silently — the model then
// answered about an image it never received, inventing a fluent, confident,
// entirely wrong result. Nothing errored, which is what made it dangerous.
//
// The image must therefore reach the backend as an image_url part on a
// role:"user" message that follows the tool reply, and the tool reply itself must
// still be present so every tool_call is answered.
func TestAnthropicImageToolResultSurvives(t *testing.T) {
	// 1x1 GIF — small enough to inline, real enough to round-trip.
	const b64 = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[{"index":0,"message":{"content":"a receipt"},"finish_reason":"stop"}]}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","max_tokens":64,"messages":[` +
		`{"role":"user","content":"read this receipt"},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Read","input":{"path":"/tmp/r.gif"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[` +
		`{"type":"image","source":{"type":"base64","media_type":"image/gif","data":"` + b64 + `"}}` +
		`]}]}` +
		`]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var sent struct {
		Messages []struct {
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			ToolCallID string          `json:"tool_call_id"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(be.lastBody, &sent); err != nil {
		t.Fatalf("forwarded body: %v", err)
	}

	// The pixels must be somewhere in the forwarded body. This is the assertion
	// that fails against the pre-fix translator.
	if !strings.Contains(string(be.lastBody), b64) {
		t.Fatalf("image data dropped from the forwarded body — the model would answer blind:\n%s", be.lastBody)
	}

	// Every tool_call still needs its tool reply, or the request is invalid.
	var toolMsg, userImageMsg int
	for _, m := range sent.Messages {
		if m.Role == "tool" && m.ToolCallID == "toolu_1" {
			toolMsg++
			if len(m.Content) == 0 || string(m.Content) == `""` {
				t.Fatalf("tool reply is empty; the model cannot tell the call succeeded: %s", be.lastBody)
			}
		}
		if m.Role == "user" && strings.Contains(string(m.Content), "image_url") {
			userImageMsg++
		}
	}
	if toolMsg != 1 {
		t.Fatalf("expected exactly one tool reply for toolu_1, got %d: %s", toolMsg, be.lastBody)
	}
	if userImageMsg != 1 {
		t.Fatalf("expected the image on one role:\"user\" message, got %d: %s", userImageMsg, be.lastBody)
	}

	// It must be a proper data: URI, not raw base64 — that is what an OpenAI
	// backend actually accepts.
	if !strings.Contains(string(be.lastBody), "data:image/gif;base64,") {
		t.Fatalf("image not encoded as a data: URI: %s", be.lastBody)
	}
}

// TestAnthropicTextToolResultUnchanged pins the common path: a text-only tool
// result must still produce exactly one role:"tool" message and no extra user
// turn. This is what the original tool-use work verified, and it must not
// regress while making room for images.
func TestAnthropicTextToolResultUnchanged(t *testing.T) {
	be := &fakeBackend{name: "b1", protocol: model.ProtocolOpenAI, fn: okResponse(`{"id":"c","model":"up-oai","choices":[{"index":0,"message":{"content":"ok"},"finish_reason":"stop"}]}`)}
	srv := toolServer(t, be)

	rec := srv.post(`{"model":"alias-oai","max_tokens":64,"messages":[` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"calc","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"42"}]}` +
		`]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	var sent struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(be.lastBody, &sent); err != nil {
		t.Fatalf("forwarded body: %v", err)
	}
	if len(sent.Messages) != 2 {
		t.Fatalf("text tool result should stay 2 messages (assistant+tool), got %d: %s", len(sent.Messages), be.lastBody)
	}
	if sent.Messages[1].Role != "tool" || !strings.Contains(string(sent.Messages[1].Content), "42") {
		t.Fatalf("text tool result lost: %s", be.lastBody)
	}
}

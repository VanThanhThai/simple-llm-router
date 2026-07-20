package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/mattbucci/simple-llm-router/internal/model"
	"github.com/mattbucci/simple-llm-router/internal/observability"
)

// ----- inbound adapter: Anthropic Messages -> canonical OpenAI shape ---------

// anthropicInMessage is the part of an Anthropic message the adapter reads.
type anthropicInMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// parseAnthropicRequest is the inbound adapter for Anthropic consumers. It
// translates the Anthropic Messages body into the canonical OpenAI-shaped Raw map
// the router operates on (ADR-0016): the top-level system field becomes a
// system-role message, stop_sequences becomes stop, and content blocks map to
// OpenAI content parts (ADR-0008). The response-side translation back to
// Anthropic shape is handled by anthropicSink.
func parseAnthropicRequest(body []byte) (*model.ChatRequest, error) {
	var in map[string]json.RawMessage
	if err := json.Unmarshal(body, &in); err != nil {
		return nil, model.ErrBadRequest("request body is not valid JSON")
	}

	out := map[string]json.RawMessage{}
	// ConsumerBody keeps the original Anthropic bytes verbatim so an Anthropic
	// consumer routed to an Anthropic backend takes the same-protocol native relay
	// (ADR-0016 full passthrough) instead of the lossy canonical translation; the
	// OpenAI-canonical out/Raw below is still populated for the cross-protocol case
	// (Anthropic consumer -> OpenAI backend).
	req := &model.ChatRequest{Consumer: model.ProtocolAnthropic, ConsumerBody: body}

	if m, ok := in["model"]; ok {
		out["model"] = m
		_ = json.Unmarshal(m, &req.Model)
	}
	if s, ok := in["stream"]; ok {
		out["stream"] = s
		_ = json.Unmarshal(s, &req.Stream)
	}
	if mt, ok := in["max_tokens"]; ok {
		out["max_tokens"] = mt
	}
	// Common sampling parameters carry over by the same name.
	for _, k := range []string{"temperature", "top_p"} {
		if v, ok := in[k]; ok {
			out[k] = v
		}
	}
	// Anthropic stop_sequences -> OpenAI stop.
	if ss, ok := in["stop_sequences"]; ok {
		out["stop"] = ss
	}
	// Tool use crosses the canonical boundary: Anthropic tool definitions and
	// tool_choice have direct OpenAI equivalents, so an agent loop survives the
	// cross-protocol path (ADR-0016 revised).
	if t, ok := in["tools"]; ok {
		if tools := anthropicToolsToOpenAI(t); len(tools) > 0 {
			if raw, err := json.Marshal(tools); err == nil {
				out["tools"] = raw
			}
		}
	}
	if tc, ok := in["tool_choice"]; ok {
		choice, parallel := anthropicToolChoiceToOpenAI(tc)
		if choice != nil {
			if raw, err := json.Marshal(choice); err == nil {
				out["tool_choice"] = raw
			}
		}
		// Anthropic spells "one tool at a time" as a flag on tool_choice;
		// OpenAI spells it as a top-level parameter.
		if parallel != nil {
			if raw, err := json.Marshal(*parallel); err == nil {
				out["parallel_tool_calls"] = raw
			}
		}
	}
	// Routing-control plugins are honored on either endpoint and stripped before
	// forwarding (ADR-0001).
	if p, ok := in["plugins"]; ok {
		req.Plugins = model.ParsePlugins(p)
	}

	var inMsgs []anthropicInMessage
	if m, ok := in["messages"]; ok {
		_ = json.Unmarshal(m, &inMsgs)
	}

	var systemText string
	if sys, ok := in["system"]; ok {
		systemText = anthropicSystemText(sys)
	}

	// Build only the OpenAI-canonical messages array (the single source of truth in
	// Raw). The Anthropic top-level system field is hoisted into a leading
	// system-role message; the parsed model.Message view, when a downstream path
	// needs it, is derived lazily from this via ChatRequest.CanonicalMessages — so
	// there is no per-message marshal->unmarshal round-trip here (ADR-0014, ADR-0016).
	oaiMsgs := make([]json.RawMessage, 0, len(inMsgs)+1)
	if systemText != "" {
		raw, _ := json.Marshal(map[string]any{"role": "system", "content": systemText})
		oaiMsgs = append(oaiMsgs, raw)
	}
	for _, m := range inMsgs {
		oaiMsgs = append(oaiMsgs, translateAnthropicMessage(m.Role, m.Content)...)
	}

	msgsRaw, _ := json.Marshal(oaiMsgs)
	out["messages"] = msgsRaw
	req.Raw = out
	return req, nil
}

// anthropicSystemText flattens an Anthropic system field (a string or an array of
// text blocks) to plain text for use as an OpenAI system message.
func anthropicSystemText(raw json.RawMessage) string {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || string(t) == "null" {
		return ""
	}
	switch t[0] {
	case '"':
		var s string
		_ = json.Unmarshal(t, &s)
		return s
	case '[':
		var blocks []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(t, &blocks); err == nil {
			var sb strings.Builder
			for _, b := range blocks {
				if b.Text == "" {
					continue
				}
				if sb.Len() > 0 {
					sb.WriteByte('\n')
				}
				sb.WriteString(b.Text)
			}
			return sb.String()
		}
	}
	return ""
}

// anthropicToolsToOpenAI maps Anthropic tool definitions to OpenAI function
// tools. The only real difference is the wrapper: Anthropic puts name /
// description / input_schema at the top level, OpenAI nests them under
// "function" and calls the schema "parameters".
func anthropicToolsToOpenAI(raw json.RawMessage) []any {
	var tools []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil
	}
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		fn := map[string]any{}
		var name string
		if v, ok := t["name"]; ok {
			_ = json.Unmarshal(v, &name)
		}
		if name == "" {
			continue // a nameless tool is not addressable; skip it
		}
		fn["name"] = name
		if v, ok := t["description"]; ok {
			var desc string
			if json.Unmarshal(v, &desc) == nil && desc != "" {
				fn["description"] = desc
			}
		}
		if v, ok := t["input_schema"]; ok {
			var schema any
			if json.Unmarshal(v, &schema) == nil && schema != nil {
				fn["parameters"] = schema
			}
		}
		out = append(out, map[string]any{"type": "function", "function": fn})
	}
	return out
}

// anthropicToolChoiceToOpenAI maps an Anthropic tool_choice to the OpenAI form,
// plus the OpenAI parallel_tool_calls parameter when Anthropic's
// disable_parallel_tool_use flag is set. A nil choice means "not expressible" —
// the caller then forwards nothing, leaving the provider default.
func anthropicToolChoiceToOpenAI(raw json.RawMessage) (choice any, parallel *bool) {
	var tc struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse *bool  `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, nil
	}
	if tc.DisableParallelToolUse != nil {
		enabled := !*tc.DisableParallelToolUse
		parallel = &enabled
	}
	switch tc.Type {
	case "auto":
		return "auto", parallel
	case "any":
		// Anthropic "any" = must call some tool; OpenAI spells that "required".
		return "required", parallel
	case "none":
		return "none", parallel
	case "tool":
		if tc.Name == "" {
			return nil, parallel
		}
		return map[string]any{"type": "function", "function": map[string]any{"name": tc.Name}}, parallel
	}
	return nil, parallel
}

// translateAnthropicMessage maps one Anthropic message to one or more OpenAI
// messages. Most turns map 1:1, but tool results do not: Anthropic carries them
// as tool_result blocks inside a user turn, whereas OpenAI requires a separate
// message per result with role "tool". A turn holding several tool_result blocks
// therefore fans out into several OpenAI messages (ADR-0016 revised).
func translateAnthropicMessage(role string, content json.RawMessage) []json.RawMessage {
	t := bytes.TrimSpace(content)
	if len(t) == 0 || t[0] != '[' {
		raw, _ := json.Marshal(map[string]any{"role": role, "content": translateAnthropicContent(content)})
		return []json.RawMessage{raw}
	}

	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(t, &blocks); err != nil {
		raw, _ := json.Marshal(map[string]any{"role": role, "content": ""})
		return []json.RawMessage{raw}
	}

	parts := make([]any, 0, len(blocks))
	toolCalls := make([]any, 0)
	toolMsgs := make([]json.RawMessage, 0)
	for _, b := range blocks {
		switch anthropicBlockType(b) {
		case "tool_use":
			if call := anthropicToolUseToOpenAI(b); call != nil {
				toolCalls = append(toolCalls, call)
			}
		case "tool_result":
			if msg := anthropicToolResultToOpenAI(b); msg != nil {
				toolMsgs = append(toolMsgs, msg)
			}
		default:
			parts = append(parts, translateAnthropicBlock(b))
		}
	}

	out := make([]json.RawMessage, 0, len(toolMsgs)+1)
	// Tool results must precede any prose in the same turn: OpenAI requires each
	// "tool" message to follow the assistant turn that requested it, and text the
	// user added alongside belongs after those results.
	out = append(out, toolMsgs...)
	if len(parts) > 0 || len(toolCalls) > 0 {
		m := map[string]any{"role": role}
		if len(parts) > 0 {
			m["content"] = parts
		} else {
			// An assistant turn that is purely tool calls carries no content;
			// OpenAI expects null rather than an empty array here.
			m["content"] = nil
		}
		if len(toolCalls) > 0 {
			m["tool_calls"] = toolCalls
		}
		raw, _ := json.Marshal(m)
		out = append(out, raw)
	}
	if len(out) == 0 {
		raw, _ := json.Marshal(map[string]any{"role": role, "content": ""})
		out = append(out, raw)
	}
	return out
}

// anthropicBlockType reads a content block's "type" discriminator.
func anthropicBlockType(b map[string]json.RawMessage) string {
	var typ string
	if t, ok := b["type"]; ok {
		_ = json.Unmarshal(t, &typ)
	}
	return typ
}

// anthropicToolUseToOpenAI maps a tool_use block to an OpenAI tool call. The
// structured "input" object becomes OpenAI's JSON-string "arguments".
func anthropicToolUseToOpenAI(b map[string]json.RawMessage) any {
	var id, name string
	if v, ok := b["id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	if v, ok := b["name"]; ok {
		_ = json.Unmarshal(v, &name)
	}
	if name == "" {
		return nil
	}
	args := "{}"
	if v, ok := b["input"]; ok && len(bytes.TrimSpace(v)) > 0 {
		args = string(v)
	}
	return map[string]any{
		"id":       id,
		"type":     "function",
		"function": map[string]any{"name": name, "arguments": args},
	}
}

// anthropicToolResultToOpenAI maps a tool_result block to an OpenAI "tool"
// message. OpenAI's tool content is plain text, so block content is flattened.
func anthropicToolResultToOpenAI(b map[string]json.RawMessage) json.RawMessage {
	var id string
	if v, ok := b["tool_use_id"]; ok {
		_ = json.Unmarshal(v, &id)
	}
	if id == "" {
		return nil
	}
	text := ""
	if v, ok := b["content"]; ok {
		text = anthropicToolResultText(v)
	}
	raw, err := json.Marshal(map[string]any{
		"role":         "tool",
		"tool_call_id": id,
		"content":      text,
	})
	if err != nil {
		return nil
	}
	return raw
}

// anthropicToolResultText flattens tool_result content — a string, or an array of
// blocks — to the plain text OpenAI expects. Non-text blocks (e.g. an image
// result) are skipped rather than rendered as JSON noise.
func anthropicToolResultText(raw json.RawMessage) string {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || string(t) == "null" {
		return ""
	}
	if t[0] == '"' {
		var s string
		_ = json.Unmarshal(t, &s)
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(t, &blocks); err != nil {
		return string(t)
	}
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type != "text" || b.Text == "" {
			continue
		}
		if sb.Len() > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString(b.Text)
	}
	return sb.String()
}

// translateAnthropicContent maps Anthropic message content (string or content
// blocks) to the OpenAI content form (string or content parts). Unknown block
// types are passed through best-effort rather than dropped (ADR-0008, ADR-0016).
func translateAnthropicContent(raw json.RawMessage) any {
	t := bytes.TrimSpace(raw)
	if len(t) == 0 || string(t) == "null" {
		return ""
	}
	switch t[0] {
	case '"':
		var s string
		if err := json.Unmarshal(t, &s); err == nil {
			return s
		}
		return ""
	case '[':
		var blocks []map[string]json.RawMessage
		if err := json.Unmarshal(t, &blocks); err != nil {
			return ""
		}
		parts := make([]any, 0, len(blocks))
		for _, b := range blocks {
			parts = append(parts, translateAnthropicBlock(b))
		}
		return parts
	default:
		var s string
		if err := json.Unmarshal(t, &s); err == nil {
			return s
		}
		return ""
	}
}

// translateAnthropicBlock maps one Anthropic content block to an OpenAI content
// part. text -> text; image (base64/url source) -> image_url; anything else is
// reproduced as a generic object so nothing is silently dropped.
func translateAnthropicBlock(b map[string]json.RawMessage) any {
	var typ string
	if t, ok := b["type"]; ok {
		_ = json.Unmarshal(t, &typ)
	}
	switch typ {
	case "text":
		var text string
		if t, ok := b["text"]; ok {
			_ = json.Unmarshal(t, &text)
		}
		return map[string]any{"type": "text", "text": text}
	case "image":
		if src, ok := b["source"]; ok {
			var s struct {
				Type      string `json:"type"`
				MediaType string `json:"media_type"`
				Data      string `json:"data"`
				URL       string `json:"url"`
			}
			if err := json.Unmarshal(src, &s); err == nil {
				url := s.URL
				if s.Type == "base64" && s.Data != "" {
					url = "data:" + s.MediaType + ";base64," + s.Data
				}
				return map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}}
			}
		}
		return map[string]any{"type": "text", "text": ""}
	default:
		generic := make(map[string]any, len(b))
		for k, v := range b {
			var anyv any
			_ = json.Unmarshal(v, &anyv)
			generic[k] = anyv
		}
		return generic
	}
}

// ----- response sink: OpenAI-canonical output -> Anthropic Messages shape ----

// anthropicSink translates the OpenAI-canonical output the router produces into
// the Anthropic Messages response shape (ADR-0016), for both unary and SSE.
// Streaming translation is stateful within a single stream (ADR-0007): it tracks
// whether the message has been opened and what stop reason/usage to report.
type anthropicSink struct {
	w     http.ResponseWriter
	rc    *http.ResponseController
	wrote bool
	extra map[string]string

	started bool
	msgID   string
	model   string
	stop    string
	out     int
	in      int

	// Content-block bookkeeping. An Anthropic stream is a sequence of indexed
	// blocks, each opened and closed in turn, whereas OpenAI streams text and
	// tool-call fragments interleaved on one channel — so the sink allocates
	// indices and tracks which block is currently open. Single-goroutine per
	// request, so plain fields suffice (ADR-0015).
	nextIndex int         // next block index to allocate
	textIndex int         // index of the text block
	textOpen  bool        // is the text block currently open?
	toolIndex map[int]int // OpenAI tool_call index -> Anthropic block index
	openTool  int         // currently open tool block, -1 when none
}

// newAnthropicSink builds a translating sink over w.
func newAnthropicSink(w http.ResponseWriter) *anthropicSink {
	return &anthropicSink{w: w, rc: http.NewResponseController(w), openTool: -1}
}

// Wrote reports whether any byte has reached the client yet.
func (s *anthropicSink) Wrote() bool { return s.wrote }

// SetHeader buffers a router-set response header (e.g. X-Router-Model) to be
// emitted when the response is committed. Single-goroutine per request, so no
// synchronization is needed (ADR-0015).
func (s *anthropicSink) SetHeader(key, value string) {
	if s.extra == nil {
		s.extra = make(map[string]string, 2)
	}
	s.extra[key] = value
}

// WriteResponse translates a complete OpenAI chat completion (or error) into the
// Anthropic unary response shape (ADR-0016).
func (s *anthropicSink) WriteResponse(status int, header http.Header, body []byte) error {
	s.w.Header().Set("Content-Type", "application/json")
	applyExtraHeaders(s.w.Header(), s.extra)
	s.w.WriteHeader(status)
	s.wrote = true
	if status >= 400 {
		_, err := s.w.Write(anthropicErrorBody(status, body))
		return err
	}
	_, err := s.w.Write(translateCompletionToAnthropic(body))
	return err
}

// WriteRawResponse relays a complete upstream response verbatim — no translation —
// for the same-protocol native relay path (Anthropic->Anthropic full passthrough,
// ADR-0016/ADR-0001). The reply is already Anthropic-shaped, so the router's
// double-translation is bypassed and the upstream status/Content-Type/body are
// carried through unchanged, preserving tools, tool_choice, top_k, metadata, and
// cache_control. It is part of the router.ResponseSink raw-relay surface.
func (s *anthropicSink) WriteRawResponse(status int, header http.Header, body []byte) error {
	ct := ""
	if header != nil {
		ct = header.Get("Content-Type")
	}
	if ct == "" {
		ct = "application/json"
	}
	s.w.Header().Set("Content-Type", ct)
	applyExtraHeaders(s.w.Header(), s.extra)
	s.w.WriteHeader(status)
	s.wrote = true
	_, err := s.w.Write(body)
	return err
}

// StartRawStream commits the SSE response for the native relay path (ADR-0007).
// Unlike StartStream it does not arm the translating stream machinery: the
// upstream's own Anthropic event stream is copied through verbatim by WriteRawChunk.
func (s *anthropicSink) StartRawStream() error {
	setStreamHeaders(s.w.Header())
	applyExtraHeaders(s.w.Header(), s.extra)
	s.w.WriteHeader(http.StatusOK)
	s.wrote = true
	_ = s.rc.Flush()
	return nil
}

// WriteRawChunk relays raw upstream SSE bytes to the consumer verbatim and flushes
// (no reframing, no translation), preserving the Anthropic "event:"/"data:" framing
// the consumer's SDK expects on the same-protocol native path (ADR-0007/ADR-0016).
func (s *anthropicSink) WriteRawChunk(p []byte) error {
	s.wrote = true
	if _, err := s.w.Write(p); err != nil {
		return err
	}
	_ = s.rc.Flush()
	return nil
}

// StartStream writes the SSE response headers and flushes (ADR-0007). Anthropic
// stream events are emitted lazily once the first chunk arrives, since the model
// id is only known then.
func (s *anthropicSink) StartStream() error {
	setStreamHeaders(s.w.Header())
	applyExtraHeaders(s.w.Header(), s.extra)
	s.w.WriteHeader(http.StatusOK)
	s.wrote = true
	_ = s.rc.Flush()
	return nil
}

// WriteEvent translates one OpenAI streaming chunk into Anthropic stream events
// (ADR-0007, ADR-0016). The OpenAI [DONE] sentinel is skipped; EndStream emits
// the Anthropic terminators.
func (s *anthropicSink) WriteEvent(data []byte) error {
	if isDone(data) {
		return nil
	}
	var chunk oaiChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil // ignore an unparseable chunk best-effort
	}
	if chunk.Model != "" {
		s.model = chunk.Model
	}
	if !s.started {
		if err := s.emitStart(); err != nil {
			return err
		}
	}
	if len(chunk.Choices) > 0 {
		ch := chunk.Choices[0]
		if text := contentText(ch.Delta.Content); text != "" {
			if err := s.writeTextDelta(text); err != nil {
				return err
			}
		}
		for _, tc := range ch.Delta.ToolCalls {
			if err := s.writeToolCallDelta(tc); err != nil {
				return err
			}
		}
		if ch.FinishReason != nil && *ch.FinishReason != "" {
			s.stop = openAIFinishToAnthropicStop(*ch.FinishReason)
		}
	}
	if chunk.Usage != nil {
		if chunk.Usage.CompletionTokens > 0 {
			s.out = chunk.Usage.CompletionTokens
		}
		if chunk.Usage.PromptTokens > 0 {
			s.in = chunk.Usage.PromptTokens
		}
	}
	return nil
}

// EndStream emits the Anthropic stream terminators (ADR-0007).
func (s *anthropicSink) EndStream() error {
	if !s.started {
		if err := s.emitStart(); err != nil {
			return err
		}
	}
	// Close whatever is still open, in the order it was opened.
	for s.textOpen || s.openTool >= 0 {
		if err := s.closeOpenBlock(); err != nil {
			return err
		}
	}
	stop := s.stop
	if stop == "" {
		stop = "end_turn"
	}
	if err := s.emit("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": s.out},
	}); err != nil {
		return err
	}
	return s.emit("message_stop", map[string]any{"type": "message_stop"})
}

// emitStart opens the Anthropic message: message_start + content_block_start.
func (s *anthropicSink) emitStart() error {
	s.started = true
	if s.msgID == "" {
		s.msgID = "msg_" + observability.NewRequestID()
	}
	if err := s.emit("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            s.msgID,
			"type":          "message",
			"role":          "assistant",
			"model":         s.model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         map[string]any{"input_tokens": s.in, "output_tokens": 0},
		},
	}); err != nil {
		return err
	}
	s.textIndex = 0
	s.textOpen = true
	s.nextIndex = 1
	return s.emit("content_block_start", map[string]any{
		"type":          "content_block_start",
		"index":         0,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

// closeOpenBlock emits content_block_stop for whichever block is open, if any.
// Anthropic closes a block before the next one opens, so this runs whenever the
// stream switches between text and a tool call, and again at EndStream.
func (s *anthropicSink) closeOpenBlock() error {
	switch {
	case s.textOpen:
		s.textOpen = false
		return s.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": s.textIndex})
	case s.openTool >= 0:
		idx := s.openTool
		s.openTool = -1
		return s.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": idx})
	}
	return nil
}

// writeTextDelta appends to the text block, reopening a fresh one if a tool call
// has closed it in the meantime (a provider may interleave prose and calls).
func (s *anthropicSink) writeTextDelta(text string) error {
	if !s.textOpen {
		if err := s.closeOpenBlock(); err != nil {
			return err
		}
		s.textIndex = s.nextIndex
		s.nextIndex++
		if err := s.emit("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         s.textIndex,
			"content_block": map[string]any{"type": "text", "text": ""},
		}); err != nil {
			return err
		}
		s.textOpen = true
	}
	return s.emit("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": s.textIndex,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

// writeToolCallDelta translates one OpenAI tool-call fragment. The first fragment
// for a given OpenAI index opens an Anthropic tool_use block; subsequent
// fragments stream the accumulating arguments as input_json_delta, which is how
// an Anthropic consumer reconstructs the call's input object.
func (s *anthropicSink) writeToolCallDelta(tc oaiToolCallDelta) error {
	if s.toolIndex == nil {
		s.toolIndex = make(map[int]int, 2)
	}
	idx, seen := s.toolIndex[tc.Index]
	if !seen {
		if err := s.closeOpenBlock(); err != nil {
			return err
		}
		idx = s.nextIndex
		s.nextIndex++
		s.toolIndex[tc.Index] = idx
		if err := s.emit("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    tc.ID,
				"name":  tc.Function.Name,
				"input": map[string]any{},
			},
		}); err != nil {
			return err
		}
		s.openTool = idx
		// A tool call means the turn ends in tool_use, even if the provider
		// never sends a finish_reason (some stop the stream abruptly).
		s.stop = "tool_use"
	}
	if tc.Function.Arguments == "" {
		return nil
	}
	return s.emit("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": idx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": tc.Function.Arguments},
	})
}

// emit writes one Anthropic SSE event ("event:" + "data:") and flushes.
func (s *anthropicSink) emit(event string, payload any) error {
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	s.wrote = true
	if _, err := io.WriteString(s.w, "event: "+event+"\n"); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := s.w.Write(b); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n\n")); err != nil {
		return err
	}
	_ = s.rc.Flush()
	return nil
}

// ----- OpenAI response shapes the translation reads -------------------------

// oaiToolCall is a complete tool call on a unary OpenAI response.
type oaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// oaiToolCallDelta is a streaming fragment of a tool call. OpenAI splits one call
// across many chunks — the id and name arrive once, then "arguments" accumulates
// a character at a time — and keys the fragments by Index.
type oaiToolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oaiCompletion struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content   json.RawMessage `json:"content"`
			ToolCalls []oaiToolCall   `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

type oaiChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content   json.RawMessage    `json:"content"`
			ToolCalls []oaiToolCallDelta `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// translateCompletionToAnthropic converts a unary OpenAI completion into the
// Anthropic Messages response shape (best-effort, ADR-0016).
func translateCompletionToAnthropic(body []byte) []byte {
	var comp oaiCompletion
	_ = json.Unmarshal(body, &comp)

	id := comp.ID
	if id == "" {
		id = "msg_" + observability.NewRequestID()
	} else {
		id = "msg_" + strings.TrimPrefix(id, "chatcmpl-")
	}

	text := ""
	stop := "end_turn"
	var toolCalls []oaiToolCall
	if len(comp.Choices) > 0 {
		text = contentText(comp.Choices[0].Message.Content)
		stop = openAIFinishToAnthropicStop(comp.Choices[0].FinishReason)
		toolCalls = comp.Choices[0].Message.ToolCalls
	}

	// Text first, then one tool_use block per call — the block order Anthropic
	// consumers expect. Emitting stop_reason "tool_use" with no tool_use block
	// would be a malformed response, so the two are derived together.
	content := make([]any, 0, len(toolCalls)+1)
	if text != "" || len(toolCalls) == 0 {
		content = append(content, map[string]any{"type": "text", "text": text})
	}
	for _, tc := range toolCalls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Function.Name,
			"input": toolArgumentsToInput(tc.Function.Arguments),
		})
	}
	if len(toolCalls) > 0 {
		stop = "tool_use"
	}

	out, err := json.Marshal(map[string]any{
		"id":            id,
		"type":          "message",
		"role":          "assistant",
		"model":         comp.Model,
		"content":       content,
		"stop_reason":   stop,
		"stop_sequence": nil,
		"usage": map[string]any{
			"input_tokens":  comp.Usage.PromptTokens,
			"output_tokens": comp.Usage.CompletionTokens,
		},
	})
	if err != nil {
		return body
	}
	return out
}

// anthropicErrorBody wraps an OpenAI-style error into the Anthropic error
// envelope (best-effort, ADR-0016/ADR-0019). Anthropic clients match on the
// error type, so it is derived from the status rather than always reported as a
// generic api_error.
func anthropicErrorBody(status int, body []byte) []byte {
	msg := "request failed"
	var e struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &e); err == nil && e.Error.Message != "" {
		msg = e.Error.Message
	}
	out, err := json.Marshal(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": anthropicErrorType(status), "message": msg},
	})
	if err != nil {
		return body
	}
	return out
}

// anthropicErrorType maps an HTTP status to the Anthropic error type string.
func anthropicErrorType(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// contentText extracts a plain-text view of an OpenAI content value (string or
// parts), never assuming it is a string (ADR-0008).
func contentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var c model.Content
	if err := json.Unmarshal(raw, &c); err != nil {
		return ""
	}
	return c.Text()
}

// toolArgumentsToInput turns an OpenAI tool call's JSON-string "arguments" into
// the structured object Anthropic's tool_use "input" expects. Providers
// occasionally emit an empty or malformed string; an empty object keeps the
// response well-formed rather than failing the whole translation.
func toolArgumentsToInput(arguments string) any {
	if strings.TrimSpace(arguments) == "" {
		return map[string]any{}
	}
	var input any
	if err := json.Unmarshal([]byte(arguments), &input); err != nil || input == nil {
		return map[string]any{}
	}
	return input
}

// openAIFinishToAnthropicStop maps an OpenAI finish_reason to the Anthropic
// stop_reason.
func openAIFinishToAnthropicStop(finish string) string {
	switch finish {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	default:
		// "stop", "content_filter", "", and anything unknown collapse to the
		// natural end (best-effort, ADR-0016).
		return "end_turn"
	}
}

package bridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"kolibri-llm/internal/bridge/ops"
)

// Handler returns the bridge's HTTP API.
//
//	POST /v1/chat/completions          OpenAI's request, without tools: the bridge brings its own
//	POST /v1/executions/{id}/cancel    ends a request; with {"call_id":"t1c0"} only that operation
//	GET  /v1/bridge/tools              the tools with their output schemas and the limits
//	GET  /health
func (b *Bridge) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", b.handleChat)
	mux.HandleFunc("POST /v1/executions/{id}/cancel", b.handleCancel)
	mux.HandleFunc("GET /v1/bridge/tools", b.handleTools)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"status":"ok"}`)
	})
	return mux
}

// statusOf maps a stop reason to the status of a non-streaming response.
func statusOf(reason StopReason) int {
	switch reason {
	case StopCompleted:
		return http.StatusOK
	case StopTurnLimit, StopCallLimit, StopTruncated, StopUnparsedCall:
		return http.StatusUnprocessableEntity
	case StopTurnTimeout, StopRequestTimeout:
		return http.StatusGatewayTimeout
	case StopCancelled:
		return 499 // nginx's "client closed request": no standard status says this
	case StopInternal:
		return http.StatusInternalServerError
	}
	return http.StatusBadGateway
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// errorBody has the shape of llama-server's error object, and a stop reason in its type.
func errorBody(status int, kind, message string) map[string]any {
	return map[string]any{"error": map[string]any{"code": status, "type": kind, "message": message}}
}

func (b *Bridge) handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, b.cfg.Limits.MaxRequestBytes))
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeJSON(w, http.StatusRequestEntityTooLarge, errorBody(http.StatusRequestEntityTooLarge, "request_too_large",
				fmt.Sprintf("the body is past %d bytes", b.cfg.Limits.MaxRequestBytes)))
			return
		}
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", err.Error()))
		return
	}
	c, reqErr := b.parseChat(body)
	if reqErr != nil {
		writeJSON(w, reqErr.Status, errorBody(reqErr.Status, reqErr.Type, reqErr.Message))
		return
	}
	id := r.Header.Get("X-Request-Id")
	if c.stream {
		b.serveStream(w, r, id, c)
		return
	}
	res, err := b.start(r.Context(), id, c, Callbacks{})
	if err != nil {
		var refused *RequestError
		if errors.As(err, &refused) {
			writeJSON(w, refused.Status, errorBody(refused.Status, refused.Type, refused.Message))
			return
		}
		writeJSON(w, http.StatusInternalServerError, errorBody(http.StatusInternalServerError, "internal", err.Error()))
		return
	}
	if res.Stop != StopCompleted {
		out := errorBody(statusOf(res.Stop), string(res.Stop), res.Failure.Message)
		out["kolibri_bridge"] = res
		writeJSON(w, statusOf(res.Stop), out)
		return
	}
	writeJSON(w, http.StatusOK, completion(res))
}

// completion is OpenAI's chat.completion of a finished request, and the bridge's account of it.
func completion(res *Result) map[string]any {
	message := map[string]any{"role": "assistant", "content": res.Content}
	if res.Reasoning != "" {
		message["reasoning_content"] = res.Reasoning
	}
	return map[string]any{
		"id": res.RequestID, "object": "chat.completion", "created": time.Now().Unix(), "model": res.Model,
		"choices": []map[string]any{{"index": 0, "finish_reason": res.Finish, "message": message}},
		"usage": map[string]any{
			"prompt_tokens": res.Usage.PromptTokens, "completion_tokens": res.Usage.CompletionTokens, "total_tokens": res.Usage.TotalTokens,
			"prompt_tokens_details": map[string]any{"cached_tokens": res.Usage.CachedTokens},
		},
		"kolibri_bridge": res,
	}
}

// sseWriter serializes events: deltas and trace events come from the run's goroutine, a cancel call's
// trace event from another one. It sends the headers with the first event.
type sseWriter struct {
	mu      sync.Mutex
	w       http.ResponseWriter
	started bool
}

func (s *sseWriter) begin() {
	if !s.started {
		s.started = true
		s.w.Header().Set("Content-Type", "text/event-stream")
		s.w.Header().Set("X-Accel-Buffering", "no")
		s.w.WriteHeader(http.StatusOK)
	}
}

func (s *sseWriter) write(text string) {
	s.begin()
	_, _ = io.WriteString(s.w, text) // a client that left cancels the request through its context
	if f, ok := s.w.(http.Flusher); ok {
		f.Flush()
	}
}

func (s *sseWriter) event(name string, v any) {
	encoded, err := json.Marshal(v)
	if err != nil {
		return
	}
	var b strings.Builder
	if name != "" {
		fmt.Fprintf(&b, "event: %s\n", name)
	}
	fmt.Fprintf(&b, "data: %s\n\n", encoded)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.write(b.String())
}

func (s *sseWriter) done() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.write("data: [DONE]\n\n")
}

// refuse answers a refused request. Refusal comes before the first event, so a status still fits.
func (s *sseWriter) refuse(status int, body any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	writeJSON(s.w, status, body)
}

func chunkOf(id, model string, delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion.chunk", "created": time.Now().Unix(), "model": model,
		"choices": []map[string]any{{"index": 0, "delta": delta, "finish_reason": finish}},
	}
}

// serveStream answers like llama-server: chunks of the text, a finish chunk and [DONE], or an error
// event and no [DONE]. The trace goes along as named "kolibri_bridge" events, which a client that
// only looks at unnamed data can ignore. A failure the client should see as a status has to happen
// before the first byte; once the stream is open it is an error event.
func (b *Bridge) serveStream(w http.ResponseWriter, r *http.Request, id string, c *chat) {
	out := &sseWriter{w: w}
	res, err := b.start(r.Context(), id, c, Callbacks{
		Delta: func(d Delta) {
			delta := map[string]any{}
			if d.Content != "" {
				delta["content"] = d.Content
			}
			if d.Reasoning != "" {
				delta["reasoning_content"] = d.Reasoning
			}
			out.event("", chunkOf(d.RequestID, d.Model, delta, nil))
		},
		Event: func(e Event) { out.event("kolibri_bridge", e) },
	})
	var refused *RequestError
	switch {
	case errors.As(err, &refused):
		out.refuse(refused.Status, errorBody(refused.Status, refused.Type, refused.Message))
	case err != nil:
		out.refuse(http.StatusInternalServerError, errorBody(http.StatusInternalServerError, "internal", err.Error()))
	case res.Stop != StopCompleted:
		out.event("", errorBody(statusOf(res.Stop), string(res.Stop), res.Failure.Message))
	default:
		out.event("", chunkOf(res.RequestID, res.Model, map[string]any{}, res.Finish))
		out.done()
	}
}

func (b *Bridge) handleCancel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	var req struct {
		CallID string `json:"call_id"`
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 4<<10))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", err.Error()))
		return
	}
	if len(raw) > 0 && json.Unmarshal(raw, &req) != nil {
		writeJSON(w, http.StatusBadRequest, errorBody(http.StatusBadRequest, "invalid_request_error", "the body is not JSON"))
		return
	}
	if req.CallID != "" {
		cancelled, err := b.CancelCall(id, req.CallID)
		if err != nil {
			writeJSON(w, http.StatusNotFound, errorBody(http.StatusNotFound, "not_found", err.Error()))
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"request_id": id, "call_id": req.CallID, "cancelled": cancelled})
		return
	}
	if !b.Cancel(id) {
		writeJSON(w, http.StatusNotFound, errorBody(http.StatusNotFound, "not_found", fmt.Sprintf("no running request %q", id)))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"request_id": id, "cancelled": true})
}

// Manifest is what the bridge offers: the protocol version, the limits and each operation's schemas.
func (b *Bridge) Manifest() map[string]any {
	var list []map[string]any
	for _, o := range b.reg.Allowed() {
		list = append(list, map[string]any{"name": o.Name, "description": o.Description, "input": o.Input(), "output": o.Output()})
	}
	lim := b.cfg.Limits
	return map[string]any{
		"protocol": ops.Version,
		"limits": map[string]any{
			"max_turns": lim.MaxTurns, "max_calls": lim.MaxCalls, "max_calls_per_turn": lim.MaxCallsPerTurn,
			"max_turn_tokens": lim.MaxTurnTokens, "max_turn_bytes": lim.MaxTurnBytes,
			"op_timeout_ms": lim.OpTimeout.Milliseconds(), "turn_timeout_ms": lim.TurnTimeout.Milliseconds(),
			"request_timeout_ms": lim.RequestTimeout.Milliseconds(),
			"max_arg_bytes":      b.cfg.OpLimits.MaxArgBytes, "max_int_bits": b.cfg.OpLimits.MaxIntBits,
			"max_array_len": b.cfg.OpLimits.MaxArrayLen, "max_result_bytes": b.cfg.OpLimits.MaxResultBytes,
		},
		"ops": list,
	}
}

func (b *Bridge) handleTools(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, b.Manifest())
}

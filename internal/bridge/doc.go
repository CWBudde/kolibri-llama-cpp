// Package bridge runs deterministic operations for a model behind llama-server's OpenAI-compatible
// API (issue #59, Phase A). It is runtime tool execution, not hidden-state integration: a model
// reaches an operation only by writing a tool call into its output, and reads the result only as
// text in its next prompt. Nothing here patches llama.cpp or touches the weights.
//
// A request goes through turns. A turn is one streaming completion. llama-server parses the
// model's <tool_call> blocks into tool_calls, and a turn that ends with finish_reason "tool_calls"
// is the only thing that executes anything. Then, per call, the bridge runs the operation through
// internal/coproc, appends the result as a tool message and starts the next turn, until the model
// answers without a call or a limit stops it.
//
// "Pause" is the end of an HTTP response and "resume" a new request whose prompt llama-server
// renders again from the messages. The generation is not suspended in place, whatever of the KV
// cache survives is what the rendered prompt shares with the last one.
//
// The bridge fails closed. A call counts only if the turn that carried it finished cleanly: a cut
// stream, a turn out of tokens and a stream that ends without [DONE] execute nothing, even when the
// arguments received so far look complete. A failed operation goes back to the model as a result
// with a machine-readable code. A failed turn ends the request with a machine-readable stop reason.
package bridge

# Execution bridge

**Result: the bridge of issue #59 exists and runs tool calls end to end through real llama-server on random tiny weights, but it has not met a real model. The issue's acceptance criterion, unmodified Kolibri completing a deterministic multi-step task through the bridge, has NOT been shown.** The test for it is written, is skipped by default and has not run on a real weight.

This is runtime tool execution, not hidden-state integration. A model reaches an operation only by writing a tool call into its output, and reads the result only as text in its next prompt. Nothing here patches llama.cpp, touches the weights or reads a hidden state.

## What exists and what doesn't

| Task of #59 | State |
|---|---|
| Inference architecture and insertion point | done: [Where it sits](#where-it-sits) |
| Versioned protocol: request IDs, typed input and output, errors, cancellation | done: `internal/bridge/ops`, `POST /v1/executions/{id}/cancel` |
| Detect a complete request, pause, execute, feed back, resume | done against the fake and against real llama-server on tiny weights. **Not on a real model** |
| Deterministic operations: arithmetic, arrays, hashing | done, 15 operations |
| Optional WebAssembly, FFT or PDE plugins | **not done.** No code runs here that isn't in this repo |
| Allowlist, limits, no filesystem or network, trace | done, with the limits named below. **No OS sandbox** |
| Tests: malformed requests, timeouts, reproducibility, multi-step calls | done, with `-race` |
| Unmodified Kolibri completes a multi-step task | **NOT shown.** `TestAcceptanceRealModel` is written and has not run |
| PLAN's benchmark of correctness, latency, token overhead and failure recovery | **not done.** No timing in this document is a result |

The fake shows the bridge handles what a server sends. The tiny checkpoint shows real llama-server accepts what the bridge sends. Neither shows that Kolibri plans these calls, copies 36 digits correctly or reads a result the right way. Only the real run can.

## Where it sits

```
client ──HTTP──> kolibri-bridge ──HTTP, streaming──> llama-server --jinja
                   │  plan: internal/coproc              template, grammar, parser
                   └─ operations: internal/bridge/ops
```

The bridge is a client of `/v1/chat/completions`, and a server of the same API. It adds the tools and runs the loop. A completion holds the last turn's text only. A stream carries the text of every turn as it arrives, so that a client sees `Let me add.3` where a completion says `3`. A client that wants one answer asks for a completion, or reads the turn boundaries from the trace events.

**Why the tool-call path:** Kolibri's template, `<tool_call>` tokens, grammar and parser already work ([chat.md](chat.md)), and the real checkpoint has returned a parsed call through them ([real-checkpoint.md](real-checkpoint.md)). A new special token would need tokenizer and template work. A hook in llama-server's slot loop would be a patch to carry, and would only change what a pause costs, not what the model can do.

**What that costs:** a pause isn't a suspended generation. It is the end of an HTTP response, and a resume is a new request whose prompt llama-server renders again from the messages. What of the KV cache survives is the prefix the two prompts share. On the tiny checkpoint the second turn of a loop reported 230 of 320 prompt tokens cached. The template also writes a call's arguments back normalized, not as generated, so everything from there on is decoded again. `coproc.Generator` is the seam where a real in-place pause would go; here a `Step` is a whole completion, so the scheduler's "no token while BLOCKED" holds per turn, not per token.

## How a request runs

A **turn** is one streaming completion, a **call** is one operation.

1. The bridge sends the messages, the tools and `stream: true`.
2. It reads the stream to `[DONE]`. Text goes on to a streaming client as it arrives. Fragments of a call are only collected.
3. The turn's outcome decides:

| The turn ended with | The request |
|---|---|
| `finish_reason: "tool_calls"` and calls | runs them, if the limits allow |
| `finish_reason: "stop"`, no calls | ends as `completed`, unless the content has `<tool_call>` in it (`unparsed_call`) |
| `finish_reason: "length"` | ends as `truncated`. **No call runs,** even one that looks complete |
| no `[DONE]`, a cut connection | ends as `stream_broken`. No call runs |
| an error event or an HTTP error | ends as `upstream_error` or `upstream_http`, with the server's error type |

4. For each call in order: `coproc.Execute`, and then for each in order `Wait` and `ReadResult`. Calls of one turn run concurrently. Their results reach the model in call order, whichever finishes first, because the template doesn't render `tool_call_id`: order is all a model has to go by.
5. `ReadResult` delivers a tool message, and the next turn starts.

The bridge replaces the server's random call IDs with `t<turn>c<index>`, so that the bytes sent upstream reproduce. It sends `reasoning_content` back, or the template renders an empty think block where the model wrote one.

## Protocol, version 1

An operation is an OpenAI tool of the same name, so the grammar constrains each one's arguments. Its input is the arguments object; it may also hold `"v": 1`, and any other value is `unsupported_version`. What the model reads, always one of:

```json
{"v":1,"id":"t1c0","op":"int_mul","ok":true,"result":{"value":"121932631356500531347203169112635269"}}
{"v":1,"id":"t1c0","op":"int_div","ok":false,"error":{"code":"division_by_zero","message":"the divisor is 0"}}
```

`op` appears only if it names a registered operation. Encoded responses are deterministic, and they escape `<`, so that text a model wrote can't close the template's `<tool_response>` from inside an error message.

| Error code | Meaning |
|---|---|
| `malformed_arguments` | not one JSON object: bad syntax, invalid UTF-8, a duplicate key, nesting past the limit, trailing data |
| `invalid_argument` | valid JSON, wrong field, type or value |
| `unknown_op`, `op_not_allowed` | no such operation; it exists but the allowlist excludes it |
| `unsupported_version` | the request names another version |
| `input_too_large`, `result_too_large` | past a limit |
| `overflow` | an integer result past `MaxIntBits` |
| `division_by_zero` | the divisor is 0 |
| `timeout`, `cancelled` | past the deadline; the operation or its request was cancelled |
| `internal` | a bug here, such as a panic in an operation |

A failed operation goes back to the model, which may repair it. A failed turn ends the request. The stop reasons are `completed`, `turn_limit`, `call_limit`, `truncated`, `unparsed_call`, `stream_broken`, `upstream_unreachable`, `upstream_http`, `upstream_error`, `bad_upstream`, `response_too_large`, `turn_timeout`, `request_timeout`, `cancelled` and `internal`.

**HTTP.** `POST /v1/chat/completions` takes OpenAI's request, with `X-Request-Id` to name it. A success is a `chat.completion` plus a `kolibri_bridge` object: stop reason, turns, calls, every message and the trace. A failure is an error status with the same object, whose `content` is what the failed turn wrote before it failed and whose `finish_reason` is absent. With `"stream": true` the answer comes in llama-server's chunks with every turn's text, the trace as named `kolibri_bridge` events, and a failure as an error event without `[DONE]`.

- `POST /v1/executions/{id}/cancel` ends a request. The upstream connection is cut, which stops llama-server's generation. The answer doesn't wait for the request's client: its trace gets the `cancel_requested` event at once, and its stream at the next event.
- The same with `{"call_id":"t1c0"}` cancels one operation: the model reads `cancelled` as its result and the request goes on. The answer's `cancelled` is false if it already had a result. An empty or null `call_id` is HTTP 400, not a cancel of the whole request: only a body without the field means that.
- `GET /v1/bridge/tools` lists the version, the limits and every operation's input and output schema. `kolibri-bridge -ops` prints the same.

A client's request may hold `messages`, `stream`, `tool_choice` (only `auto`), `n` (only 1), `max_tokens` (within the limit), `model`, `user`, `seed`, `stop`, `logit_bias`, `reasoning_effort`, `parallel_tool_calls`, `cache_prompt`, `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `presence_penalty`, `frequency_penalty`, `repeat_penalty` and `repeat_last_n`. These pass through as written. **Any other field is refused with HTTP 400.** That covers `tools`, `functions`, `function_call`, `grammar`, `json_schema`, `response_format`, `n_predict`, `max_completion_tokens`, `stream_options`, `continue_final_message` and `add_generation_prompt`, and it covers what llama-server reads under other names, such as `chat_template_kwargs` and `n_cmpl`. A list of what to refuse kept missing those, so this is a list of what to accept.

## Operations

| Operation | Input | Output |
|---|---|---|
| `int_add`, `int_sub`, `int_mul`, `int_gcd` | `a`, `b` | `value` |
| `int_div` | `a`, `b` | `quotient`, `remainder`; truncates toward zero, the remainder has `a`'s sign |
| `int_pow` | `base`, `exponent` (`>= 0`) | `value`; `0^0` is 1 |
| `int_cmp` | `a`, `b` | `order`: -1, 0, 1 |
| `array_sum`, `array_product` | `values` | `value`; of an empty array, 0 and 1 |
| `array_min`, `array_max` | `values`, not empty | `value` |
| `array_sort` | `values`, `descending` (optional) | `values` |
| `array_reverse` | `values` | `values` |
| `hash_sha256`, `hash_sha512` | `data`, `encoding` (`utf8`, `hex`, `base64`; absent means `utf8`, empty is refused) | `hex`; `base64` is strict: no line breaks, no non-zero trailing bits |

**Arithmetic is exact, with these limits:**

- Integers only. `1.5`, `1.0` and `1e3` are `invalid_argument`. There is no floating point and no rational type.
- Up to 4096 bits (`-max-int-bits`), which every number of 1233 decimal digits fits in. A larger input is `input_too_large`, a larger result `overflow`, also a partial sum or product whose final value would fit. Nothing is rounded or wrapped.
- Both sides use decimal strings: an optional minus and digits, with no leading zero and no `-0`.
- A JSON integer is accepted within int64. Past that it is refused, even though the operation could compute it, as the next section explains.

A JSON string with a lone surrogate escape (`"\ud800"`) decodes to U+FFFD, as everywhere in encoding/json, so `hash_*` hashes that replacement and not what the model wrote. The real server can't parse such arguments, so the echo wraps them, as the next section says, and the model reads the result.

## What the real server showed

These came from llama-server on the `cmd/kolibri-tiny` checkpoint. They shaped the code and are tests now.

| Finding | Consequence | Test |
|---|---|---|
| The server parses a call's arguments to render the next prompt. `9223372036854775808` comes out as `-9223372036854775808`, `99999999999999999999999` as `1e+23` | JSON numbers past int64 are refused. Integers go as strings, which survive | `TestTinyCheckpointCorruptsLargeJSONNumbers` |
| The grammar for `{"type":"integer"}` allows 16 digits (`common/json-schema-to-grammar.cpp`), while a string pattern is unbounded | The schema uses a string with a pattern. A model under the integer grammar couldn't write a longer number | `TestTinyCheckpointCompilesEverySchema` checks the grammar |
| Arguments that aren't JSON make the next turn HTTP 500. So do arguments that encoding/json takes and the server's parser doesn't: a lone surrogate escape, and a number that overflows a double | The echo wraps all of these as `{"_malformed":"..."}`, so the model gets to read its result. That can be a successful one, as with `hash_*` and a surrogate. Other JSON goes back as it is, also when it isn't an object: the server takes `5` and `null` | `TestTinyCheckpointRefusesRawMalformedArguments`, `TestTinyCheckpointAcceptsTheEchoOfMalformedArguments`, `TestEchoArguments` |
| `chat_template_kwargs` overrides the template's variables: a `messages` key replaces the rendered messages, and the bridge delivers results by appending to them | Refused | `TestTinyCheckpointTemplateKwargsReplaceTheMessages`, `TestClientRequestsThatAreRefused` |
| A turn out of tokens returns a complete-looking call and a second one that is only `{`, with `finish_reason: "length"` | Only `"tool_calls"` runs anything | `TestCapturedCutTurnExecutesNothing` |
| Text the parser can't read is an error event after a 200, without `[DONE]`. A repeated `<tool_call>` token comes back as plain content | `upstream_error`, `unparsed_call` | `TestCapturedParseErrorMidStream`, `TestAnAnswerCutOrWithAnUnparsedCall` |
| `tool_choice: "required"` allows free text before the first call: the grammar's root has a free-text rule (`until-182?`) ahead of `tool-calls`. It doesn't make a call come first | The bridge uses `auto`. Only the tiny test uses `required`, with a prefilled `<tool_call>` | `forceCall` |
| Under `required`, `parallel_tool_calls: false` didn't stop several calls in one turn | The limits count every call, and the bridge doesn't send the field by itself | `TestCallLimits` |
| The tools cost prompt tokens: 51 without tools, 605 with the four of the acceptance task, 1931 with all 15, on llama.cpp `90311e5a2` | `-allow` matters. The tiny checkpoint's 2048-token context barely fits them all | `TestTinyCheckpointCompilesEverySchema` |

`testdata/bridge` holds the captured bytes. `TestCaptureFixtures` regenerates them, and `TestFakeHasTheCapturedShapes` holds the test server to their key paths.

## Limits

| Limit | Default | Past it |
|---|---|---|
| Turns per request | 16 | `turn_limit`. The last turn's calls don't run: nothing could read the result |
| Calls per request, per turn | 64, 8 | `call_limit`. All of a turn runs or none of it |
| Time of an operation | 5 s | the model reads `timeout` |
| Time of a turn, of a request | 15 min, 1 h | `turn_timeout`, `request_timeout` |
| Time of one write to a client | 30 s | `cancelled`: the client went away, and its slot is free |
| Bytes of a turn's response, of a client's request | 8 MiB, 16 MiB | `response_too_large`, HTTP 413 |
| `max_tokens` of a turn | 4096 | refused |
| Arguments, array length, integer, result | 256 KiB, 4096, 4096 bits, 1 MiB | the model reads the error |
| Open requests, running operations | 8, 8 | HTTP 429; operations queue |

**What the limits don't do:**

- **A write to a client can't be interrupted.** A turn's text goes to a stream on the request's own goroutine, so that a client that stops reading parks the turn in the write, where neither a timeout nor a cancel reaches it. The write's deadline ends that. So a stuck client can hold its request for up to one write timeout past the turn timeout, the request timeout or a cancel call. A cancel call answers at once all the same.
- **Memory and time inside an operation are bounded by input size, not enforced.** There is no rlimit or separate process. The operations can't be interrupted inside a `math/big` call, so the checks run before the work. After a timeout the worker's context ends and its result is ignored, but a worker stuck in a call would keep its goroutine. No real operation can get stuck, since every input is bounded.
- **"No filesystem or network" holds because the operations are plain Go functions in a registry.** The model picks a name from a list and passes JSON. `TestImportsStayPure` keeps `os`, `net`, `time` and random sources out of the direct imports of `internal/bridge/ops`. That is not a sandbox against hostile code, and there is no way to load code.
- **The bridge talks to one upstream.** It ignores proxy settings and doesn't follow redirects.
- **The API has no authentication.** It listens on localhost by default. Anyone who can reach it can cancel requests and read answers.

The trace of a request holds every step in order: start, each turn with its finish reason and token counts, each call with its arguments and SHA-256, each result, each refused call, each cancel request and the end. Text is clipped at 4 KiB, while the hashes cover all of it. `-trace FILE` writes it as JSON lines.

## Tests

```sh
go test -race -count=1 ./internal/bridge/... ./cmd/kolibri-bridge/
```

All pass, and the real-server tests below are skipped without their environment variables. The test server replays `testdata/bridge` or a script, which covers:

- malformed arguments, unknown and disallowed operations, overflow and division by zero;
- a model that never stops calling;
- a stream cut before a call, inside the arguments, inside an event, after the finish chunk and before `[DONE]`;
- a hanging server, a slow operation, cancellation during an operation and during a turn, a client that leaves, and a client that stops reading, with a stream or a whole answer;
- parallel calls finishing in the other order, sixteen requests at once;
- eight runs of one script giving the same bytes upstream, the same messages and the same canonical trace, with the server's random IDs in play.

`FuzzExecute` ran 1.5 million executions without a panic or an unstable answer.

### On real llama-server with tiny weights

No memory to speak of (about 500 MiB) and no real weight. The test forces calls with a prefilled `<tool_call>` and `tool_choice: "required"`, so it says nothing about what a model would choose.

```sh
D=$(mktemp -d)
TOK=$(.venv/bin/python -c 'import sys; sys.path.insert(0, "tools/tokenizer"); from common import tokenizer_dir; print(tokenizer_dir())')
go run ./cmd/kolibri-tiny -out $D/kolibri-tiny -tokenizer-dir "$TOK" -seed 1
.venv/bin/python third_party/llama.cpp/convert_hf_to_gguf.py $D/kolibri-tiny --outtype f32 --outfile $D/kolibri-tiny-f32.gguf
third_party/llama.cpp/build/bin/llama-server -m $D/kolibri-tiny-f32.gguf --host 127.0.0.1 --port 18731 -c 2048 -np 1 -ngl 0 --no-webui --jinja &
KOLIBRI_BRIDGE_TINY_URL=http://127.0.0.1:18731 go test -race -count=1 -v -run TestTinyCheckpoint ./internal/bridge/
```

All seven tests pass. A random model called `array_sum`, `int_div` and `hash_sha256`, the operations ran, the real template rendered their results, and two runs gave the same canonical trace.

### The acceptance test, which has not run

`TestAcceptanceRealModel` asks the model to multiply `987654321987654321` by `123456789123456789`, add 1, and take the SHA-256 of the sum's digits. The prompt asks for the digest only, and a model can't work that out in its head. The test passes only with `stop` = `completed`, the three calls in order with exact operands, each result exact, and exactly the expected digest in the answer. The expected values are literals computed outside this repo's code. A mistake the model repairs doesn't fail the test but shows up in the trace. `TestAcceptanceJudge` shows on the fake that a wrong digest, a miscopied number, a skipped step, an answer without tools, steps out of order and a truncated end all fail.

It needs a real weight, llama-server built from the fork, and the machine to itself: **a resident run wires about 33 GiB. Two have panicked this 48 GiB Mac, so check that no other run holds a weight.**

```sh
third_party/llama.cpp/build/bin/llama-server -m ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf \
    -ngl 99 -c 16384 -np 1 --jinja --host 127.0.0.1 --port 8099 &
KOLIBRI_BRIDGE_ACCEPT_URL=http://127.0.0.1:8099 KOLIBRI_BRIDGE_ACCEPT_TRACE=accept-trace.jsonl \
    go test -race -count=1 -timeout 3h -v -run TestAcceptanceRealModel ./internal/bridge/
```

It refuses a server whose template isn't the released one (sha256 `9ba35d4b…`), and a weight with `tiny` in its file name. Its plumbing has run on random weights only, under another file name, and it failed as it should: stop reason `truncated`, 0 of 3 steps, a 722-token prompt. That says nothing about Kolibri. `KOLIBRI_BRIDGE_ACCEPT_RUNS=2` repeats the task and requires one canonical trace. Prompt caching may make that fail on a server that decodes a cached prefix differently, which would be worth knowing. `KOLIBRI_BRIDGE_ACCEPT_EFFORT` sets `reasoning_effort`, default `low`.

The outcome is open. Kolibri may use the tools, may stop after a step, may miscopy a number or may ignore them. Each of those is a clean, machine-readable result of this test.

## Do the tests catch regressions?

Each of these was applied alone to the code and reverted. All 28 made `go test -race -count=1 ./internal/bridge/... ./cmd/kolibri-bridge/` fail. The first run had one survivor: the overflow check inside a sum or product, because the check at the end still reported the overflow. `TestLimits` now pins it, and the table lists the test that catches it now.

| Mutation | Failed tests |
|---|---|
| Turn limit off | `TestAModelThatNeverStopsCalling` |
| Call limit off | `TestCallLimits` |
| Calls-per-turn limit off, in the stream | `TestCallLimits` |
| Calls-per-turn limit off, in a whole response | `TestCallLimits` |
| Operation timeout ignored | `TestOperationTimeout`, `TestCancelUnknownCallOfARunningRequest` |
| Turn timeout ignored | `TestTimeouts` |
| Request timeout ignored | `TestTimeouts` |
| Allowlist not enforced | `TestAllowlist`, `TestBadCallsGoBackToTheModel` |
| Argument size not checked | `TestLimits` |
| Overflow not checked | `TestLimits` |
| Overflow not checked inside a sum or product | `TestLimits` |
| Division by zero not checked | `TestBadCallsGoBackToTheModel`, `TestMalformedAndInvalidArguments`, `TestResponseEnvelope` |
| Array length not checked | `TestLimits` |
| Integer input size not checked | `TestLimits` |
| Result size not checked | `TestLimits` |
| JSON numbers past int64 accepted | `TestBadCallsGoBackToTheModel`, `TestMalformedAndInvalidArguments` |
| Duplicate keys accepted | `TestBadCallsGoBackToTheModel`, `TestMalformedAndInvalidArguments` |
| A truncated turn runs its calls | `TestCapturedCutTurnExecutesNothing`, `TestAnAnswerCutOrWithAnUnparsedCall`, `TestHTTPFailures` |
| A stream without `[DONE]` counts | `TestAStreamThatBreaks`, `TestHTTPFailures` |
| An unparsed marker counts as an answer | `TestAnAnswerCutOrWithAnUnparsedCall` |
| Response size not limited | `TestUpstreamProblems` |
| Redirects followed | `TestUpstreamProblems` |
| Malformed arguments echoed raw | `TestBadCallsGoBackToTheModel` |
| Reasoning not sent back | `TestMultiStepChain` |
| The server's random IDs go back | `TestReproducible` and 8 more |
| Results in reverse call order | `TestParallelCallsKeepTheirOrder`, `TestManyRequestsAtOnce`, `TestCancelACallThatIsDone` |
| Cancel does nothing | `TestCancelDuringATurn`, `TestCancelDuringAnOperation`, `TestHTTPCancel` |
| Request body not limited | `TestHTTPRefusals` |

A later review ran more mutations against that suite and found ten that survived: the call limit off by one, four refused fields left out of the test, `Close` not waiting for workers, a dropped cancel event, an unbounded upstream error body, events after `request_end`, and proxy settings kept from the environment. So the table above was never all the coverage it may look like. The fixes that followed came with these 38, each applied alone to a copy of the code, and all of them make `go test -count=1 ./internal/bridge/... ./cmd/kolibri-bridge/` fail. They ran without `-race`. The first run had one survivor: a write deadline set at the start of every response, which turned out to do nothing, because net/http clears a connection's deadline when a response ends. That code is gone. The 28 above were not rerun after these changes.

| Mutation | Failed tests |
|---|---|
| No write deadline | `TestHTTPCompletionClientThatStopsReading`, `TestHTTPStreamingClientThatStopsReading` |
| A failed write does not end the request | `TestAFailedWriteEndsTheRequest` |
| Writes go on after a failed one | `TestAFailedWriteEndsTheRequest` |
| Deadline not renewed per stream write | `TestHTTPDeadlinesAreForOneWrite` |
| Deadline not renewed before a whole answer | `TestHTTPDeadlinesAreForOneWrite` |
| Cancel writes to the client itself | `TestHTTPCancelWhileTheClientIsStuck` |
| A cancel of one call writes to the client itself | `TestHTTPCancelWhileTheClientIsStuck` |
| Trace lock held during the callbacks | `TestCancelDoesntWaitForACallback` |
| Echo trusts json.Valid | `TestArgumentsTheServerCantParseStillReachTheModel`, `TestEchoArguments` |
| Echo: overflow not checked | `TestArgumentsTheServerCantParseStillReachTheModel`, `TestEchoArguments` |
| Echo: lone low surrogate not checked | `TestEchoArguments` |
| Echo: lone high surrogate not checked | `TestEchoArguments` |
| chat_template_kwargs accepted | `TestClientRequestsThatAreRefused` |
| n_cmpl accepted | `TestClientRequestsThatAreRefused` |
| json_schema accepted | `TestClientRequestsThatAreRefused` |
| function_call accepted | `TestClientRequestsThatAreRefused` |
| continue_final_message accepted | `TestClientRequestsThatAreRefused` |
| add_generation_prompt accepted | `TestClientRequestsThatAreRefused` |
| An accepted field dropped | `TestClientFieldsThatPassThrough` |
| A failed turn keeps the turn before it | `TestAFailedTurnIsTheLastTurn`, `TestAStreamThatBreaks` |
| A failed turn keeps its finish reason | `TestAFailedTurnIsTheLastTurn` |
| A failed stream drops its text | `TestAFailedTurnIsTheLastTurn`, `TestAStreamThatBreaks` |
| A generated ID can be a duplicate | `TestGeneratedIDsStepAroundClientIDs` |
| An empty call_id cancels the request | `TestHTTPCancel` |
| A null call_id cancels the request | `TestHTTPCancel` |
| Canonical keeps the request ID | `TestCanonicalIgnoresGeneratedIDs` |
| Failed turn has no turn_end | `TestCancelDuringATurn`, `TestAStreamThatBreaks` and 7 more |
| A result without its call_end | `TestAcceptanceJudge`, `TestMultiStepChain` |
| Call limit off by one | `TestCallLimits` |
| Close does not wait for workers | `TestClosingTheBridgeEndsRequests` |
| A cancel of one call leaves no event | `TestCancelOneCall`, `TestHTTPCancelWhileTheClientIsStuck` |
| Upstream error body unbounded | `TestUpstreamProblems` |
| Events after request_end kept | `TestNothingAfterTheEndOfARequest` |
| Proxy settings kept | `TestUpstreamProblems` |
| A closed request keeps its slot | `TestAdmission`, `TestHTTPStreamingClientThatStopsReading` |
| Hash: empty encoding accepted | `TestMalformedAndInvalidArguments` |
| Hash: non-zero trailing bits accepted | `TestMalformedAndInvalidArguments` |
| Hash: base64 line breaks accepted | `TestMalformedAndInvalidArguments` |

// Package coproc is the deterministic scheduler between a Kolibri request and a sandboxed
// coprocessor (issue #64, phases 1 to 4). A controller plans typed actions; the scheduler owns
// execution, blocking, wakeup and result delivery. The controller is scripted here and the
// coprocessor and token generator are mocks: nothing in this package touches a model or llama.cpp.
//
// Completion is event driven, a software-interrupt analogy and not an interruption of a forward
// pass. A worker publishes a CompletionEvent to a bounded channel. One goroutine consumes it,
// validates request, job and generation, and moves the matching blocked request to READY in one
// critical section. The request's driver resumes in its next Apply, which is the boundary between
// two tokens. Nothing polls, on the scheduler's side or the controller's.
//
// # What the scheduler guarantees
//
//   - No token is generated while a request is BLOCKED: every action but a job's external
//     cancellation needs READY.
//   - A job starts at most once per idempotency key within a request. A retry with the same key
//     gets the first job back, even after it failed, because its side effects may have happened.
//   - WAIT unblocks only on the matching job's completion, failure, cancellation or timeout.
//   - A job's first terminal state stands. Duplicate events, and events after a timeout, a
//     cancellation or a teardown, change nothing. A completion before WAIT stays in the registry.
//   - A dependency must be declared before the job that names it, so the graph is acyclic by
//     construction: a cycle needs a forward edge, and a forward edge is an unknown ID. A job whose
//     dependency fails ends in ERROR without starting.
//   - Requests, jobs, running workers, the event queue, results and actions all have limits.
//
// # What it cannot guarantee
//
// The scheduler enforces the dependencies a controller declares. It cannot detect undeclared
// semantic ones: a controller that plans a job without naming the job whose result it needs gets
// a job that runs too early or on the wrong input, and an answer generated without a WAIT is not
// held back. This design does not make outputs free of hallucination if the controller plans
// incorrectly.
//
// A worker must honor its context. After a timeout or cancellation the scheduler frees the
// worker's slot and ignores its late events, but it cannot stop a worker that keeps running.
package coproc

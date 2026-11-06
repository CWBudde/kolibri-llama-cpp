# Coprocessor scheduler

**Result: the scheduler of issue #64 exists and passes its acceptance test against a mock (phases 1 to 4). It has not met a model.** `internal/coproc` runs a mocked one-second operation through EXECUTE, WAIT, completion, READ_RESULT and answer, and returns the exact result with no answer token before the second has passed. The same holds with a timeout, an error, a duplicate event, a mismatched job ID, and two independent jobs plus a dependent one.

`go test -race -count=20 ./internal/coproc/` passes. The tests run in `testing/synctest` bubbles: the clock is virtual, so they don't sleep, and a lost wakeup is a reported deadlock instead of a hang.

## What exists and what doesn't

| Phase of #64 | State |
|---|---|
| 1. Typed controller, scheduler and coprocessor API, mock coprocessor, state-machine tests | done |
| 2. One delayed calculation, EXECUTE to answer | done, on the virtual clock |
| 3. Timeout, error, cancel, retry and idempotency tests, no tokens while BLOCKED | done |
| 4. Two independent jobs and one dependent job, synchronous waits | done |
| 5. llama.cpp inference lifecycle, KV cache and event delivery | **not started.** `Generator` is the seam; the only "KV cache" here is a mock's token counter |
| 6. Token-context delivery against a hidden-state adapter, asynchronous execution | **not started** |

The controller is a script, so nothing here shows that Kolibri can plan these actions. No sandbox exists: `Coprocessor` is an interface and `Mock` its only implementation. No result reaches a model.

## How it works

A controller returns typed actions: `Continue`, `Execute{Job, IdempotencyKey, Program, Args, Dependencies, Timeout}`, `Wait{Job}`, `ReadResult{Job}`. A `Request` applies them one at a time, and every action needs READY.

```
READY -- Continue, ReadResult --> GENERATING -- token or delivery done --> READY
READY -- Wait, job has no outcome --> BLOCKED -- the job's outcome --> READY
any state -- Close --> CLOSED (ends the machine)

IDLE --> RUNNING --> DONE | ERROR | TIMEOUT | CANCELLED
IDLE --> ERROR (a dependency failed) | CANCELLED
```

A worker publishes `CompletionEvent{RequestID, JobID, Generation, Status, Result, Error}` to a bounded channel, and one goroutine consumes it. It looks up the request and the job, compares the generation, and rejects a status that isn't terminal. Then, in one critical section, it settles the job, frees the slot, moves a request blocked on exactly this job to READY and passes the outcome to the dependents. A job's deadline is an ordinary event on the same channel, so a timeout and a completion race in one place. Nothing polls: the waiter sits on a channel and the controller isn't called while its request is BLOCKED.

The safe boundary is the next `Apply`. A completion during a token updates only its job, and the request keeps GENERATING until the token is finished. WAIT reports only the outcome. The result becomes visible to the model in READ_RESULT, through `Generator.Deliver`.

| Invariant | How it holds |
|---|---|
| No token while BLOCKED | every action is gated on READY, under the mutex |
| A job starts at most once per key | a retry gets the first job back, also under a new job ID and after a failure; IDLE to RUNNING happens once |
| WAIT unblocks only on its job | the transition compares the waited job with the settled one |
| Duplicates are harmless | a job's first terminal state stands, including its result |
| Stale events can't resurrect a request | a closed request is gone, and a generation is unique in a scheduler even if the request and job IDs come back |
| No lost wakeup | WAIT checks the registry under the mutex that the completion is recorded under |
| No job runs on an unknown or failed dependency | a dependency must be declared first; a failure ends its dependents in ERROR unstarted |
| Bounded resources | `Config`: requests, job IDs per request, running jobs, result bytes, queued events, actions per request, timeout |

## Decisions and limits

- **The scheduler enforces declared dependencies. It can't detect undeclared semantic ones.** A controller that plans a job without naming the job whose result it needs gets a job that runs too early or on the wrong input. An answer generated without a WAIT isn't held back. This design doesn't make outputs free of hallucination if the controller plans incorrectly.
- **Cycles can't be declared, so there is no cycle search.** A dependency must already exist, so every edge points backwards. A cycle's first forward edge fails as `ErrUnknownDependency`, a self-edge as `ErrCycle`, and a changed retry that would add a back edge as `ErrIdempotencyConflict`. A plan has to list its jobs in dependency order.
- **Retrying a failed job needs a new key.** The same key returns the same failure, because the first run's side effects may have happened.
- **Idempotency keys are per request and in memory.** They don't survive a teardown or a restart.
- **A worker must honor its context.** After a timeout or cancellation its slot is free and its events are ignored, but the scheduler can't stop a worker that keeps running, so `MaxRunning` doesn't bound such workers.
- **A job's timeout counts its run time, not its time in the queue.** The queue stays bounded because every running job has a deadline.
- **`MaxResultBytes` doesn't cover error values.** A worker can still report an arbitrarily long error.
- **WAIT takes one job and one mutex guards the whole scheduler.** The registry already holds several jobs per request, so a wait-any fits later. The mutex hasn't been measured under load.
- **Publishing isn't authenticated.** Validation protects against stale and misrouted events, not against a worker forging another job's ID and generation. That is for the sandbox boundary to solve.

## Do the tests catch regressions?

Each of these was applied alone to the code and reverted. Every one failed `go test -race -count=1`. A deadlock ends a run at its first failing test, so those two rows list only that test.

| Mutation | Failed tests |
|---|---|
| Generation not compared | `TestOnlyTheMatchingEventUnblocks`, `TestAcceptanceMismatchedJobID`, `TestStaleEventsDontResurrectAClosedRequest` |
| First outcome doesn't stand | 10 tests, among them `TestDuplicateCompletionsAreHarmless`, `TestAcceptanceDuplicateEvent`, `TestCancelRacingCompletion` |
| Completion before WAIT not retained | deadlock in `TestTwoIndependentJobsAndADependentOne` |
| Any job's outcome wakes the waiter | `TestOnlyTheMatchingEventUnblocks`, `TestTwoIndependentJobsAndADependentOne` and 3 more |
| No idempotency lookup | `TestIdempotentRetries`, `TestCyclesCannotBeDeclared`, `TestLimits` |
| No READY gate on actions | `TestNoActionRunsWhileBlocked` |
| Dependent starts before its dependencies | `TestRandomPlansRunInDependencyOrder`, `TestTwoIndependentJobsAndADependentOne` |
| Failure not propagated | deadlock in `TestFailurePropagatesThroughTheGraph` |
| Unknown dependency accepted | `TestUnknownDependencyIsRejected`, `TestCyclesCannotBeDeclared`, `TestRandomPlansRunInDependencyOrder` |
| Close leaves the request registered | `TestStaleEventsDontResurrectAClosedRequest`, `TestLimits` |
| Deadline fires twice too late | `TestAcceptanceTimeout`, `TestTimeoutIsDeliveredAndALateCompletionIsDropped`, `TestLimits` |
| Settling leaves the worker running | `TestCancelWakesTheWaiterAndStopsTheWorker`, `TestShutdown` |
| No limit on running jobs | `TestLimits`, `TestShutdown`, `TestCancelBeforeStart` |
| READ_RESULT doesn't deliver | 9 tests, among them `TestAcceptanceOneSecondOperation` and `TestReadResultDeliversToTheModelAndCanRepeat` |

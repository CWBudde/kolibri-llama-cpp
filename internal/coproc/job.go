package coproc

import (
	"context"
	"errors"
)

// RequestID names a request. An ID can be used again once its request is closed.
type RequestID string

// JobID names a job within its request. The controller chooses it.
type JobID string

// Generation identifies one start of one job. It is unique within a scheduler, so a worker's event
// can't match a later job that reuses the request and job IDs. A job that hasn't started has
// generation 0.
type Generation uint64

// JobState is a job's lifecycle state. The first terminal state is final.
type JobState int

const (
	JobIdle      JobState = iota // registered, not started: waiting for dependencies or a slot
	JobRunning                   // started on the coprocessor
	JobDone                      // finished with a result
	JobError                     // failed, by the program or through a dependency
	JobTimeout                   // ran past its deadline
	JobCancelled                 // cancelled, directly or by the request's teardown
)

// Terminal reports whether the state is final.
func (s JobState) Terminal() bool { return s >= JobDone && s <= JobCancelled }

func (s JobState) String() string {
	switch s {
	case JobIdle:
		return "IDLE"
	case JobRunning:
		return "RUNNING"
	case JobDone:
		return "DONE"
	case JobError:
		return "ERROR"
	case JobTimeout:
		return "TIMEOUT"
	case JobCancelled:
		return "CANCELLED"
	}
	return "INVALID"
}

// RequestState is the scheduler's state of a request.
type RequestState int

const (
	Ready      RequestState = iota // at a safe boundary, the next action can run
	Generating                     // the inference context is in use, a completion waits for the boundary
	Blocked                        // waiting for one job, no action runs and no token is generated
	Closed                         // torn down; not a state of the issue's machine, it ends the machine
)

func (s RequestState) String() string {
	switch s {
	case Ready:
		return "READY"
	case Generating:
		return "GENERATING"
	case Blocked:
		return "BLOCKED"
	case Closed:
		return "CLOSED"
	}
	return "INVALID"
}

// Errors reported in Observation.Err, Observation.JobErr or by the scheduler's methods.
var (
	ErrInvalidAction       = errors.New("coproc: invalid action")
	ErrUnknownJob          = errors.New("coproc: unknown job")
	ErrUnknownDependency   = errors.New("coproc: unknown dependency")
	ErrCycle               = errors.New("coproc: dependency cycle")
	ErrIdempotencyConflict = errors.New("coproc: job ID or idempotency key already used by a different job")
	ErrTooManyJobs         = errors.New("coproc: too many jobs in the request")
	ErrTooManyRequests     = errors.New("coproc: too many open requests")
	ErrDuplicateRequest    = errors.New("coproc: request ID already open")
	ErrNotReady            = errors.New("coproc: request is not READY")
	ErrNotComplete         = errors.New("coproc: job has no outcome yet")
	ErrRequestClosed       = errors.New("coproc: request closed")
	ErrStopped             = errors.New("coproc: scheduler stopped")
	ErrActionLimit         = errors.New("coproc: request used up its actions")
	ErrJobTimeout          = errors.New("coproc: job timed out")
	ErrJobCancelled        = errors.New("coproc: job cancelled")
	ErrDependencyFailed    = errors.New("coproc: dependency failed")
	ErrResultTooLarge      = errors.New("coproc: result too large")
)

// JobSpec is what the coprocessor gets to start a job.
type JobSpec struct {
	RequestID      RequestID
	ID             JobID
	IdempotencyKey string
	Generation     Generation
	Program        string
	Args           []string
	DepResults     map[JobID]string // the result of every dependency, under the name its Execute gave it
}

// CompletionEvent is a worker's report of a job's outcome. Result is set for JobDone, Error for
// the other outcomes.
type CompletionEvent struct {
	RequestID  RequestID
	JobID      JobID
	Generation Generation
	Status     JobState
	Result     string
	Error      error
}

// Coprocessor runs jobs. The scheduler calls Start at most once per job.
type Coprocessor interface {
	// Start must return promptly and run the job on its own goroutine: it runs on the goroutine
	// that consumes completion events. An error means the job did not start. The worker reports
	// its outcome through publish, which blocks while the event queue is full and returns
	// ErrStopped once the scheduler has stopped. ctx ends when the job's outcome is settled by a
	// timeout, a cancellation or a teardown, and the worker must then stop.
	Start(ctx context.Context, job JobSpec, publish func(CompletionEvent) error) error
}

// JobInfo is a snapshot of a job.
type JobInfo struct {
	ID             JobID
	IdempotencyKey string
	State          JobState
	Generation     Generation
	Dependencies   []JobID
	Result         string
	Err            error
}

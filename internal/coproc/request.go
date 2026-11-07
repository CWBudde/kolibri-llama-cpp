package coproc

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"time"
)

// Generator stands in for the inference context of a request: its decode position, KV cache and
// prompt. Phase 5 puts llama.cpp behind it. The scheduler never calls it while the request is
// BLOCKED and never calls both methods at once.
type Generator interface {
	// Step generates one token and returns at a boundary between tokens.
	Step(ctx context.Context) (string, error)
	// Deliver makes a job's result or error visible to the model. Resuming inference alone doesn't.
	Deliver(ctx context.Context, msg ResultMessage) error
}

// Request is one request's state machine and job registry. Its methods are safe for concurrent
// use, but actions run one at a time: an Apply that finds the request not READY is rejected.
type Request struct {
	s   *Scheduler
	id  RequestID
	gen Generator

	// ctx ends when the request closes, which aborts a running generator call.
	ctx    context.Context
	cancel context.CancelFunc

	// Guarded by s.mu.
	state   RequestState
	waiting *job          // the job a BLOCKED request waits for
	wake    chan struct{} // closed when BLOCKED ends
	jobs    map[JobID]*job
	byKey   map[string]*job
	order   []*job // canonical jobs in order of declaration
	actions int
}

// Open registers a request in READY. Close ends it.
func (s *Scheduler) Open(id RequestID, gen Generator) (*Request, error) {
	if gen == nil {
		return nil, errors.New("coproc: no generator")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch {
	case s.stopped:
		return nil, ErrStopped
	case s.requests[id] != nil:
		return nil, fmt.Errorf("%w: %s", ErrDuplicateRequest, id)
	case len(s.requests) >= s.cfg.MaxRequests:
		return nil, ErrTooManyRequests
	}
	r := &Request{
		s: s, id: id, gen: gen,
		jobs: make(map[JobID]*job), byKey: make(map[string]*job),
	}
	r.ctx, r.cancel = context.WithCancel(context.Background())
	s.requests[id] = r
	return r, nil
}

// ID returns the request's ID.
func (r *Request) ID() RequestID { return r.id }

// State returns the request's state.
func (r *Request) State() RequestState {
	r.s.mu.Lock()
	defer r.s.mu.Unlock()
	return r.state
}

// Job returns a snapshot of a job, by its ID or an alias.
func (r *Request) Job(id JobID) (JobInfo, bool) {
	r.s.mu.Lock()
	defer r.s.mu.Unlock()
	j, ok := r.jobs[id]
	if !ok {
		return JobInfo{}, false
	}
	return j.info(), true
}

// Run drives the request with a controller until the controller finishes, the request closes, its
// actions run out or ctx ends. emit, which may be nil, gets the user-visible answer tokens. It runs
// before the request leaves GENERATING, so a token can't go out after a second driver has moved the
// request on, and a token that was generated goes out even if ctx ends right after it. Actions from
// inside emit are rejected as not READY. Run doesn't close the request.
func (r *Request) Run(ctx context.Context, c Controller, emit func(string)) error {
	var obs Observation
	for {
		a, ok := c.Next(obs)
		if !ok {
			return nil
		}
		obs = r.apply(ctx, a, emit)
		if obs.Err != nil && (errors.Is(obs.Err, ErrRequestClosed) || errors.Is(obs.Err, ErrActionLimit)) {
			return obs.Err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
	}
}

// Apply runs one action in READY and returns its observation. A Wait blocks the caller until the
// job has its outcome. If ctx ends during a Wait, the awaited job is cancelled, as nothing is left
// to wait for it. Failures are reported in the observation, never by leaving the request stuck.
func (r *Request) Apply(ctx context.Context, a Action) Observation {
	return r.apply(ctx, a, nil)
}

func (r *Request) apply(ctx context.Context, a Action, emit func(string)) Observation {
	if err := r.admit(); err != nil {
		return Observation{Action: a, Err: err}
	}
	var obs Observation
	switch a := a.(type) {
	case Continue:
		obs = r.applyContinue(ctx, emit)
	case Execute:
		obs = r.applyExecute(a)
	case Wait:
		obs = r.applyWait(ctx, a)
	case ReadResult:
		obs = r.applyReadResult(ctx, a)
	default:
		obs.Err = fmt.Errorf("%w: %T", ErrInvalidAction, a)
	}
	obs.Action = a
	return obs
}

// admit is the gate of every action, and the reason nothing generates while BLOCKED.
func (r *Request) admit() error {
	r.s.mu.Lock()
	defer r.s.mu.Unlock()
	switch {
	case r.state == Closed:
		return ErrRequestClosed
	case r.state != Ready:
		return fmt.Errorf("%w: %s", ErrNotReady, r.state)
	case r.actions >= r.s.cfg.MaxActionsPerRequest:
		return ErrActionLimit
	}
	r.actions++
	return nil
}

// useInference claims the inference context, moving READY to GENERATING. A completion that arrives
// meanwhile updates only its job; the request picks it up at its next action. The returned
// function ends the claim and returns ErrRequestClosed if the request closed meanwhile.
func (r *Request) useInference(ctx context.Context) (context.Context, func() error, error) {
	r.s.mu.Lock()
	defer r.s.mu.Unlock()
	if r.state != Ready {
		if r.state == Closed {
			return nil, nil, ErrRequestClosed
		}
		return nil, nil, fmt.Errorf("%w: %s", ErrNotReady, r.state)
	}
	r.state = Generating
	ctx, cancel := context.WithCancel(ctx)
	stop := context.AfterFunc(r.ctx, cancel)
	return ctx, func() error {
		stop()
		cancel()
		r.s.mu.Lock()
		defer r.s.mu.Unlock()
		if r.state == Closed {
			return ErrRequestClosed
		}
		r.state = Ready
		return nil
	}, nil
}

func (r *Request) applyContinue(ctx context.Context, emit func(string)) Observation {
	var obs Observation
	ctx, release, err := r.useInference(ctx)
	if err != nil {
		obs.Err = err
		return obs
	}
	obs.Token, obs.Err = r.gen.Step(ctx)
	if obs.Err == nil && emit != nil && r.State() != Closed {
		emit(obs.Token)
	}
	if err := release(); err != nil {
		return Observation{Err: err} // the token of a closed request is not an answer
	}
	return obs
}

func (r *Request) applyExecute(a Execute) Observation {
	obs := Observation{Job: a.Job}
	cfg := r.s.cfg
	if a.Job == "" || a.Program == "" || a.Timeout < 0 || a.Timeout > cfg.MaxTimeout {
		obs.Err = fmt.Errorf("%w: job %q program %q timeout %v", ErrInvalidAction, a.Job, a.Program, a.Timeout)
		return obs
	}
	if a.Timeout == 0 {
		a.Timeout = cfg.DefaultTimeout
	}
	if a.IdempotencyKey == "" {
		a.IdempotencyKey = string(a.Job)
	}
	r.s.update(func() {
		switch r.state {
		case Ready:
		case Closed:
			obs.Err = ErrRequestClosed
			return
		default:
			obs.Err = fmt.Errorf("%w: %s", ErrNotReady, r.state)
			return
		}
		var j *job
		j, obs.Existing, obs.Err = r.register(a)
		if obs.Err == nil {
			obs.Job = j.id
		}
	})
	if obs.Err == nil {
		// Read after the update, which may have moved the job from queued to RUNNING.
		info, _ := r.Job(obs.Job)
		obs.Status, obs.JobErr = info.State, info.Err
	}
	return obs
}

// register adds a job to the graph, or returns the existing job of its idempotency key. A
// dependency must already exist, so every new edge points backwards and the graph can't get a
// cycle. A self-edge is the one cycle an action can name in one go, and it gets its own error.
func (r *Request) register(a Execute) (j *job, existing bool, err error) {
	if slices.Contains(a.Dependencies, a.Job) {
		return nil, false, fmt.Errorf("%w: job %s depends on itself", ErrCycle, a.Job)
	}
	var deps []*job
	for _, id := range a.Dependencies {
		d, ok := r.jobs[id]
		if !ok {
			return nil, false, fmt.Errorf("%w: job %s depends on %s", ErrUnknownDependency, a.Job, id)
		}
		deps = append(deps, d)
	}
	slices.SortFunc(deps, func(x, y *job) int { return cmp.Compare(x.id, y.id) })
	deps = slices.Compact(deps)
	depNames := slices.Compact(slices.Sorted(slices.Values(a.Dependencies)))

	byID, hasID := r.jobs[a.Job]
	byKey, hasKey := r.byKey[a.IdempotencyKey]
	switch {
	case hasKey && hasID && byKey != byID, !hasKey && hasID:
		return nil, false, fmt.Errorf("%w: job %s", ErrIdempotencyConflict, a.Job)
	case hasKey && !byKey.sameSpec(a, deps):
		return nil, false, fmt.Errorf("%w: key %q", ErrIdempotencyConflict, a.IdempotencyKey)
	case hasKey:
		if !hasID {
			if len(r.jobs) >= r.s.cfg.MaxJobsPerRequest {
				return nil, false, ErrTooManyJobs
			}
			r.jobs[a.Job] = byKey // a retry under a new ID: WAIT on it means the first job
		}
		return byKey, true, nil
	}
	if len(r.jobs) >= r.s.cfg.MaxJobsPerRequest {
		return nil, false, ErrTooManyJobs
	}

	j = &job{
		req: r, id: a.Job, key: a.IdempotencyKey, program: a.Program,
		args: slices.Clone(a.Args), timeout: a.Timeout, deps: deps, depNames: depNames,
	}
	r.jobs[a.Job] = j
	r.byKey[a.IdempotencyKey] = j
	r.order = append(r.order, j)

	var failed *job
	for _, d := range deps {
		switch {
		case d.state == JobDone:
		case d.state.Terminal():
			failed = d
		default:
			j.pending++
			d.dependents = append(d.dependents, j)
		}
	}
	switch {
	case failed != nil:
		r.s.settle(j, JobError, "", dependencyError(failed.id, failed.err))
	case j.pending == 0:
		r.s.queue = append(r.s.queue, j)
	}
	return j, false, nil
}

func (j *job) sameSpec(a Execute, deps []*job) bool {
	return j.program == a.Program && j.timeout == a.Timeout &&
		slices.Equal(j.args, a.Args) && slices.Equal(j.deps, deps)
}

func (r *Request) applyWait(ctx context.Context, a Wait) Observation {
	obs := Observation{Job: a.Job}
	s := r.s

	s.mu.Lock()
	j, ok := r.jobs[a.Job]
	switch {
	case r.state == Closed: // Close can land after admit
		s.mu.Unlock()
		obs.Err = ErrRequestClosed
		return obs
	case !ok:
		s.mu.Unlock()
		obs.Err = fmt.Errorf("%w: %s", ErrUnknownJob, a.Job)
		return obs
	case r.state != Ready:
		obs.Err = fmt.Errorf("%w: %s", ErrNotReady, r.state)
		s.mu.Unlock()
		return obs
	case j.state.Terminal():
		// The completion came first. It stayed in the registry, so no wakeup was lost.
		obs.Job, obs.Status, obs.JobErr = j.id, j.state, j.err
		s.mu.Unlock()
		return obs
	}
	r.state = Blocked
	r.waiting = j
	wake := make(chan struct{})
	r.wake = wake
	s.mu.Unlock()

	select {
	case <-wake:
	case <-ctx.Done():
		s.update(func() {
			if r.state == Blocked && r.waiting == j {
				s.settle(j, JobCancelled, "", outcomeError(JobCancelled, context.Cause(ctx)))
			}
		})
		<-wake // settle or a concurrent teardown has closed it
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	obs.Job, obs.Status, obs.JobErr = j.id, j.state, j.err
	if r.state == Closed {
		obs.Err = ErrRequestClosed
	}
	return obs
}

func (r *Request) applyReadResult(ctx context.Context, a ReadResult) Observation {
	obs := Observation{Job: a.Job}
	r.s.mu.Lock()
	j, ok := r.jobs[a.Job]
	var msg ResultMessage
	if ok {
		obs.Job, obs.Status, obs.JobErr, obs.Result = j.id, j.state, j.err, j.result
		msg = ResultMessage{Job: j.id, Status: j.state, Result: j.result, Err: j.err}
	}
	r.s.mu.Unlock()
	switch {
	case !ok:
		obs.Err = fmt.Errorf("%w: %s", ErrUnknownJob, a.Job)
		return obs
	case !msg.Status.Terminal():
		obs.Err = fmt.Errorf("%w: %s is %s", ErrNotComplete, a.Job, msg.Status)
		return obs
	}
	ctx, release, err := r.useInference(ctx)
	if err != nil {
		obs.Err = err
		return obs
	}
	obs.Err = r.gen.Deliver(ctx, msg)
	if err := release(); err != nil {
		obs.Err = err
	}
	return obs
}

// Cancel cancels a job that has no outcome yet, from any goroutine, also while the request waits
// for it. A job that already has one keeps it. Its dependents end in ERROR.
func (r *Request) Cancel(id JobID) error {
	var err error
	r.s.update(func() {
		j, ok := r.jobs[id]
		switch {
		case r.state == Closed:
			err = ErrRequestClosed
		case !ok:
			err = fmt.Errorf("%w: %s", ErrUnknownJob, id)
		default:
			r.s.settle(j, JobCancelled, "", ErrJobCancelled)
		}
	})
	return err
}

// Close tears the request down: its jobs without an outcome are cancelled, a waiter is released
// with ErrRequestClosed, a running generator call is cancelled and its token discarded. Later
// events for the request are dropped, also if a new request reuses the ID. Close can be called
// again.
func (r *Request) Close() {
	s := r.s
	s.update(func() {
		if r.state == Closed {
			return
		}
		if r.wake != nil {
			close(r.wake)
			r.wake = nil
			r.waiting = nil
		}
		r.state = Closed
		cause := fmt.Errorf("%w: %w", ErrJobCancelled, ErrRequestClosed)
		// Dependents first, so a job ends as CANCELLED rather than as its dependency's failure.
		for _, j := range slices.Backward(r.order) {
			s.settle(j, JobCancelled, "", cause)
		}
		delete(s.requests, r.id)
		r.cancel()
	})
}

// job is the registry's record. The scheduler's mutex guards it.
type job struct {
	req        *Request
	id         JobID
	key        string
	program    string
	args       []string
	timeout    time.Duration
	deps       []*job  // sorted by ID
	depNames   []JobID // the names the Execute gave them, which may be aliases
	dependents []*job
	pending    int // dependencies not yet DONE

	state      JobState
	generation Generation
	result     string
	err        error
	cancel     context.CancelFunc // ends the worker's context
	timer      *time.Timer        // the deadline
}

func (j *job) info() JobInfo {
	info := JobInfo{
		ID: j.id, IdempotencyKey: j.key, State: j.state, Generation: j.generation,
		Result: j.result, Err: j.err,
	}
	for _, d := range j.deps {
		info.Dependencies = append(info.Dependencies, d.id)
	}
	return info
}

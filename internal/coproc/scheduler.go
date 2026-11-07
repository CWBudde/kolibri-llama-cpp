package coproc

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"
)

// Config bounds a scheduler's resources. Every field must be positive.
type Config struct {
	MaxRequests          int           // open requests
	MaxJobsPerRequest    int           // job IDs in a request, aliases included
	MaxRunning           int           // jobs running at once, the others queue in order
	MaxResultBytes       int           // a larger result turns the job into an ERROR
	EventBuffer          int           // completion events queued; a worker blocks beyond this
	MaxActionsPerRequest int           // actions a request may apply, tokens included
	DefaultTimeout       time.Duration // for a job that sets none
	MaxTimeout           time.Duration // a job may not ask for more
}

// DefaultConfig returns limits for a small prototype setup.
func DefaultConfig() Config {
	return Config{
		MaxRequests:          64,
		MaxJobsPerRequest:    256,
		MaxRunning:           8,
		MaxResultBytes:       1 << 20,
		EventBuffer:          64,
		MaxActionsPerRequest: 1 << 20,
		DefaultTimeout:       30 * time.Second,
		MaxTimeout:           10 * time.Minute,
	}
}

func (c Config) validate() error {
	if c.MaxRequests <= 0 || c.MaxJobsPerRequest <= 0 || c.MaxRunning <= 0 || c.MaxResultBytes <= 0 ||
		c.EventBuffer <= 0 || c.MaxActionsPerRequest <= 0 || c.DefaultTimeout <= 0 ||
		c.DefaultTimeout > c.MaxTimeout {
		return fmt.Errorf("coproc: invalid config %+v", c)
	}
	return nil
}

// Stats counts what happened to the completion events.
type Stats struct {
	Accepted        int // settled a running job
	Late            int // right job and generation, but the job already had its outcome
	UnknownRequest  int // no open request of that ID
	UnknownJob      int // no job of that ID in the request
	StaleGeneration int // the job's generation is a different one
	Invalid         int // malformed: not a terminal status, or DONE with an error
}

// Scheduler owns the requests, their job registries and the completion channel. One mutex guards
// all the state and no external code runs under it.
type Scheduler struct {
	cfg      Config
	cp       Coprocessor
	events   chan CompletionEvent
	stop     chan struct{}
	loopDone chan struct{}
	stopOnce sync.Once

	mu       sync.Mutex
	requests map[RequestID]*Request
	queue    []*job // jobs that can start, waiting for a slot, in order; all IDLE, of open requests
	stale    bool   // a job left IDLE outside dispatch, so the queue may hold it
	running  int
	lastGen  Generation
	stopped  bool
	stats    Stats
	claims   map[*Request]chan struct{} // running generator calls, also of closed requests
}

// NewScheduler starts a scheduler. Shutdown ends it.
func NewScheduler(cp Coprocessor, cfg Config) (*Scheduler, error) {
	if cp == nil {
		return nil, errors.New("coproc: no coprocessor")
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	s := &Scheduler{
		cfg:      cfg,
		cp:       cp,
		events:   make(chan CompletionEvent, cfg.EventBuffer),
		stop:     make(chan struct{}),
		loopDone: make(chan struct{}),
		requests: make(map[RequestID]*Request),
		claims:   make(map[*Request]chan struct{}),
	}
	go s.loop()
	return s, nil
}

// Publish queues a completion event. It is safe for concurrent use, blocks while the queue is full
// and returns ErrStopped once the scheduler has stopped.
func (s *Scheduler) Publish(ev CompletionEvent) error {
	select {
	case <-s.stop:
		return ErrStopped
	default:
	}
	select {
	case s.events <- ev:
		return nil
	case <-s.stop:
		return ErrStopped
	}
}

// Stats returns the event counters.
func (s *Scheduler) Stats() Stats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.stats
}

// Shutdown closes every request, which cancels their jobs and releases their waiters, and stops
// consuming events. Events still queued refer to closed requests and are dropped. It returns once
// the consumer has ended and every generator call has returned, or ctx is done, and can be called
// again. After a nil return no generator is in use. A generator or an emit callback that calls it
// must pass a context that ends, since Shutdown then waits for its caller.
func (s *Scheduler) Shutdown(ctx context.Context) error {
	s.stopOnce.Do(func() {
		s.mu.Lock()
		s.stopped = true
		open := make([]*Request, 0, len(s.requests))
		for _, r := range s.requests {
			open = append(open, r)
		}
		s.mu.Unlock()
		for _, r := range open {
			r.Close()
		}
		close(s.stop)
	})
	select {
	case <-s.loopDone:
	case <-ctx.Done():
		return ctx.Err()
	}
	// No claim can start now: every request is closed and Open is refused.
	s.mu.Lock()
	claims := slices.Collect(maps.Values(s.claims))
	s.mu.Unlock()
	for _, claim := range claims {
		select {
		case <-claim:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

// loop is the only consumer of the channel. It blocks until an event arrives.
func (s *Scheduler) loop() {
	defer close(s.loopDone)
	for {
		select {
		case ev := <-s.events:
			s.update(func() { s.accept(ev) })
		case <-s.stop:
			return
		}
	}
}

// update runs f under the mutex, then starts the jobs that became runnable, outside it.
func (s *Scheduler) update(f func()) {
	s.mu.Lock()
	f()
	starts := s.dispatch()
	s.mu.Unlock()
	for _, st := range starts {
		if st.ctx.Err() != nil {
			continue // settled before the start, so it never needs to run
		}
		if err := s.cp.Start(st.ctx, st.spec, s.Publish); err != nil {
			s.update(func() { s.startFailed(st, err) })
		}
	}
}

// accept validates one event and applies it. Everything it rejects is only counted: the event
// can't change a request.
func (s *Scheduler) accept(ev CompletionEvent) {
	r, ok := s.requests[ev.RequestID]
	if !ok {
		s.stats.UnknownRequest++
		return
	}
	j, ok := r.jobs[ev.JobID]
	if !ok || j.id != ev.JobID {
		s.stats.UnknownJob++
		return
	}
	if ev.Generation != j.generation {
		s.stats.StaleGeneration++
		return
	}
	if !ev.Status.Terminal() || (ev.Status == JobDone && ev.Error != nil) {
		s.stats.Invalid++
		return
	}
	if j.state.Terminal() {
		s.stats.Late++
		return
	}
	if j.state != JobRunning {
		s.stats.StaleGeneration++ // generation 0: the job has not started
		return
	}
	s.stats.Accepted++
	switch {
	case ev.Status != JobDone:
		s.settle(j, ev.Status, "", outcomeError(ev.Status, ev.Error))
	case len(ev.Result) > s.cfg.MaxResultBytes:
		s.settle(j, JobError, "", fmt.Errorf("%w: %d bytes, limit %d", ErrResultTooLarge, len(ev.Result), s.cfg.MaxResultBytes))
	default:
		s.settle(j, JobDone, ev.Result, nil)
	}
}

// outcomeError makes sure a failed outcome carries an error that names its kind.
func outcomeError(status JobState, err error) error {
	var kind error
	switch status {
	case JobTimeout:
		kind = ErrJobTimeout
	case JobCancelled:
		kind = ErrJobCancelled
	default:
		if err == nil {
			return errors.New("coproc: job failed without an error")
		}
		return err
	}
	switch {
	case err == nil:
		return kind
	case errors.Is(err, kind):
		return err
	}
	return fmt.Errorf("%w: %w", kind, err)
}

// settle gives a job its outcome, once. In one critical section it frees the slot, moves a request
// blocked on exactly this job to READY, and passes the outcome on to the dependents.
func (s *Scheduler) settle(j *job, state JobState, result string, err error) {
	if j.state.Terminal() {
		return
	}
	if j.state == JobRunning {
		s.running--
		j.timer.Stop()
		j.cancel()
	} else {
		s.stale = true
	}
	j.state, j.result, j.err = state, result, err

	if r := j.req; r.state == Blocked && r.waiting == j {
		r.state = Ready
		r.waiting = nil
		close(r.wake)
		r.wake = nil
	}
	for _, d := range j.dependents {
		if state != JobDone {
			s.settle(d, JobError, "", dependencyError(j.id, err))
			continue
		}
		d.pending--
		if d.pending == 0 && d.state == JobIdle {
			s.queue = append(s.queue, d)
		}
	}
}

func dependencyError(dep JobID, err error) error {
	return fmt.Errorf("%w: %s: %w", ErrDependencyFailed, dep, err)
}

// start is a job marked RUNNING under the mutex, to be handed to the coprocessor outside it.
type start struct {
	job  *job
	ctx  context.Context
	spec JobSpec
}

// dispatch marks queued jobs RUNNING as far as there are slots. That transition happens once per
// job, which is what makes a start at most once. The deadline starts here and reaches the
// consumer as an ordinary event, so it races a completion in one place, the channel.
func (s *Scheduler) dispatch() []start {
	// Every settle runs in an update, so a settled job is out of the queue when the mutex is released.
	// This is what bounds the queue: slots can stay taken for MaxTimeout while clients keep closing.
	if s.stale {
		s.queue = slices.DeleteFunc(s.queue, func(j *job) bool { return j.state != JobIdle })
		s.stale = false
	}
	if s.stopped {
		return nil
	}
	var starts []start
	for s.running < s.cfg.MaxRunning && len(s.queue) > 0 {
		j := s.queue[0]
		s.queue[0] = nil // the backing array must not keep a closed request alive
		s.queue = s.queue[1:]
		s.lastGen++
		s.running++
		j.state = JobRunning
		j.generation = s.lastGen

		ctx, cancel := context.WithCancel(context.Background())
		j.cancel = cancel
		deadline := CompletionEvent{
			RequestID: j.req.id, JobID: j.id, Generation: j.generation, Status: JobTimeout,
			Error: fmt.Errorf("%w after %v", ErrJobTimeout, j.timeout),
		}
		j.timer = time.AfterFunc(j.timeout, func() { _ = s.Publish(deadline) })

		spec := JobSpec{
			RequestID: j.req.id, ID: j.id, IdempotencyKey: j.key, Generation: j.generation,
			Program: j.program, Args: slices.Clone(j.args), DepResults: make(map[JobID]string, len(j.depNames)),
		}
		for _, name := range j.depNames {
			spec.DepResults[name] = j.req.jobs[name].result
		}
		starts = append(starts, start{job: j, ctx: ctx, spec: spec})
	}
	return starts
}

// startFailed settles a job the coprocessor could not start.
func (s *Scheduler) startFailed(st start, err error) {
	if st.job.generation == st.spec.Generation && st.job.state == JobRunning {
		s.settle(st.job, JobError, "", fmt.Errorf("coproc: start: %w", err))
	}
}

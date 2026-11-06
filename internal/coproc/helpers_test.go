package coproc

import (
	"context"
	"errors"
	"strconv"
	"sync"
	"testing"
	"time"
)

// All tests with time run in a testing/synctest bubble: the clock is virtual, a 1 s mock job takes
// no real time, and synctest.Wait returns once every goroutine in the bubble is durably blocked. A
// lost wakeup is a deadlock, which the bubble reports as a failure instead of hanging.

// sum adds its arguments and its dependencies' results.
func sum(args []string, deps map[JobID]string) (string, error) {
	total := 0
	for _, a := range args {
		v, err := strconv.Atoi(a)
		if err != nil {
			return "", err
		}
		total += v
	}
	for _, d := range deps {
		v, err := strconv.Atoi(d)
		if err != nil {
			return "", err
		}
		total += v
	}
	return strconv.Itoa(total), nil
}

// mul multiplies its arguments.
func mul(args []string, _ map[JobID]string) (string, error) {
	total := 1
	for _, a := range args {
		v, err := strconv.Atoi(a)
		if err != nil {
			return "", err
		}
		total *= v
	}
	return strconv.Itoa(total), nil
}

var errProgram = errors.New("division by zero")

func fails(_ []string, _ map[JobID]string) (string, error) { return "", errProgram }

// programs is the mock's default set.
func programs() map[string]MockProgram {
	return map[string]MockProgram{
		"sum":    {Run: sum},
		"mul":    {Run: mul},
		"slow":   {Delay: time.Second, Run: mul},
		"hang":   {Delay: 24 * time.Hour},
		"fail":   {Delay: time.Second, Run: fails},
		"manual": {Manual: true},
	}
}

// setup returns a scheduler that shuts down, and whose workers must end, before the bubble does.
func setup(t *testing.T, change func(*Config)) (*Scheduler, *Mock) {
	t.Helper()
	m := &Mock{Programs: programs()}
	cfg := DefaultConfig()
	if change != nil {
		change(&cfg)
	}
	s, err := NewScheduler(m, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := s.Shutdown(context.Background()); err != nil {
			t.Error(err)
		}
		m.Wait()
	})
	return s, m
}

func open(t *testing.T, s *Scheduler, id RequestID, g Generator) *Request {
	t.Helper()
	r, err := s.Open(id, g)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// must applies an action that has to succeed.
func must(t *testing.T, r *Request, a Action) Observation {
	t.Helper()
	obs := r.Apply(context.Background(), a)
	if obs.Err != nil {
		t.Fatalf("%T%+v: %v", a, a, obs.Err)
	}
	return obs
}

// waitAsync starts a Wait on its own goroutine and returns its observation's channel.
func waitAsync(r *Request, id JobID) <-chan Observation {
	ch := make(chan Observation, 1)
	go func() { ch <- r.Apply(context.Background(), Wait{Job: id}) }()
	return ch
}

func jobInfo(t *testing.T, r *Request, id JobID) JobInfo {
	t.Helper()
	info, ok := r.Job(id)
	if !ok {
		t.Fatalf("no job %s", id)
	}
	return info
}

func wantState(t *testing.T, r *Request, want RequestState) {
	t.Helper()
	if got := r.State(); got != want {
		t.Fatalf("request is %s, want %s", got, want)
	}
}

func wantJob(t *testing.T, r *Request, id JobID, want JobState) {
	t.Helper()
	if got := jobInfo(t, r, id).State; got != want {
		t.Fatalf("job %s is %s, want %s", id, got, want)
	}
}

func wantErr(t *testing.T, got, want error) {
	t.Helper()
	if !errors.Is(got, want) {
		t.Fatalf("error %v, want %v", got, want)
	}
}

// done is a valid completion of a job as it stands.
func done(t *testing.T, r *Request, id JobID, result string) CompletionEvent {
	t.Helper()
	info := jobInfo(t, r, id)
	return CompletionEvent{RequestID: r.ID(), JobID: info.ID, Generation: info.Generation, Status: JobDone, Result: result}
}

// publish sends events and lets the scheduler consume them all.
func publish(t *testing.T, s *Scheduler, sync func(), events ...CompletionEvent) {
	t.Helper()
	for _, ev := range events {
		if err := s.Publish(ev); err != nil {
			t.Fatal(err)
		}
	}
	sync()
}

// timedTokens records user-visible tokens with their virtual time.
type timedTokens struct {
	mu     sync.Mutex
	start  time.Time
	tokens []string
	at     []time.Duration
}

func newTimedTokens() *timedTokens { return &timedTokens{start: time.Now()} }

func (tt *timedTokens) emit(tok string) {
	tt.mu.Lock()
	defer tt.mu.Unlock()
	tt.tokens = append(tt.tokens, tok)
	tt.at = append(tt.at, time.Since(tt.start))
}

func (tt *timedTokens) snapshot() ([]string, []time.Duration) {
	tt.mu.Lock()
	defer tt.mu.Unlock()
	return append([]string(nil), tt.tokens...), append([]time.Duration(nil), tt.at...)
}

// checkedGen fails the test if the generator is used in any state but GENERATING. It holds the
// invariant at the only place tokens come from.
type checkedGen struct {
	*MockGenerator
	t   *testing.T
	req *Request
}

func (g *checkedGen) check() {
	if st := g.req.State(); st != Generating {
		g.t.Errorf("generator used while %s", st)
	}
}

func (g *checkedGen) Step(ctx context.Context) (string, error) {
	g.check()
	return g.MockGenerator.Step(ctx)
}

func (g *checkedGen) Deliver(ctx context.Context, msg ResultMessage) error {
	g.check()
	return g.MockGenerator.Deliver(ctx, msg)
}

func openChecked(t *testing.T, s *Scheduler, id RequestID, preamble ...string) (*Request, *checkedGen) {
	t.Helper()
	g := &checkedGen{MockGenerator: &MockGenerator{Preamble: preamble}, t: t}
	g.req = open(t, s, id, g)
	return g.req, g
}

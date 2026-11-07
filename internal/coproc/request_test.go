package coproc

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func TestStateNames(t *testing.T) {
	for s, want := range map[JobState]string{
		JobIdle: "IDLE", JobRunning: "RUNNING", JobDone: "DONE", JobError: "ERROR",
		JobTimeout: "TIMEOUT", JobCancelled: "CANCELLED", JobState(99): "INVALID",
	} {
		if s.String() != want {
			t.Errorf("%d: %s, want %s", s, s, want)
		}
	}
	for s, want := range map[RequestState]string{
		Ready: "READY", Generating: "GENERATING", Blocked: "BLOCKED", Closed: "CLOSED", RequestState(99): "INVALID",
	} {
		if s.String() != want {
			t.Errorf("%d: %s, want %s", s, s, want)
		}
	}
	for s, terminal := range map[JobState]bool{
		JobIdle: false, JobRunning: false, JobDone: true, JobError: true, JobTimeout: true, JobCancelled: true,
	} {
		if s.Terminal() != terminal {
			t.Errorf("%s: terminal %v", s, s.Terminal())
		}
	}
}

func TestScriptPlaysItsActionsAndRecords(t *testing.T) {
	sc := &Script{Actions: []Action{Continue{}, Wait{Job: "a"}}}
	first, _ := sc.Next(Observation{})
	second, _ := sc.Next(Observation{Token: "x"})
	_, more := sc.Next(Observation{Token: "y"})
	if first != (Continue{}) || second != (Wait{Job: "a"}) || more {
		t.Fatalf("got %v, %v, more %v", first, second, more)
	}
	if len(sc.Observations) != 2 || sc.Observations[0].Token != "x" || sc.Observations[1].Token != "y" {
		t.Fatalf("observations %+v", sc.Observations)
	}
}

func TestNewSchedulerRejectsBadConfig(t *testing.T) {
	for name, change := range map[string]func(*Config){
		"requests": func(c *Config) { c.MaxRequests = 0 },
		"jobs":     func(c *Config) { c.MaxJobsPerRequest = 0 },
		"running":  func(c *Config) { c.MaxRunning = 0 },
		"result":   func(c *Config) { c.MaxResultBytes = 0 },
		"events":   func(c *Config) { c.EventBuffer = 0 },
		"actions":  func(c *Config) { c.MaxActionsPerRequest = 0 },
		"timeout":  func(c *Config) { c.DefaultTimeout = 0 },
		"maximum":  func(c *Config) { c.MaxTimeout = c.DefaultTimeout - 1 },
	} {
		cfg := DefaultConfig()
		change(&cfg)
		if _, err := NewScheduler(&Mock{}, cfg); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	if _, err := NewScheduler(nil, DefaultConfig()); err == nil {
		t.Error("nil coprocessor: no error")
	}
}

// Every action needs READY, so a blocked request can't produce a token or do anything else.
func TestNoActionRunsWhileBlocked(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, g := openChecked(t, s, "r", "before", "after")
		must(t, r, Continue{})
		must(t, r, Execute{Job: "a", Program: "manual"})
		must(t, r, Execute{Job: "ready", Program: "sum", Args: []string{"1"}})
		synctest.Wait()

		woken := waitAsync(r, "a")
		synctest.Wait()
		wantState(t, r, Blocked)

		ctx := context.Background()
		for _, a := range []Action{
			Continue{},
			Execute{Job: "b", Program: "manual"},
			Wait{Job: "ready"},
			ReadResult{Job: "ready"},
		} {
			obs := r.Apply(ctx, a)
			wantErr(t, obs.Err, ErrNotReady)
			if obs.Token != "" {
				t.Errorf("%T produced token %q", a, obs.Token)
			}
		}
		if g.Decoded() != 1 {
			t.Errorf("%d tokens decoded, want the one before WAIT", g.Decoded())
		}
		if len(g.Delivered()) != 0 {
			t.Errorf("delivered %v while blocked", g.Delivered())
		}
		if m.StartCount("b") != 0 {
			t.Error("job b started while blocked")
		}
		wantState(t, r, Blocked)

		publish(t, s, synctest.Wait, done(t, r, "a", "1"))
		if obs := <-woken; obs.Err != nil || obs.Status != JobDone {
			t.Fatalf("wait: %+v", obs)
		}
		// The KV cache position survived the wait: the next token is the next one.
		if tok := must(t, r, Continue{}).Token; tok != "after" {
			t.Fatalf("token %q", tok)
		}
	})
}

// blockingGen holds a token until the test releases it, like a slow decode.
type blockingGen struct {
	release chan struct{}
	entered chan struct{}
}

func (g *blockingGen) Step(context.Context) (string, error) {
	close(g.entered)
	<-g.release
	return "tok", nil
}

func (g *blockingGen) Deliver(context.Context, ResultMessage) error { return nil }

// A completion during a token doesn't touch the request; WAIT then finds it at the boundary.
func TestCompletionDuringGenerationWaitsForTheBoundary(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		g := &blockingGen{release: make(chan struct{}), entered: make(chan struct{})}
		r := open(t, s, "r", g)
		must(t, r, Execute{Job: "a", Program: "manual"})
		synctest.Wait()

		token := make(chan Observation, 1)
		go func() { token <- r.Apply(context.Background(), Continue{}) }()
		<-g.entered
		wantState(t, r, Generating)

		publish(t, s, synctest.Wait, done(t, r, "a", "7"))
		wantJob(t, r, "a", JobDone)
		wantState(t, r, Generating) // still inside the token

		wantErr(t, r.Apply(context.Background(), Wait{Job: "a"}).Err, ErrNotReady)

		close(g.release)
		if obs := <-token; obs.Err != nil || obs.Token != "tok" {
			t.Fatalf("token: %+v", obs)
		}
		wantState(t, r, Ready)
		if obs := must(t, r, Wait{Job: "a"}); obs.Status != JobDone {
			t.Fatalf("wait: %+v", obs)
		}
	})
}

func TestCloseDuringGenerationDiscardsTheToken(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		g := &blockingGen{release: make(chan struct{}), entered: make(chan struct{})}
		r := open(t, s, "r", g)
		token := make(chan Observation, 1)
		go func() { token <- r.Apply(context.Background(), Continue{}) }()
		<-g.entered
		r.Close()
		close(g.release)
		obs := <-token
		wantErr(t, obs.Err, ErrRequestClosed)
		if obs.Token != "" {
			t.Fatalf("token %q from a closed request", obs.Token)
		}
		wantState(t, r, Closed)
	})
}

func TestRejectedActionsLeaveTheRequestReady(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r")
		ctx := context.Background()
		must(t, r, Execute{Job: "a", Program: "manual"})
		synctest.Wait()

		wantErr(t, r.Apply(ctx, Wait{Job: "nope"}).Err, ErrUnknownJob)
		wantErr(t, r.Apply(ctx, ReadResult{Job: "nope"}).Err, ErrUnknownJob)
		wantErr(t, r.Apply(ctx, ReadResult{Job: "a"}).Err, ErrNotComplete)
		wantErr(t, r.Apply(ctx, Execute{Program: "sum"}).Err, ErrInvalidAction)
		wantErr(t, r.Apply(ctx, Execute{Job: "x"}).Err, ErrInvalidAction)
		wantErr(t, r.Apply(ctx, Execute{Job: "x", Program: "sum", Timeout: -1}).Err, ErrInvalidAction)
		wantErr(t, r.Apply(ctx, nil).Err, ErrInvalidAction)
		wantErr(t, r.Apply(ctx, Continue{}).Err, ErrGeneratorExhausted)
		wantState(t, r, Ready)
		if _, ok := r.Job("x"); ok {
			t.Error("a rejected EXECUTE registered its job")
		}
		if len(g.Delivered()) != 0 {
			t.Errorf("READ_RESULT of an unfinished job delivered %v", g.Delivered())
		}
	})
}

func TestReadResultDeliversToTheModelAndCanRepeat(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "sum", Args: []string{"40", "2"}})
		must(t, r, Wait{Job: "a"})
		if len(g.Delivered()) != 0 {
			t.Fatal("WAIT made the result visible; only READ_RESULT may")
		}
		for range 2 {
			obs := must(t, r, ReadResult{Job: "a"})
			if obs.Result != "42" || obs.Status != JobDone {
				t.Fatalf("read: %+v", obs)
			}
		}
		if len(g.Delivered()) != 2 {
			t.Fatalf("delivered %+v", g.Delivered())
		}
	})
}

// A generator's failure is the action's failure and must not cost the request its state.
type failingGen struct{ err error }

func (g failingGen) Step(context.Context) (string, error)         { return "", g.err }
func (g failingGen) Deliver(context.Context, ResultMessage) error { return g.err }

func TestGeneratorErrorsAreObserved(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		boom := errors.New("decode failed")
		r := open(t, s, "r", failingGen{boom})
		must(t, r, Execute{Job: "a", Program: "sum"})
		must(t, r, Wait{Job: "a"})
		ctx := context.Background()
		wantErr(t, r.Apply(ctx, Continue{}).Err, boom)
		wantErr(t, r.Apply(ctx, ReadResult{Job: "a"}).Err, boom)
		wantState(t, r, Ready)
	})
}

// The controller is not asked anything while its request is blocked: no model-side polling.
func TestControllerIsNotPolledWhileBlocked(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r")
		var mu sync.Mutex
		calls := 0
		sc := &Script{Actions: []Action{
			Execute{Job: "a", Program: "slow", Args: []string{"2"}},
			Wait{Job: "a"},
		}}
		counting := ControllerFunc(func(obs Observation) (Action, bool) {
			mu.Lock()
			calls++
			mu.Unlock()
			return sc.Next(obs)
		})
		finished := make(chan error, 1)
		go func() { finished <- r.Run(context.Background(), counting, nil) }()

		time.Sleep(999 * time.Millisecond)
		synctest.Wait()
		wantState(t, r, Blocked)
		mu.Lock()
		atBlock := calls
		mu.Unlock()
		if atBlock != 2 {
			t.Fatalf("controller called %d times before BLOCKED was reached, want 2", atBlock)
		}
		if g.Decoded() != 0 {
			t.Fatal("generator ran during the wait")
		}
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
		if calls != 3 {
			t.Fatalf("controller called %d times in total, want 3", calls)
		}
	})
}

func TestRunStopsWhenTheRequestCloses(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		sc := &Script{Actions: []Action{Execute{Job: "a", Program: "hang"}, Wait{Job: "a"}, Continue{}}}
		finished := make(chan error, 1)
		go func() { finished <- r.Run(context.Background(), sc, nil) }()
		synctest.Wait()
		wantState(t, r, Blocked)
		r.Close()
		wantErr(t, <-finished, ErrRequestClosed)
		wantErr(t, r.Apply(context.Background(), Continue{}).Err, ErrRequestClosed)
	})
}

// Close can land between an action's gate and its critical section, where only applyWait can see
// it. A waiter must then hear of the close: a retryable ErrNotReady makes Run report success for a
// job that never finished.
func TestWaitReportsAClosedRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		g := &blockingGen{release: make(chan struct{}), entered: make(chan struct{})}
		r := open(t, s, "r", g)
		must(t, r, Execute{Job: "a", Program: "manual"})
		synctest.Wait()
		ctx := context.Background()

		// Any other state stays retryable.
		token := make(chan Observation, 1)
		go func() { token <- r.Apply(ctx, Continue{}) }()
		<-g.entered
		wantErr(t, r.applyWait(ctx, Wait{Job: "a"}).Err, ErrNotReady)
		close(g.release)
		<-token

		r.Close()
		wantErr(t, r.applyWait(ctx, Wait{Job: "a"}).Err, ErrRequestClosed)
		wantErr(t, r.applyWait(ctx, Wait{Job: "nope"}).Err, ErrRequestClosed)
	})
}

// Run emits a token inside the GENERATING claim, so it can't go out after a second driver has moved
// the request on.
func TestRunEmitsInsideTheGeneratingClaim(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r", "a", "b")
		must(t, r, Execute{Job: "j", Program: "manual"})
		synctest.Wait()
		var emitted []string
		sc := &Script{Actions: []Action{Continue{}, Continue{}}}
		err := r.Run(context.Background(), sc, func(tok string) {
			emitted = append(emitted, tok)
			wantState(t, r, Generating)
			wantErr(t, r.Apply(context.Background(), Wait{Job: "j"}).Err, ErrNotReady)
		})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(emitted, []string{"a", "b"}) {
			t.Fatalf("emitted %v", emitted)
		}
		wantState(t, r, Ready)
	})
}

// The token of a request that closed during its Step is discarded, by Run as by Apply.
func TestRunDoesNotEmitTheTokenOfAClosedRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		g := &blockingGen{release: make(chan struct{}), entered: make(chan struct{})}
		r := open(t, s, "r", g)
		finished := make(chan error, 1)
		go func() {
			finished <- r.Run(context.Background(), &Script{Actions: []Action{Continue{}}}, func(tok string) {
				t.Errorf("emitted %q from a closed request", tok)
			})
		}()
		<-g.entered
		r.Close()
		close(g.release)
		wantErr(t, <-finished, ErrRequestClosed)
	})
}

// cancellingGen ends a context right after generating a token.
type cancellingGen struct {
	*MockGenerator
	cancel context.CancelFunc
	after  int
}

func (g *cancellingGen) Step(ctx context.Context) (string, error) {
	tok, err := g.MockGenerator.Step(ctx)
	if g.Decoded() == g.after {
		g.cancel()
	}
	return tok, err
}

// A token that was generated goes out even if ctx ends right after it, so the answer doesn't fall
// behind the decode position.
func TestRunEmitsATokenGeneratedBeforeCtxEnded(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		g := &cancellingGen{MockGenerator: &MockGenerator{Preamble: []string{"a", "b", "c"}}, cancel: cancel, after: 2}
		r := open(t, s, "r", g)
		var emitted []string
		sc := &Script{Actions: []Action{Continue{}, Continue{}, Continue{}}}
		wantErr(t, r.Run(ctx, sc, func(tok string) { emitted = append(emitted, tok) }), context.Canceled)
		if g.Decoded() != 2 || !slices.Equal(emitted, []string{"a", "b"}) {
			t.Fatalf("decoded %d, emitted %v", g.Decoded(), emitted)
		}
	})
}

// A context that has ended starts nothing: not through Run, and not through Apply.
func TestDeadContextStartsNoJob(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		sc := &Script{Actions: []Action{Execute{Job: "run", Program: "slow"}}}
		wantErr(t, r.Run(ctx, sc, nil), context.Canceled)
		wantErr(t, r.Apply(ctx, Execute{Job: "apply", Program: "slow"}).Err, context.Canceled)
		synctest.Wait()

		for _, id := range []JobID{"run", "apply"} {
			if _, ok := r.Job(id); ok {
				t.Errorf("job %s registered under a dead context", id)
			}
			if n := m.StartCount(string(id)); n != 0 {
				t.Errorf("job %s started %d times", id, n)
			}
		}
		wantState(t, r, Ready)
		if sc.next != 0 {
			t.Errorf("controller asked %d times under a dead context", sc.next)
		}
	})
}

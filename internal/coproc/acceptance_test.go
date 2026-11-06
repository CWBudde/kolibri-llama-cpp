package coproc

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"testing/synctest"
	"time"
)

// runScript drives a script on its own goroutine and returns when Run has.
func runScript(t *testing.T, r *Request, sc *Script, tokens *timedTokens) <-chan error {
	t.Helper()
	finished := make(chan error, 1)
	go func() { finished <- r.Run(context.Background(), sc, tokens.emit) }()
	return finished
}

// The issue's acceptance test: a mocked one-second operation returns an exact result. The request
// starts it, waits without emitting dependent text, wakes on the correct completion, reads the
// result and only then answers.
func TestAcceptanceOneSecondOperation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, g := openChecked(t, s, "r", "computing")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Continue{}, // text that depends on no job may come first
			Execute{Job: "calc", Program: "slow", Args: []string{"6", "7"}},
			Wait{Job: "calc"},
			ReadResult{Job: "calc"},
			Continue{},
			Continue{},
		}}
		finished := runScript(t, r, sc, tokens)

		time.Sleep(999 * time.Millisecond)
		synctest.Wait()
		wantState(t, r, Blocked)
		wantJob(t, r, "calc", JobRunning)
		if got, _ := tokens.snapshot(); !reflect.DeepEqual(got, []string{"computing"}) {
			t.Fatalf("tokens while blocked: %q", got)
		}
		if g.Decoded() != 1 {
			t.Fatalf("decoded %d while blocked", g.Decoded())
		}

		if err := <-finished; err != nil {
			t.Fatal(err)
		}
		got, at := tokens.snapshot()
		if !reflect.DeepEqual(got, []string{"computing", "result", "42"}) {
			t.Fatalf("answer %q", got)
		}
		if !reflect.DeepEqual(at, []time.Duration{0, time.Second, time.Second}) {
			t.Fatalf("token times %v", at)
		}

		read := sc.Observations[3]
		if read.Status != JobDone || read.Result != "42" || read.JobErr != nil || read.Err != nil {
			t.Fatalf("read: %+v", read)
		}
		if st := s.Stats(); st != (Stats{Accepted: 1}) {
			t.Fatalf("stats %+v", st)
		}
		if m.StartCount("calc") != 1 {
			t.Fatalf("%d starts", m.StartCount("calc"))
		}
		wantState(t, r, Ready)
	})
}

// A job that takes too long: the same script, and the model is told so instead of an answer.
func TestAcceptanceTimeout(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		m.Programs["late"] = MockProgram{Delay: 2 * time.Second, Run: mul}
		r, g := openChecked(t, s, "r")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Execute{Job: "calc", Program: "late", Args: []string{"6", "7"}, Timeout: time.Second},
			Wait{Job: "calc"},
			ReadResult{Job: "calc"},
			Continue{},
			Continue{},
		}}
		if err := <-runScript(t, r, sc, tokens); err != nil {
			t.Fatal(err)
		}
		got, at := tokens.snapshot()
		if !reflect.DeepEqual(got, []string{"error", "TIMEOUT"}) {
			t.Fatalf("answer %q", got)
		}
		if at[0] != time.Second {
			t.Fatalf("first token at %v", at[0])
		}
		read := sc.Observations[2]
		if read.Status != JobTimeout || read.Result != "" {
			t.Fatalf("read: %+v", read)
		}
		wantErr(t, read.JobErr, ErrJobTimeout)
		wantErr(t, g.Delivered()[0].Err, ErrJobTimeout)

		// The worker was cancelled, and past its own delay nothing changes.
		time.Sleep(5 * time.Second)
		synctest.Wait()
		wantJob(t, r, "calc", JobTimeout)
	})
}

// A job that fails: the error comes back explicitly, word for word.
func TestAcceptanceError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Execute{Job: "calc", Program: "fail"},
			Wait{Job: "calc"},
			ReadResult{Job: "calc"},
			Continue{},
			Continue{},
		}}
		if err := <-runScript(t, r, sc, tokens); err != nil {
			t.Fatal(err)
		}
		if got, _ := tokens.snapshot(); !reflect.DeepEqual(got, []string{"error", "ERROR"}) {
			t.Fatalf("answer %q", got)
		}
		waited := sc.Observations[1]
		if waited.Status != JobError || !errors.Is(waited.JobErr, errProgram) {
			t.Fatalf("wait: %+v", waited)
		}
		if !errors.Is(g.Delivered()[0].Err, errProgram) {
			t.Fatalf("delivered %+v", g.Delivered())
		}
	})
}

// A worker that reports three times: one answer, the same one.
func TestAcceptanceDuplicateEvent(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		m.Programs["noisy"] = MockProgram{Delay: time.Second, Run: mul, Repeat: 2}
		r, _ := openChecked(t, s, "r")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Execute{Job: "calc", Program: "noisy", Args: []string{"6", "7"}},
			Wait{Job: "calc"},
			ReadResult{Job: "calc"},
			Continue{},
			Continue{},
		}}
		if err := <-runScript(t, r, sc, tokens); err != nil {
			t.Fatal(err)
		}
		if got, _ := tokens.snapshot(); !reflect.DeepEqual(got, []string{"result", "42"}) {
			t.Fatalf("answer %q", got)
		}
		m.Wait()
		synctest.Wait()
		if st := s.Stats(); st != (Stats{Accepted: 1, Late: 2}) {
			t.Fatalf("stats %+v", st)
		}
	})
}

// An event with a mismatched job ID arrives first: nothing moves until the right one comes.
func TestAcceptanceMismatchedJobID(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Execute{Job: "calc", Program: "manual"},
			Execute{Job: "other", Program: "manual"},
			Wait{Job: "calc"},
			ReadResult{Job: "calc"},
			Continue{},
			Continue{},
		}}
		finished := runScript(t, r, sc, tokens)
		synctest.Wait()
		wantState(t, r, Blocked)

		match := done(t, r, "calc", "42")
		mismatched := match
		mismatched.JobID = "other"
		publish(t, s, synctest.Wait, mismatched)
		wantState(t, r, Blocked)
		unknown := match
		unknown.JobID = "elsewhere"
		publish(t, s, synctest.Wait, unknown)
		wantState(t, r, Blocked)
		if got, _ := tokens.snapshot(); len(got) != 0 {
			t.Fatalf("tokens %q", got)
		}
		if st := s.Stats(); st != (Stats{StaleGeneration: 1, UnknownJob: 1}) {
			t.Fatalf("stats %+v", st)
		}

		publish(t, s, synctest.Wait, match)
		if err := <-finished; err != nil {
			t.Fatal(err)
		}
		if got, _ := tokens.snapshot(); !reflect.DeepEqual(got, []string{"result", "42"}) {
			t.Fatalf("answer %q", got)
		}
	})
}

// Phase 4: two independent jobs run side by side, a third runs on both results, still with
// synchronous waits.
func TestTwoIndependentJobsAndADependentOne(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		m.Programs["oneSecond"] = MockProgram{Delay: time.Second, Run: mul}
		m.Programs["twoSeconds"] = MockProgram{Delay: 2 * time.Second, Run: sum}
		m.Programs["combine"] = MockProgram{Delay: time.Second, Run: sum}
		r, g := openChecked(t, s, "r")
		tokens := newTimedTokens()
		sc := &Script{Actions: []Action{
			Execute{Job: "a", Program: "oneSecond", Args: []string{"6", "7"}},
			Execute{Job: "b", Program: "twoSeconds", Args: []string{"1", "2"}},
			Execute{Job: "c", Program: "combine", Dependencies: []JobID{"a", "b"}},
			Wait{Job: "b"}, // the slower one first: a's completion in between must not wake this
			Wait{Job: "a"}, // retained
			Wait{Job: "c"},
			ReadResult{Job: "c"},
			Continue{},
			Continue{},
		}}
		finished := runScript(t, r, sc, tokens)

		time.Sleep(1500 * time.Millisecond) // a is done, b still runs, c hasn't started
		synctest.Wait()
		wantJob(t, r, "a", JobDone)
		wantJob(t, r, "b", JobRunning)
		wantJob(t, r, "c", JobIdle)
		wantState(t, r, Blocked)
		if g.Decoded() != 0 {
			t.Fatal("a token before the dependent job finished")
		}

		time.Sleep(time.Second) // 2.5 s: b is done, c runs on the two results
		synctest.Wait()
		wantJob(t, r, "b", JobDone)
		wantJob(t, r, "c", JobRunning)
		wantState(t, r, Blocked)

		if err := <-finished; err != nil {
			t.Fatal(err)
		}
		got, at := tokens.snapshot()
		if !reflect.DeepEqual(got, []string{"result", "45"}) { // 6*7 + (1+2)
			t.Fatalf("answer %q", got)
		}
		if at[0] != 3*time.Second { // 2 s in parallel, then 1 s; 4 s would mean a and b ran in series
			t.Fatalf("answered at %v", at[0])
		}

		starts := m.Starts()
		if len(starts) != 3 || starts[2].ID != "c" {
			t.Fatalf("starts %+v", starts)
		}
		want := map[JobID]string{"a": "42", "b": "3"}
		if !reflect.DeepEqual(starts[2].DepResults, want) {
			t.Fatalf("dependency results %v", starts[2].DepResults)
		}
	})
}

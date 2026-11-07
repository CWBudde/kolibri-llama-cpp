package coproc

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"testing/synctest"
	"time"
	"weak"
)

// WAIT unblocks only on its own job's outcome. Every event that doesn't match is counted and
// changes nothing, a real completion of another job included.
func TestOnlyTheMatchingEventUnblocks(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		other, _ := openChecked(t, s, "other")
		must(t, r, Execute{Job: "a", Program: "manual"})
		must(t, r, Execute{Job: "b", Program: "manual"})
		must(t, other, Execute{Job: "a", Program: "manual"})
		synctest.Wait()

		woken := waitAsync(r, "a")
		synctest.Wait()
		wantState(t, r, Blocked)

		match := done(t, r, "a", "right")
		foreign := done(t, other, "a", "foreign")
		wrongGen := match
		wrongGen.Generation += 1000
		wrongJob := match
		wrongJob.JobID = "nope"
		wrongRequest := match
		wrongRequest.RequestID = "nobody"
		sameIDsOtherGen := match
		sameIDsOtherGen.Generation = foreign.Generation // the other request's job of the same ID
		running := match
		running.Status = JobRunning
		idle := match
		idle.Status = JobIdle
		doneWithError := match
		doneWithError.Error = errors.New("both")

		publish(t, s, synctest.Wait, wrongGen, wrongJob, wrongRequest, sameIDsOtherGen, running, idle, doneWithError)
		wantState(t, r, Blocked)
		wantJob(t, r, "a", JobRunning)
		want := Stats{UnknownRequest: 1, UnknownJob: 1, StaleGeneration: 2, Invalid: 3}
		if got := s.Stats(); got != want {
			t.Fatalf("stats %+v, want %+v", got, want)
		}

		// A true completion, of the wrong job.
		publish(t, s, synctest.Wait, done(t, r, "b", "other job"))
		wantJob(t, r, "b", JobDone)
		wantState(t, r, Blocked)

		// The same job ID in another request, completing for real.
		publish(t, s, synctest.Wait, foreign)
		wantJob(t, other, "a", JobDone)
		wantState(t, r, Blocked)
		wantJob(t, r, "a", JobRunning)

		publish(t, s, synctest.Wait, match)
		if obs := <-woken; obs.Err != nil || obs.Status != JobDone || obs.Job != "a" {
			t.Fatalf("wait: %+v", obs)
		}
		wantState(t, r, Ready)
		if got := must(t, r, ReadResult{Job: "a"}).Result; got != "right" {
			t.Fatalf("result %q", got)
		}
	})
}

// An event for a job that has not started can't settle it, also with its generation 0.
func TestEventForAJobNotStartedIsDropped(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "manual"})
		must(t, r, Execute{Job: "b", Program: "manual", Dependencies: []JobID{"a"}})
		synctest.Wait()
		publish(t, s, synctest.Wait, CompletionEvent{RequestID: "r", JobID: "b", Status: JobDone})
		wantJob(t, r, "b", JobIdle)
		if got := s.Stats().StaleGeneration; got != 1 {
			t.Fatalf("stale %d", got)
		}
	})
}

func TestDuplicateCompletionsAreHarmless(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "manual"})
		must(t, r, Execute{Job: "next", Program: "manual"})
		synctest.Wait()
		first := done(t, r, "a", "first")
		second := first
		second.Result = "second"
		failure := first
		failure.Status, failure.Result, failure.Error = JobError, "", errors.New("late")

		woken := waitAsync(r, "a")
		synctest.Wait()
		publish(t, s, synctest.Wait, first, first, second, failure)
		if obs := <-woken; obs.Status != JobDone {
			t.Fatalf("wait: %+v", obs)
		}
		if got := must(t, r, ReadResult{Job: "a"}); got.Result != "first" || got.JobErr != nil {
			t.Fatalf("read: %+v", got)
		}

		// Duplicates can't wake a later wait on another job either.
		blocked := waitAsync(r, "next")
		synctest.Wait()
		publish(t, s, synctest.Wait, first, second)
		wantState(t, r, Blocked)

		st := s.Stats()
		if st.Accepted != 1 || st.Late != 5 {
			t.Fatalf("stats %+v", st)
		}
		if len(g.Delivered()) != 1 {
			t.Fatalf("delivered %+v", g.Delivered())
		}
		publish(t, s, synctest.Wait, done(t, r, "next", "x"))
		<-blocked
	})
}

// The completion comes first and stays in the registry: WAIT returns without blocking at all.
func TestCompletionBeforeWaitIsRetained(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "slow", Args: []string{"6", "7"}})
		time.Sleep(5 * time.Second)
		synctest.Wait()
		wantJob(t, r, "a", JobDone)

		before := time.Now()
		obs := must(t, r, Wait{Job: "a"})
		if obs.Status != JobDone || time.Since(before) != 0 {
			t.Fatalf("wait %+v after %v", obs, time.Since(before))
		}
		wantState(t, r, Ready)
		if got := must(t, r, ReadResult{Job: "a"}).Result; got != "42" {
			t.Fatalf("result %q", got)
		}
	})
}

// The completion and the WAIT race for real, on separate goroutines, with no time in between. A
// lost wakeup leaves the WAIT blocked forever, which the bubble reports as a deadlock.
func TestCompletionRacingWaitLosesNoWakeup(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, func(c *Config) { c.MaxJobsPerRequest = 1000 })
		r, _ := openChecked(t, s, "r")
		for i := range 500 {
			id := JobID(fmt.Sprint("j", i))
			must(t, r, Execute{Job: id, Program: "sum", Args: []string{strconv.Itoa(i)}})
			obs := must(t, r, Wait{Job: id})
			if obs.Status != JobDone {
				t.Fatalf("wait %s: %+v", id, obs)
			}
			if got := must(t, r, ReadResult{Job: id}).Result; got != strconv.Itoa(i) {
				t.Fatalf("job %s: result %q", id, got)
			}
		}
	})
}

func TestTimeoutIsDeliveredAndALateCompletionIsDropped(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "manual", Timeout: time.Second})
		must(t, r, Execute{Job: "b", Program: "manual"})
		synctest.Wait()
		late := done(t, r, "a", "too late")

		start := time.Now()
		obs := must(t, r, Wait{Job: "a"})
		if obs.Status != JobTimeout || time.Since(start) != time.Second {
			t.Fatalf("wait %+v after %v", obs, time.Since(start))
		}
		wantErr(t, obs.JobErr, ErrJobTimeout)

		// The late event can neither change the outcome nor wake a later wait.
		blocked := waitAsync(r, "b")
		synctest.Wait()
		publish(t, s, synctest.Wait, late)
		wantState(t, r, Blocked)
		wantJob(t, r, "a", JobTimeout)
		if got := s.Stats().Late; got != 1 {
			t.Fatalf("late %d", got)
		}
		publish(t, s, synctest.Wait, done(t, r, "b", "x"))
		<-blocked
		wantJob(t, r, "a", JobTimeout)
		if obs := must(t, r, ReadResult{Job: "a"}); obs.Result != "" || !errors.Is(obs.JobErr, ErrJobTimeout) {
			t.Fatalf("read: %+v", obs)
		}
	})
}

// Timer and worker fire at the same virtual instant. Either may win; the outcome has to be one of
// them, whole, and it has to stay.
func TestTimeoutRacingCompletion(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, func(c *Config) { c.MaxJobsPerRequest = 1000 })
		m.Programs["tie"] = MockProgram{Delay: time.Second, Run: mul}
		r, _ := openChecked(t, s, "r")
		for i := range 100 {
			id := JobID(fmt.Sprint("j", i))
			must(t, r, Execute{Job: id, Program: "tie", Args: []string{"3"}, Timeout: time.Second})
			obs := must(t, r, Wait{Job: id})
			switch obs.Status {
			case JobDone:
				if got := must(t, r, ReadResult{Job: id}).Result; got != "3" {
					t.Fatalf("result %q", got)
				}
			case JobTimeout:
				wantErr(t, obs.JobErr, ErrJobTimeout)
			default:
				t.Fatalf("wait: %+v", obs)
			}
			time.Sleep(time.Millisecond)
			synctest.Wait()
			wantJob(t, r, id, obs.Status)
			wantState(t, r, Ready)
		}
	})
}

func TestCancelWakesTheWaiterAndStopsTheWorker(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "hang"})
		woken := waitAsync(r, "a")
		synctest.Wait()
		wantState(t, r, Blocked)

		start := time.Now()
		if err := r.Cancel("a"); err != nil {
			t.Fatal(err)
		}
		obs := <-woken
		if obs.Status != JobCancelled || obs.Err != nil {
			t.Fatalf("wait: %+v", obs)
		}
		wantErr(t, obs.JobErr, ErrJobCancelled)
		m.Wait() // the 24 h worker has stopped, without the clock moving
		if time.Since(start) != 0 {
			t.Fatalf("took %v", time.Since(start))
		}
		synctest.Wait()
		wantJob(t, r, "a", JobCancelled)
		if got := s.Stats().Late; got != 1 {
			t.Fatalf("late %d: the worker's CANCELLED report should be a late event", got)
		}

		// Cancel of a finished job keeps its outcome; of an unknown job says so.
		must(t, r, Execute{Job: "b", Program: "sum", Args: []string{"1"}})
		must(t, r, Wait{Job: "b"})
		if err := r.Cancel("b"); err != nil {
			t.Fatal(err)
		}
		wantJob(t, r, "b", JobDone)
		wantErr(t, r.Cancel("nope"), ErrUnknownJob)
	})
}

// Cancel and completion race for real. One wins, the waiter wakes once, and it stays that way.
func TestCancelRacingCompletion(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, func(c *Config) { c.MaxJobsPerRequest = 1000 })
		r, _ := openChecked(t, s, "r")
		for i := range 300 {
			id := JobID(fmt.Sprint("j", i))
			must(t, r, Execute{Job: id, Program: "manual"})
			synctest.Wait()
			ev := done(t, r, id, "won")
			woken := waitAsync(r, id)
			synctest.Wait()

			var wg sync.WaitGroup
			wg.Go(func() {
				if err := s.Publish(ev); err != nil {
					t.Error(err)
				}
			})
			wg.Go(func() {
				if err := r.Cancel(id); err != nil {
					t.Error(err)
				}
			})
			wg.Wait()
			obs := <-woken
			synctest.Wait()
			if obs.Status != JobDone && obs.Status != JobCancelled {
				t.Fatalf("wait: %+v", obs)
			}
			wantJob(t, r, id, obs.Status)
			wantState(t, r, Ready)
		}
	})
}

func TestContextEndDuringWaitCancelsTheJob(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, g := openChecked(t, s, "r", "never")
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		sc := &Script{Actions: []Action{Execute{Job: "a", Program: "hang"}, Wait{Job: "a"}, Continue{}}}
		finished := make(chan error, 1)
		go func() { finished <- r.Run(ctx, sc, nil) }()
		time.Sleep(time.Second)
		synctest.Wait()
		wantState(t, r, Blocked)

		cancel()
		wantErr(t, <-finished, context.Canceled)
		info := jobInfo(t, r, "a")
		if info.State != JobCancelled {
			t.Fatalf("job %+v", info)
		}
		wantErr(t, info.Err, ErrJobCancelled)
		wantErr(t, info.Err, context.Canceled)
		wantState(t, r, Ready)
		if g.Decoded() != 0 {
			t.Fatal("a token after the context ended")
		}
	})
}

// A worker that outlives its request reports to nobody, also when the IDs are in use again.
func TestStaleEventsDontResurrectAClosedRequest(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "manual"})
		must(t, r, Execute{Job: "dep", Program: "manual", Dependencies: []JobID{"a"}})
		synctest.Wait()
		stale := done(t, r, "a", "stale")
		woken := waitAsync(r, "a")
		synctest.Wait()

		r.Close()
		obs := <-woken
		wantErr(t, obs.Err, ErrRequestClosed)
		wantState(t, r, Closed)
		for _, id := range []JobID{"a", "dep"} {
			info := jobInfo(t, r, id)
			if info.State != JobCancelled {
				t.Fatalf("job %s: %+v", id, info)
			}
			wantErr(t, info.Err, ErrRequestClosed)
		}
		r.Close() // a second close is a no-op
		wantErr(t, r.Cancel("a"), ErrRequestClosed)

		publish(t, s, synctest.Wait, stale)
		wantState(t, r, Closed)
		wantJob(t, r, "a", JobCancelled)
		if got := s.Stats().UnknownRequest; got != 1 {
			t.Fatalf("unknown request %d", got)
		}

		// The same request and job IDs again: only the new generation counts.
		again, _ := openChecked(t, s, "r")
		must(t, again, Execute{Job: "a", Program: "manual"})
		synctest.Wait()
		blocked := waitAsync(again, "a")
		synctest.Wait()
		publish(t, s, synctest.Wait, stale)
		wantState(t, again, Blocked)
		wantJob(t, again, "a", JobRunning)
		if got := s.Stats().StaleGeneration; got != 1 {
			t.Fatalf("stale generation %d", got)
		}
		publish(t, s, synctest.Wait, done(t, again, "a", "fresh"))
		<-blocked
		if got := must(t, again, ReadResult{Job: "a"}).Result; got != "fresh" {
			t.Fatalf("result %q", got)
		}
	})
}

func TestShutdown(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, func(c *Config) { c.MaxRunning = 1 })
		r, _ := openChecked(t, s, "r")
		other, _ := openChecked(t, s, "other")
		must(t, r, Execute{Job: "a", Program: "hang"})
		must(t, other, Execute{Job: "queued", Program: "sum"}) // no slot: MaxRunning is 1
		synctest.Wait()
		woken := waitAsync(r, "a")
		synctest.Wait()
		ev := done(t, r, "a", "late")

		start := time.Now()
		if err := s.Shutdown(context.Background()); err != nil {
			t.Fatal(err)
		}
		obs := <-woken
		wantErr(t, obs.Err, ErrRequestClosed)
		m.Wait()
		if time.Since(start) != 0 {
			t.Fatalf("shutdown took %v", time.Since(start))
		}
		wantState(t, r, Closed)
		wantState(t, other, Closed)
		wantJob(t, other, "queued", JobCancelled)
		if m.StartCount("queued") != 0 {
			t.Fatal("a queued job started during shutdown")
		}

		wantErr(t, s.Publish(ev), ErrStopped)
		_, err := s.Open("new", &MockGenerator{})
		wantErr(t, err, ErrStopped)
		wantErr(t, r.Apply(context.Background(), Continue{}).Err, ErrRequestClosed)
		if err := s.Shutdown(context.Background()); err != nil {
			t.Fatal(err)
		}
	})
}

// A worker stuck on a full queue gets out when the scheduler stops.
func TestShutdownReleasesBlockedPublishers(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, func(c *Config) { c.EventBuffer = 1 })
		var wg sync.WaitGroup
		for range 20 {
			wg.Go(func() {
				for s.Publish(CompletionEvent{RequestID: "nobody"}) == nil {
				}
			})
		}
		if err := s.Shutdown(context.Background()); err != nil {
			t.Fatal(err)
		}
		wg.Wait()
	})
}

// stuckCoprocessor breaks Start's contract: it blocks the goroutine that calls it.
type stuckCoprocessor struct{ release chan struct{} }

func (c stuckCoprocessor) Start(_ context.Context, job JobSpec, _ func(CompletionEvent) error) error {
	if job.ID == "stuck" {
		<-c.release
	}
	return nil
}

// A consumer stuck in such a Start can't end, and Shutdown gives up on its context instead of
// hanging with it.
func TestShutdownHonorsItsContext(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		c := stuckCoprocessor{release: make(chan struct{})}
		s, err := NewScheduler(c, DefaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		r := open(t, s, "r", &MockGenerator{})
		must(t, r, Execute{Job: "a", Program: "any"})
		must(t, r, Execute{Job: "stuck", Program: "any", Dependencies: []JobID{"a"}})
		synctest.Wait()
		// The consumer starts the dependent job, and that Start doesn't return.
		publish(t, s, synctest.Wait, done(t, r, "a", "x"))

		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		wantErr(t, s.Shutdown(ctx), context.DeadlineExceeded)

		close(c.release)
		if err := s.Shutdown(context.Background()); err != nil {
			t.Fatal(err)
		}
	})
}

func TestStartFailureIsAnExplicitError(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		obs := must(t, r, Execute{Job: "a", Program: "no such program"})
		if obs.Status != JobError || !strings.Contains(obs.JobErr.Error(), "no such program") {
			t.Fatalf("execute: %+v", obs)
		}
		obs = must(t, r, Wait{Job: "a"})
		if obs.Status != JobError {
			t.Fatalf("wait: %+v", obs)
		}
		if m.StartCount("a") != 0 {
			t.Fatal("recorded a start")
		}
		// The slot came back.
		must(t, r, Execute{Job: "b", Program: "sum"})
		if obs := must(t, r, Wait{Job: "b"}); obs.Status != JobDone {
			t.Fatalf("wait: %+v", obs)
		}
	})
}

// A worker reporting its own outcome gets it recorded with the kind intact.
func TestWorkerReportedOutcomes(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		for i, c := range []struct {
			status JobState
			err    error
			want   error
		}{
			{JobTimeout, nil, ErrJobTimeout},
			{JobTimeout, errProgram, errProgram},
			{JobCancelled, nil, ErrJobCancelled},
			{JobCancelled, ErrJobCancelled, ErrJobCancelled},
			{JobError, errProgram, errProgram},
		} {
			id := JobID(fmt.Sprint("j", i))
			must(t, r, Execute{Job: id, Program: "manual"})
			synctest.Wait()
			ev := done(t, r, id, "")
			ev.Status, ev.Error = c.status, c.err
			publish(t, s, synctest.Wait, ev)
			info := jobInfo(t, r, id)
			if info.State != c.status {
				t.Fatalf("%+v: job %+v", c, info)
			}
			wantErr(t, info.Err, c.want)
		}
		must(t, r, Execute{Job: "silent", Program: "manual"})
		synctest.Wait()
		ev := done(t, r, "silent", "")
		ev.Status = JobError
		publish(t, s, synctest.Wait, ev)
		if jobInfo(t, r, "silent").Err == nil {
			t.Fatal("ERROR without an error")
		}
	})
}

func TestLimits(t *testing.T) {
	t.Run("requests", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, _ := setup(t, func(c *Config) { c.MaxRequests = 1 })
			r := open(t, s, "a", &MockGenerator{})
			_, err := s.Open("a", &MockGenerator{})
			wantErr(t, err, ErrDuplicateRequest)
			_, err = s.Open("b", &MockGenerator{})
			wantErr(t, err, ErrTooManyRequests)
			if _, err := s.Open("c", nil); err == nil {
				t.Fatal("a request without a generator")
			}
			r.Close()
			open(t, s, "a", &MockGenerator{}) // the slot and the ID are free again
		})
	})
	t.Run("jobs", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, _ := setup(t, func(c *Config) { c.MaxJobsPerRequest = 2 })
			r, _ := openChecked(t, s, "r")
			must(t, r, Execute{Job: "a", Program: "sum"})
			must(t, r, Execute{Job: "b", Program: "sum"})
			wantErr(t, r.Apply(context.Background(), Execute{Job: "c", Program: "sum"}).Err, ErrTooManyJobs)
			wantErr(t, r.Apply(context.Background(), Execute{Job: "alias", IdempotencyKey: "a", Program: "sum"}).Err, ErrTooManyJobs)
			if !must(t, r, Execute{Job: "a", Program: "sum"}).Existing {
				t.Fatal("a retry within the limit should find its job")
			}
		})
	})
	t.Run("running", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, func(c *Config) { c.MaxRunning = 2 })
			r, _ := openChecked(t, s, "r")
			for _, id := range []JobID{"a", "b", "c"} {
				must(t, r, Execute{Job: id, Program: "slow"})
			}
			synctest.Wait()
			wantJob(t, r, "c", JobIdle)
			if len(m.Starts()) != 2 {
				t.Fatalf("%d started", len(m.Starts()))
			}
			start := time.Now()
			must(t, r, Wait{Job: "c"})
			if time.Since(start) != 2*time.Second {
				t.Fatalf("third job finished after %v, want 2 s: 1 s in the queue, 1 s running", time.Since(start))
			}
			if got := m.Starts(); got[2].ID != "c" {
				t.Fatalf("starts %+v", got)
			}
		})
	})
	t.Run("a timed out job frees its slot", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, _ := setup(t, func(c *Config) { c.MaxRunning = 1 })
			r, _ := openChecked(t, s, "r")
			must(t, r, Execute{Job: "a", Program: "hang", Timeout: time.Second})
			must(t, r, Execute{Job: "b", Program: "sum"})
			start := time.Now()
			must(t, r, Wait{Job: "b"})
			if time.Since(start) != time.Second {
				t.Fatalf("took %v", time.Since(start))
			}
		})
	})
	t.Run("result", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, func(c *Config) { c.MaxResultBytes = 4 })
			m.Programs["big"] = MockProgram{Run: func([]string, map[JobID]string) (string, error) { return "12345", nil }}
			r, _ := openChecked(t, s, "r")
			must(t, r, Execute{Job: "a", Program: "big"})
			obs := must(t, r, Wait{Job: "a"})
			if obs.Status != JobError {
				t.Fatalf("wait: %+v", obs)
			}
			wantErr(t, obs.JobErr, ErrResultTooLarge)
		})
	})
	t.Run("actions", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, _ := setup(t, func(c *Config) { c.MaxActionsPerRequest = 2 })
			r, _ := openChecked(t, s, "r")
			forever := ControllerFunc(func(Observation) (Action, bool) { return Wait{Job: "nope"}, true })
			wantErr(t, r.Run(context.Background(), forever, nil), ErrActionLimit)
		})
	})
	t.Run("timeout", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, _ := setup(t, func(c *Config) { c.MaxTimeout = time.Minute; c.DefaultTimeout = time.Second })
			r, _ := openChecked(t, s, "r")
			wantErr(t, r.Apply(context.Background(), Execute{Job: "a", Program: "sum", Timeout: time.Hour}).Err, ErrInvalidAction)
			must(t, r, Execute{Job: "b", Program: "hang"})
			start := time.Now()
			if obs := must(t, r, Wait{Job: "b"}); obs.Status != JobTimeout || time.Since(start) != time.Second {
				t.Fatalf("default timeout: %+v after %v", obs, time.Since(start))
			}
		})
	})
	t.Run("event queue", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, func(c *Config) { c.EventBuffer = 1; c.MaxRunning = 20 })
			m.Programs["noisy"] = MockProgram{Delay: time.Second, Run: mul, Repeat: 3}
			r, _ := openChecked(t, s, "r")
			var ids []JobID
			for i := range 20 {
				id := JobID(fmt.Sprint("j", i))
				ids = append(ids, id)
				must(t, r, Execute{Job: id, Program: "noisy", Args: []string{"2"}})
			}
			for _, id := range ids {
				if obs := must(t, r, Wait{Job: id}); obs.Status != JobDone {
					t.Fatalf("wait %s: %+v", id, obs)
				}
			}
			m.Wait()
			synctest.Wait()
			if got := s.Stats(); got.Accepted != 20 || got.Late != 60 {
				t.Fatalf("stats %+v", got)
			}
		})
	})
}

// The queue holds only jobs that can still start. Closing a request or cancelling a job takes its
// entries out at once, however long the running jobs keep the slots.
func TestQueueDropsSettledJobs(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, func(c *Config) { c.MaxRunning, c.MaxRequests, c.MaxJobsPerRequest = 1, 2, 4 })
		queued := func() int {
			s.mu.Lock()
			defer s.mu.Unlock()
			return len(s.queue)
		}
		holder := open(t, s, "holder", &MockGenerator{})
		must(t, holder, Execute{Job: "hold", Program: "manual"})
		synctest.Wait()

		for range 10 {
			r := open(t, s, "r", &MockGenerator{})
			for i := range 4 {
				must(t, r, Execute{Job: JobID(strconv.Itoa(i)), Program: "sum"})
			}
			if got := queued(); got != 4 {
				t.Fatalf("%d jobs queued, want 4", got)
			}
			r.Close()
			if got := queued(); got != 0 {
				t.Fatalf("%d jobs queued after Close, want 0", got)
			}
		}

		r := open(t, s, "r", &MockGenerator{})
		must(t, r, Execute{Job: "cancelled", Program: "sum"})
		must(t, r, Execute{Job: "left", Program: "sum", Args: []string{"3"}})
		if err := r.Cancel("cancelled"); err != nil {
			t.Fatal(err)
		}
		if got := queued(); got != 1 {
			t.Fatalf("%d jobs queued after Cancel, want 1", got)
		}

		// The job that is left still runs when the slot frees.
		publish(t, s, synctest.Wait, done(t, holder, "hold", "x"))
		wantJob(t, r, "left", JobDone)
		wantJob(t, r, "cancelled", JobCancelled)
	})
}

// A closed request must be collectable while the scheduler lives on. The queue's backing array is
// the one place that could keep it: a job taken off the head stays in the slot behind the slice.
func TestClosedRequestIsNotRetained(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, func(c *Config) { c.MaxRunning = 2 })
		holder := open(t, s, "holder", &MockGenerator{})
		must(t, holder, Execute{Job: "hold", Program: "manual"})
		synctest.Wait()

		ref := func() weak.Pointer[Request] {
			r := open(t, s, "r", &MockGenerator{})
			must(t, r, Execute{Job: "dispatched", Program: "manual"})
			synctest.Wait()
			// The second slot is taken, so these two stay queued until Close.
			must(t, r, Execute{Job: "queued", Program: "manual"})
			must(t, r, Execute{Job: "queued too", Program: "manual"})
			r.Close()
			return weak.Make(r)
		}()
		runtime.GC()
		if ref.Value() != nil {
			t.Fatal("the scheduler still holds a closed request")
		}
		wantState(t, holder, Ready) // keeps the scheduler's other request in use up to here
	})
}

// A worker owns the Args it gets. Changing them mustn't change the registry's record, or an
// identical retry reads as a conflict and a controller answering that with a new key runs the side
// effect twice.
func TestWorkerChangingItsArgsDoesNotChangeTheJob(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		m.Programs["scribble"] = MockProgram{Run: func(args []string, _ map[JobID]string) (string, error) {
			args[0] = "changed"
			return "ok", nil
		}}
		r, _ := openChecked(t, s, "r")
		a := Execute{Job: "a", Program: "scribble", Args: []string{"1"}}
		must(t, r, a)
		must(t, r, Wait{Job: "a"})
		if obs := must(t, r, a); !obs.Existing {
			t.Fatalf("retry: %+v", obs)
		}
		if n := m.StartCount("a"); n != 1 {
			t.Fatalf("started %d times", n)
		}
	})
}

// Once Shutdown has returned nil no generator call is running, so the caller can free the inference
// contexts. A generator that doesn't return on its context holds Shutdown up to Shutdown's own
// context, whether the request was still open or had been closed before.
func TestShutdownWaitsForARunningGeneratorCall(t *testing.T) {
	for name, closeFirst := range map[string]bool{"open": false, "closed": true} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				s, _ := setup(t, nil)
				g := &blockingGen{release: make(chan struct{}), entered: make(chan struct{})}
				r := open(t, s, "r", g)
				token := make(chan Observation, 1)
				go func() { token <- r.Apply(context.Background(), Continue{}) }()
				<-g.entered
				if closeFirst {
					r.Close() // returns during the call
				}

				ctx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				wantErr(t, s.Shutdown(ctx), context.DeadlineExceeded)
				wantState(t, r, Closed)

				close(g.release)
				wantErr(t, (<-token).Err, ErrRequestClosed)
				if err := s.Shutdown(context.Background()); err != nil {
					t.Fatal(err)
				}
			})
		})
	}
}

// A settled job's deadline is off: nothing fires later, and nothing stays armed up to MaxTimeout.
func TestDeadlineOfASettledJobDoesNotFire(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, _ := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "a", Program: "sum", Timeout: time.Second})
		must(t, r, Wait{Job: "a"})
		time.Sleep(2 * time.Second)
		synctest.Wait()
		wantJob(t, r, "a", JobDone)
		if got, want := s.Stats(), (Stats{Accepted: 1}); got != want {
			t.Fatalf("stats %+v, want %+v", got, want)
		}
	})
}

// gateCoprocessor holds the Start of one job, the state in which a later job of the same dispatch
// is RUNNING and not yet started.
type gateCoprocessor struct {
	*Mock
	hold    JobID
	entered chan struct{}
	release chan struct{}
}

func (c gateCoprocessor) Start(ctx context.Context, job JobSpec, publish func(CompletionEvent) error) error {
	if job.ID == c.hold {
		close(c.entered)
		<-c.release
	}
	return c.Mock.Start(ctx, job, publish)
}

// A job settled between its dispatch and its Start is not started: its side effect would come after
// its CANCELLED.
func TestJobSettledBeforeItsStartDoesNotStart(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		m := &Mock{Programs: programs()}
		gate := gateCoprocessor{Mock: m, hold: "a", entered: make(chan struct{}), release: make(chan struct{})}
		s, err := NewScheduler(gate, DefaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := s.Shutdown(context.Background()); err != nil {
				t.Error(err)
			}
			m.Wait()
		})
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "p", Program: "manual"})
		synctest.Wait()
		must(t, r, Execute{Job: "a", Program: "manual", Dependencies: []JobID{"p"}})
		must(t, r, Execute{Job: "b", Program: "manual", Dependencies: []JobID{"p"}})

		// p's completion dispatches a and b at once, and Start holds on a.
		if err := s.Publish(done(t, r, "p", "1")); err != nil {
			t.Fatal(err)
		}
		<-gate.entered
		wantJob(t, r, "b", JobRunning)
		if err := r.Cancel("b"); err != nil {
			t.Fatal(err)
		}
		close(gate.release)
		synctest.Wait()

		wantJob(t, r, "b", JobCancelled)
		if n := m.StartCount("b"); n != 0 {
			t.Fatalf("job b started %d times after its CANCELLED", n)
		}
		if n := m.StartCount("a"); n != 1 {
			t.Fatalf("job a started %d times", n)
		}
	})
}

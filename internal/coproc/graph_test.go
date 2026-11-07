package coproc

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"strconv"
	"sync"
	"testing"
	"testing/synctest"
	"time"
)

func TestUnknownDependencyIsRejected(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		obs := r.Apply(context.Background(), Execute{Job: "a", Program: "sum", Dependencies: []JobID{"nope"}})
		wantErr(t, obs.Err, ErrUnknownDependency)
		if _, ok := r.Job("a"); ok {
			t.Fatal("rejected job is registered")
		}
		synctest.Wait()
		if len(m.Starts()) != 0 {
			t.Fatal("rejected job started")
		}

		// A job of another request is unknown here, whatever its ID.
		other, _ := openChecked(t, s, "other")
		must(t, other, Execute{Job: "elsewhere", Program: "sum"})
		obs = r.Apply(context.Background(), Execute{Job: "b", Program: "sum", Dependencies: []JobID{"elsewhere"}})
		wantErr(t, obs.Err, ErrUnknownDependency)
	})
}

// Dependencies must exist already, so a cycle can't be declared: its first forward edge is an
// unknown ID, a self-edge is named as what it is, and an existing job's edges can't change.
func TestCyclesCannotBeDeclared(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		ctx := context.Background()

		wantErr(t, r.Apply(ctx, Execute{Job: "self", Program: "sum", Dependencies: []JobID{"self"}}).Err, ErrCycle)
		if _, ok := r.Job("self"); ok {
			t.Fatal("self-cycle registered")
		}

		// x -> y -> x, in both declaration orders.
		wantErr(t, r.Apply(ctx, Execute{Job: "x", Program: "sum", Dependencies: []JobID{"y"}}).Err, ErrUnknownDependency)
		must(t, r, Execute{Job: "y", Program: "manual"})
		must(t, r, Execute{Job: "x", Program: "manual", Dependencies: []JobID{"y"}})
		obs := r.Apply(ctx, Execute{Job: "y", Program: "manual", Dependencies: []JobID{"x"}})
		wantErr(t, obs.Err, ErrIdempotencyConflict)
		if deps := jobInfo(t, r, "y").Dependencies; len(deps) != 0 {
			t.Fatalf("y got the back edge: %v", deps)
		}

		// p -> q -> o -> p: closing the longer cycle through an alias changes nothing either.
		must(t, r, Execute{Job: "o", Program: "manual"})
		must(t, r, Execute{Job: "q", Program: "manual", Dependencies: []JobID{"o"}})
		must(t, r, Execute{Job: "p", Program: "manual", Dependencies: []JobID{"q"}})
		obs = r.Apply(ctx, Execute{Job: "o2", IdempotencyKey: "o", Program: "manual", Dependencies: []JobID{"p"}})
		wantErr(t, obs.Err, ErrIdempotencyConflict)

		synctest.Wait()
		for _, st := range m.Starts() {
			if st.ID == "x" || st.ID == "p" || st.ID == "q" {
				t.Fatalf("job %s started with its dependencies open", st.ID)
			}
		}
	})
}

// Random plans, forward references included: whatever gets accepted starts in dependency order and
// computes exactly what an independent evaluation of the plan says.
func TestRandomPlansRunInDependencyOrder(t *testing.T) {
	for seed := range uint64(20) {
		t.Run(strconv.FormatUint(seed, 10), func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				rng := rand.New(rand.NewPCG(seed, 1))
				s, m := setup(t, func(c *Config) { c.MaxRunning = 1 + rng.IntN(4) })
				m.Programs["work"] = MockProgram{Delay: time.Millisecond, Run: sum}
				r, _ := openChecked(t, s, "r")

				want := map[JobID]int{}
				deps := map[JobID][]JobID{}
				var accepted []JobID
				for i := range 60 {
					id := JobID(fmt.Sprint("j", i))
					var named []JobID
					for range rng.IntN(4) {
						// Half a step beyond the declared range: some of these are forward edges.
						named = append(named, JobID(fmt.Sprint("j", rng.IntN(i+1+i/2))))
					}
					obs := r.Apply(context.Background(), Execute{
						Job: id, Program: "work", Args: []string{strconv.Itoa(i)}, Dependencies: named,
					})
					if obs.Err != nil {
						if !errors.Is(obs.Err, ErrUnknownDependency) && !errors.Is(obs.Err, ErrCycle) {
							t.Fatalf("%s: %v", id, obs.Err)
						}
						continue
					}
					total := i
					seen := map[JobID]bool{}
					for _, d := range named {
						if _, known := want[d]; !known {
							t.Fatalf("%s accepted with unknown dependency %s", id, d)
						}
						if !seen[d] {
							seen[d] = true
							total += want[d]
							deps[id] = append(deps[id], d)
						}
					}
					want[id] = total
					accepted = append(accepted, id)
				}

				for _, id := range accepted {
					if obs := must(t, r, Wait{Job: id}); obs.Status != JobDone {
						t.Fatalf("wait %s: %+v", id, obs)
					}
					if got := must(t, r, ReadResult{Job: id}).Result; got != strconv.Itoa(want[id]) {
						t.Fatalf("%s: result %s, want %d", id, got, want[id])
					}
				}
				at := map[JobID]int{}
				for i, st := range m.Starts() {
					at[st.ID] = i
				}
				if len(at) != len(accepted) {
					t.Fatalf("%d jobs started, %d accepted", len(at), len(accepted))
				}
				for id, ds := range deps {
					for _, d := range ds {
						if at[d] >= at[id] {
							t.Fatalf("%s started before its dependency %s", id, d)
						}
					}
				}
			})
		})
	}
}

func TestFailurePropagatesThroughTheGraph(t *testing.T) {
	for _, c := range []struct {
		name    string
		program string
		timeout time.Duration
		cancel  bool
		state   JobState
		cause   error
	}{
		{name: "error", program: "fail", state: JobError, cause: errProgram},
		{name: "timeout", program: "hang", timeout: time.Second, state: JobTimeout, cause: ErrJobTimeout},
		{name: "cancel", program: "hang", cancel: true, state: JobCancelled, cause: ErrJobCancelled},
	} {
		t.Run(c.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				s, m := setup(t, nil)
				r, _ := openChecked(t, s, "r")
				must(t, r, Execute{Job: "root", Program: c.program, Timeout: c.timeout})
				must(t, r, Execute{Job: "mid", Program: "sum", Dependencies: []JobID{"root"}})
				must(t, r, Execute{Job: "leaf", Program: "sum", Dependencies: []JobID{"mid"}})
				must(t, r, Execute{Job: "free", Program: "sum", Args: []string{"5"}})

				// The waiter sits on the end of the chain and is woken by the root's failure.
				woken := waitAsync(r, "leaf")
				synctest.Wait()
				wantState(t, r, Blocked)
				if c.cancel {
					if err := r.Cancel("root"); err != nil {
						t.Fatal(err)
					}
				}
				obs := <-woken
				if obs.Err != nil || obs.Status != JobError {
					t.Fatalf("wait: %+v", obs)
				}
				wantErr(t, obs.JobErr, ErrDependencyFailed)
				wantErr(t, obs.JobErr, c.cause)

				wantJob(t, r, "root", c.state)
				wantJob(t, r, "mid", JobError)
				wantJob(t, r, "free", JobDone)
				for _, id := range []string{"mid", "leaf"} {
					if m.StartCount(id) != 0 {
						t.Fatalf("job %s started although its dependency failed", id)
					}
				}

				// A job declared on a dependency that has failed already never starts either.
				late := must(t, r, Execute{Job: "late", Program: "sum", Dependencies: []JobID{"root", "free"}})
				if late.Status != JobError {
					t.Fatalf("late: %+v", late)
				}
				wantErr(t, late.JobErr, c.cause)
				if m.StartCount("late") != 0 {
					t.Fatal("late job started")
				}
			})
		})
	}
}

// A job that was queued and gets cancelled doesn't start, and its slot goes to the next one.
func TestCancelBeforeStart(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, func(c *Config) { c.MaxRunning = 1 })
		r, _ := openChecked(t, s, "r")
		must(t, r, Execute{Job: "running", Program: "slow"})
		must(t, r, Execute{Job: "queued", Program: "sum"})
		must(t, r, Execute{Job: "after", Program: "sum"})
		if err := r.Cancel("queued"); err != nil {
			t.Fatal(err)
		}
		must(t, r, Wait{Job: "after"})
		if m.StartCount("queued") != 0 {
			t.Fatal("the cancelled job started")
		}
		wantJob(t, r, "queued", JobCancelled)
		if jobInfo(t, r, "queued").Generation != 0 {
			t.Fatal("the cancelled job got a generation")
		}
	})
}

func TestIdempotentRetries(t *testing.T) {
	t.Run("same action", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			a := Execute{Job: "a", Program: "slow", Args: []string{"2"}}
			if must(t, r, a).Existing {
				t.Fatal("the first EXECUTE found a job")
			}
			if !must(t, r, a).Existing {
				t.Fatal("the retry didn't find the job")
			}
			must(t, r, Wait{Job: "a"})
			if !must(t, r, a).Existing {
				t.Fatal("the retry after completion didn't find the job")
			}
			synctest.Wait()
			if m.StartCount("a") != 1 {
				t.Fatalf("%d starts", m.StartCount("a"))
			}
		})
	})
	t.Run("new job ID with the same key", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			must(t, r, Execute{Job: "first", IdempotencyKey: "charge-card", Program: "slow", Args: []string{"9"}})
			obs := must(t, r, Execute{Job: "retry", IdempotencyKey: "charge-card", Program: "slow", Args: []string{"9"}})
			if !obs.Existing || obs.Job != "first" {
				t.Fatalf("retry: %+v", obs)
			}
			if obs := must(t, r, Wait{Job: "retry"}); obs.Status != JobDone || obs.Job != "first" {
				t.Fatalf("wait: %+v", obs)
			}
			if got := must(t, r, ReadResult{Job: "retry"}).Result; got != "9" {
				t.Fatalf("result %q", got)
			}
			if m.StartCount("charge-card") != 1 {
				t.Fatalf("%d starts", m.StartCount("charge-card"))
			}
		})
	})
	t.Run("conflicts", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			ctx := context.Background()
			must(t, r, Execute{Job: "a", IdempotencyKey: "k", Program: "slow", Args: []string{"1"}})
			must(t, r, Execute{Job: "b", Program: "slow"})
			for name, a := range map[string]Execute{
				"other args":    {Job: "c", IdempotencyKey: "k", Program: "slow", Args: []string{"2"}},
				"other program": {Job: "c", IdempotencyKey: "k", Program: "sum", Args: []string{"1"}},
				"other timeout": {Job: "c", IdempotencyKey: "k", Program: "slow", Args: []string{"1"}, Timeout: time.Minute},
				"other deps":    {Job: "c", IdempotencyKey: "k", Program: "slow", Args: []string{"1"}, Dependencies: []JobID{"b"}},
				"ID, other key": {Job: "a", IdempotencyKey: "k2", Program: "slow", Args: []string{"1"}},
				"key of b":      {Job: "a", IdempotencyKey: "b", Program: "slow"},
			} {
				if err := r.Apply(ctx, a).Err; !errors.Is(err, ErrIdempotencyConflict) {
					t.Errorf("%s: %v", name, err)
				}
			}
			synctest.Wait()
			if len(m.Starts()) != 2 {
				t.Fatalf("%d starts", len(m.Starts()))
			}
		})
	})
	t.Run("a failed job stays failed", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			a := Execute{Job: "a", Program: "fail"}
			must(t, r, a)
			must(t, r, Wait{Job: "a"})
			retry := must(t, r, a)
			if !retry.Existing || retry.Status != JobError || !errors.Is(retry.JobErr, errProgram) {
				t.Fatalf("retry: %+v", retry)
			}
			// Its side effects may have happened. Running it again takes a new key, on purpose.
			again := must(t, r, Execute{Job: "a2", IdempotencyKey: "second try", Program: "fail"})
			if again.Existing {
				t.Fatalf("new key: %+v", again)
			}
			must(t, r, Wait{Job: "a2"})
			if m.StartCount("a") != 1 || m.StartCount("second try") != 1 {
				t.Fatalf("starts %d and %d", m.StartCount("a"), m.StartCount("second try"))
			}
		})
	})
	t.Run("keys are per request", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			other, _ := openChecked(t, s, "other")
			must(t, r, Execute{Job: "a", Program: "sum"})
			must(t, other, Execute{Job: "a", Program: "sum"})
			must(t, r, Wait{Job: "a"})
			must(t, other, Wait{Job: "a"})
			if m.StartCount("a") != 2 {
				t.Fatalf("%d starts", m.StartCount("a"))
			}
		})
	})
	t.Run("concurrent retries", func(t *testing.T) {
		synctest.Test(t, func(t *testing.T) {
			s, m := setup(t, nil)
			r, _ := openChecked(t, s, "r")
			var wg sync.WaitGroup
			for range 64 {
				wg.Go(func() {
					obs := r.Apply(context.Background(), Execute{Job: "a", Program: "slow", Args: []string{"3"}})
					if obs.Err != nil || obs.Job != "a" {
						t.Errorf("execute: %+v", obs)
					}
				})
			}
			wg.Wait()
			must(t, r, Wait{Job: "a"})
			if m.StartCount("a") != 1 {
				t.Fatalf("%d starts", m.StartCount("a"))
			}
		})
	})
}

// A worker finds a dependency's result under the name its Execute gave it, also if that name is an
// alias, and gets it once: a program that adds up its dependencies mustn't count an alias twice.
func TestDependencyNamedByAnAlias(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s, m := setup(t, nil)
		r, _ := openChecked(t, s, "r")
		first := Execute{Job: "first", IdempotencyKey: "k", Program: "sum", Args: []string{"4", "5"}}
		must(t, r, first)
		retry := first
		retry.Job = "retry"
		if obs := must(t, r, retry); !obs.Existing || obs.Job != "first" {
			t.Fatalf("retry: %+v", obs)
		}
		must(t, r, Wait{Job: "first"})

		deps := func(job JobID) map[JobID]string {
			t.Helper()
			for _, spec := range m.Starts() {
				if spec.ID == job {
					return spec.DepResults
				}
			}
			t.Fatalf("job %s did not start", job)
			return nil
		}

		c := Execute{Job: "c", Program: "sum", Dependencies: []JobID{"retry"}}
		must(t, r, c)
		must(t, r, Wait{Job: "c"})
		if got := deps("c"); len(got) != 1 || got["retry"] != "9" {
			t.Fatalf("job c got %v, want retry:9 alone", got)
		}
		if got := jobInfo(t, r, "c"); got.State != JobDone || got.Result != "9" {
			t.Fatalf("job c: %+v", got)
		}

		both := Execute{Job: "both", Program: "sum", Dependencies: []JobID{"retry", "first", "retry"}}
		must(t, r, both)
		must(t, r, Wait{Job: "both"})
		if got := deps("both"); len(got) != 2 || got["retry"] != "9" || got["first"] != "9" {
			t.Fatalf("job both got %v, want first:9 and retry:9", got)
		}

		// Both names mean one job, so naming it the other way is still a retry and starts nothing.
		c.Dependencies = []JobID{"first"}
		if obs := must(t, r, c); !obs.Existing {
			t.Fatalf("retry of c: %+v", obs)
		}
		if n := m.StartCount("c"); n != 1 {
			t.Fatalf("job c started %d times", n)
		}
	})
}

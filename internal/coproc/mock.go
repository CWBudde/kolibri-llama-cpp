package coproc

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// MockProgram is one program of the mock coprocessor.
type MockProgram struct {
	Delay time.Duration // run time; a cancelled job stops at once
	// Run computes the result from the arguments and the dependencies' results.
	Run func(args []string, deps map[JobID]string) (string, error)
	// Repeat publishes the completion that many extra times, as a misbehaving worker would.
	Repeat int
	// Manual starts no worker: whoever holds the scheduler publishes the completion.
	Manual bool
}

// Mock is a coprocessor of deterministic programs. A worker that is cancelled reports it, and that
// event is usually stale by then, as a real late worker's would be.
type Mock struct {
	Programs map[string]MockProgram

	mu     sync.Mutex
	starts []JobSpec
	wg     sync.WaitGroup
}

// Start implements Coprocessor.
func (m *Mock) Start(ctx context.Context, job JobSpec, publish func(CompletionEvent) error) error {
	p, ok := m.Programs[job.Program]
	if !ok {
		return fmt.Errorf("mock: no program %q", job.Program)
	}
	m.mu.Lock()
	m.starts = append(m.starts, job)
	m.mu.Unlock()
	if p.Manual {
		return nil
	}
	m.wg.Add(1)
	go func() {
		defer m.wg.Done()
		ev := CompletionEvent{RequestID: job.RequestID, JobID: job.ID, Generation: job.Generation}
		timer := time.NewTimer(p.Delay)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			ev.Status, ev.Error = JobCancelled, context.Cause(ctx)
			_ = publish(ev)
			return
		case <-timer.C:
		}
		var err error
		if p.Run != nil {
			ev.Result, err = p.Run(job.Args, job.DepResults)
		}
		ev.Status = JobDone
		if err != nil {
			ev.Status, ev.Result, ev.Error = JobError, "", err
		}
		for range p.Repeat + 1 {
			if publish(ev) != nil {
				return
			}
		}
	}()
	return nil
}

// Wait returns once every worker has ended.
func (m *Mock) Wait() { m.wg.Wait() }

// Starts lists the jobs started so far, in order.
func (m *Mock) Starts() []JobSpec {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]JobSpec(nil), m.starts...)
}

// StartCount counts the starts under an idempotency key.
func (m *Mock) StartCount(key string) int {
	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for _, s := range m.starts {
		if s.IdempotencyKey == key {
			n++
		}
	}
	return n
}

// ErrGeneratorExhausted is a MockGenerator's answer once it has no tokens left.
var ErrGeneratorExhausted = errors.New("mock: generator has no tokens")

// MockGenerator produces fixed tokens: a preamble that depends on no job, then for every delivered
// message "result <value>" or "error <status>". Delivery is the only way a result reaches its
// tokens, as with a model that sees only its context.
type MockGenerator struct {
	Preamble []string

	mu        sync.Mutex
	queue     []string
	started   bool
	decoded   int
	delivered []ResultMessage
}

// Step implements Generator.
func (g *MockGenerator) Step(ctx context.Context) (string, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if !g.started {
		g.started = true
		g.queue = append(append([]string(nil), g.Preamble...), g.queue...)
	}
	if len(g.queue) == 0 {
		return "", ErrGeneratorExhausted
	}
	tok := g.queue[0]
	g.queue = g.queue[1:]
	g.decoded++
	return tok, nil
}

// Deliver implements Generator.
func (g *MockGenerator) Deliver(ctx context.Context, msg ResultMessage) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	g.delivered = append(g.delivered, msg)
	if msg.Status == JobDone {
		g.queue = append(g.queue, "result", msg.Result)
	} else {
		g.queue = append(g.queue, "error", msg.Status.String())
	}
	return nil
}

// Decoded counts the tokens generated so far, the mock's KV cache position.
func (g *MockGenerator) Decoded() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.decoded
}

// Delivered lists the delivered messages.
func (g *MockGenerator) Delivered() []ResultMessage {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]ResultMessage(nil), g.delivered...)
}

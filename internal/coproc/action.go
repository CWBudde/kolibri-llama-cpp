package coproc

import "time"

// Action is one typed step of a controller: Continue, Execute, Wait or ReadResult.
type Action interface{ isAction() }

// Continue generates one answer token.
type Continue struct{}

// Execute registers a job. It starts once its dependencies are DONE and a slot is free. Execute
// returns at once; only Wait blocks.
type Execute struct {
	Job            JobID
	IdempotencyKey string // defaults to the job ID; scope is the request
	Program        string
	Args           []string
	Dependencies   []JobID       // must already be declared in this request
	Timeout        time.Duration // run time limit, 0 means Config.DefaultTimeout
}

// Wait moves the request to BLOCKED until the job has an outcome. A job that already has one
// returns at once. Wait reports the outcome's status and error; the result itself comes from
// ReadResult, because event delivery and model-visible delivery are separate steps.
type Wait struct{ Job JobID }

// ReadResult delivers a finished job's result or error to the generator and to the controller. It
// never blocks: a job without an outcome gets ErrNotComplete.
type ReadResult struct{ Job JobID }

func (Continue) isAction()   {}
func (Execute) isAction()    {}
func (Wait) isAction()       {}
func (ReadResult) isAction() {}

// Observation is the outcome of an action.
type Observation struct {
	Action   Action
	Err      error    // the action itself was rejected or failed
	Job      JobID    // the job's canonical ID, which a retry under an old key maps to
	Existing bool     // Execute found the job under its idempotency key and started nothing
	Status   JobState // the job's state when the action finished
	JobErr   error    // the job's failure, for a job in a failed terminal state
	Result   string   // ReadResult of a DONE job
	Token    string   // Continue
}

// Controller decides a request's actions. Next gets the previous action's observation, a zero
// Observation at first, and returns false to finish. It is never called while the request is
// BLOCKED.
type Controller interface {
	Next(Observation) (Action, bool)
}

// ControllerFunc adapts a function to a Controller.
type ControllerFunc func(Observation) (Action, bool)

// Next calls f.
func (f ControllerFunc) Next(obs Observation) (Action, bool) { return f(obs) }

// Script is a controller that plays a fixed list of actions and records what it observed.
type Script struct {
	Actions      []Action
	Observations []Observation
	next         int
}

// Next returns the next action in the list.
func (s *Script) Next(obs Observation) (Action, bool) {
	if s.next > 0 {
		s.Observations = append(s.Observations, obs)
	}
	if s.next == len(s.Actions) {
		return nil, false
	}
	s.next++
	return s.Actions[s.next-1], true
}

// ResultMessage is the result channel into the inference context: a job's result or error, as the
// model gets to see it. A later hidden-state adapter can replace this delivery.
type ResultMessage struct {
	Job    JobID
	Status JobState
	Result string
	Err    error
}

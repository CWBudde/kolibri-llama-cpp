package locality

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"testing"
)

var updateFixture = flag.Bool("update-fixture", false, "rewrite testdata/locality/markov-fixture.json")

const markovFixture = "../../testdata/locality/markov-fixture.json"

// markovFixtureBytes replays a small random conversation through the Markov predictor, same-token
// horizon, as a Prefetcher does: before every step it predicts, after it observes the step's
// experts. The fixture holds the steps and, per parameter set, every prediction. The llama.cpp
// expert cache's C++ port of the predictor reads it (tests/test-expert-cache) and must predict the
// same experts in the same order.
func markovFixtureBytes() []byte {
	const layers, tokens, k, nExpert, top = 6, 300, 3, 24, 5
	r := rand.New(rand.NewPCG(51, 52))
	sel := randomSelections(r, layers, tokens, k, nExpert)
	answers := []Answer{{Prompt: 70, Tokens: make([]int32, 60)}, {Prompt: 180, Tokens: make([]int32, 100)}}
	spans, err := OnlineSpans(tokens, answers, true, 32)
	if err != nil {
		panic(err)
	}
	params := []MarkovParams{{HalfLife: 1, Mix: 0.75}, {HalfLife: 4, Mix: 0.25}, {HalfLife: 16, Mix: 0}, {HalfLife: 64, Mix: 1}}

	type step struct {
		First   int   `json:"first"`
		Decode  bool  `json:"decode"`
		Layer   int   `json:"layer"`
		Experts []int `json:"experts"`
	}
	var steps []step
	for _, s := range spans {
		for l := range layers {
			steps = append(steps, step{s.First, s.Phase == Decode, l, stepExperts(sel, s, l, nil)})
		}
	}

	var b bytes.Buffer
	line := func(v any) {
		j, err := json.Marshal(v)
		if err != nil {
			panic(err)
		}
		b.Write(j)
	}
	fmt.Fprintf(&b, "{\"layers\": %d, \"n_expert\": %d, \"k\": %d,\n\"params\": ", layers, nExpert, top)
	line(params)
	b.WriteString(",\n\"steps\": [\n")
	for i, s := range steps {
		line(s)
		if i < len(steps)-1 {
			b.WriteString(",")
		}
		b.WriteString("\n")
	}
	// per parameter set and step: the predicted layer and experts, layer -1 for no prediction
	b.WriteString("],\n\"predictions\": [\n")
	for pi, p := range params {
		m := NewMarkov(layers, nExpert, p)
		var out []int
		var preds [][]int
		for i, s := range steps {
			info := StepInfo{i, spans[i/layers], s.Layer}
			target, ids := m.Predict(SameToken, info, top, out[:0])
			if ids == nil {
				target = -1
			}
			preds = append(preds, append([]int{target}, ids...))
			m.Observe(info, s.Experts)
		}
		line(preds)
		if pi < len(params)-1 {
			b.WriteString(",")
		}
		b.WriteString("\n")
	}
	b.WriteString("]}\n")
	return b.Bytes()
}

func TestMarkovFixture(t *testing.T) {
	want := markovFixtureBytes()
	if *updateFixture {
		if err := os.WriteFile(markovFixture, want, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	got, err := os.ReadFile(markovFixture)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s differs from the predictor's own predictions; rerun with -update-fixture if the predictor changed on purpose", markovFixture)
	}
	// the fixture must exercise the predictor: every parameter set predicts often
	var f struct {
		Predictions [][][]int `json:"predictions"`
	}
	if err := json.Unmarshal(got, &f); err != nil {
		t.Fatal(err)
	}
	for pi, preds := range f.Predictions {
		n := 0
		for _, p := range preds {
			if p[0] >= 0 {
				n++
			}
		}
		if n < 100 {
			t.Fatalf("parameter set %d predicts only %d times", pi, n)
		}
	}
}

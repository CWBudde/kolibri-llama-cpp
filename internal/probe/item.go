// Package probe holds the dataset and the readout of the layer-depth probe of issue #63: which
// Kolibri layers carry usable information about whether to execute a program and what class of
// operation is needed. It generates and verifies the prompts, reads captured activations, and
// trains and reports regularized linear probes under grouped cross-validation. It observes only:
// nothing here changes a weight or runs a coprocessor. See docs/layer-probe.md.
package probe

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// The categories of the issue, and the label values the dataset uses.
const (
	CategoryNoExecution  = "no_execution"
	CategoryArithmetic   = "exact_arithmetic"
	CategoryAlgorithm    = "small_algorithm"
	CategoryNonExecution = "explicit_non_execution"

	FamilyNone       = "none"
	FamilyArithmetic = "arithmetic"
	FamilyAlgorithm  = "algorithm"

	SplitPilot = "pilot"
	SplitSmoke = "smoke"

	ConstraintNone   = "none"
	ConstraintForbid = "forbid"
	ConstraintDemand = "demand"

	NeedNone      = "none"
	NeedHelpful   = "helpful"
	NeedNecessary = "necessary"

	// NoExactAnswer is expected_answer_or_oracle of a prompt with no exact answer to check.
	NoExactAnswer = "no-exact-answer"

	// LabelPolicy names the policy behind execute_expected: execute when a prompt asks for an exact
	// number or for an algorithm's result and does not forbid it. It says "helpful", not
	// "necessary"; execution_need keeps the difference.
	LabelPolicy = "execute-when-exact-result-asked-v1"
)

// Families are the operation family classes in the order the probe uses.
var Families = []string{FamilyNone, FamilyArithmetic, FamilyAlgorithm}

// Item is one prompt. The first eight fields are the issue's schema, in its order. The capture's
// metadata (model and tokenizer revision, layer index, activation dtype) belongs to a capture, not
// to a prompt, and is in its capture.json.
type Item struct {
	ID              string `json:"id"`
	Category        string `json:"category"`
	Prompt          string `json:"prompt"`
	ExecuteExpected bool   `json:"execute_expected"`
	OperationFamily string `json:"operation_family"`
	Difficulty      string `json:"difficulty"`
	ParaphraseGroup string `json:"paraphrase_group"`
	Split           string `json:"split"`

	Language   string `json:"language"`
	TemplateID string `json:"template_id"`
	// NumericSeed drew the group's numbers; null for a prompt without drawn numbers.
	NumericSeed  *uint32 `json:"numeric_seed"`
	PromptLength int     `json:"prompt_length"` // in characters
	// ExplicitExecutionConstraint is what the prompt itself says: forbid (do not compute), demand
	// (compute exactly) or none.
	ExplicitExecutionConstraint string `json:"explicit_execution_constraint"`
	ExpectedAnswerOrOracle      string `json:"expected_answer_or_oracle"`

	// Operation is the operation the prompt asks for or only mentions, "" if none: a category D
	// prompt mentions one (family none, nothing is needed), and that is its temptation.
	Operation string `json:"operation"`
	// ExecutionNeed separates "helpful" from "strictly necessary", as the issue asks.
	ExecutionNeed string `json:"execution_need"`
	LabelPolicy   string `json:"label_policy"`
	// HardNegative marks a negative with a tempting keyword or number.
	HardNegative bool `json:"hard_negative"`
	// VariedAxis is the one axis this row varies against its group's first row; "base" for that row.
	VariedAxis string `json:"varied_axis"`
}

// MarshalJSONL writes items as JSON lines. The bytes are the file's definition: no HTML escaping,
// the fields in struct order, one line per item.
func MarshalJSONL(items []Item) ([]byte, error) {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	for _, it := range items {
		if err := enc.Encode(it); err != nil {
			return nil, err
		}
	}
	return b.Bytes(), nil
}

// ReadJSONL reads items, refusing unknown fields, so a schema change cannot pass unnoticed.
func ReadJSONL(r io.Reader) ([]Item, error) {
	var out []Item
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for n := 1; sc.Scan(); n++ {
		dec := json.NewDecoder(bytes.NewReader(sc.Bytes()))
		dec.DisallowUnknownFields()
		var it Item
		if err := dec.Decode(&it); err != nil {
			return nil, fmt.Errorf("line %d: %w", n, err)
		}
		out = append(out, it)
	}
	return out, sc.Err()
}

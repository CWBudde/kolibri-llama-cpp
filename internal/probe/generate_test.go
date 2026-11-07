package probe

import (
	"bytes"
	"os"
	"regexp"
	"strings"
	"testing"
)

const smokePath = "../../testdata/probe/smoke.jsonl"

func TestRegenerateBytes(t *testing.T) {
	want, err := os.ReadFile(smokePath)
	if err != nil {
		t.Fatal(err)
	}
	got, err := MarshalJSONL(Generate(Seed))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("regenerating %s gives other bytes; run go run ./cmd/kolibri-probe generate, unless a real capture has read the file", smokePath)
	}
	other, err := MarshalJSONL(Generate(Seed + 1))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(other, want) {
		t.Fatal("another seed gives the same bytes: the numbers are not drawn from the seed")
	}
}

func TestReadJSONLRoundTrip(t *testing.T) {
	items := Generate(Seed)
	data, err := MarshalJSONL(items)
	if err != nil {
		t.Fatal(err)
	}
	back, err := ReadJSONL(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	again, err := MarshalJSONL(back)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, again) {
		t.Fatal("JSONL round trip changed the bytes")
	}
	if _, err := ReadJSONL(strings.NewReader(`{"id":"X","nope":1}` + "\n")); err == nil {
		t.Fatal("an unknown field passed")
	}
}

// The issue's example row and its four pilot prompts, word for word.
func TestPilotIsTheIssues(t *testing.T) {
	want := map[string]string{
		"A-001": "Was ist die Hauptstadt von Frankreich?",
		"B-001": "Berechne 17893 mal 24761 exakt.",
		"C-001": "Sortiere die Zahlen 17, 3, 29, 8, 11 aufsteigend.",
		"D-001": "Erkläre den euklidischen Algorithmus, aber rechne nichts aus.",
	}
	items := Generate(Seed)
	pilot := 0
	for _, it := range items {
		if it.Split != SplitPilot {
			continue
		}
		pilot++
		if want[it.ID] != it.Prompt {
			t.Errorf("%s: prompt %q, want %q", it.ID, it.Prompt, want[it.ID])
		}
		if it.ID == "B-001" {
			if it.Category != "exact_arithmetic" || !it.ExecuteExpected || it.OperationFamily != "arithmetic" ||
				it.Difficulty != "medium" || it.ParaphraseGroup != "multiplication-01" {
				t.Errorf("B-001 differs from the issue's example: %+v", it)
			}
		}
	}
	if pilot != 4 || len(items) != 44 {
		t.Fatalf("%d pilot prompts of %d, want 4 of 44", pilot, len(items))
	}
}

func TestDatasetVerifies(t *testing.T) {
	items := Generate(Seed)
	for _, err := range VerifyAnswers(items) {
		t.Error(err)
	}
	for _, err := range VerifyStructure(items) {
		t.Error(err)
	}
}

// The oracle's answers on values worked out by hand, so that it does not merely agree with the
// generator.
func TestOracleKnownValues(t *testing.T) {
	for _, c := range []struct{ op, prompt, want string }{
		{"multiplication", "Berechne 17893 mal 24761 exakt.", "443048573"},
		{"multiplication", "Berechne 1.000.000 × 1.001 exakt.", "1001000000"},
		{"multiplication", "Compute 1,000,000 × 1,001 exactly.", "1001000000"},
		{"multiplication", "Multipliziere 10 000 mit 20 000.", "200000000"},
		{"exponentiation", "Wie lautet der exakte Wert von 2^61?", "2305843009213693952"},
		{"exponentiation", "Berechne 3 hoch 4.", "81"},
		{"factorial", "Berechne 20! exakt.", "2432902008176640000"},
		{"gcd", "Bestimme den ggT von 1071 und 462.", "21"},
		{"primality", "Ist 7919 eine Primzahl?", "true"},
		{"primality", "Is 1001 a prime number?", "false"},
		{"fibonacci", "Berechne die Fibonacci-Zahl F(10) mit F(1) = F(2) = 1.", "55"},
		{"fibonacci", "Give the exact Fibonacci number F(90), with F(1) = F(2) = 1.", "2880067194370816120"},
		{"sorting", "Sortiere die Zahlen 17, 3, 29, 8, 11 aufsteigend.", "3, 8, 11, 17, 29"},
		{"sorting", "Sortiere die Zahlen 4, 9, 1 absteigend.", "9, 4, 1"},
		{"filtering_even", "Behalte aus der Liste 12, 5, 8, 21 nur die geraden Zahlen.", "12, 8"},
		{"filtering_greater", "From the list 12, 60, 8, 90, keep only the numbers greater than 50.", "60, 90"},
	} {
		got, err := Oracle(c.op, c.prompt)
		if err != nil {
			t.Errorf("%q: %v", c.prompt, err)
		} else if got != c.want {
			t.Errorf("%q: %s, want %s", c.prompt, got, c.want)
		}
	}
}

// Control M78: a wrong answer or a changed operand fails the oracle, on every row with an answer.
func TestM78OracleCatchesCorruption(t *testing.T) {
	digit := regexp.MustCompile(`\d`)
	for i, it := range Generate(Seed) {
		switch it.Category {
		case CategoryArithmetic, CategoryAlgorithm:
		default:
			continue
		}
		wrong := it
		wrong.ExpectedAnswerOrOracle = "wrong" + it.ExpectedAnswerOrOracle
		if len(VerifyAnswers([]Item{wrong})) == 0 {
			t.Errorf("row %d (%s): a wrong answer passed", i, it.ID)
		}
		// the answer stays, a number in the prompt moves: the oracle reads the prompt, not the generator
		changed := it
		loc := digit.FindStringIndex(it.Prompt)
		d := byte('1')
		if it.Prompt[loc[0]] == '1' {
			d = '2'
		}
		changed.Prompt = it.Prompt[:loc[0]] + string(d) + it.Prompt[loc[1]:]
		if len(VerifyAnswers([]Item{changed})) == 0 && it.Operation != "primality" && !strings.HasPrefix(it.Operation, "filtering") {
			t.Errorf("row %d (%s): a changed operand passed: %q", i, it.ID, changed.Prompt)
		}
	}
	// a row of a kind that has no answer
	for _, it := range Generate(Seed) {
		if it.Category != CategoryNonExecution {
			continue
		}
		bad := it
		bad.Prompt = "Berechne 3 hoch 4."
		if len(VerifyAnswers([]Item{bad})) == 0 {
			t.Errorf("%s: a category D prompt that does not forbid execution passed", it.ID)
		}
	}
}

func TestStructureRefuses(t *testing.T) {
	for name, edit := range map[string]func([]Item){
		"duplicate prompt":   func(s []Item) { s[5].Prompt = s[6].Prompt },
		"template shared":    func(s []Item) { s[5].TemplateID = s[20].TemplateID },
		"seed shared":        func(s []Item) { s[20].NumericSeed = s[30].NumericSeed },
		"label flipped":      func(s []Item) { s[20].ExecuteExpected = !s[20].ExecuteExpected },
		"near-identical":     func(s []Item) { s[7].Prompt = s[4].Prompt + " Kurz." },
		"pilot group reused": func(s []Item) { s[4].ParaphraseGroup = s[0].ParaphraseGroup },
		"count":              func(s []Item) { s[4].Split = SplitPilot },
	} {
		items := Generate(Seed)
		edit(items)
		if len(VerifyStructure(items)) == 0 {
			t.Errorf("%s: no error", name)
		}
	}
}

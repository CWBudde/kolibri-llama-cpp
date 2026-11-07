package probe

import (
	"bytes"
	"os"
	"regexp"
	"strings"
	"testing"
)

const smokePath = "../../testdata/probe/smoke.jsonl"

func TestCommittedDatasetHash(t *testing.T) {
	data, err := os.ReadFile(smokePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := SHA256Hex(data); got != CommittedDatasetSHA256 {
		t.Fatalf("%s has sha256 %s, the definitions say %s", smokePath, got, CommittedDatasetSHA256)
	}
}

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

// The oracle reads the operation from the prompt, so a row's operation field cannot choose what it
// computes: an answer that follows a wrong operation fails too.
func TestOracleReadsTheOperation(t *testing.T) {
	for prompt, want := range map[string]string{
		"Was ist 36 mal 49?":                          "multiplication",
		"Berechne das exakte Produkt 1 × 2.":          "multiplication",
		"Berechne 6 hoch 21.":                         "exponentiation",
		"Wie lautet der exakte Wert von 2^3?":         "exponentiation",
		"Wie groß ist 15! genau?":                     "factorial",
		"What is the factorial of 12?":                "factorial",
		"Bestimme den ggT von 12 und 18.":             "gcd",
		"Is 1001 a prime number?":                     "primality",
		"Berechne die Fibonacci-Zahl F(5).":           "fibonacci",
		"Sortiere absteigend: 1, 2.":                  "sorting",
		"Welche Zahlen der Liste 1 sind gerade?":      "filtering_even",
		"From the list 1, keep those greater than 0.": "filtering_greater",
	} {
		if got, err := PromptOperation(prompt); err != nil || got != want {
			t.Errorf("%q: %q, %v, want %q", prompt, got, err, want)
		}
	}
	for _, prompt := range []string{"Was ist die Hauptstadt von Spanien?", "Berechne 3 mal 4 hoch 5."} {
		if got, err := PromptOperation(prompt); err == nil {
			t.Errorf("%q: %q, want a refusal", prompt, got)
		}
	}
	for _, it := range Generate(Seed) {
		if !it.ExecuteExpected {
			continue
		}
		other := "gcd"
		if it.Operation == other {
			other = "multiplication"
		}
		wrong := it
		wrong.Operation = other
		if len(VerifyAnswers([]Item{wrong})) == 0 {
			t.Errorf("%s: operation %s passed", it.ID, other)
		}
		if it.ID == "B-002" {
			wrong.ExpectedAnswerOrOracle = "1" // what gcd(36, 49) gives
			if len(VerifyAnswers([]Item{wrong})) == 0 {
				t.Errorf("%s: an answer that follows a wrong operation passed", it.ID)
			}
		}
	}
}

func byID(items []Item, id string) *Item {
	for i := range items {
		if items[i].ID == id {
			return &items[i]
		}
	}
	panic("no item " + id)
}

// A refusal carries its partner's numbers and operation, and execution_need follows the numbers.
func TestPairsAndNeed(t *testing.T) {
	for name, edit := range map[string]func([]Item){
		"other numbers": func(s []Item) {
			byID(s, "D-002").Prompt = "Erkläre das schriftliche Multiplizieren am Beispiel 11 mal 13, aber rechne das Ergebnis nicht aus."
		},
		"other order": func(s []Item) {
			byID(s, "D-005").Prompt = "Erkläre, wie der euklidische Algorithmus den ggT von 1680 und 1020 findet, aber rechne nichts aus."
		},
		"no numbers":      func(s []Item) { byID(s, "D-007").Prompt = "Erkläre Bubblesort, aber sortiere nicht." },
		"other operation": func(s []Item) { byID(s, "D-002").Operation = "gcd" },
		"no partner":      func(s []Item) { byID(s, "D-002").ParaphraseGroup = "alone-01" },
	} {
		items := Generate(Seed)
		edit(items)
		if len(verifyPairs(items)) == 0 {
			t.Errorf("%s: verifyPairs passed", name)
		}
		if len(VerifyStructure(items)) == 0 {
			t.Errorf("%s: VerifyStructure passed", name)
		}
	}
	for id, need := range map[string]string{"C-002": NeedNecessary, "B-002": NeedNecessary, "B-003": NeedHelpful, "C-006": NeedHelpful} {
		items := Generate(Seed)
		bad := byID(items, id)
		bad.ExecutionNeed = need
		if len(VerifyAnswers([]Item{*bad})) == 0 {
			t.Errorf("%s: execution_need %s passed", id, need)
		}
	}
}

func TestStructureRefuses(t *testing.T) {
	for name, edit := range map[string]func([]Item){
		"duplicate prompt": func(s []Item) { s[5].Prompt = s[6].Prompt },
		"template shared":  func(s []Item) { s[5].TemplateID = s[20].TemplateID },
		"seed shared":      func(s []Item) { s[20].NumericSeed = s[30].NumericSeed },
		"label flipped":    func(s []Item) { s[20].ExecuteExpected = !s[20].ExecuteExpected },
		"near-identical":   func(s []Item) { s[7].Prompt = s[4].Prompt + " Kurz." },
		// frames of two groups: one word, or a few, apart; each was once in the dataset
		"frame, smoke": func(s []Item) {
			byID(s, "C-009").Prompt = "Behalte aus der Liste 81, 92, 7, 90, 74, 55 nur die geraden Zahlen."
		},
		"frame, pilot and smoke": func(s []Item) { byID(s, "A-002").Prompt = "Was ist die Hauptstadt von Spanien?" },
		"frame, pilot's sort": func(s []Item) {
			byID(s, "C-002").Prompt = "Sortiere die Zahlen 53, 97, 40, 51 absteigend."
		},
		"frame, pilot's product": func(s []Item) { byID(s, "B-003").Prompt = "Berechne 9.694.976 × 5.310 exakt." },
		"pilot group reused":     func(s []Item) { s[4].ParaphraseGroup = s[0].ParaphraseGroup },
		"count":                  func(s []Item) { s[4].Split = SplitPilot },
	} {
		items := Generate(Seed)
		edit(items)
		if len(VerifyStructure(items)) == 0 {
			t.Errorf("%s: no error", name)
		}
	}
}

// kolibri-admission replays the Stage A workloads through the expert slots of the streaming cache
// and asks what the slots hold when a prompt's prefill ends (PLAN Phase 11, Stage E4, admission):
// how well the slots that a waves prefill leaves serve the decode that follows, and whether
// seeding them differently at that point would serve it better.
//
// It reads what cmd/kolibri-policy reads: a model's selection files and token IDs in -dir, the
// answers of the traces in -traces, and the train/test splits in -splits. A trace is replayed in
// the order an online server meets it (locality.OnlineSpans, with -ubatch): each prompt's new
// tokens in ubatches, then the answer token by token. Prefill ubatches run through a model of the
// fork's waves (locality.ReplayAdmission), decoded tokens through its plain remap, in -slots slots
// per layer. The whole trace is replayed, and only the decoded tokens of the test suffix are
// scored, as in kolibri-policy.
//
// At the end of each prefill phase one variant sets the slots:
//
//   - today: as the prefill left them;
//   - lfu: the experts the prompt's prefill used most;
//   - recency: the experts of the prompt's last tokens;
//   - decode-set: what the slots held before the prefill, as if it had bypassed them;
//   - future-lfu: the experts the following decode uses most. Not implementable, and a frequency
//     seed that reads the answer, not an upper bound: another resident set or order can hit more.
//
// It prints a Markdown report, also written to -md, and writes admission-<model>.json to -out.
// It loads no weights.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"kolibri-llm/internal/locality"
)

// Model are the fields of a model in testdata/locality/expert-bytes.json the replay needs.
type Model struct {
	NExpert int `json:"n_expert"`
	Layers  []struct {
		BytesPerExpert int64 `json:"bytes_per_expert"`
	} `json:"layers"`
}

// SplitManifest is testdata/locality/splits.json: the rule and each workload's split.
type SplitManifest struct {
	Rule      string                   `json:"rule"`
	Workloads map[string]SplitWorkload `json:"workloads"`
}

// SplitWorkload is a workload's declared split: the first TrainAnswers of its Answers answers are
// training, the test suffix starts at FirstTestToken.
type SplitWorkload struct {
	Answers        int `json:"answers"`
	TrainAnswers   int `json:"train_answers"`
	FirstTestToken int `json:"first_test_token"`
}

// split checks a declared split against the rule (the first ⌊n/2⌋ answers train) and the trace,
// and returns the index of the first test span, as cmd/kolibri-policy does.
func split(d SplitWorkload, spans []locality.Span, answers []locality.Answer) (int, error) {
	if d.Answers != len(answers) || d.TrainAnswers != len(answers)/2 {
		return 0, fmt.Errorf("%d of %d answers train, the trace has %d answers and the rule trains on %d",
			d.TrainAnswers, d.Answers, len(answers), len(answers)/2)
	}
	cut, err := locality.SplitAfter(spans, answers, d.TrainAnswers)
	if err != nil {
		return 0, err
	}
	if spans[cut].First != d.FirstTestToken {
		return 0, fmt.Errorf("the test suffix starts at token %d, declared %d", spans[cut].First, d.FirstTestToken)
	}
	return cut, nil
}

// Assumptions are the settings a report was made with.
type Assumptions struct {
	Ubatch    int    `json:"ubatch"`
	Batch     int    `json:"batch"`
	Batches   string `json:"batches"` // server or ubatch
	Waves     bool   `json:"waves"`
	LastLayer string `json:"last_layer"` // outputs or all
	First     []int  `json:"first"`
	SplitRule string `json:"split_rule,omitempty"`
}

// Row is one variant's replay of one workload at one slot count, scored over the test suffix.
type Row struct {
	Model    string `json:"model"`
	Workload string `json:"workload"`
	Slots    int    `json:"slots"`
	Variant  string `json:"variant"`
	Layers   int    `json:"layers"`
	locality.AdmissionResult
}

// Output is admission-<model>.json.
type Output struct {
	Command     string      `json:"command"`
	Assumptions Assumptions `json:"assumptions"`
	Rows        []Row       `json:"rows"`
	Totals      []Row       `json:"totals"`
}

// evaluate replays a workload with every variant at every slot count.
func evaluate(model, wl string, sel locality.Selections, spans []locality.Span, cut, nExpert int, slots []int,
	a Assumptions) ([]Row, error) {
	var rows []Row
	for _, n := range slots {
		cfg := locality.AdmissionConfig{Slots: n, NExpert: nExpert, Waves: a.Waves,
			LastLayerOutputs: a.LastLayer == "outputs", First: a.First}
		for _, v := range locality.AdmissionVariants {
			r, err := locality.ReplayAdmission(sel, spans, cut, cfg, v)
			if err != nil {
				return nil, fmt.Errorf("%d slots, %v: %w", n, v, err)
			}
			rows = append(rows, Row{model, wl, n, v.String(), sel.Layers, r})
		}
	}
	return rows, nil
}

// totals adds the rows of all workloads up, per model, slot count and variant, in the order of
// their first rows.
func totals(rows []Row) []Row {
	var out []Row
	at := map[[3]string]int{}
	for _, r := range rows {
		key := [3]string{r.Model, strconv.Itoa(r.Slots), r.Variant}
		i, ok := at[key]
		if !ok {
			at[key] = len(out)
			t := r
			t.Workload = "all"
			t.First = append([]locality.AdmissionWindow(nil), r.First...)
			out = append(out, t)
			continue
		}
		t := &out[i]
		add := func(to *locality.AdmissionWindow, w locality.AdmissionWindow) {
			to.Tokens += w.Tokens
			to.Needed += w.Needed
			to.Hits += w.Hits
		}
		for j, w := range r.First {
			add(&t.First[j], w)
		}
		add(&t.All, r.All)
		add(&t.Steady, r.Steady)
		t.Prompts += r.Prompts
		t.Seeded += r.Seeded
		t.SeedLoads += r.SeedLoads
		t.PrefillNeeded += r.PrefillNeeded
		t.PrefillLoads += r.PrefillLoads
		t.WaveSteps += r.WaveSteps
	}
	return out
}

func short(model string) string { return strings.TrimPrefix(model, "Kolibri-1-") }

func variantLabel(v string) string {
	if v == locality.AdmitOracle.String() {
		return v + " (not implementable)"
	}
	return v
}

func window(w locality.AdmissionWindow) string {
	return fmt.Sprintf("%.1f%% | %.1f", 100*w.HitRate(), w.MissesPerToken())
}

func writeMarkdown(w io.Writer, command string, a Assumptions, rows []Row) {
	fmt.Fprintf(w, "# Admission: the slots after a waves prefill, replayed\n\n")
	fmt.Fprintf(w, "Generated by `%s`; do not edit by hand.\n\n", command)
	fmt.Fprintf(w, "An offline replay of the Stage A traces through a model of the fork's expert slots "+
		"(`internal/locality/admission.go`): prefill in ubatches of %d tokens, ", a.Ubatch)
	if a.Batches == "server" {
		fmt.Fprintf(w, "cut as llama-server cuts a prompt (batches of %d tokens that also end %d and 4 tokens before the "+
			"prompt does, so its last 4 tokens are a ubatch of their own), ", a.Batch, 4+a.Ubatch)
	}
	if a.Waves {
		fmt.Fprintf(w, "in waves (`--expert-cache-waves`) when a ubatch can touch more experts than a layer has slots, ")
	}
	fmt.Fprintf(w, "decoded tokens through the plain LRU remap, a token's experts pinned together. ")
	if a.LastLayer == "outputs" {
		fmt.Fprintf(w, "The last layer runs a prompt's last token only, as the graph does. ")
	}
	fmt.Fprintf(w, "No prefetcher. The whole trace is replayed and only the decoded tokens of the test suffixes "+
		"(`testdata/locality/splits.json`) are scored.\n\n")
	fmt.Fprintf(w, "When a prompt's prefill ends, a variant sets the slots:\n\n"+
		"- `today`: as the prefill left them.\n"+
		"- `lfu`: the experts the prompt's prefill used most, the most used the most recent.\n"+
		"- `recency`: the experts of the prompt's last tokens, walking back until the slots are full.\n"+
		"- `decode-set`: what the slots held before the prefill began, as if it had bypassed them.\n"+
		"- `future-lfu`: the experts the following decode uses most. **Not implementable**: it reads the answer. "+
		"It is a frequency seed, not an upper bound of seeding: a set or order chosen against the decode's "+
		"sequence can hit more.\n\n"+
		"A seeding variant keeps residents it does not replace, older than its seeds.\n\n")
	fmt.Fprintf(w, "**Columns.** hit: experts found resident of experts needed, over all layers. miss/tok: experts "+
		"loaded per decoded token, summed over the layers. The windows are each answer's first decoded tokens; "+
		"\"tokens %d+\" is the steady-state reference. seed loads: experts the variant seeds that the prefill "+
		"had not left resident, per prompt summed over the layers, and per prompt and layer; they are loads on top "+
		"of the prefill's, before the first decoded token. For `decode-set` they are what restoring costs here; a "+
		"prefill that truly bypasses the slots loads none of them and needs its experts elsewhere.\n", a.First[len(a.First)-1]+1)

	all := append(append([]Row(nil), rows...), totals(rows)...)
	type table struct {
		model string
		slots int
	}
	var tables []table
	seen := map[table]bool{}
	for _, r := range rows {
		if t := (table{r.Model, r.Slots}); !seen[t] {
			seen[t] = true
			tables = append(tables, t)
		}
	}
	for _, t := range tables {
		var layers, prompts, tokens int
		for _, r := range all {
			if r.Model == t.model && r.Slots == t.slots && r.Workload == "all" && r.Variant == "today" {
				layers, prompts, tokens = r.Layers, r.Prompts, r.All.Tokens
			}
		}
		fmt.Fprintf(w, "\n## %s, %d slots per layer\n\n", short(t.model), t.slots)
		fmt.Fprintf(w, "%d layers; the test suffixes hold %d prompts and %d decoded tokens.\n\n", layers, prompts, tokens)
		head, rule := "| workload | variant |", "|---|---|"
		for _, n := range a.First {
			head += fmt.Sprintf(" first %d: hit | miss/tok |", n)
			rule += "---:|---:|"
		}
		head += fmt.Sprintf(" all: hit | miss/tok | tokens %d+: hit | miss/tok | seed loads/prompt | per layer |",
			a.First[len(a.First)-1]+1)
		rule += "---:|---:|---:|---:|---:|---:|"
		fmt.Fprintf(w, "%s\n%s\n", head, rule)
		for _, r := range all {
			if r.Model != t.model || r.Slots != t.slots {
				continue
			}
			wl := r.Workload
			if wl == "all" {
				wl = "**all**"
			}
			fmt.Fprintf(w, "| %s | %s |", wl, variantLabel(r.Variant))
			for _, f := range r.First {
				fmt.Fprintf(w, " %s |", window(f))
			}
			perPrompt := float64(r.SeedLoads) / float64(max(r.Prompts, 1))
			fmt.Fprintf(w, " %s | %s | %.0f | %.1f |\n", window(r.All), window(r.Steady), perPrompt,
				perPrompt/float64(max(r.Layers, 1)))
		}
	}
}

// Validation is a replay of a workload's first tokens as one prompt from empty slots, and the
// prefill counters of a server run of the same prompt, if given.
type Validation struct {
	Model    string                  `json:"model"`
	Workload string                  `json:"workload"`
	Tokens   int                     `json:"tokens"`
	Spans    []locality.Span         `json:"-"`
	Startup  int                     `json:"startup_decodes"`
	Replay   locality.PrefillCounts  `json:"replay"`
	Runtime  *locality.PrefillCounts `json:"runtime,omitempty"`
	// Decoded are the trace's tokens after the prompt replayed as decoded tokens, and Decode their
	// counters: comparable with a run only if it generated exactly these tokens.
	Decoded       int                     `json:"decoded"`
	Decode        locality.PrefillCounts  `json:"decode"`
	RuntimeDecode *locality.PrefillCounts `json:"runtime_decode,omitempty"`
}

// validate replays the first n tokens of sel as one prompt, in the ubatches the server cuts it
// into (or plain ubatches, given batches "ubatch"), after startup stand-ins for the server's
// startup decodes, and then the next decoded tokens of the trace as decoded tokens.
func validate(sel locality.Selections, n, decoded int, startup locality.Selections, nExpert, slots int,
	a Assumptions) (Validation, error) {
	if n < 1 || decoded < 0 || n+decoded > sel.Tokens {
		return Validation{}, fmt.Errorf("the first %d and %d decoded of %d tokens", n, decoded, sel.Tokens)
	}
	spans := locality.ServerSpans(0, n, a.Batch, a.Ubatch)
	if a.Batches == "ubatch" {
		spans = nil
		for t := 0; t < n; t += a.Ubatch {
			spans = append(spans, locality.Span{Phase: locality.Prefill, First: t, End: min(t+a.Ubatch, n)})
		}
	}
	cfg := locality.AdmissionConfig{Slots: slots, NExpert: nExpert, Waves: a.Waves, LastLayerOutputs: a.LastLayer == "outputs",
		First: a.First}
	all := spans
	for t := n; t < n+decoded; t++ {
		all = append(all[:len(all):len(all)], locality.Span{Phase: locality.Decode, First: t, End: t + 1})
	}
	c, d, err := locality.ReplayPrefill(sel, all, startup, cfg)
	return Validation{Tokens: n, Spans: spans, Startup: startup.Tokens / 2, Replay: c, Decoded: decoded, Decode: d}, err
}

// standIn is the stand-in for n startup decodes of a server: the trace's first two tokens, n times.
func standIn(sel locality.Selections, n int) locality.Selections {
	out := locality.Selections{Layers: sel.Layers, Tokens: 2 * n, K: sel.K}
	for l := range sel.Layers {
		for range n {
			out.IDs = append(out.IDs, sel.IDs[l*sel.Tokens*sel.K:(l*sel.Tokens+2)*sel.K]...)
		}
	}
	return out
}

// runtimeCounts reads the prefill and decode counters of a server's --expert-cache-stats file.
func runtimeCounts(path string) (prefill, decode *locality.PrefillCounts, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var stats struct {
		Prefill *locality.PrefillCounts `json:"prefill"`
		Decode  *locality.PrefillCounts `json:"decode"`
	}
	if err := json.Unmarshal(b, &stats); err != nil || stats.Prefill == nil || stats.Decode == nil {
		return nil, nil, fmt.Errorf("%s: no prefill and decode counters (%v)", path, err)
	}
	return stats.Prefill, stats.Decode, nil
}

func writeValidation(w io.Writer, slots int, vs []Validation) {
	fmt.Fprintf(w, "The first tokens of a workload as one prompt, from empty slots, replayed and as a server run counted "+
		"them (`--expert-cache-stats`), summed over the layers. Ubatches in parentheses are startup decodes of two tokens.\n\n")
	fmt.Fprintf(w, "| model | workload | phase | tokens | ubatches | source | steps | needed | hits | misses | evictions | waves | waves built |\n")
	fmt.Fprintf(w, "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
	rows := func(v Validation, phase string, tokens int, sizes string, d locality.PrefillCounts, r *locality.PrefillCounts) {
		row := func(source string, c locality.PrefillCounts) {
			fmt.Fprintf(w, "| %s | %s | %s | %d | %s | %s | %d | %d | %d | %d | %d | %d | %d |\n", short(v.Model), v.Workload,
				phase, tokens, sizes, source, c.Steps, c.Needed, c.Hits, c.Misses, c.Evictions, c.Waves, c.WavesBuilt)
		}
		row(fmt.Sprintf("replay, %d slots", slots), d)
		if r != nil {
			row("runtime", *r)
			row("replay - runtime", locality.PrefillCounts{Steps: d.Steps - r.Steps, Needed: d.Needed - r.Needed,
				Hits: d.Hits - r.Hits, Misses: d.Misses - r.Misses, Evictions: d.Evictions - r.Evictions,
				Waves: d.Waves - r.Waves, WavesBuilt: d.WavesBuilt - r.WavesBuilt})
		}
	}
	for _, v := range vs {
		var sizes []string
		for range v.Startup {
			sizes = append(sizes, "(2)")
		}
		for _, s := range v.Spans {
			sizes = append(sizes, strconv.Itoa(s.End-s.First))
		}
		rows(v, "prefill", v.Tokens, strings.Join(sizes, "+"), v.Replay, v.Runtime)
		if v.Decoded > 0 {
			rows(v, "decode", v.Decoded, "", v.Decode, v.RuntimeDecode)
		}
	}
}

func readJSON(path string, v any) {
	b, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		log.Fatalf("%s: %v", path, err)
	}
}

// ints reads a comma-separated list of positive counts.
func ints(s string) ([]int, error) {
	var out []int
	for f := range strings.SplitSeq(s, ",") {
		c, err := strconv.Atoi(f)
		if err != nil || c < 1 {
			return nil, fmt.Errorf("%q is not a positive count", f)
		}
		out = append(out, c)
	}
	return out, nil
}

func main() {
	home, _ := os.UserHomeDir()
	dir := flag.String("dir", filepath.Join(home, "models", "eval", "locality"), "selection files and token IDs")
	models := flag.String("models", "Kolibri-1-IQ3_XXS-IQ4_XS-down-imx,Kolibri-1-Q8_0", "comma-separated GGUF names without .gguf")
	workloads := flag.String("workloads", "coding,research,hr,medtech", "comma-separated workloads")
	traces := flag.String("traces", "testdata/locality", "the workload traces")
	splitsPath := flag.String("splits", "testdata/locality/splits.json", "the workloads' train/test splits")
	bytesPath := flag.String("bytes", "testdata/locality/expert-bytes.json", "the models' expert counts")
	slotsFlag := flag.String("slots", "128", "slots per layer, comma-separated")
	ubatch := flag.Int("ubatch", 512, "prefill ubatch size (llama-server's default)")
	firstFlag := flag.String("first", "16,64", "the windows scored: an answer's first decoded tokens, ascending")
	waves := flag.Bool("waves", true, "prefill in waves (--expert-cache-waves); without, a ubatch that exceeds the slots fails")
	lastLayer := flag.String("last-layer", "outputs", "what the last layer runs of a prompt: outputs (its last token, "+
		"as the graph does) or all")
	validateFirst := flag.Int("validate-first", 0, "instead of the report: replay each workload's first N tokens as one "+
		"prompt from empty slots and print the prefill counters, to compare with a server run of that prompt")
	nBatch := flag.Int("batch", 2048, "the server's batch size")
	batches := flag.String("batches", "server", "how a prompt is cut: server (batches of -batch tokens that also end "+
		"4+ubatch and 4 tokens before the prompt does, as llama-server cuts them, each in ubatches) or ubatch (plain ubatches)")
	runtimeStats := flag.String("runtime-stats", "", "-validate-first: workload=stats.json pairs of server runs to compare with")
	validateDecode := flag.Int("validate-decode", 0, "-validate-first: also replay the trace's next N tokens as decoded "+
		"tokens; comparable with a run only if it generated exactly them")
	startup := flag.Int("validate-startup", 0, "-validate-first: stand-ins for a fresh server's startup decodes (it runs 2: "+
		"warmup and the seq_rm probe, of two tokens each), replayed as the trace's first two tokens")
	startupNPY := flag.String("validate-startup-npy", "", "-validate-first: the startup decodes' recorded selections "+
		"instead of stand-ins, [layers][2n][k]: n decodes of two tokens, in the order the server runs them")
	md := flag.String("md", "", "write the Markdown report here")
	outDir := flag.String("out", "", "write admission-<model>.json here, created if missing")
	flag.Parse()
	slots, err := ints(*slotsFlag)
	if err != nil {
		log.Fatalf("-slots: %v", err)
	}
	first, err := ints(*firstFlag)
	if err != nil {
		log.Fatalf("-first: %v", err)
	}
	if *lastLayer != "outputs" && *lastLayer != "all" {
		log.Fatal("-last-layer must be outputs or all")
	}
	var splits SplitManifest
	readJSON(*splitsPath, &splits)
	var known map[string]Model
	readJSON(*bytesPath, &known)
	if *batches != "server" && *batches != "ubatch" || *nBatch < 1 || *ubatch < 1 {
		log.Fatal("-batches must be server or ubatch, -batch and -ubatch positive")
	}
	a := Assumptions{Ubatch: *ubatch, Batch: *nBatch, Batches: *batches, Waves: *waves, LastLayer: *lastLayer, First: first,
		SplitRule: splits.Rule}
	command := "go run ./cmd/kolibri-admission"
	flag.Visit(func(f *flag.Flag) {
		if f.Name != "md" && f.Name != "out" {
			command += fmt.Sprintf(" -%s %s", f.Name, f.Value)
		}
	})

	if *validateFirst > 0 {
		runtime := map[string]string{}
		if *runtimeStats != "" {
			// the pairs name a workload only, so they belong to one model
			if n := len(strings.Split(*models, ",")); n != 1 {
				log.Fatalf("-runtime-stats takes one model's runs, -models names %d", n)
			}
			for f := range strings.SplitSeq(*runtimeStats, ",") {
				wl, path, ok := strings.Cut(f, "=")
				if !ok {
					log.Fatalf("-runtime-stats: %q is not workload=path", f)
				}
				runtime[wl] = path
			}
		}
		for _, n := range slots {
			var vs []Validation
			for name := range strings.SplitSeq(*models, ",") {
				m, ok := known[name]
				if !ok {
					log.Fatalf("%s: no expert count in %s", name, *bytesPath)
				}
				for wl := range strings.SplitSeq(*workloads, ",") {
					sel, err := locality.ReadNPYFile(filepath.Join(*dir, fmt.Sprintf("%s.%s.experts.npy", wl, name)))
					if err != nil {
						log.Fatal(err)
					}
					pre := standIn(sel, *startup)
					if *startupNPY != "" {
						if pre, err = locality.ReadNPYFile(*startupNPY); err != nil {
							log.Fatal(err)
						}
					}
					v, err := validate(sel, *validateFirst, *validateDecode, pre, m.NExpert, n, a)
					if err != nil {
						log.Fatalf("%s %s: %v", name, wl, err)
					}
					v.Model, v.Workload = name, wl
					if path, ok := runtime[wl]; ok {
						if v.Runtime, v.RuntimeDecode, err = runtimeCounts(path); err != nil {
							log.Fatal(err)
						}
						if v.Decoded == 0 {
							v.RuntimeDecode = nil
						}
					}
					vs = append(vs, v)
				}
			}
			writeValidation(os.Stdout, n, vs)
		}
		return
	}

	var rows []Row
	names := strings.Split(*models, ",")
	for _, name := range names {
		m, ok := known[name]
		if !ok {
			log.Fatalf("%s: no expert count in %s", name, *bytesPath)
		}
		for wl := range strings.SplitSeq(*workloads, ",") {
			sel, err := locality.ReadNPYFile(filepath.Join(*dir, fmt.Sprintf("%s.%s.experts.npy", wl, name)))
			if err != nil {
				log.Fatal(err)
			}
			if sel.Layers != len(m.Layers) {
				log.Fatalf("%s %s: %d layers of selections, %d of expert sizes", name, wl, sel.Layers, len(m.Layers))
			}
			f, err := os.Open(filepath.Join(*dir, wl+".tokens.npy"))
			if err != nil {
				log.Fatal(err)
			}
			ids, err := locality.ReadTokens(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s tokens: %v", wl, err)
			}
			f, err = os.Open(filepath.Join(*traces, wl+".json"))
			if err != nil {
				log.Fatal(err)
			}
			answers, ended, err := locality.ReadAnswers(f)
			f.Close()
			if err != nil {
				log.Fatalf("%s trace: %v", wl, err)
			}
			if len(ids) != sel.Tokens {
				log.Fatalf("FAIL %s: %d token IDs, %d tokens of selections", wl, len(ids), sel.Tokens)
			}
			if err := locality.CheckAnswers(ids, answers); err != nil {
				log.Fatalf("FAIL %s: %v", wl, err)
			}
			spans, err := locality.OnlineSpans(sel.Tokens, answers, ended, *ubatch)
			if err != nil {
				log.Fatalf("%s: %v", wl, err)
			}
			if *batches == "server" {
				spans = locality.ServerPrefill(spans, *nBatch, *ubatch)
			}
			d, ok := splits.Workloads[wl]
			if !ok {
				log.Fatalf("%s: no split in %s", wl, *splitsPath)
			}
			cut, err := split(d, spans, answers)
			if err != nil {
				log.Fatalf("FAIL %s: the split disagrees with %s: %v", wl, *splitsPath, err)
			}
			fmt.Fprintf(os.Stderr, "PASS %s %s: %d answers hold their tokens, split after %d at token %d, as in %s\n",
				short(name), wl, len(answers), d.TrainAnswers, d.FirstTestToken, filepath.Base(*splitsPath))
			r, err := evaluate(name, wl, sel, spans, cut, m.NExpert, slots, a)
			if err != nil {
				log.Fatalf("%s %s: %v", name, wl, err)
			}
			rows = append(rows, r...)
		}
	}

	writeMarkdown(os.Stdout, command, a, rows)
	if *md != "" {
		f, err := os.Create(*md)
		if err != nil {
			log.Fatal(err)
		}
		writeMarkdown(f, command, a, rows)
		if err := f.Close(); err != nil {
			log.Fatal(err)
		}
	}
	if *outDir != "" {
		if err := os.MkdirAll(*outDir, 0o755); err != nil {
			log.Fatal(err)
		}
		for _, name := range names {
			var mine []Row
			for _, r := range rows {
				if r.Model == name {
					mine = append(mine, r)
				}
			}
			b, err := json.MarshalIndent(Output{command, a, mine, totals(mine)}, "", " ")
			if err != nil {
				log.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(*outDir, "admission-"+name+".json"), append(b, '\n'), 0o644); err != nil {
				log.Fatal(err)
			}
		}
	}
}

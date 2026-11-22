package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"kolibri-llm/internal/gguf"
	"kolibri-llm/internal/quantsens"
)

// writeModel writes a two-layer Kolibri-shaped GGUF whose experts and
// attention have the given types, every tensor filled with fill.
func writeModel(t *testing.T, path string, exps, attn gguf.Type, fill byte) {
	t.Helper()
	type tn struct {
		name string
		typ  gguf.Type
		dims []uint64
	}
	ts := []tn{{"token_embd.weight", gguf.Q8_0, []uint64{256, 4}}}
	for _, l := range []string{"0", "1"} {
		p := "blk." + l + "."
		ts = append(ts,
			tn{p + "attn_norm.weight", gguf.F32, []uint64{256}},
			tn{p + "attn_q.weight", attn, []uint64{256, 256}},
			tn{p + "attn_k.weight", attn, []uint64{256, 64}},
			tn{p + "attn_v.weight", attn, []uint64{256, 64}},
			tn{p + "attn_output.weight", attn, []uint64{256, 256}},
			tn{p + "ffn_gate_inp.weight", gguf.F32, []uint64{256, 8}},
			tn{p + "ffn_gate_exps.weight", exps, []uint64{256, 64, 8}},
			tn{p + "ffn_up_exps.weight", exps, []uint64{256, 64, 8}},
			tn{p + "ffn_down_exps.weight", exps, []uint64{256, 256, 8}},
		)
	}
	ts = append(ts, tn{"output.weight", gguf.Q8_0, []uint64{256, 4}})
	var infos []gguf.Tensor
	var sizes []int64
	var off uint64
	for _, x := range ts {
		g := gguf.Tensor{Name: x.name, Dims: x.dims, Type: x.typ, Offset: off}
		n, err := g.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		infos = append(infos, g)
		sizes = append(sizes, n)
		off = (off + uint64(n) + 31) / 32 * 32
	}
	var buf bytes.Buffer
	if _, err := gguf.WriteHeader(&buf, 32, map[string]any{"general.architecture": "kolibri1"}, infos); err != nil {
		t.Fatal(err)
	}
	for _, n := range sizes {
		buf.Write(bytes.Repeat([]byte{fill}, int(n)))
		buf.Write(make([]byte, (n+31)/32*32-n))
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fakePerplexity prints a KLD summary whose KLD is 1e-6 per KiB the model
// is smaller than 1 GiB, so a smaller file always scores worse.
const fakePerplexity = `#!/bin/sh
size=$(wc -c < "$2" | tr -d ' ')
kld=$(awk -v s="$size" 'BEGIN { printf "%.9f", (1073741824 - s) / 1024 * 1e-6 }')
echo "Mean ln(PPL(Q)/PPL(base))     :   0.001000 ±   0.000100"
echo "Mean    KLD: $kld ±   0.000100"
echo "Same top p: 90.000 ±  0.100 %"
`

func gibArg(bytes int64) string { return strconv.FormatFloat(float64(bytes)/gib, 'g', -1, 64) }

func TestMeasureAllocateBuild(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.gguf")
	q4 := filepath.Join(dir, "q4.gguf")
	q8 := filepath.Join(dir, "q8.gguf")
	writeModel(t, base, gguf.IQ4_XS, gguf.Q8_0, 1)
	writeModel(t, q4, gguf.Q4_0, gguf.Q4_0, 2)
	writeModel(t, q8, gguf.Q8_0, gguf.Q8_0, 3)
	ppl := filepath.Join(dir, "llama-perplexity")
	if err := os.WriteFile(ppl, []byte(fakePerplexity), 0o755); err != nil {
		t.Fatal(err)
	}
	donors := q4 + "," + q8
	sens := filepath.Join(dir, "sens.jsonl")
	work := filepath.Join(dir, "work")
	measure := []string{"-base", base, "-donors", donors, "-scope", "layer", "-perplexity", ppl,
		"-kld-base", "unused.bin", "-work", work, "-out", sens, "-min-free", "0"}
	var out bytes.Buffer
	if err := cmdMeasure(measure, &out); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	recs, err := readRecords(sens)
	if err != nil {
		t.Fatal(err)
	}
	// Per layer: gate/up/down to Q4_0 and Q8_0 (6), attn to Q4_0 only (Q8_0
	// is the base's); token_embd and output are Q8_0 in every file, so they
	// have no run. 2*7 = 14, plus the base.
	if len(recs) != 15 || recs[0].Scope != "base" {
		t.Fatalf("%d records:\n%s", len(recs), out.String())
	}
	for _, r := range recs[1:] {
		if smaller := r.Bytes < r.BaseBytes; (r.DKL > 0) != smaller {
			t.Fatalf("%s %s: bytes %d vs base %d but ΔKL %g", r.Item, r.Type, r.Bytes, r.BaseBytes, r.DKL)
		}
	}
	if _, err := os.Stat(filepath.Join(work, "candidate.gguf")); !os.IsNotExist(err) {
		t.Fatalf("candidate left behind: %v", err)
	}
	// A second run resumes with nothing left to do.
	out.Reset()
	if err := cmdMeasure(measure, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "0 runs to do") {
		t.Fatalf("resume:\n%s", out.String())
	}
	if again, _ := readRecords(sens); len(again) != 15 {
		t.Fatalf("resume appended: %d records", len(again))
	}

	bf, _ := gguf.Open(base)
	var total int64
	for _, x := range bf.Tensors {
		n, _ := x.Bytes()
		total += n
	}
	// At the base's own size (plus a little), keeping the base costs
	// nothing, so the plan must fit and predict no added KL.
	plan := filepath.Join(dir, "plan.json")
	types := filepath.Join(dir, "types.txt")
	out.Reset()
	if err := cmdAllocate([]string{"-base", base, "-sens", sens, "-budget", gibArg(total + 4096), "-unit", "1",
		"-out", plan, "-types-out", types}, &out); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(plan)
	var p quantsens.Plan
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatal(err)
	}
	if p.Total > p.Budget || p.DKL > 1e-12 || len(p.Picks) != 2*4+2 {
		t.Fatalf("plan %+v\n%s", p, out.String())
	}
	if tt, _ := os.ReadFile(types); strings.Count(string(tt), "\n") != 2*(3+4)+2 {
		t.Fatalf("types file:\n%s", tt)
	}
	// A tight budget forces downgrades.
	tight := filepath.Join(dir, "tight.json")
	out.Reset()
	if err := cmdAllocate([]string{"-base", base, "-sens", sens, "-budget", gibArg(total * 9 / 10), "-unit", "1",
		"-out", tight}, &out); err != nil {
		t.Fatal(err)
	}
	raw, _ = os.ReadFile(tight)
	var tp quantsens.Plan
	if err := json.Unmarshal(raw, &tp); err != nil {
		t.Fatal(err)
	}
	if tp.Total > tp.Budget || tp.DKL <= 0 {
		t.Fatalf("tight plan %+v", tp)
	}

	// Build the tight plan; every item must have its planned type.
	built := filepath.Join(dir, "built.gguf")
	out.Reset()
	if err := cmdBuild([]string{"-base", base, "-donors", donors, "-plan", tight, "-out", built}, &out); err != nil {
		t.Fatal(err)
	}
	gf, err := gguf.Open(built)
	if err != nil {
		t.Fatal(err)
	}
	items, _ := quantsens.Items(gf, "layer")
	pick := map[string]string{}
	for _, pk := range tp.Picks {
		pick[pk.Item] = pk.Type
	}
	var builtTotal int64
	for _, it := range items {
		n, typ, err := quantsens.ItemBytes(gf, it)
		if err != nil || typ != pick[it.Name] {
			t.Fatalf("%s: %s, want %s (%v)", it.Name, typ, pick[it.Name], err)
		}
		builtTotal += n
	}
	if builtTotal+tp.Fixed != tp.Total {
		t.Fatalf("built item bytes %d + fixed %d != plan total %d", builtTotal, tp.Fixed, tp.Total)
	}
	err = cmdBuild([]string{"-base", base, "-donors", donors, "-plan", tight, "-out", built}, &out)
	if err == nil || !strings.Contains(err.Error(), "exists") {
		t.Fatalf("build over an existing file: %v", err)
	}
}

func TestSpliceCommand(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.gguf")
	q4 := filepath.Join(dir, "q4.gguf")
	writeModel(t, base, gguf.IQ4_XS, gguf.Q8_0, 1)
	writeModel(t, q4, gguf.Q4_0, gguf.Q4_0, 2)
	out := filepath.Join(dir, "out.gguf")
	var sink bytes.Buffer
	if err := cmdSplice([]string{"-base", base, "-out", out, "-take", q4 + `=^blk\.1\.ffn_.*_exps`}, &sink); err != nil {
		t.Fatal(err)
	}
	f, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	for _, x := range f.Tensors {
		taken := strings.HasPrefix(x.Name, "blk.1.ffn_") && strings.HasSuffix(x.Name, "_exps.weight")
		if taken && x.Type != gguf.Q4_0 {
			t.Fatalf("%s is %s, want Q4_0", x.Name, x.Type)
		}
		if !taken && strings.HasSuffix(x.Name, "_exps.weight") && x.Type != gguf.IQ4_XS {
			t.Fatalf("%s is %s, want the base's IQ4_XS", x.Name, x.Type)
		}
	}
	if err := cmdSplice([]string{"-base", base, "-out", filepath.Join(dir, "x.gguf"), "-take", q4 + "=^nothing$"}, &sink); err == nil {
		t.Fatal("a -take matching nothing was accepted")
	}
	if err := cmdSplice([]string{"-base", base, "-out", out, "-take", q4 + "=exps"}, &sink); err == nil {
		t.Fatal("overwrote an existing output")
	}
}

// Band-scope records price every layer of their band in allocate.
func TestMeasureAllocateBand(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base.gguf")
	q4 := filepath.Join(dir, "q4.gguf")
	writeModel(t, base, gguf.IQ4_XS, gguf.Q8_0, 1)
	writeModel(t, q4, gguf.Q4_0, gguf.Q4_0, 2)
	ppl := filepath.Join(dir, "llama-perplexity")
	if err := os.WriteFile(ppl, []byte(fakePerplexity), 0o755); err != nil {
		t.Fatal(err)
	}
	sens := filepath.Join(dir, "sens.jsonl")
	var out bytes.Buffer
	if err := cmdMeasure([]string{"-base", base, "-donors", q4, "-scope", "band:2", "-perplexity", ppl,
		"-kld-base", "unused.bin", "-work", filepath.Join(dir, "work"), "-out", sens, "-min-free", "0"}, &out); err != nil {
		t.Fatalf("%v\n%s", err, out.String())
	}
	recs, _ := readRecords(sens)
	// One band of both layers: gate, up, down, attn to Q4_0; plus the base.
	if len(recs) != 5 || recs[1].Scope != "band:2" || !strings.HasPrefix(recs[1].Item, "blk.0-1.") {
		t.Fatalf("%d records %+v\n%s", len(recs), recs, out.String())
	}
	bf, _ := gguf.Open(base)
	var total int64
	for _, x := range bf.Tensors {
		n, _ := x.Bytes()
		total += n
	}
	plan := filepath.Join(dir, "plan.json")
	out.Reset()
	if err := cmdAllocate([]string{"-base", base, "-sens", sens, "-budget", gibArg(total * 9 / 10), "-unit", "1", "-out", plan}, &out); err != nil {
		t.Fatal(err)
	}
	raw, _ := os.ReadFile(plan)
	var p quantsens.Plan
	if err := json.Unmarshal(raw, &p); err != nil {
		t.Fatal(err)
	}
	down := 0
	for _, pk := range p.Picks {
		if pk.Type == "Q4_0" {
			down++
			if !pk.Derived {
				t.Fatalf("%s priced from its own record", pk.Item)
			}
		}
	}
	if down == 0 || p.Total > p.Budget {
		t.Fatalf("plan %+v\n%s", p, out.String())
	}
}

func TestCalibrateCommand(t *testing.T) {
	dir := t.TempDir()
	proxy := filepath.Join(dir, "proxy.jsonl")
	target := filepath.Join(dir, "target.jsonl")
	write := func(path string, recs ...quantsens.Record) {
		for _, r := range recs {
			if err := appendRecord(path, r); err != nil {
				t.Fatal(err)
			}
		}
	}
	write(proxy, quantsens.Record{Scope: "base"},
		quantsens.Record{Scope: "kind", Item: "attn", Type: "Q4_0", DKL: 0.01},
		quantsens.Record{Scope: "band:2", Item: "blk.0-1.attn", Type: "Q4_0", DKL: 0.004})
	write(target, quantsens.Record{Scope: "base", KLD: 0.1},
		quantsens.Record{Scope: "kind", Item: "attn", Type: "Q4_0", DKL: 0.03})
	out := filepath.Join(dir, "cal.jsonl")
	var buf bytes.Buffer
	if err := cmdCalibrate([]string{"-proxy", proxy, "-target", target, "-out", out}, &buf); err != nil {
		t.Fatal(err)
	}
	recs, _ := readRecords(out)
	if len(recs) != 2 || math.Abs(recs[1].DKL-0.012) > 1e-12 {
		t.Fatalf("calibrated %+v\n%s", recs, buf.String())
	}
	if err := cmdCalibrate([]string{"-proxy", proxy, "-target", target, "-out", out}, &buf); err == nil {
		t.Fatal("calibrate wrote over its output")
	}
}

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"kolibri-llm/internal/probe"
)

// generate reproduces the committed dataset, and verify accepts it and refuses a changed answer.
func TestGenerateAndVerify(t *testing.T) {
	out := filepath.Join(t.TempDir(), "sub", "smoke.jsonl")
	if err := generate([]string{"-out", out}); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(filepath.Join("..", "..", defaultDataset))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("generate does not reproduce the committed dataset")
	}
	if err := verify([]string{"-dataset", out}); err != nil {
		t.Fatal(err)
	}
	bad := bytes.Replace(got, []byte(`"expected_answer_or_oracle":"443048573"`), []byte(`"expected_answer_or_oracle":"443048574"`), 1)
	if bytes.Equal(bad, got) {
		t.Fatal("the pilot's answer is not in the dataset")
	}
	if err := os.WriteFile(out, bad, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verify([]string{"-dataset", out}); err == nil {
		t.Fatal("verify accepted a wrong answer")
	}
}

// writeTestCapture writes a capture of noise, which is all a report's bookkeeping needs.
func writeTestCapture(t *testing.T, dir, dataset string) {
	t.Helper()
	data, err := os.ReadFile(dataset)
	if err != nil {
		t.Fatal(err)
	}
	items, err := readDataset(dataset)
	if err != nil {
		t.Fatal(err)
	}
	r := rand.New(rand.NewPCG(1, 1))
	m := probe.Matrix{Rows: len(items), Dim: 8, Data: make([]float32, len(items)*8)}
	for i := range m.Data {
		m.Data[i] = float32(r.NormFloat64())
	}
	var npy bytes.Buffer
	if err := probe.WriteNpy(&npy, m); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "layer-6.npy"), npy.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := func(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
	rows := make([]map[string]any, len(items))
	for i, it := range items {
		rows[i] = map[string]any{"id": it.ID, "n_tokens": 10, "last_token": 108}
	}
	js, err := json.Marshal(map[string]any{
		"schema": probe.CaptureSchema, "dataset_sha256": sum(data), "logits_check": "equal", "repeatable": true, "n_layer": 6,
		"last_tokens": []int{108}, "template_is_released": true, "rows": rows,
		"graph_check": map[string]any{"l_out-5": map[string]any{
			"sources": []string{"node_9", "ffn_inp-5"}, "read_by": "norm",
			"post_ffn": map[string]any{"ops": []string{"MUL", "RMS_NORM", "ADD"}, "weight": "blk.5.post_ffw_norm.weight",
				"norm": "norm-5", "experts": []string{"ffn_moe_out-5", "ffn_shexp_out-5"}},
		}},
		"layers": []map[string]any{{"one_based": 6, "zero_based": 5, "node": "l_out-5", "file": "layer-6.npy",
			"sha256": sum(npy.Bytes()), "dtype": "float32", "shape": []int{len(items), 8}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "capture.json"), js, 0o644); err != nil {
		t.Fatal(err)
	}
}

// A report never replaces another, and moved definitions neither take report.md's name nor pass
// as the committed ones.
func TestReportKeepsCommittedDefinitions(t *testing.T) {
	dataset := filepath.Join("..", "..", defaultDataset)
	dir := t.TempDir()
	writeTestCapture(t, dir, dataset)
	base := []string{"-capture", dir, "-dataset", dataset}
	if err := report(base); err != nil {
		t.Fatal(err)
	}
	committed, err := os.ReadFile(filepath.Join(dir, "report.md"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(committed), "not the committed definitions") {
		t.Error("the committed report says it deviates")
	}
	if err := report(base); err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Errorf("a second report: %v, want a refusal", err)
	}
	moved := append(base[:len(base):len(base)], "-l2", "0.1", "-shuffles", "2")
	if err := report(moved); err != nil {
		t.Fatal(err)
	}
	after, err := os.ReadFile(filepath.Join(dir, "report.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(committed, after) {
		t.Error("a report with moved options changed report.md")
	}
	found, err := filepath.Glob(filepath.Join(dir, "report-deviation-*.md"))
	if err != nil || len(found) != 1 {
		t.Fatalf("deviation reports: %v, %v", found, err)
	}
	md, err := os.ReadFile(found[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(md), "These are not the committed definitions: L2 0.1 (committed 0.01), shuffles 2 (committed 20).") {
		t.Errorf("the deviation report does not say so:\n%s", md)
	}
	if err := report(moved); err == nil {
		t.Error("the same deviation replaced its report")
	}
	if err := report([]string{"-capture", dir, "-dataset", dataset, "-out", t.TempDir()}); err != nil {
		t.Errorf("a new output directory: %v", err)
	}
}

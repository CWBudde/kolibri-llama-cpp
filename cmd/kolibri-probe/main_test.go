package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
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

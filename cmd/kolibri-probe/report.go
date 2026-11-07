package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"kolibri-llm/internal/probe"
)

func report(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	capture := fs.String("capture", "", "capture directory (tools/probe/capture.py)")
	dataset := fs.String("dataset", defaultDataset, "dataset JSONL the capture read")
	folds := fs.Int("folds", probe.DefaultOptions.Folds, "grouped cross-validation folds")
	l2 := fs.Float64("l2", probe.DefaultOptions.L2, "L2 strength")
	shuffles := fs.Int("shuffles", probe.DefaultOptions.Shuffles, "shuffled-label runs per probe")
	out := fs.String("out", "", "write report.md and report.json here (default: the capture directory)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *capture == "" {
		fs.Usage()
		os.Exit(2)
	}
	data, err := os.ReadFile(*dataset)
	if err != nil {
		return err
	}
	items, err := readDataset(*dataset)
	if err != nil {
		return err
	}
	if err := check(items); err != nil {
		return err
	}
	c, layers, err := probe.ReadCapture(*capture, data, items)
	if err != nil {
		return err
	}
	var named []probe.NamedMatrix
	for _, l := range layers {
		named = append(named, probe.NamedMatrix{Name: fmt.Sprintf("layer-%d", l.Layer.OneBased), Matrix: l.Matrix})
	}
	opt := probe.Options{Folds: *folds, L2: *l2, Shuffles: *shuffles, Seed: probe.Seed}
	rep, err := probe.Evaluate(items, named, opt)
	if err != nil {
		return err
	}
	rep.Capture = c.Raw
	dir := *out
	if dir == "" {
		dir = *capture
	}
	js, err := rep.JSON()
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "report.json"), js, 0o644); err != nil {
		return err
	}
	md := rep.Markdown()
	if err := os.WriteFile(filepath.Join(dir, "report.md"), []byte(md), 0o644); err != nil {
		return err
	}
	fmt.Print(md)
	fmt.Printf("\nPASS report on %d rows in %d groups: %s\n", rep.NRows, rep.NGroups, filepath.Join(dir, "report.md"))
	return nil
}

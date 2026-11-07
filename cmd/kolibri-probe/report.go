package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"kolibri-llm/internal/probe"
)

func report(args []string) error {
	fs := flag.NewFlagSet("report", flag.ExitOnError)
	capture := fs.String("capture", "", "capture directory (tools/probe/capture.py)")
	dataset := fs.String("dataset", defaultDataset, "dataset JSONL the capture read")
	folds := fs.Int("folds", probe.DefaultOptions.Folds, "grouped cross-validation folds")
	l2 := fs.Float64("l2", probe.DefaultOptions.L2, "L2 strength")
	shuffles := fs.Int("shuffles", probe.DefaultOptions.Shuffles, "shuffled-label runs per probe")
	out := fs.String("out", "", "write the report here (default: the capture directory)")
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
	opt := probe.Options{Folds: *folds, L2: *l2, Shuffles: *shuffles, Seed: probe.Seed, DatasetSHA256: probe.SHA256Hex(data)}
	dir := *out
	if dir == "" {
		dir = *capture
	}
	base := filepath.Join(dir, opt.ReportName())
	for _, ext := range []string{".md", ".json"} {
		if _, err := os.Stat(base + ext); err == nil {
			return fmt.Errorf("%s already exists, and a report must not replace another: move it away to report again", base+ext)
		} else if !os.IsNotExist(err) {
			return err
		}
	}
	rep, err := probe.Evaluate(items, named, opt)
	if err != nil {
		return err
	}
	rep.Capture = c.Raw
	rep.CaptureFacts = c.Facts()
	js, err := rep.JSON()
	if err != nil {
		return err
	}
	if err := writeNew(base+".json", js); err != nil {
		return err
	}
	md := rep.Markdown()
	if err := writeNew(base+".md", []byte(md)); err != nil {
		return err
	}
	fmt.Print(md)
	if len(rep.Deviations) > 0 {
		fmt.Printf("\nDEVIATION report on %d rows in %d groups, not the committed definitions (%s): %s.md\n", rep.NRows, rep.NGroups, strings.Join(rep.Deviations, ", "), base)
		return nil
	}
	fmt.Printf("\nPASS report on %d rows in %d groups: %s.md\n", rep.NRows, rep.NGroups, base)
	return nil
}

// writeNew writes a file that must not exist yet.
func writeNew(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

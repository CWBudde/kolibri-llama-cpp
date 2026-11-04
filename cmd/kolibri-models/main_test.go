package main

import (
	"bytes"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// fixture is a root with a master "w.gguf" on a cold root, not yet linked.
func fixture(t *testing.T) (store, *bytes.Buffer, []byte) {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("HOME", dir)
	var out bytes.Buffer
	s := store{root: filepath.Join(dir, "models"), cold: filepath.Join(dir, "cold"), out: &out}
	for _, d := range []string{s.root, s.cold} {
		if err := os.Mkdir(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	data := bytes.Repeat([]byte("GGUFweights"), 5000)
	write(t, filepath.Join(s.cold, "w.gguf"), data)
	return s, &out, data
}

func write(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func link(t *testing.T, s store, name string) {
	t.Helper()
	if err := os.Symlink(filepath.Join(s.cold, name), filepath.Join(s.root, name)); err != nil {
		t.Fatal(err)
	}
}

func isRegular(t *testing.T, path string) bool {
	t.Helper()
	li, err := os.Lstat(path)
	if err != nil {
		t.Fatal(err)
	}
	return li.Mode().IsRegular()
}

func TestPullReplacesTheSymlink(t *testing.T) {
	s, _, data := fixture(t)
	master := filepath.Join(s.cold, "w.gguf")
	when := time.Date(2026, 10, 23, 2, 41, 0, 0, time.UTC)
	if err := os.Chtimes(master, when, when); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(master, 0o640); err != nil {
		t.Fatal(err)
	}
	link(t, s, "w.gguf")
	if err := s.pull("w.gguf"); err != nil {
		t.Fatal(err)
	}
	entry := filepath.Join(s.root, "w.gguf")
	li, err := os.Lstat(entry)
	if err != nil {
		t.Fatal(err)
	}
	if !li.Mode().IsRegular() {
		t.Fatalf("mode %v, want a regular file", li.Mode())
	}
	if !li.ModTime().Equal(when) || li.Mode().Perm() != 0o640 {
		t.Errorf("mtime %v mode %v, want the master's", li.ModTime(), li.Mode())
	}
	got, err := os.ReadFile(entry)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Error("content differs")
	}
	leftovers, _ := filepath.Glob(filepath.Join(s.root, ".*"))
	if len(leftovers) != 0 {
		t.Errorf("left behind: %v", leftovers)
	}
}

func TestPullRefusals(t *testing.T) {
	s, _, _ := fixture(t)
	entry := filepath.Join(s.root, "w.gguf")
	if err := os.Symlink(filepath.Join(s.root, "other.gguf"), entry); err != nil {
		t.Fatal(err)
	}
	if err := s.pull("w.gguf"); err == nil {
		t.Error("a symlink to elsewhere: no error")
	}
	if err := s.pull("missing.gguf"); err == nil {
		t.Error("no master: no error")
	}
	if err := s.pull("../w.gguf"); err == nil {
		t.Error("a path as name: no error")
	}
}

func TestPullKeepsAHotFile(t *testing.T) {
	s, out, _ := fixture(t)
	entry := filepath.Join(s.root, "w.gguf")
	write(t, entry, []byte("written locally"))
	if err := s.pull("w.gguf"); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(entry)
	if string(got) != "written locally" || !strings.Contains(out.String(), "already hot") {
		t.Errorf("content %q, output %q", got, out)
	}
}

func TestOffloadReplacesAVerifiedCopy(t *testing.T) {
	s, _, data := fixture(t)
	entry := filepath.Join(s.root, "w.gguf")
	write(t, entry, data)
	if err := s.offload("w.gguf"); err != nil {
		t.Fatal(err)
	}
	target, err := os.Readlink(entry)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(s.cold, "w.gguf"); target != want {
		t.Errorf("target %s, want %s", target, want)
	}
	if err := s.offload("w.gguf"); err != nil {
		t.Errorf("a second offload: %v", err)
	}
}

// A refused offload must leave the only local copy exactly as it was.
func TestOffloadRefusals(t *testing.T) {
	s, _, data := fixture(t)

	differs := bytes.Clone(data)
	differs[len(differs)-1] ^= 1
	entry := filepath.Join(s.root, "w.gguf")
	write(t, entry, differs)
	if err := s.offload("w.gguf"); err == nil {
		t.Error("same size, other bytes: no error")
	}
	if got, _ := os.ReadFile(entry); !isRegular(t, entry) || !bytes.Equal(got, differs) {
		t.Error("same size, other bytes: file changed")
	}

	write(t, entry, data[:100])
	if err := s.offload("w.gguf"); err == nil {
		t.Error("other size: no error")
	}

	local := filepath.Join(s.root, "new.gguf")
	write(t, local, data)
	if err := s.offload("new.gguf"); err == nil {
		t.Error("no master: no error")
	}
	if !isRegular(t, local) {
		t.Error("no master: file replaced")
	}

	dir := filepath.Join(s.root, "d.gguf")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(s.cold, "d.gguf"), data)
	if err := s.offload("d.gguf"); err == nil {
		t.Error("a directory: no error")
	}
}

// With the root a symlink to the cold root, or a hard link between them, both
// paths are one file: its hash always matches, and replacing it would destroy
// the only copy.
func TestOffloadRefusesToReplaceTheMasterItself(t *testing.T) {
	s, _, data := fixture(t)
	master := filepath.Join(s.cold, "w.gguf")

	root := filepath.Join(filepath.Dir(s.root), "linked")
	if err := os.Symlink(s.cold, root); err != nil {
		t.Fatal(err)
	}
	if err := (store{root: root, cold: s.cold, out: s.out}).offload("w.gguf"); err == nil {
		t.Error("root links to the cold root: no error")
	}
	if !isRegular(t, master) {
		t.Fatal("root links to the cold root: master replaced")
	}

	entry := filepath.Join(s.root, "w.gguf")
	if err := os.Link(master, entry); err != nil {
		t.Fatal(err)
	}
	if err := s.offload("w.gguf"); err == nil {
		t.Error("hard link: no error")
	}
	if got, err := os.ReadFile(master); err != nil || !bytes.Equal(got, data) || !isRegular(t, master) {
		t.Fatalf("hard link: master damaged (%v)", err)
	}
}

func TestOffloadAddsAMissingLink(t *testing.T) {
	s, _, _ := fixture(t)
	if err := s.offload("w.gguf"); err != nil {
		t.Fatal(err)
	}
	if target, err := os.Readlink(filepath.Join(s.root, "w.gguf")); err != nil || target != filepath.Join(s.cold, "w.gguf") {
		t.Errorf("target %q, %v", target, err)
	}
}

func TestPullClonesFromAHotPartner(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("APFS clones")
	}
	s, out, _ := fixture(t)
	size := 3 << 20
	data := bytes.Repeat([]byte{7}, size)
	header := bytes.Clone(data)
	copy(header, "other header")
	write(t, filepath.Join(s.cold, "x.gguf"), data)
	write(t, filepath.Join(s.root, "x-kolibri.gguf"), header)
	link(t, s, "x.gguf")
	if err := s.pull("x.gguf"); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(s.root, "x.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Error("content differs")
	}
	if strings.Contains(out.String(), "no clone") {
		t.Skipf("no clones on this file system: %s", out)
	}
	if !strings.Contains(out.String(), "cloned from x-kolibri.gguf, 4096 bytes rewritten") {
		t.Errorf("output %q, want one block rewritten", out)
	}
}

func TestPartner(t *testing.T) {
	for name, want := range map[string]string{
		"X.gguf":         "X-kolibri.gguf",
		"X-kolibri.gguf": "X.gguf",
		"notes.txt":      "",
	} {
		if got := partner(name); got != want {
			t.Errorf("partner(%q) = %q, want %q", name, got, want)
		}
	}
}

func TestStatus(t *testing.T) {
	s, out, data := fixture(t)
	link(t, s, "w.gguf")
	write(t, filepath.Join(s.root, "new.gguf"), data)
	if err := os.Mkdir(filepath.Join(s.root, "eval"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/nonexistent/gone.gguf", filepath.Join(s.root, "gone.gguf")); err != nil {
		t.Fatal(err)
	}
	if err := s.status(); err != nil {
		t.Fatal(err)
	}
	lines := out.String()
	for _, want := range []string{"LOCAL ONLY", "local", "absent"} {
		if !strings.Contains(lines, want) {
			t.Errorf("no %q in\n%s", want, lines)
		}
	}
}

func TestCheck(t *testing.T) {
	s, out, data := fixture(t)
	hot := filepath.Join(s.root, "hot.gguf")
	write(t, hot, data)
	// /dev is its own file system, so it stands in for the cold volume.
	cold := filepath.Join(s.root, "cold.gguf")
	if err := os.Symlink("/dev/null", cold); err != nil {
		t.Fatal(err)
	}
	m := filepath.Join(s.root, "m.json")
	write(t, m, []byte(`{"configs": [
		{"name": "a", "model": "`+hot+`"},
		{"name": "a2", "model": "`+hot+`"},
		{"name": "b", "model": "`+cold+`"}
	]}`))

	if err := s.check([]string{m}, []string{"a", "a2"}); err != nil {
		t.Errorf("hot arms: %v", err)
	}
	if n := strings.Count(out.String(), "hot "); n != 1 {
		t.Errorf("output %q, want one line per weight", out)
	}
	if err := s.check([]string{m}, nil); err == nil {
		t.Error("a cold arm: no error")
	}
	if err := s.check([]string{m}, []string{"typo"}); err == nil {
		t.Error("an unknown config: no error")
	}
	if err := s.check([]string{"hot.gguf"}, nil); err != nil {
		t.Errorf("a file name in the root: %v", err)
	}
	if err := s.check([]string{"absent.gguf"}, nil); err == nil {
		t.Error("an absent weight: no error")
	}
}

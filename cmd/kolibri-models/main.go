// Command kolibri-models moves model weights between the two storage tiers
// under ~/models, without ever losing the only copy of one. A hot weight is a
// real file on the home volume. A cold weight is a symlink to the complete
// master set on the cold volume, whose root comes from -cold or
// $KOLIBRI_COLD_ROOT.
//
//	kolibri-models status
//	kolibri-models check testdata/bench/expert-cache.json   # exit 1 if a weight is cold
//	kolibri-models -only iq3-cache128,iq3-cpu-moe check testdata/bench/expert-cache.json
//	kolibri-models pull Kolibri-1-Q8_0.gguf
//	kolibri-models offload Kolibri-1-Q8_0.gguf
//
// pull copies a master to the root under a temporary name, compares SHA-256
// and only then renames it over the symlink. If the file's clone pair
// (X.gguf and X-kolibri.gguf, see cmd/kolibri-rekey) is already hot, the copy
// is an APFS clone of it with only the differing blocks rewritten, so the pair
// keeps sharing its blocks.
//
// offload replaces a hot file with a symlink to its master, and only if the
// master exists and has the same SHA-256. Anything else is refused and left
// alone, so a file written locally and never copied to the cold volume cannot
// be lost. A file that changes while it is being compared is refused as well
// (replaceWithLink). Both are single files of the root: directories are not
// moved.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"

	"kolibri-llm/internal/tier"
)

const (
	coldEnv   = "KOLIBRI_COLD_ROOT"
	blockSize = 4096
	gib       = 1 << 30
)

type store struct {
	root, cold string // absolute; cold may be empty, for status and check
	out        io.Writer
}

func expandHome(p string) string {
	if rest, ok := strings.CutPrefix(p, "~/"); ok {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, rest)
		}
	}
	return p
}

func newStore(root, cold string, out io.Writer) (store, error) {
	r, err := filepath.Abs(expandHome(root))
	if err != nil {
		return store{}, err
	}
	s := store{root: r, out: out}
	if cold == "" {
		return s, nil
	}
	if s.cold, err = filepath.Abs(expandHome(cold)); err != nil {
		return store{}, err
	}
	return s, nil
}

// paths checks that name is a plain file name and returns its entry in the root
// and its master. With the root and the cold root being one directory, the two
// would be one file and offload would replace a weight with a link to itself.
func (s store) paths(name string) (entry, master string, err error) {
	if s.cold == "" {
		return "", "", fmt.Errorf("no cold root: use -cold or $%s", coldEnv)
	}
	if name == "." || name == ".." || name != filepath.Base(name) {
		return "", "", fmt.Errorf("%q: want a file name in %s", name, s.root)
	}
	ri, err := os.Stat(s.root)
	if err != nil {
		return "", "", err
	}
	ci, err := os.Stat(s.cold)
	if err != nil {
		return "", "", fmt.Errorf("%w (is the cold volume mounted?)", err)
	}
	if os.SameFile(ri, ci) {
		return "", "", fmt.Errorf("%s and %s are the same directory", s.root, s.cold)
	}
	return filepath.Join(s.root, name), filepath.Join(s.cold, name), nil
}

// status lists the root's entries by tier.
func (s store) status() error {
	entries, err := os.ReadDir(s.root)
	if err != nil {
		return err
	}
	var hot int64
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		path := filepath.Join(s.root, e.Name())
		li, err := os.Lstat(path)
		if err != nil {
			return err
		}
		what, size, note := s.describe(path, li)
		if what == "hot" && li.Mode().IsRegular() {
			hot += size
		}
		fmt.Fprintf(s.out, "%-8s %8.1f GiB  %s%s\n", what, float64(size)/gib, e.Name(), note)
	}
	fmt.Fprintf(s.out, "hot files: %.1f GiB apparent size (clone pairs share their blocks)\n", float64(hot)/gib)
	if s.cold == "" {
		fmt.Fprintf(s.out, "no cold root: masters not checked (-cold or $%s)\n", coldEnv)
	}
	return nil
}

func (s store) describe(path string, li fs.FileInfo) (what string, size int64, note string) {
	if li.Mode()&os.ModeSymlink != 0 {
		target, err := os.Readlink(path)
		if err != nil {
			return "broken", 0, "  " + err.Error()
		}
		note = " -> " + target
		if _, err := os.Stat(path); err != nil {
			return "absent", 0, note + "  (cold volume not mounted?)"
		}
	}
	loc, err := tier.Locate(path)
	if err != nil {
		return "broken", 0, "  " + err.Error()
	}
	fi, err := os.Stat(path)
	if err != nil {
		return "broken", 0, "  " + err.Error()
	}
	what = "cold"
	if loc.Home {
		what = "hot"
	}
	if fi.IsDir() {
		if loc.Home {
			what = "local"
			note += "  (directory, not managed: keep it, it may hold the only copy of results)"
		}
		return what, 0, note
	}
	if li.Mode().IsRegular() && s.cold != "" {
		mi, err := os.Stat(filepath.Join(s.cold, filepath.Base(path)))
		switch {
		case errors.Is(err, fs.ErrNotExist):
			note += "  LOCAL ONLY: no master"
		case err != nil:
			note += "  master unreadable: " + err.Error()
		case mi.Size() != fi.Size():
			note += "  DIFFERS from its master (size)"
		}
	}
	return what, fi.Size(), note
}

// matrix is the part of a kolibri-bench matrix that names weights.
type matrix struct {
	Configs []struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	} `json:"configs"`
}

// weights lists the weights that a matrix's configs read, restricted to only
// if it is not empty (kolibri-bench's -only).
func weights(path string, only []string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m matrix
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	var out, names []string
	for _, c := range m.Configs {
		names = append(names, c.Name)
		if len(only) > 0 && !slices.Contains(only, c.Name) {
			continue
		}
		if p := expandHome(c.Model); !slices.Contains(out, p) {
			out = append(out, p)
		}
	}
	for _, name := range only {
		if !slices.Contains(names, name) {
			return nil, fmt.Errorf("%s: no config %q", path, name)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s: no weights", path)
	}
	return out, nil
}

// check prints where each weight resolves and fails unless all are on the home
// volume, which a timing run needs.
func (s store) check(args, only []string) error {
	bad := 0
	for _, arg := range args {
		paths := []string{arg}
		if strings.HasSuffix(arg, ".json") {
			var err error
			if paths, err = weights(arg, only); err != nil {
				return err
			}
		} else if !strings.Contains(arg, "/") {
			paths = []string{filepath.Join(s.root, arg)}
		}
		for _, p := range paths {
			loc, err := tier.Locate(p)
			switch {
			case err != nil:
				bad++
				fmt.Fprintf(s.out, "absent %s: %v\n", p, err)
			case !loc.Home:
				bad++
				fmt.Fprintf(s.out, "cold   %s -> %s\n", p, loc)
			default:
				fmt.Fprintf(s.out, "hot    %s -> %s\n", p, loc)
			}
		}
	}
	if bad > 0 {
		return fmt.Errorf("%d weight(s) not on the home volume: pull them before a timing run", bad)
	}
	return nil
}

// partner is the other name of a clone pair: X.gguf and X-kolibri.gguf.
func partner(name string) string {
	stem, ok := strings.CutSuffix(name, ".gguf")
	if !ok {
		return ""
	}
	if s, ok := strings.CutSuffix(stem, "-kolibri"); ok {
		return s + ".gguf"
	}
	return stem + "-kolibri.gguf"
}

func (s store) pull(name string) error {
	entry, master, err := s.paths(name)
	if err != nil {
		return err
	}
	mi, err := os.Stat(master)
	if err != nil {
		return fmt.Errorf("%w (is the cold volume mounted?)", err)
	}
	if !mi.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", master)
	}
	li, err := os.Lstat(entry)
	switch {
	case errors.Is(err, fs.ErrNotExist):
	case err != nil:
		return err
	case li.Mode().IsRegular():
		fmt.Fprintf(s.out, "%s is already hot\n", name)
		return nil
	case li.Mode()&os.ModeSymlink == 0:
		return fmt.Errorf("%s is neither a file nor a symlink: left alone", entry)
	default:
		target, err := os.Readlink(entry)
		if err != nil {
			return err
		}
		if target != master {
			return fmt.Errorf("%s points at %s, not at its master %s: left alone", entry, target, master)
		}
	}

	tmp := filepath.Join(s.root, "."+name+".incoming")
	if err := os.Remove(tmp); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	defer os.Remove(tmp)
	if err := s.copyIn(name, master, mi, tmp); err != nil {
		return err
	}
	if err := os.Chmod(tmp, mi.Mode().Perm()); err != nil {
		return err
	}
	if err := os.Chtimes(tmp, mi.ModTime(), mi.ModTime()); err != nil {
		return err
	}
	if err := sameContent(master, tmp); err != nil {
		return err
	}
	if err := os.Rename(tmp, entry); err != nil {
		return err
	}
	fmt.Fprintf(s.out, "%s is hot (SHA-256 matches its master)\n", name)
	return nil
}

// copyIn writes tmp: a clone of the hot partner if there is one, else a copy.
func (s store) copyIn(name, master string, mi fs.FileInfo, tmp string) error {
	if p := partner(name); p != "" {
		base := filepath.Join(s.root, p)
		if bi, err := os.Lstat(base); err == nil && bi.Mode().IsRegular() && bi.Size() == mi.Size() {
			rewritten, err := cloneOver(base, master, tmp)
			if err == nil {
				fmt.Fprintf(s.out, "%s: cloned from %s, %d bytes rewritten\n", name, p, rewritten)
				return nil
			}
			fmt.Fprintf(s.out, "%s: no clone (%v), copying\n", name, err)
			if err := os.Remove(tmp); err != nil && !errors.Is(err, fs.ErrNotExist) {
				return err
			}
		}
	}
	var fsi syscall.Statfs_t
	if err := syscall.Statfs(s.root, &fsi); err != nil {
		return err
	}
	if free := int64(fsi.Bavail) * int64(fsi.Bsize); free < mi.Size() {
		return fmt.Errorf("%s needs %.1f GiB, %.1f GiB are free", name, float64(mi.Size())/gib, float64(free)/gib)
	}
	return copyFile(master, tmp)
}

// cloneOver makes dst an APFS clone of base, then rewrites only the blocks that
// differ from src, so dst shares every other block with base.
func cloneOver(base, src, dst string) (rewritten int64, err error) {
	// The absolute path, since a GNU cp earlier on PATH reads -c differently.
	if out, err := exec.Command("/bin/cp", "-c", base, dst).CombinedOutput(); err != nil {
		return 0, fmt.Errorf("cp -c: %v: %s", err, bytes.TrimSpace(out))
	}
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	if err := os.Chmod(dst, 0o644); err != nil {
		return 0, err
	}
	out, err := os.OpenFile(dst, os.O_RDWR, 0)
	if err != nil {
		return 0, err
	}
	defer out.Close()

	want, have := make([]byte, 4<<20), make([]byte, 4<<20)
	for off := int64(0); ; {
		n, rerr := io.ReadFull(in, want)
		if n > 0 {
			if _, err := out.ReadAt(have[:n], off); err != nil {
				return rewritten, err
			}
			for i := 0; i < n; i += blockSize {
				end := min(i+blockSize, n)
				if !bytes.Equal(want[i:end], have[i:end]) {
					if _, err := out.WriteAt(want[i:end], off+int64(i)); err != nil {
						return rewritten, err
					}
					rewritten += int64(end - i)
				}
			}
			off += int64(n)
		}
		if rerr == io.EOF || rerr == io.ErrUnexpectedEOF {
			break
		}
		if rerr != nil {
			return rewritten, rerr
		}
	}
	return rewritten, out.Sync()
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	if err := out.Sync(); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func (s store) offload(name string) error {
	entry, master, err := s.paths(name)
	if err != nil {
		return err
	}
	mi, err := os.Stat(master)
	if err != nil {
		return fmt.Errorf("no master, so %s stays: %w (copy it to the cold root first; is the cold volume mounted?)", name, err)
	}
	if !mi.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", master)
	}
	aside := filepath.Join(s.root, "."+name+".offload")
	if _, err := os.Lstat(aside); err == nil {
		return fmt.Errorf("%s exists, perhaps from an interrupted offload, and may be the only copy of %s: move it back or delete it first", aside, name)
	} else if !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	li, err := os.Lstat(entry)
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// A master without an entry: nothing to verify, only the link to add.
		li = nil
	case err != nil:
		return err
	case li.Mode()&os.ModeSymlink != 0:
		target, err := os.Readlink(entry)
		if err != nil {
			return err
		}
		if target != master {
			return fmt.Errorf("%s points at %s, not at its master %s: left alone", entry, target, master)
		}
		fmt.Fprintf(s.out, "%s is already cold\n", name)
		return nil
	case !li.Mode().IsRegular():
		return fmt.Errorf("%s is not a regular file: left alone", entry)
	case os.SameFile(li, mi):
		return fmt.Errorf("%s and %s are one file: left alone", entry, master)
	case li.Size() != mi.Size():
		return fmt.Errorf("%s has %d bytes, its master %d: left alone", entry, li.Size(), mi.Size())
	default:
		if err := sameContent(entry, master); err != nil {
			return fmt.Errorf("%w: left alone", err)
		}
	}

	if err := replaceWithLink(entry, master, aside, li, mi); err != nil {
		return err
	}
	fmt.Fprintf(s.out, "%s is cold -> %s\n", name, master)
	if p := partner(name); p != "" {
		if pi, err := os.Lstat(filepath.Join(s.root, p)); err == nil && pi.Mode().IsRegular() {
			fmt.Fprintf(s.out, "%s is still hot and may share its blocks, so little space is freed until it goes too\n", p)
		}
	}
	return nil
}

// replaceWithLink makes entry a symlink to master, given the file infos taken
// before the two were hashed. A nil hashed means that entry did not exist.
//
// Hashing a large weight takes minutes, and another process may replace or
// modify the entry meanwhile. So entry is first renamed to aside, which pins
// the exact file that would be deleted, and that file must still be the one
// that was hashed. The link is created without overwriting anything, and a
// file that fails either step goes back to entry. Entry is missing for the
// moment in between, so a reader opening it then gets ENOENT.
//
// A writer that keeps a descriptor open past the check is not detected: nothing
// here can see it without locks that other tools do not take.
func replaceWithLink(entry, master, aside string, hashed, masterHashed fs.FileInfo) error {
	mi, err := os.Stat(master)
	if err != nil {
		return err
	}
	if !sameVersion(masterHashed, mi) {
		return fmt.Errorf("%s changed while it was being compared: left alone", master)
	}
	if hashed == nil {
		if err := os.Symlink(master, entry); err != nil {
			return fmt.Errorf("%w: left alone", err)
		}
		return nil
	}
	if err := os.Rename(entry, aside); err != nil {
		return err
	}
	ai, err := os.Lstat(aside)
	if err == nil && !sameVersion(hashed, ai) {
		err = fmt.Errorf("%s changed while it was being compared", entry)
	}
	if err == nil {
		err = os.Symlink(master, entry)
	}
	if err != nil {
		// Link, not rename: a file that has appeared at entry stays.
		if lerr := os.Link(aside, entry); lerr != nil {
			return fmt.Errorf("%w; the file is now at %s, since it could not go back: %v", err, aside, lerr)
		}
		if rerr := os.Remove(aside); rerr != nil {
			return fmt.Errorf("%w; %s is a leftover link to the same file: %v", err, aside, rerr)
		}
		return fmt.Errorf("%w: left alone", err)
	}
	return os.Remove(aside)
}

// sameVersion says whether b is still the file that a described: the same
// inode, and neither its size nor its modification time has moved.
func sameVersion(a, b fs.FileInfo) bool {
	return os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

// sameContent compares the SHA-256 of two files.
func sameContent(a, b string) error {
	ha, err := hashFile(a)
	if err != nil {
		return err
	}
	hb, err := hashFile(b)
	if err != nil {
		return err
	}
	if !bytes.Equal(ha, hb) {
		return fmt.Errorf("SHA-256 of %s and %s differ", a, b)
	}
	return nil
}

func hashFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 4<<20)); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

func main() {
	log.SetFlags(0)
	root := flag.String("root", "~/models", "the directory that tools read weights from")
	cold := flag.String("cold", os.Getenv(coldEnv), "the root of the master weights on the cold volume")
	only := flag.String("only", "", "check: only the matrix configs with these comma-separated names")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: kolibri-models [flags] status | check MATRIX.json|WEIGHT... | pull FILE | offload FILE")
		flag.PrintDefaults()
	}
	flag.Parse()
	args := flag.Args()
	if len(args) == 0 {
		flag.Usage()
		os.Exit(2)
	}
	s, err := newStore(*root, *cold, os.Stdout)
	if err != nil {
		log.Fatal(err)
	}
	var names []string
	if *only != "" {
		names = strings.Split(*only, ",")
	}
	switch cmd, rest := args[0], args[1:]; {
	case cmd == "status" && len(rest) == 0:
		err = s.status()
	case cmd == "check" && len(rest) > 0:
		err = s.check(rest, names)
	case cmd == "pull" && len(rest) == 1:
		err = s.pull(rest[0])
	case cmd == "offload" && len(rest) == 1:
		err = s.offload(rest[0])
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
}

//go:build darwin

package memctl

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"syscall"
	"unsafe"
)

// PageSize is the VM page size (16 KiB on Apple silicon).
func PageSize() int { return syscall.Getpagesize() }

// Wired is anonymous memory locked into RAM: the compressor and swap cannot
// take it back, so it lowers the memory left for everything else by its size.
type Wired struct {
	mem []byte
}

// Wire maps n bytes of anonymous memory, writes every page and locks it.
func Wire(n int64) (*Wired, error) {
	if n <= 0 {
		return nil, fmt.Errorf("wire %d bytes", n)
	}
	mem, err := syscall.Mmap(-1, 0, int(n), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, fmt.Errorf("mmap %d bytes: %w", n, err)
	}
	ps := PageSize()
	for i := 0; i < len(mem); i += ps {
		mem[i] = byte(i/ps) | 1
	}
	if err := syscall.Mlock(mem); err != nil {
		syscall.Munmap(mem)
		return nil, fmt.Errorf("mlock %d bytes: %w", n, err)
	}
	return &Wired{mem: mem}, nil
}

// Release unlocks and unmaps the memory; a second call does nothing.
func (w *Wired) Release() error {
	if w.mem == nil {
		return nil
	}
	err := errors.Join(syscall.Munlock(w.mem), syscall.Munmap(w.mem))
	w.mem = nil
	return err
}

// Residency is the part of a file the page cache holds.
type Residency struct {
	Pages    int // the file's pages
	Resident int // of those, the pages in memory
}

// mapFile maps a whole file read-only and shared; a file of size 0 gives nil.
func mapFile(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return nil, nil
	}
	mem, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	return mem, nil
}

// Resident counts the pages of a file in the page cache (mincore); it maps
// the file but reads none of it.
func Resident(path string) (Residency, error) {
	mem, err := mapFile(path)
	if err != nil || mem == nil {
		return Residency{}, err
	}
	defer syscall.Munmap(mem)
	ps := PageSize()
	vec := make([]byte, (len(mem)+ps-1)/ps)
	_, _, errno := syscall.Syscall(syscall.SYS_MINCORE, uintptr(unsafe.Pointer(&mem[0])), uintptr(len(mem)), uintptr(unsafe.Pointer(&vec[0])))
	if errno != 0 {
		return Residency{}, fmt.Errorf("mincore %s: %w", path, errno)
	}
	r := Residency{Pages: len(vec)}
	for _, v := range vec {
		if v&1 != 0 { // MINCORE_INCORE
			r.Resident++
		}
	}
	return r, nil
}

// Evict drops a file's clean pages from the page cache: msync(MS_INVALIDATE)
// on a shared mapping of all of it. It needs no root, unlike purge. Pages
// another process has wired or mapped may stay.
func Evict(path string) error {
	mem, err := mapFile(path)
	if err != nil || mem == nil {
		return err
	}
	defer syscall.Munmap(mem)
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&mem[0])), uintptr(len(mem)), uintptr(syscall.MS_INVALIDATE))
	if errno != 0 {
		return fmt.Errorf("msync %s: %w", path, errno)
	}
	return nil
}

var wiredRe = regexp.MustCompile(`(?m)^Pages wired down:\s+(\d+)\.`)

func parseWiredPages(out string) (int64, error) {
	m := wiredRe.FindStringSubmatch(out)
	if m == nil {
		return 0, errors.New("vm_stat: no wired pages line")
	}
	return strconv.ParseInt(m[1], 10, 64)
}

// WiredPages is the system's wired page count, from vm_stat.
func WiredPages() (int64, error) {
	out, err := exec.Command("vm_stat").Output()
	if err != nil {
		return 0, fmt.Errorf("vm_stat: %w", err)
	}
	return parseWiredPages(string(out))
}

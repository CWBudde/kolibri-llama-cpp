// Command kolibri-memhog wires memory and holds it until SIGINT or SIGTERM,
// the memory ceiling of the expert cache benchmark (cmd/kolibri-bench). Wired
// pages cannot be compressed or swapped, so the system has exactly that much
// less for everything else; mlock needs no root on macOS, up to
// vm.user_wire_limit.
//
//	kolibri-memhog -wire 16GiB
//
// Once the memory is wired it prints one line, "wired N pages (X GiB), system
// wired A -> B", which the benchmark waits for.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"kolibri-llm/internal/memctl"
)

// parseSize reads a byte count with an optional KiB, MiB or GiB suffix.
func parseSize(s string) (int64, error) {
	mult := int64(1)
	for _, u := range []struct {
		suffix string
		mult   int64
	}{{"GiB", 1 << 30}, {"MiB", 1 << 20}, {"KiB", 1 << 10}} {
		if rest, ok := strings.CutSuffix(s, u.suffix); ok {
			s, mult = rest, u.mult
			break
		}
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 {
		return 0, fmt.Errorf("size %q: want a positive number with an optional KiB, MiB or GiB", s)
	}
	return int64(v * float64(mult)), nil
}

func main() {
	size := flag.String("wire", "", "memory to wire, e.g. 16GiB")
	flag.Parse()
	n, err := parseSize(*size)
	if err != nil {
		flag.Usage()
		log.Fatal(err)
	}
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	w0, err := memctl.WiredPages()
	if err != nil {
		log.Fatal(err)
	}
	w, err := memctl.Wire(n)
	if err != nil {
		log.Fatal(err)
	}
	w1, err := memctl.WiredPages()
	if err != nil {
		w.Release()
		log.Fatal(err)
	}
	fmt.Printf("wired %d pages (%.2f GiB), system wired %d -> %d\n", n/int64(memctl.PageSize()), float64(n)/(1<<30), w0, w1)
	<-sig
	if err := w.Release(); err != nil {
		log.Fatal(err)
	}
	fmt.Println("released")
}

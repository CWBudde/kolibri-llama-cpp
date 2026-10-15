// Command kolibri-evict drops files from the page cache, for the cold runs of
// the expert cache benchmark (cmd/kolibri-bench), and reports how many of their
// pages were in memory before and after. It needs no root, unlike purge:
// msync(MS_INVALIDATE) on a shared mapping of each file (internal/memctl).
// Pages another process maps or has wired may stay, so evict before the
// server starts.
//
//	kolibri-evict ~/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf
//	kolibri-evict -check ~/models/Kolibri-1-Q8_0.gguf
package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"kolibri-llm/internal/memctl"
)

func main() {
	check := flag.Bool("check", false, "only report the resident pages")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: kolibri-evict [-check] FILE...")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}
	mib := float64(memctl.PageSize()) / (1 << 20)
	for _, path := range flag.Args() {
		before, err := memctl.Resident(path)
		if err != nil {
			log.Fatal(err)
		}
		if *check {
			fmt.Printf("%s: %d of %d pages resident (%.0f MiB)\n", path, before.Resident, before.Pages, float64(before.Resident)*mib)
			continue
		}
		if err := memctl.Evict(path); err != nil {
			log.Fatal(err)
		}
		after, err := memctl.Resident(path)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s: %d of %d pages resident before, %d after (%.0f MiB evicted)\n",
			path, before.Resident, before.Pages, after.Resident, float64(before.Resident-after.Resident)*mib)
	}
}

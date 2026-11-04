# Working in this repository

Where things live, and the rules for the model weights. The project itself is in
[README.md](README.md) and [PLAN.md](PLAN.md).

## Where things live

- **Code is local.** Checkouts and worktrees are real directories on the internal drive, never
  symlinks to another volume. `third_party/llama.cpp` is a gitignored checkout of the fork with its
  own branches and uncommitted work, which the outer `git status` does not show.
- **`~/models` is a real local directory, and every `~/models/<file>` path stays valid.** Code and
  docs hard-code these paths. Each weight is on one of two tiers:
  - **hot:** a real file on the internal drive;
  - **cold:** a symlink to its master on the cold volume, an external drive.
- **The cold volume holds the master of every weight.** A hot weight is a verified copy of its
  master, so offloading it loses nothing.
- **`~/models/eval/` is a local working directory, not a tier.** Benchmarks read their workloads
  from it and write their results to it. Those results may exist nowhere else: never delete, offload
  or link it. Copy it to the cold volume when a stage ends, under a dated name:
  `ditto ~/models/eval "$KOLIBRI_COLD_ROOT/eval.backup-<date>"`.

## Hot or cold

```sh
go run ./cmd/kolibri-models status   # every entry's tier
readlink ~/models/<file>             # no output: hot; a path: cold, and that is where it lives
```

Keep hot only what the current stage reads, and start with the smaller set: pulling later is cheap.
`X.gguf` is the current name (architecture `kolibri1`). `X-kolibri.gguf` has the old architecture,
which upstream's model rejects (see docs/upstream.md), so it stays cold unless a task needs it.

## Moving a weight

The tool takes the cold root from `-cold` or `$KOLIBRI_COLD_ROOT`.

```sh
go run ./cmd/kolibri-models pull <file>
go run ./cmd/kolibri-models offload <file>
```

Both compare SHA-256 before replacing anything. If either refuses, find out why instead of working
around it: `offload` refuses because the local file differs from its master or has none.

- Do not `rm` or overwrite a hot weight by hand, and do not replace it with a symlink yourself.
- Never write through a `~/models` path that may be cold: the symlink leads to the master. Give
  converters, quantizers and `kolibri-rekey` a new output name.
- A new weight is local-only (`status` says `LOCAL ONLY`) until it is copied to the cold root. Then
  `offload` can verify it.

## Clone pairs

`X.gguf` and `X-kolibri.gguf` are APFS clones that share every block except the rewritten header.
A plain `cp` or `rsync` of both uses twice the space. `pull` keeps a pair shared when the partner is
already hot. Offloading one of a hot pair frees almost nothing until the other goes too. To clone by
hand use `/bin/cp -c`, since a GNU `cp` earlier on `PATH` reads `-c` differently.

## Benchmarks

I/O timings depend on where a weight physically lives, and a cold read over the external link is not
comparable with an internal one.

- **A timing run reads its weights from the internal drive.** Check before starting; this exits 1 if
  a weight is not on the home volume:

  ```sh
  go run ./cmd/kolibri-models check testdata/bench/<matrix>.json   # add -only a,b to match the run
  ```

- **Each run records the location.** `kolibri-evict` prints where the weight resolves, and
  `kolibri-bench` keeps that in `run.json` under `evict`. A run marked `NOT the home volume` is not a
  timing result. For a run outside `kolibri-bench`, write down `realpath <file>` with the results.
- **Results go under `~/models/eval/`,** so that writing them does not touch the external link.

## When the cold volume is absent

Cold entries are then dangling symlinks: `No such file or directory`, and `status` says `absent`.
Nothing is lost. Hot weights, `eval/` and the code keep working.

Mount the volume, or carry on with what is hot and say what could not run. Do not delete the
links, re-download or reconvert a weight, or put a file in a link's place.

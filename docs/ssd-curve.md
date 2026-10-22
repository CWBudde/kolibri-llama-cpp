# SSD random-read curve

Written by `cmd/kolibri-ssd` (PLAN Phase 11, Stage E1). Each cell reads a fixed volume at distinct,
non-overlapping random offsets inside the GGUF's routed-expert tensors; QD is the number of goroutines
issuing blocking `pread`s on one shared descriptor, as the expert cache's I/O threads do. `nocache` sets
`F_NOCACHE` like patch 0011; `buffered` goes through the page cache. MB/s is 10^6 bytes per second of
wall time; latencies are per op (one chunk, or one expert) in ms.

## Run 2026-10-22T02:41:00+02:00

- Command: `/Users/christian/models/eval/ssd/kolibri-ssd -json /Users/christian/models/eval/ssd/sweep.jsonl`
- File: `/Users/christian/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf` (35,556,257,600 bytes; experts 32,636,928,000 bytes, 50 layers x 384)
- Expert slices (layer 0): gate IQ3_XXS 501,760, up IQ3_XXS 501,760, down IQ4_XS 696,320 bytes
- Machine: model Mac17,9; cpu Apple M5 Pro; memory 48 GiB; os 27.0.1; ssd APPLE SSD AP1024Z
- Power: Now drawing from 'AC Power'; page size 16384; seed 1; evict before each cell: true

## Run 2026-10-22T02:41:33+02:00

- Command: `/Users/christian/models/eval/ssd/kolibri-ssd -expert -json /Users/christian/models/eval/ssd/expert.jsonl`
- File: `/Users/christian/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf` (35,556,257,600 bytes; experts 32,636,928,000 bytes, 50 layers x 384)
- Expert slices (layer 0): gate IQ3_XXS 501,760, up IQ3_XXS 501,760, down IQ4_XS 696,320 bytes
- Machine: model Mac17,9; cpu Apple M5 Pro; memory 48 GiB; os 27.0.1; ssd APPLE SSD AP1024Z
- Power: Now drawing from 'AC Power'; page size 16384; seed 1; evict before each cell: true

## Run 2026-10-22T02:41:46+02:00

- Command: `/Users/christian/models/eval/ssd/kolibri-ssd -no-evict -modes buffered -chunks 64KiB,256KiB -json /Users/christian/models/eval/ssd/m41-warm.jsonl`
- File: `/Users/christian/models/Kolibri-1-IQ3_XXS-IQ4_XS-down-imx.gguf` (35,556,257,600 bytes; experts 32,636,928,000 bytes, 50 layers x 384)
- Expert slices (layer 0): gate IQ3_XXS 501,760, up IQ3_XXS 501,760, down IQ4_XS 696,320 bytes
- Machine: model Mac17,9; cpu Apple M5 Pro; memory 48 GiB; os 27.0.1; ssd APPLE SSD AP1024Z
- Power: Now drawing from 'AC Power'; page size 16384; seed 1; evict before each cell: false

## raw, nocache, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per chunk.

| chunk | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 64 KiB | 652 (9946)<br>0.10 / 0.18 / 2.42 | 2321 (35413)<br>0.11 / 0.17 / 0.27 | 6040 (92166)<br>0.16 / 0.29 / 2.48 |
| 256 KiB | 2062 (7867)<br>0.12 / 0.16 / 2.26 | 6004 (22905)<br>0.17 / 0.24 / 2.35 | 12015 (45832)<br>0.35 / 0.51 / 0.67 |
| 501,760 B | 2509 (5001)<br>0.19 / 0.27 / 2.43 | 6299 (12554)<br>0.31 / 0.44 / 2.48 | 11896 (23708)<br>0.63 / 2.92 / 3.29 |
| 696,320 B | 3514 (5047)<br>0.19 / 0.25 / 2.35 | 11482 (16489)<br>0.24 / 0.34 / 0.46 | 12846 (18449)<br>0.82 / 2.96 / 3.12 |
| 8 MiB | 10643 (1269)<br>0.75 / 2.92 / 2.92 | 13242 (1579)<br>2.37 / 5.11 / 5.11 | 13927 (1660)<br>9.30 / 10.49 / 10.49 |

Resident pages of 2,170,182: at most 0 before a timed pass, 0 after one.

## raw, buffered, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per chunk.

| chunk | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 64 KiB | 636 (9705)<br>0.10 / 0.14 / 2.25 | 2301 (35104)<br>0.11 / 0.17 / 0.36 | 6036 (92105)<br>0.16 / 0.29 / 2.37 |
| 256 KiB | 1870 (7132)<br>0.14 / 0.17 / 2.27 | 5855 (22333)<br>0.18 / 0.25 / 0.42 | 11985 (45718)<br>0.35 / 0.52 / 0.67 |
| 501,760 B | 2949 (5878)<br>0.17 / 0.21 / 2.26 | 9246 (18427)<br>0.21 / 0.28 / 2.55 | 12105 (24124)<br>0.62 / 2.49 / 2.90 |
| 696,320 B | 3046 (4375)<br>0.23 / 0.26 / 2.63 | 10623 (15256)<br>0.25 / 0.35 / 2.53 | 12854 (18460)<br>0.85 / 1.04 / 1.24 |
| 8 MiB | 10334 (1232)<br>0.77 / 3.06 / 3.06 | 13292 (1584)<br>2.35 / 5.15 / 5.15 | 12634 (1506)<br>10.33 / 13.11 / 13.11 |

Resident pages of 2,170,182: at most 0 before a timed pass, 33,153 after one.

## expert-three, nocache, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per expert.

| expert | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 1,699,840 B | 2776 (1633)<br>0.61 / 0.71 / 0.97 | 9448 (5558)<br>0.68 / 3.06 / 3.23 | 12471 (7337)<br>2.05 / 3.99 / 4.13 |

Resident pages of 2,170,182: at most 0 before a timed pass, 0 after one.

## expert-contiguous, nocache, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per expert.

| expert | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 1,699,840 B | 5479 (3223)<br>0.28 / 0.49 / 2.50 | 12911 (7595)<br>0.49 / 2.72 / 2.82 | 13160 (7742)<br>1.95 / 4.08 / 4.26 |

Resident pages of 2,170,182: at most 0 before a timed pass, 2,040 after one.

## expert-three, buffered, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per expert.

| expert | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 1,699,840 B | 2855 (1680)<br>0.59 / 0.64 / 2.90 | 9446 (5557)<br>0.68 / 2.94 / 3.13 | 11975 (7045)<br>2.13 / 4.05 / 4.35 |

Resident pages of 2,170,182: at most 0 before a timed pass, 33,627 after one.

## expert-contiguous, buffered, evicted (cold)

MB/s (ops/s), then p50 / p99 / max latency in ms per expert.

| expert | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 1,699,840 B | 5426 (3192)<br>0.30 / 0.35 / 2.81 | 13076 (7692)<br>0.49 / 2.73 / 2.89 | 13519 (7953)<br>1.97 / 2.34 / 2.44 |

Resident pages of 2,170,182: at most 0 before a timed pass, 32,760 after one.

## raw, buffered, not evicted, warmed (-no-evict)

MB/s (ops/s), then p50 / p99 / max latency in ms per chunk.

| chunk | QD 1 | QD 4 | QD 16 |
|---:|---:|---:|---:|
| 64 KiB | 22343 (340921)<br>0.00 / 0.00 / 0.01 | 56168 (857061)<br>0.00 / 0.01 / 0.02 | 52831 (806134)<br>0.02 / 0.04 / 4.36 |
| 256 KiB | 31498 (120154)<br>0.01 / 0.02 / 0.12 | 101824 (388427)<br>0.01 / 0.02 / 0.02 | 133256 (508330)<br>0.03 / 0.08 / 0.31 |

Resident pages of 2,170,182: at most 218,848 before a timed pass, 218,848 after one.

## Control M41: eviction matters

- PASS raw buffered 64 KiB QD 1: warm 22343 MB/s, evicted 636 MB/s, 35.1x
- PASS raw buffered 64 KiB QD 4: warm 56168 MB/s, evicted 2301 MB/s, 24.4x
- PASS raw buffered 64 KiB QD 16: warm 52831 MB/s, evicted 6036 MB/s, 8.8x
- PASS raw buffered 256 KiB QD 1: warm 31498 MB/s, evicted 1870 MB/s, 16.8x
- PASS raw buffered 256 KiB QD 4: warm 101824 MB/s, evicted 5855 MB/s, 17.4x
- PASS raw buffered 256 KiB QD 16: warm 133256 MB/s, evicted 11985 MB/s, 11.1x

package main

import (
	"fmt"
	"io"
	"math"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// read is one pread: n bytes at file offset off into buffer buf at dst.
type read struct {
	Off int64
	N   int64
	Buf int
	Dst int64
}

// op is what one worker does at a time and what one latency sample covers:
// a raw chunk is one read; an expert is three (gate, up, down) or one.
type op []read

// readerAt is the file: *os.File's ReadAt is pread on a shared descriptor,
// safe for concurrent use.
type readerAt interface {
	ReadAt(p []byte, off int64) (int, error)
}

// timing is one cell's measurement.
type timing struct {
	Ops       int
	Bytes     int64
	Seconds   float64
	Latencies []time.Duration
	TimeCap   bool // stopped by the time cap before all ops ran
}

// alignedBuf allocates n bytes starting on an align boundary (a page:
// F_NOCACHE reads into page-aligned memory can go to the device directly).
func alignedBuf(n, align int64) []byte {
	b := make([]byte, n+align)
	skip := (align - int64(uintptr(unsafe.Pointer(&b[0])))%align) % align
	return b[skip : skip+n : skip+n]
}

// bufSizes are the buffer sizes a worker needs for ops.
func bufSizes(ops []op) []int64 {
	var s []int64
	for _, o := range ops {
		for _, r := range o {
			for len(s) <= r.Buf {
				s = append(s, 0)
			}
			s[r.Buf] = max(s[r.Buf], r.Dst+r.N)
		}
	}
	return s
}

// runCell runs ops with qd workers on one file: each worker takes the next
// op, issues its reads one after another with blocking preads and records
// the op's latency. Workers stop taking ops after maxDur (0: no cap).
func runCell(f readerAt, ops []op, qd int, maxDur time.Duration, align int64) (timing, error) {
	if qd < 1 {
		return timing{}, fmt.Errorf("queue depth %d", qd)
	}
	sizes := bufSizes(ops)
	bufs := make([][][]byte, qd)
	for w := range bufs {
		for _, n := range sizes {
			bufs[w] = append(bufs[w], alignedBuf(n, align))
		}
	}
	var (
		next     atomic.Int64
		capped   atomic.Bool
		firstErr error
		errOnce  sync.Once
		wg       sync.WaitGroup
	)
	lat := make([][]time.Duration, qd)
	bytes := make([]int64, qd)
	start := time.Now()
	for w := range qd {
		wg.Go(func() {
			for {
				i := next.Add(1) - 1
				if i >= int64(len(ops)) {
					return
				}
				if maxDur > 0 && time.Since(start) > maxDur {
					capped.Store(true)
					return
				}
				t0 := time.Now()
				for _, r := range ops[i] {
					if err := readFull(f, bufs[w][r.Buf][r.Dst:r.Dst+r.N], r.Off); err != nil {
						errOnce.Do(func() { firstErr = err })
						return
					}
				}
				lat[w] = append(lat[w], time.Since(t0))
				for _, r := range ops[i] {
					bytes[w] += r.N
				}
			}
		})
	}
	wg.Wait()
	t := timing{Seconds: time.Since(start).Seconds(), TimeCap: capped.Load()}
	for w := range qd {
		t.Latencies = append(t.Latencies, lat[w]...)
		t.Bytes += bytes[w]
	}
	t.Ops = len(t.Latencies)
	return t, firstErr
}

// readFull reads len(p) bytes at off; ReadAt already retries short reads.
func readFull(f readerAt, p []byte, off int64) error {
	n, err := f.ReadAt(p, off)
	if n == len(p) {
		return nil
	}
	if err == nil || err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return fmt.Errorf("pread of %d bytes at %d: %d read: %w", len(p), off, n, err)
}

// percentile is the nearest-rank percentile, as in cmd/kolibri-stream.
func percentile(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	s := slices.Clone(d)
	slices.Sort(s)
	rank := int(math.Ceil(p / 100 * float64(len(s))))
	return s[min(max(rank, 1), len(s))-1]
}

// stats are a cell's figures.
type stats struct {
	MBps   float64 `json:"mb_per_s"` // 10^6 bytes per second of wall time
	IOPS   float64 `json:"iops"`     // ops per second: chunks, or experts in expert mode
	P50Us  float64 `json:"p50_us"`
	P99Us  float64 `json:"p99_us"`
	MaxUs  float64 `json:"max_us"`
	MeanUs float64 `json:"mean_us"`
}

func summarize(t timing) stats {
	us := func(d time.Duration) float64 { return float64(d) / float64(time.Microsecond) }
	var s stats
	if t.Seconds > 0 {
		s.MBps = float64(t.Bytes) / 1e6 / t.Seconds
		s.IOPS = float64(t.Ops) / t.Seconds
	}
	if len(t.Latencies) > 0 {
		var sum time.Duration
		for _, d := range t.Latencies {
			sum += d
		}
		s.P50Us = us(percentile(t.Latencies, 50))
		s.P99Us = us(percentile(t.Latencies, 99))
		s.MaxUs = us(slices.Max(t.Latencies))
		s.MeanUs = us(sum / time.Duration(len(t.Latencies)))
	}
	return s
}

// rawOps makes one single-read op per offset.
func rawOps(offs []int64, chunk int64) []op {
	ops := make([]op, len(offs))
	for i, o := range offs {
		ops[i] = op{{Off: o, N: chunk}}
	}
	return ops
}

// slots is how many expert slots a worker cycles through: like the cache's
// slot tensors, the destination of slot s is s*slice bytes into the buffer,
// so most destinations are not page-aligned.
const slots = 8

// expertThreeOps reads each expert as patch 0011 does: three preads, gate,
// up and down, each into its own slot buffer.
func expertThreeOps(l *layout, ids []expertID) []op {
	ops := make([]op, len(ids))
	for i, id := range ids {
		s := int64(i % slots)
		o := make(op, 3)
		for k := range kinds {
			n := l.Layers[id.Layer][k].Slice
			o[k] = read{Off: l.sliceOffset(id.Layer, k, id.Expert), N: n, Buf: k, Dst: s * n}
		}
		ops[i] = o
	}
	return ops
}

// expertContiguousOps reads each expert's bytes with one pread from a
// random aligned offset: what a side file with the experts repacked
// contiguously would cost.
func expertContiguousOps(l *layout, ids []expertID, offs []int64) []op {
	ops := make([]op, len(ids))
	for i, id := range ids {
		n := l.expertBytes(id.Layer)
		ops[i] = op{{Off: offs[i], N: n, Dst: int64(i%slots) * n}}
	}
	return ops
}

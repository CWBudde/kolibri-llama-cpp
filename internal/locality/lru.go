package locality

// LRU cache simulation (PLAN Phase 8): how often a cache of recently used experts would already
// hold the experts a workload selects, before any streaming I/O exists.
//
// An access's stack distance is the number of distinct other keys accessed since the key's
// previous access (-1 for a first access). A fully associative LRU cache of capacity C hits
// exactly the accesses with a distance below C (Mattson et al. 1970), so one pass over a trace
// answers every capacity. Unlike Stats.StackDistance, which treats a token's experts as one set,
// the accesses here are sequential: a token's experts in the order the router lists them.

// Flatten lists a layer's selections as one access sequence, token by token.
func Flatten(sel [][]int) []int {
	var out []int
	for _, experts := range sel {
		out = append(out, experts...)
	}
	return out
}

// SharedKeys lists the selections of all layers as one access sequence for a cache shared by the
// layers, in decode order: token by token, and within a token layer by layer. Expert e of layer
// l is key l*nExpert+e; layers[i] is the layer of access i.
func SharedKeys(s Selections, nExpert int) (keys, layers []int) {
	keys = make([]int, 0, s.Layers*s.Tokens*s.K)
	layers = make([]int, 0, cap(keys))
	for t := range s.Tokens {
		for l := range s.Layers {
			for _, e := range s.IDs[(l*s.Tokens+t)*s.K : (l*s.Tokens+t+1)*s.K] {
				keys = append(keys, l*nExpert+int(e))
				layers = append(layers, l)
			}
		}
	}
	return keys, layers
}

// StackDistances returns each access's LRU stack distance, -1 for a key's first access. Keys are in
// [0, nKey).
//
// A Fenwick tree over the access times marks, for every key, the time of its latest access; the
// distinct keys accessed strictly between a key's previous access and now are the marks in
// between.
func StackDistances(keys []int, nKey int) []int {
	n := len(keys)
	tree := make([]int, n+1)
	add := func(i, v int) {
		for i++; i <= n; i += i & -i {
			tree[i] += v
		}
	}
	prefix := func(i int) int { // marks at times [0, i)
		s := 0
		for ; i > 0; i -= i & -i {
			s += tree[i]
		}
		return s
	}
	last := make([]int, nKey)
	for k := range last {
		last[k] = -1
	}
	dist := make([]int, n)
	for i, k := range keys {
		if p := last[k]; p < 0 {
			dist[i] = -1
		} else {
			dist[i] = prefix(i) - prefix(p+1)
			add(p, -1)
		}
		add(i, 1)
		last[k] = i
	}
	return dist
}

// Hits returns, for each capacity, how many accesses an LRU cache of that capacity hits: those
// with a stack distance in [0, capacity).
func Hits(dist []int, capacities []int) []int {
	maxCap := 0
	for _, c := range capacities {
		maxCap = max(maxCap, c)
	}
	byDist := make([]int, maxCap)
	for _, d := range dist {
		if d >= 0 && d < maxCap {
			byDist[d]++
		}
	}
	below := make([]int, maxCap+1) // below[c]: accesses with a distance under c
	for d, n := range byDist {
		below[d+1] = below[d] + n
	}
	out := make([]int, len(capacities))
	for i, c := range capacities {
		out[i] = below[max(c, 0)]
	}
	return out
}

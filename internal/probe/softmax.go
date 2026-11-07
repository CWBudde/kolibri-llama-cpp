package probe

import "math"

// fit is an L2-regularized multinomial logistic regression, solved in the span of the training
// rows: the weights are w_c = sum_i a_ic x_i, so with far more dimensions (2,560) than rows (about
// 32) the solver needs only the training rows' inner products. It minimizes
//
//	(1/n) sum_i CE(softmax(f_i), y_i) + (l2/2) sum_c |w_c|^2,  f_ic = sum_j K_ij a_jc + b_c
//
// by Nesterov's method from zero, which is deterministic. The bias b is not regularized.
type fit struct {
	alpha     []float64 // n x k
	bias      []float64 // k
	iters     int
	converged bool
}

const (
	maxIters = 20000
	// gradTol bounds the largest entry of the gradient at convergence.
	gradTol = 1e-8
)

// largestEigen is the largest eigenvalue of a symmetric n x n matrix, by power iteration, at least
// what it reaches after a fixed number of steps; the margin in fitSoftmax covers the rest.
func largestEigen(m []float64, n int) float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = 1 / math.Sqrt(float64(n))
	}
	var lambda float64
	next := make([]float64, n)
	for range 200 {
		for i := range n {
			var s float64
			for j := range n {
				s += m[i*n+j] * v[j]
			}
			next[i] = s
		}
		var norm float64
		for _, x := range next {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			return 0
		}
		lambda = norm
		for i := range v {
			v[i] = next[i] / norm
		}
	}
	return lambda
}

// softmaxInto writes the class probabilities of logits f.
func softmaxInto(f, p []float64) {
	top := math.Inf(-1)
	for _, x := range f {
		top = max(top, x)
	}
	var sum float64
	for i, x := range f {
		p[i] = math.Exp(x - top)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
}

// gradient writes the gradient at (alpha, bias) in alpha's coordinates: for a training row i and a
// class c, (p_ic - y_ic)/n + l2 a_ic, and for the bias sum_i (p_ic - y_ic)/n. It returns the largest
// entry in absolute value, scaled by n for the rows, so that the test does not depend on n.
func gradient(kern []float64, y []int, n, k int, l2 float64, alpha, bias, gAlpha, gBias []float64) float64 {
	f, p := make([]float64, k), make([]float64, k)
	clear(gBias)
	var worst float64
	for i := range n {
		for c := range k {
			f[c] = bias[c]
		}
		for j := range n {
			kij := kern[i*n+j]
			for c := range k {
				f[c] += kij * alpha[j*k+c]
			}
		}
		softmaxInto(f, p)
		for c := range k {
			r := p[c]
			if y[i] == c {
				r--
			}
			r /= float64(n)
			gAlpha[i*k+c] = r + l2*alpha[i*k+c]
			gBias[c] += r
			worst = max(worst, math.Abs(gAlpha[i*k+c])*float64(n))
		}
	}
	for c := range k {
		worst = max(worst, math.Abs(gBias[c]))
	}
	return worst
}

// fitSoftmax trains on the n x n kernel of the training rows; y holds classes in [0, k).
func fitSoftmax(kern []float64, y []int, k int, l2 float64) fit {
	n := len(y)
	// The gradient is Lipschitz with at most (lambda_max(K)/n + 1)/2 + l2 in the weights.
	lip := 0.5*(1.01*largestEigen(kern, n)/float64(n)+1) + l2
	step := 1 / lip
	kappa := lip / l2
	momentum := (math.Sqrt(kappa) - 1) / (math.Sqrt(kappa) + 1)

	alpha, bias := make([]float64, n*k), make([]float64, k)
	prevA, prevB := make([]float64, n*k), make([]float64, k)
	lookA, lookB := make([]float64, n*k), make([]float64, k)
	gA, gB := make([]float64, n*k), make([]float64, k)
	checkA, checkB := make([]float64, n*k), make([]float64, k)
	out := fit{alpha: alpha, bias: bias}
	for it := 1; it <= maxIters; it++ {
		for i := range lookA {
			lookA[i] = alpha[i] + momentum*(alpha[i]-prevA[i])
		}
		for i := range lookB {
			lookB[i] = bias[i] + momentum*(bias[i]-prevB[i])
		}
		gradient(kern, y, n, k, l2, lookA, lookB, gA, gB)
		copy(prevA, alpha)
		copy(prevB, bias)
		for i := range alpha {
			alpha[i] = lookA[i] - step*gA[i]
		}
		for i := range bias {
			bias[i] = lookB[i] - step*gB[i]
		}
		out.iters = it
		if it%10 == 0 && gradient(kern, y, n, k, l2, alpha, bias, checkA, checkB) <= gradTol {
			out.converged = true
			break
		}
	}
	return out
}

// predict returns the class probabilities (rows x k) for a test-by-train kernel.
func (m fit) predict(kern []float64, rows, nTrain, k int) []float64 {
	out := make([]float64, rows*k)
	f := make([]float64, k)
	for i := range rows {
		copy(f, m.bias)
		for j := range nTrain {
			kij := kern[i*nTrain+j]
			for c := range k {
				f[c] += kij * m.alpha[j*k+c]
			}
		}
		softmaxInto(f, out[i*k:(i+1)*k])
	}
	return out
}

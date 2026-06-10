// SIMD Benchmark: Axiom vs GCC/LLVM -O3
// These kernels are designed to test auto-vectorization quality.
// Compile with: g++ -O3 -march=native -S simd_bench.cpp && g++ -O3 -march=native -S -foptimize-simd simd_bench.cpp

// ── Kernel 1: Array Addition (map pattern) ──────────────────────
// for i in 0..N: c[i] = a[i] + b[i]
extern "C" void vec_add(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ── Kernel 2: Array Multiply-Add (fused pattern) ────────────────
// for i in 0..N: c[i] = a[i] * b[i] + c[i]
extern "C" void vec_fma(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * b[i] + c[i];
    }
}

// ── Kernel 3: Reduction Sum ─────────────────────────────────────
// for i in 0..N: sum += a[i]
extern "C" long reduction_sum(int* __restrict__ a, int n) {
    long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// ── Kernel 4: Reduction Dot Product ─────────────────────────────
// for i in 0..N: sum += a[i] * b[i]
extern "C" long reduction_dot(int* __restrict__ a, int* __restrict__ b, int n) {
    long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += (long)a[i] * b[i];
    }
    return sum;
}

// ── Kernel 5: Scalar + Vector broadcast ─────────────────────────
// for i in 0..N: b[i] = a[i] * scalar
extern "C" void vec_scalar_mul(int* __restrict__ a, int* __restrict__ b, int scalar, int n) {
    for (int i = 0; i < n; i++) {
        b[i] = a[i] * scalar;
    }
}

// ── Kernel 6: Copy pattern ──────────────────────────────────────
// for i in 0..N: dst[i] = src[i]
extern "C" void vec_copy(int* __restrict__ src, int* __restrict__ dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// ── Kernel 7: Conditional add (masked) ──────────────────────────
// for i in 0..N: if (a[i] > 0) b[i] += a[i]
extern "C" void vec_masked_add(int* __restrict__ a, int* __restrict__ b, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] > 0) {
            b[i] += a[i];
        }
    }
}

// ── Kernel 8: Max reduction ─────────────────────────────────────
// for i in 0..N: max = max(max, a[i])
extern "C" int reduction_max(int* __restrict__ a, int n) {
    int max = a[0];
    for (int i = 1; i < n; i++) {
        if (a[i] > max) max = a[i];
    }
    return max;
}

// ── Kernel 9: Float vector add ──────────────────────────────────
extern "C" void vec_add_f64(double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ── Kernel 10: Float reduction sum ──────────────────────────────
extern "C" double reduction_sum_f64(double* __restrict__ a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

// ── Kernel 11: Triad (STREAM benchmark kernel) ──────────────────
// for i in 0..N: a[i] = b[i] + scalar * c[i]
extern "C" void stream_triad(double* __restrict__ a, double* __restrict__ b, double* __restrict__ c, double scalar, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

// ── Kernel 12: Integer XOR pattern ──────────────────────────────
extern "C" void vec_xor(int* __restrict__ a, int* __restrict__ b, int* __restrict__ c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] ^ b[i];
    }
}

#pragma once

// The EXL3 decode's device helpers: conversions, loads/stores, rotations, streams and MUL1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace {

typedef __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16 f16x16;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float f32x8;
typedef _Float16 h2 __attribute__((ext_vector_type(2)));

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
enum DType : int { F16 = 0, BF16 = 1, F32 = 2 };

// One fp32 made opaque: clang folds fp16(a*b) of fp32 into v_fma_mix with ONE rounding where the source rounds twice.
__device__ __forceinline__ float f32_value(float x) {
    asm volatile("" : "+v"(x));
    return x;
}
__device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f32_value(f)); }
__device__ __forceinline__ float bf16_bits_to_f(uint32_t b) { return __uint_as_float(b << 16); }
__device__ __forceinline__ uint32_t f_to_bf16_bits(float f) { return __bfloat16_as_ushort(__float2bfloat16(f32_value(f))); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16(f32_value(x))); }
__device__ __forceinline__ uint32_t f_to_h_bits(float f) { return __half_as_ushort(f2h(f)); }

__device__ __forceinline__ void load4(const void* p, int dtype, size_t i, float (&v)[4]) {
    if (dtype == F32) {
        const float4 u = *reinterpret_cast<const float4*>(static_cast<const float*>(p) + i);
        v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(p) + i);
        if (dtype == BF16) {
            v[0] = bf16_bits_to_f(u.x & 0xffffu); v[1] = bf16_bits_to_f(u.x >> 16);
            v[2] = bf16_bits_to_f(u.y & 0xffffu); v[3] = bf16_bits_to_f(u.y >> 16);
        } else {
            v[0] = __half2float(__ushort_as_half(u.x & 0xffffu)); v[1] = __half2float(__ushort_as_half(u.x >> 16));
            v[2] = __half2float(__ushort_as_half(u.y & 0xffffu)); v[3] = __half2float(__ushort_as_half(u.y >> 16));
        }
    }
}

__device__ __forceinline__ void store4(void* p, int dtype, size_t i, const float (&v)[4]) {
    if (dtype == F32) {
        *reinterpret_cast<float4*>(static_cast<float*>(p) + i) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (dtype == BF16) {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_bf16_bits(v[0]) | f_to_bf16_bits(v[1]) << 16, f_to_bf16_bits(v[2]) | f_to_bf16_bits(v[3]) << 16);
    } else {
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(p) + i) =
            make_uint2(f_to_h_bits(v[0]) | f_to_h_bits(v[1]) << 16, f_to_h_bits(v[2]) | f_to_h_bits(v[3]) << 16);
    }
}

// Generic pointers compile to flat_load, counting in lgkmcnt beside vmcnt, so LDS waits also wait on weight prefetch.
template <typename T>
__device__ __forceinline__ const __attribute__((address_space(1))) T* gl(const T* p) {
    return (const __attribute__((address_space(1))) T*)p;
}

// a coherent (L2) load and the acquire / release pair of the decode experts' readiness flags
__device__ __forceinline__ float ld_cg(const float* p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ int ld_acquire(const int* p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
__device__ __forceinline__ void st_release(int* p, int v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT); }

// The unscaled Hadamard of 128 fp32 values, 4 a lane (elements 4 lane + j), decode.cuh's butterfly order.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// One row's rotated 128 inputs ((x * suh) @ H) / sqrt(128), rot128_in's first butterfly an fma of the products.
__device__ __forceinline__ void rot128_in(float (&v)[4], const float (&s)[4], int lane) {
    const float p1 = v[1] * s[1], p3 = v[3] * s[3];
    const float a = fmaf(v[0], s[0], p1), b = fmaf(v[0], s[0], -p1);
    const float c = fmaf(v[2], s[2], p3), d = fmaf(v[2], s[2], -p3);
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE;
}

// One row's finished 128 columns from fp32 sums: H / sqrt(128), * svh, + bias as one fma (linear_common.cuh finish).
__device__ __forceinline__ void finish(float (&v)[4], int lane, const __half* svh, const __half* bias, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float t = v[j] * HAD_SCALE, sv = __half2float(svh[col + j]);
        v[j] = bias ? fmaf(t, sv, __half2float(bias[col + j])) : t * sv;
    }
}

// E(p): one past the last stream bit of value p's 16-bit window (format.py stream_ends; decode.cuh stream_end).
template <int K2>
__host__ __device__ constexpr int stream_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// LaneWin: a lane's four stream windows (column c = lane & 15, k half h = lane >> 4).

// Pair p0 = 32 (c % 8) + 8 q + 4 (c / 8) + 2 h covers rows 8 h + 2 q and its successor of column c.

// Each pair's window sits in words hi and lo (lo the next, wrapping), starting o bits into hi.

// The second starts D bits after the first; hi, lo, sh feed alignbit its shift mod 32.
template <int K2>
struct LaneWin {
    static constexpr int TW = 4 * K2;
    static constexpr int D = stream_end<K2>(1) - stream_end<K2>(0);
    int hi[4], lo[4], sh[4];
    __device__ __forceinline__ explicit LaneWin(int lane) {
        const int c = lane & 15, h = lane >> 4;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int p0 = 32 * (c & 7) + 8 * q + 4 * (c >> 3) + 2 * h;
            const int s = (stream_end<K2>(p0) - 16 + 128 * K2) % (128 * K2);
            const int o = s & 31;
            hi[q] = s >> 5;                                        // pair rows are 8 h + 2 q and its successor
            lo[q] = o ? (hi[q] + 1) % TW : hi[q];                  // lo = hi when the window starts a word
            sh[q] = (32 - o) & 31;                                 // v_alignbit_b32's shift is its argument mod 32
        }
    }
};

// x * 0x83DCD12D mod 2^32 of a 16-bit state. EXL3_MUL24=1 swaps two umul24s for v_mul_lo_u32; off by default.
__device__ __forceinline__ uint32_t mul1_hash(uint32_t w) {
#if defined(EXL3_MUL24) && EXL3_MUL24
    // the barrier keeps the compiler from folding the shifted product back into a v_mul_lo_u32
    uint32_t b = __umul24(w, 0x83DCu);
    asm("" : "+v"(b));
    return __umul24(w, 0xD12Du) + (b << 16);
#else
    return w * 0x83DCD12Du;
#endif
}

// MUL1's decode of two values from 32 stream bits: fp16(0x6400 + bytesum) * fp16(0x1EEE) + fp16(0xC931), fused.
template <int D>
__device__ __forceinline__ uint32_t decode_pair(uint32_t u) {
    const uint32_t w0 = u >> 16;
    const uint32_t w1 = __builtin_amdgcn_ubfe(u, 16 - D, 16);
    const uint32_t s0 = __builtin_amdgcn_sad_u8(mul1_hash(w0), 0u, 0x64006400u);   // low half: 0x6400 + bytesum
    const uint32_t s = __builtin_amdgcn_sad_hi_u8(mul1_hash(w1), 0u, s0);           // high half: the second's
    h2 hv = __builtin_bit_cast(h2, s);  // bitcast-ok: whole u32 to f16x2, no lane extract
    hv = __builtin_elementwise_fma(hv, __builtin_bit_cast(h2, 0x1eee1eeeu), __builtin_bit_cast(h2, 0xc931c931u));
    return __builtin_bit_cast(uint32_t, hv);  // bitcast-ok: whole f16x2 to u32, no lane extract
}

// This lane's tile values: k rows 8 h .. 8 h + 7 of column c as four packed pairs from the aligned windows.
template <int K2>
__device__ __forceinline__ void decode_own(const uint32_t* tw, const LaneWin<K2>& w, uint32_t (&own)[4]) {
#pragma unroll
    for (int q = 0; q < 4; ++q) own[q] = decode_pair<LaneWin<K2>::D>(__builtin_amdgcn_alignbit(tw[w.hi[q]], tw[w.lo[q]], w.sh[q]));
}

}  // namespace

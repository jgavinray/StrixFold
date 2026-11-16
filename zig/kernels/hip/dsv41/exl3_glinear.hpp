#pragma once

// The EXL3 tile loops: the epilogue's store side, the dot2 and wmma paths over the strips tiles, glinear's block body.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "dsv41/abi.h"
#include "../common/arch.hpp"
#include "exl3_dec.hpp"

namespace {

// rot_out: the finished outputs of one row's 128 columns as stored in y, rotated into the next layer's input rows.

// rot_many computes from y: y's rounding, then rot128_in with the next layer's suh.
__device__ __forceinline__ void rot_out(const tf_ds_exl3_glayer& L, const float (&v)[4], int y_dtype, int lane, size_t row, int col0) {
    float u[4], s[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        u[j] = y_dtype == BF16 ? bf16r(v[j]) : y_dtype == F16 ? __half2float(f2h(v[j])) : v[j];
    const int c = L.roff + col0 + 4 * lane;
    load4(reinterpret_cast<const void*>(L.rsuh), F16, c, s);
    rot128_in(u, s, lane);
    store4(reinterpret_cast<void*>(L.rxh), F16, row * (size_t)L.ldr + c, u);
}

// rope_heads (forward) of one row's 128 stored columns when they hold a head's rotary part.

// v rounds as stored (bf16), then re = fma(xe, cs, -(xo sn)) and im = fma(xe, sn, xo cs) on each adjacent pair.
__device__ __forceinline__ void rope_out(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    const int d0 = (col0 % L.rhd) + 4 * lane, lo = L.rhd - L.rrd;
    if (d0 < lo) return;
    const int64_t p = reinterpret_cast<const int64_t*>(L.rpos)[row];
    const float* rcos = reinterpret_cast<const float*>(L.rcos);
    const float* rsin = reinterpret_cast<const float*>(L.rsin);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int j = (d0 + 2 * q - lo) >> 1;
        const float cs = rcos[p * (L.rrd >> 1) + j], sn = rsin[p * (L.rrd >> 1) + j];
        const float xe = bf16r(v[2 * q]), xo = bf16r(v[2 * q + 1]);
        v[2 * q] = fmaf(xe, cs, -(xo * sn));
        v[2 * q + 1] = fmaf(xe, sn, xo * cs);
    }
}

__device__ __forceinline__ void finish_store(const tf_ds_exl3_glayer& L, float (&v)[4], int lane, size_t row, int col0) {
    finish(v, lane, reinterpret_cast<const __half*>(L.svh), reinterpret_cast<const __half*>(L.bias), col0 + 4 * lane);
    if (L.rcos) rope_out(L, v, lane, row, col0);
    store4(reinterpret_cast<void*>(L.y), L.y_dtype, row * (size_t)L.ldy + col0 + 4 * lane, v);
    if (L.rsuh) rot_out(L, v, L.y_dtype, lane, row, col0);
}

// The tile's B fragment for column lane & 15: this lane's eight k from its own decode, then its partner's (lane ^ 16).
__device__ __forceinline__ f16x16 bfrag_from_own(const uint32_t (&own)[4], int lane) {
    (void)lane;
    uint32_t bw[8];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        bw[q] = own[q];
        bw[4 + q] = __builtin_amdgcn_permlanex16(own[q], own[q], 0x76543210u, 0xfedcba98u, false, false);
    }
    f16x16 b;
    __builtin_memcpy(&b, bw, 32);
    return b;
}

__device__ __forceinline__ f16x16 frag8(const uint4& lo, const uint4& hi) {
    const uint32_t w[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    f16x16 a;
    __builtin_memcpy(&a, w, 32);
    return a;
}

// 16 contiguous halves (32-byte aligned) as two 16-byte loads; zero for a null row
__device__ __forceinline__ void load16h(const _Float16* p, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[0];
        hi = q[1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

// The same for an A fragment against bfrag_from_own's B: lanes 16-31 (sw = 1) take k 8-15 first
__device__ __forceinline__ void load16h_k(const _Float16* p, int sw, uint4& lo, uint4& hi) {
    if (p) {
        const auto q = gl(reinterpret_cast<const uint4*>(p));
        lo = q[sw];
        hi = q[sw ^ 1];
    } else {
        lo = hi = make_uint4(0u, 0u, 0u, 0u);
    }
}

// the gfx11 wave32 wmma fragments; arch.hpp's TF_DEVICE_WMMA_GFX11 (gfx12's layouts differ, arch.hpp:45)
#if TF_DEVICE_WMMA_GFX11
__device__ __forceinline__ f32x8 wmma(const f16x16& a, const f16x16& b, const f32x8& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
#endif

__device__ __forceinline__ void lds_fence() {
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
}

// One wave's k tiles of a 128-column strip against one row (xrow: 16 halves a k tile), without the matrix unit.

// Steps `step` words apart; a k step's words load one step ahead, stage in LDS (st: the wave's), and read back.

// A lane's decoded pairs meet the row's input pairs in v_dot2_f32_f16; the half-waves' sums add at the end.

// acc[t][0] of lanes 0-15 holds output column 16 t + c (row 0 of WMMA's C layout); the other elements stay as they are.
__device__ __forceinline__ float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(h2, a), __builtin_bit_cast(h2, b), c, false);  // bitcast-ok: whole u32 to f16x2
}

template <int K2>
__device__ __forceinline__ void strip_tiles_1row(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xrow,
                                                 uint32_t* st, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, LW = 8 * TW / 32;
    if (n <= 0) return;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    float s[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) s[t] = 0.f;
    uint32_t pf[LW];
    uint4 xa;
    {
        const uint32_t* p = tiles + (int64_t)kt0 * step;
#pragma unroll
        for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
        xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * kt0 + 8 * h));
    }
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < LW; ++i) st[i * 32 + lane] = pf[i];
        const uint4 x = xa;
        if (j + 1 < n) {
            const uint32_t* p = tiles + (int64_t)(kt0 + j + 1) * step;
#pragma unroll
            for (int i = 0; i < LW; ++i) pf[i] = __builtin_nontemporal_load(gl(p) + i * 32 + lane);
            xa = *gl(reinterpret_cast<const uint4*>(xrow + 16 * (kt0 + j + 1) + 8 * h));
        }
        lds_fence();
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            uint32_t own[4];
            decode_own<K2>(st + t * TW, win, own);
            float a = dot2(own[0], x.x, s[t]);
            a = dot2(own[1], x.y, a);
            a = dot2(own[2], x.z, a);
            s[t] = dot2(own[3], x.w, a);
        }
        __builtin_amdgcn_wave_barrier();
    }
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        const float o = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s[t]), __float_as_uint(s[t]), 0x76543210u, 0xfedcba98u, false, false));
        acc[t][0] += s[t] + o;
    }
}

// strip_tiles for 2 to 16 rows in the 1-row order, so a row's sums are the same bits whatever rows share the launch.

// A part takes TT of the strip's 8 tiles for up to RM rows: up to 8 rows take all tiles at once, more rows two parts.

// Parts load only their tiles' words; input rows (16 bytes a lane) stage one step ahead in xst, the wave's 32 uint4.

// acc as WMMA's C layout: lane l, float i -> row 2 i + (l >> 4), column 16 t + (l & 15).
template <int K2, int RM, int TT, int t0>
__device__ __forceinline__ void strip_rows_part(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                int R, uint32_t* st, uint4* xst, int lane, f32x8 (&acc)[8]) {
    constexpr int TW = 4 * K2, PW = TT * TW, RL = (PW + 31) / 32;
    const LaneWin<K2> win(lane);
    const int h = lane >> 4;
    const bool xlive = (lane >> 1) < R;
    const uint4* xl = gl(reinterpret_cast<const uint4*>(xlane + 8 * (lane & 1)));
    const uint32_t* tp = tiles + t0 * TW;
    float s[RM][TT];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int t = 0; t < TT; ++t) s[r][t] = 0.f;
    uint32_t pf[RL];
    uint4 xp = make_uint4(0u, 0u, 0u, 0u);
#define TF_DS_EXL3_ROWS_LOAD(kt_)                                                                                       \
    {                                                                                                                   \
        const uint32_t* p_ = tp + (int64_t)(kt_) * step;                                                                \
        _Pragma("unroll") for (int i = 0; i < RL; ++i) {                                                               \
            const int wi = i * 32 + lane;                                                                               \
            pf[i] = wi < PW ? __builtin_nontemporal_load(gl(p_) + wi) : 0u;                                             \
        }                                                                                                               \
        if (xlive) xp = xl[2 * (kt_)];                                                                                  \
    }
    TF_DS_EXL3_ROWS_LOAD(kt0)
    for (int j = 0; j < n; ++j) {
#pragma unroll
        for (int i = 0; i < RL; ++i) {
            const int wi = i * 32 + lane;
            if (wi < PW) st[wi] = pf[i];
        }
        xst[lane] = xp;
        if (j + 1 < n) TF_DS_EXL3_ROWS_LOAD(kt0 + j + 1)
        lds_fence();
        uint32_t own[TT][4];
#pragma unroll
        for (int t = 0; t < TT; ++t) decode_own<K2>(st + t * TW, win, own[t]);
#pragma unroll
        for (int r = 0; r < RM; ++r) {
            if (r < R) {
                const uint4 x = xst[2 * r + h];
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    float a = dot2(own[t][0], x.x, s[r][t]);
                    a = dot2(own[t][1], x.y, a);
                    a = dot2(own[t][2], x.z, a);
                    s[r][t] = dot2(own[t][3], x.w, a);
                }
            }
        }
        __builtin_amdgcn_wave_barrier();
    }
#undef TF_DS_EXL3_ROWS_LOAD
    // this lane's row 2 i + h of each pair: its own half's sum plus the partner's, whole vectors so registers hold them
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        float v[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (2 * i + 1 < RM) {
                const float s0 = s[2 * i][t], s1 = s[2 * i + 1][t];
                const float o0 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s0), __float_as_uint(s0), 0x76543210u, 0xfedcba98u, false, false));
                const float o1 = __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(s1), __float_as_uint(s1), 0x76543210u, 0xfedcba98u, false, false));
                v[i] = h ? s1 + o1 : s0 + o0;
            } else {
                v[i] = 0.f;
            }
        }
        acc[t0 + t] += (f32x8){v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
    }
}

template <int K2>
__device__ __forceinline__ void strip_tiles_rows(const uint32_t* tiles, int64_t step, int kt0, int n, const _Float16* xlane,
                                                 int R, uint32_t* st, uint4* xst, int lane, f32x8 (&acc)[8]) {
    if (n <= 0) return;
    if (R <= 8) {
        strip_rows_part<K2, 8, 8, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    } else {
        strip_rows_part<K2, 16, 4, 0>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
        __builtin_amdgcn_wave_barrier();   // the second part restages st and xst
        strip_rows_part<K2, 16, 4, 4>(tiles, step, kt0, n, xlane, R, st, xst, lane, acc);
    }
}

// glinear blocks for EXL3_WIDE_MIN rows or more: waves split the block's columns instead of K (8 / WK tiles each).

// Each wave decodes its tiles once a k tile, then applies them to every pass (up to 8) over the split's whole K range.

// SK == 1: the column tiles meet in LDS and each row's columns are finished; SK > 1 partials go to Z.

// The block completing the column block (counters[nb], left zero) sums the splits in order and finishes every row.

// From 64 rows by default (EXL3_WIDE=0: never): the wide split keeps its tiles hot across a long row set's passes.
#ifndef EXL3_WIDE_MIN
#define EXL3_WIDE_MIN 64
#endif

// the wide path needs gfx11's wmma fragments (TF_DEVICE_WMMA_GFX11); otherwise every row count runs the 16-row passes.
#if TF_DEVICE_WMMA_GFX11
template <int K2, int WK>
__device__ __forceinline__ void glinear_wide(const tf_ds_exl3_gargs& args, const tf_ds_exl3_glayer& L, float* red, uint32_t* st,
                                             int* last_sh) {
    constexpr int TW = 4 * K2, TPW = 8 / WK, RL = (TPW * TW + 31) / 32;
    const int M = args.M, NP = (M + 15) / 16;
    const _Float16* xh = reinterpret_cast<const _Float16*>(L.xh);
    float* Z = reinterpret_cast<float*>(L.Z);
    int* counters = reinterpret_cast<int*>(L.counters);
    const int K = L.K, N = L.N, SK = L.SK, NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane >> 4;
    const int per_split = (K >> 4) / SK, kt0 = split * per_split;
    const int t0 = wave * TPW;
    const uint32_t* tiles = reinterpret_cast<const uint32_t*>(L.T) + nb * L.stride_nb + t0 * TW;
    const int col0 = nb * 128;
    const LaneWin<K2> win(lane);
    f32x8 acc[8][TPW];
#pragma unroll
    for (int p = 0; p < 8; ++p)
#pragma unroll
        for (int j = 0; j < TPW; ++j) acc[p][j] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    // this lane's A row of each pass (rows past M read the last row; their outputs are dropped)
    const _Float16* xrow[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) xrow[p] = xh + (size_t)min(16 * p + (lane & 15), M - 1) * L.ldx;
    uint32_t pf[RL];
#define TF_DS_EXL3_WIDE_LOAD(kt_)                                                                                       \
    {                                                                                                                   \
        const uint32_t* q_ = tiles + (int64_t)(kt_) * L.stride_k;                                                      \
        _Pragma("unroll") for (int i = 0; i < RL; ++i) {                                                               \
            const int wi = i * 32 + lane;                                                                               \
            pf[i] = wi < TPW * TW ? __builtin_nontemporal_load(gl(q_) + wi) : 0u;                                          \
        }                                                                                                               \
    }
    TF_DS_EXL3_WIDE_LOAD(kt0)
    for (int i = 0; i < per_split; ++i) {
        const int kt = kt0 + i;
#pragma unroll
        for (int r = 0; r < RL; ++r) {
            const int wi = r * 32 + lane;
            if (wi < TPW * TW) st[wi] = pf[r];
        }
        if (i + 1 < per_split) TF_DS_EXL3_WIDE_LOAD(kt + 1)
        lds_fence();
        f16x16 B[TPW];
#pragma unroll
        for (int j = 0; j < TPW; ++j) {
            uint32_t own[4];
            decode_own<K2>(st + j * TW, win, own);
            B[j] = bfrag_from_own(own, lane);
        }
        __builtin_amdgcn_wave_barrier();
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            if (p < NP) {
                uint4 alo, ahi;
                load16h_k(xrow[p] + 16 * kt, h, alo, ahi);
                const f16x16 A = frag8(alo, ahi);
#pragma unroll
                for (int j = 0; j < TPW; ++j) acc[p][j] = wmma(A, B[j], acc[p][j]);
            }
        }
    }
#undef TF_DS_EXL3_WIDE_LOAD
    if (SK == 1) {
        // per pass: the C tiles of rows 16 p .. + 15 to red [16][128], then each row's 128 columns finished
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            if (p >= NP) break;
            const int R = min(16, M - 16 * p);
            __syncthreads();
#pragma unroll
            for (int j = 0; j < TPW; ++j)
#pragma unroll
                for (int r = 0; r < 8; ++r) red[(2 * r + h) * 128 + (t0 + j) * 16 + (lane & 15)] = acc[p][j][r];
            __syncthreads();
            for (int r = wave; r < R; r += WK) {
                float v[4];
                const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
                finish_store(L, v, lane, (size_t)(16 * p + r), col0);
            }
        }
        return;
    }
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        if (p >= NP) break;
#pragma unroll
        for (int j = 0; j < TPW; ++j)
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int row = 16 * p + 2 * r + h;
                if (row < M) Z[((size_t)split * M + row) * N + col0 + (t0 + j) * 16 + (lane & 15)] = acc[p][j][r];
            }
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *last_sh = atomicAdd(counters + nb, 1) == SK - 1;
    __syncthreads();
    if (!*last_sh) return;
    __threadfence();
    for (int r = wave; r < M; r += WK) {
        const size_t at = (size_t)r * N + col0 + 4 * lane;
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = ld_cg(Z + at + j);
        for (int q = 1; q < SK; ++q)
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += ld_cg(Z + (size_t)q * M * N + at + j);
        finish_store(L, v, lane, (size_t)r, col0);
    }
    if (threadIdx.x == 0) counters[nb] = 0;                     // every program of the block has arrived
}

#endif  // TF_DEVICE_WMMA_GFX11 (glinear_wide)

// glinear for a block of any layer of the group (linear_grouped.cu glinear_kernel): blocks first_i ..

// first_i + N_i / 128 * SK_i - 1 are layer i's; block b is column block b % (N_i / 128) of K split b / (N_i / 128).

// Wave w of WK takes k tiles split * per_warp * WK + w * per_warp, long; rows run in passes of 16.

// With SK > 1 the splits' partials (Z [SK, M, N]) sum in split order by the block completing the (pass, column block).
template <int K2, int WK>
__device__ __forceinline__ void glinear_body(const tf_ds_exl3_gargs& args) {
    constexpr int TW = 4 * K2;
    extern __shared__ __attribute__((aligned(16))) float red[];   // WK * min(M, 8) * 128 floats (the launcher's)
    __shared__ __attribute__((aligned(16))) uint32_t stage[WK][8 * TW];
    __shared__ int last_sh;
    const int M = args.M, RH = M < 8 ? M : 8;
    int li = 0;
#pragma unroll
    for (int q = 1; q < TF_DS_EXL3_GMAX; ++q)
        if (q < args.n && (int)blockIdx.x >= args.l[q].first) li = q;
    const tf_ds_exl3_glayer& L = args.l[li];
    const _Float16* xh = reinterpret_cast<const _Float16*>(L.xh);
    float* Z = reinterpret_cast<float*>(L.Z);
    int* counters = reinterpret_cast<int*>(L.counters);
    const int K = L.K, N = L.N, SK = L.SK;
    const int NB = N >> 7;
    const int b = (int)blockIdx.x - L.first;
    const int nb = b % NB, split = b / NB;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + wave * per_warp;
    const uint32_t* tiles = reinterpret_cast<const uint32_t*>(L.T) + nb * L.stride_nb;
    const int col0 = nb * 128;
#if (!defined(EXL3_WIDE) || EXL3_WIDE) && TF_DEVICE_WMMA_GFX11
    if (M >= EXL3_WIDE_MIN) {
        glinear_wide<K2, WK>(args, L, red, stage[wave], &last_sh);
        return;
    }
#endif

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = min(16, M - m0);
        f32x8 acc[8];
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[t] = (f32x8){0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        // every row in the 1-row order: a row's outputs do not depend on M; the rows' staging is the wave's red floats
        if (R == 1) {
            strip_tiles_1row<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)m0 * L.ldx, stage[wave], lane, acc);
        } else {
            const int ar = m0 + min(lane >> 1, R - 1);
            strip_tiles_rows<K2>(tiles, L.stride_k, kt0, per_warp, xh + (size_t)ar * L.ldx, R, stage[wave],
                                 reinterpret_cast<uint4*>(red + wave * 128), lane, acc);
        }

#pragma unroll
        for (int hp = 0; hp < 2; ++hp) {
            const int rlo = 8 * hp;
            if (rlo >= R) break;
            const int rn = min(R - rlo, 8);
            __syncthreads();                                    // red is reused by every half and pass
#pragma unroll
            for (int t = 0; t < 8; ++t)
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rr = 2 * i + (lane >> 4);
                    if (rr < RH) red[(wave * RH + rr) * 128 + t * 16 + (lane & 15)] = acc[t][4 * hp + i];
                }
            __syncthreads();
            if (SK == 1) {
                for (int r = wave; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish_store(L, v, lane, (size_t)(m0 + rlo + r), col0);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(Z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last_sh = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last_sh) {
                __threadfence();
                for (int r = wave; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float v[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] = ld_cg(Z + at + j);
                    for (int q = 1; q < SK; ++q)
#pragma unroll
                        for (int j = 0; j < 4; ++j) v[j] += ld_cg(Z + (size_t)q * M * N + at + j);
                    finish_store(L, v, lane, (size_t)m0 + r, col0);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                                        // red is reused in the next pass
    }
}

}  // namespace

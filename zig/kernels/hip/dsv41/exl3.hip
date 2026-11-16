// EXL3 decode linears for the dsv41 family: rot_many rotates every layer's input rows, glinear products the tiles.

// Split-K partials and wave sums are added in a fixed order, so every output is deterministic in any launch.

// Compiled with -ffp-contract=off: every fused multiply-add below is written as fmaf. Built for wave32.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "dsv41/abi.h"
#include "common/arch.hpp"
#include "exl3_dec.hpp"
#include "exl3_glinear.hpp"

// rot_in for every layer of a group (linear_grouped.cu rot_many_kernel): a wave a (layer, row, 128-wide block).
extern "C" __global__ void __launch_bounds__(128) tf_ds_exl3_rot_many(tf_ds_exl3_rargs args) {
    const int w = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    int li = 0;
#pragma unroll
    for (int q = 1; q < TF_DS_EXL3_GMAX; ++q)
        if (q < args.n && w >= args.l[q].first) li = q;
    const tf_ds_exl3_rlayer& L = args.l[li];
    const int nblk = L.K >> 7, local = w - L.first;
    if (local >= args.M * nblk) return;                       // past the last layer's rows (whole waves)
    const int row = local / nblk, k = (local % nblk) * 128 + 4 * lane;
    float v[4], s[4];
    load4(reinterpret_cast<const void*>(L.x), L.x_dtype, (size_t)row * L.ldx + k, v);
    load4(reinterpret_cast<const void*>(L.suh), F16, k, s);
    rot128_in(v, s, lane);
    store4(reinterpret_cast<void*>(L.xh), F16, (size_t)row * L.K + k, v);
}

#define TF_DS_EXL3_GLINEAR(K2_, WK_)                                                                                    \
    extern "C" __global__ void __launch_bounds__(WK_ * 32) tf_ds_exl3_glinear_k##K2_##_w##WK_(tf_ds_exl3_gargs args) { \
        glinear_body<K2_, WK_>(args);                                                                                   \
    }
TF_DS_EXL3_GLINEAR(4, 4)
TF_DS_EXL3_GLINEAR(4, 8)
TF_DS_EXL3_GLINEAR(6, 4)
TF_DS_EXL3_GLINEAR(6, 8)
TF_DS_EXL3_GLINEAR(8, 4)
TF_DS_EXL3_GLINEAR(8, 8)
TF_DS_EXL3_GLINEAR(10, 4)
TF_DS_EXL3_GLINEAR(10, 8)
TF_DS_EXL3_GLINEAR(12, 4)
TF_DS_EXL3_GLINEAR(12, 8)
TF_DS_EXL3_GLINEAR(16, 4)
TF_DS_EXL3_GLINEAR(16, 8)
#undef TF_DS_EXL3_GLINEAR

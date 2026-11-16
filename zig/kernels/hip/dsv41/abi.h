// The dsv41 group's HIP kernel ABI: the structs the decode EXL3 linear takes by value.

// One definition here, shared by every .hip file of the group and translated for Zig by the build.

// Fixed-width types only (uint64_t device addresses), natural alignment, explicit padding, size and offset assertions.

// Bump TF_DS_HIP_ABI_VERSION on any change to a struct, a kernel's parameter list or its documented semantics.
#ifndef TF_DS_HIP_ABI_H
#define TF_DS_HIP_ABI_H

#include <stddef.h>
#include <stdint.h>

#define TF_DS_HIP_ABI_VERSION 2

#ifdef __cplusplus
#define TF_DS_ASSERT(c, m) static_assert(c, m)
#else
#define TF_DS_ASSERT(c, m) _Static_assert(c, m)
#endif

// EXL3: the decode linear's by-value launcher arguments, field for field of the Zig GLayer / GArgs / RLayer / RArgs.

// The padding is explicit here and implicit (the same bytes) in Zig.
#define TF_DS_EXL3_GMAX 8

// one layer of glinear's group (linear_grouped.cu GLayer)
typedef struct {
    uint64_t xh;
    uint64_t T;
    uint64_t svh;
    uint64_t bias;
    uint64_t y;
    uint64_t Z;
    uint64_t counters;
    int64_t ldx;
    int64_t ldy;
    int64_t stride_k;
    int64_t stride_nb;
    int32_t y_dtype;
    int32_t K;
    int32_t N;
    int32_t SK;
    int32_t first;
    int32_t pad0;
    uint64_t rsuh;
    uint64_t rxh;
    int64_t ldr;
    int32_t roff;
    int32_t pad1;
    uint64_t rcos;
    uint64_t rsin;
    uint64_t rpos;
    int32_t rhd;
    int32_t rrd;
} tf_ds_exl3_glayer;
TF_DS_ASSERT(sizeof(tf_ds_exl3_glayer) == 176, "tf_ds_exl3_glayer size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, xh) == 0, "xh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, T) == 8, "T");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, svh) == 16, "svh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, bias) == 24, "bias");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, y) == 32, "y");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, Z) == 40, "Z");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, counters) == 48, "counters");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldx) == 56, "ldx");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldy) == 64, "ldy");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, stride_k) == 72, "stride_k");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, stride_nb) == 80, "stride_nb");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, y_dtype) == 88, "y_dtype");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, K) == 92, "K");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, N) == 96, "N");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, SK) == 100, "SK");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, first) == 104, "first");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, pad0) == 108, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rsuh) == 112, "rsuh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rxh) == 120, "rxh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldr) == 128, "ldr");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, roff) == 136, "roff");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, pad1) == 140, "pad1");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rcos) == 144, "rcos");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rsin) == 152, "rsin");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rpos) == 160, "rpos");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rhd) == 168, "rhd");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rrd) == 172, "rrd");

// exl3.glinear's one parameter (GArgs)
typedef struct {
    tf_ds_exl3_glayer l[TF_DS_EXL3_GMAX];
    int32_t n;
    int32_t M;
    int32_t discard;
    int32_t pad0;
} tf_ds_exl3_gargs;
TF_DS_ASSERT(sizeof(tf_ds_exl3_gargs) == 1424, "tf_ds_exl3_gargs size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, n) == 1408, "n");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, M) == 1412, "M");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, discard) == 1416, "discard");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, pad0) == 1420, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, l) == 0, "l");

// one layer of rot_many (RLayer)
typedef struct {
    uint64_t x;
    uint64_t suh;
    uint64_t xh;
    int64_t ldx;
    int32_t x_dtype;
    int32_t K;
    int32_t first;
    int32_t pad0;
} tf_ds_exl3_rlayer;
TF_DS_ASSERT(sizeof(tf_ds_exl3_rlayer) == 48, "tf_ds_exl3_rlayer size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, x) == 0, "x");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, suh) == 8, "suh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, xh) == 16, "xh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, ldx) == 24, "ldx");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, x_dtype) == 32, "x_dtype");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, K) == 36, "K");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, first) == 40, "first");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, pad0) == 44, "pad0");

// exl3.rot_many's one parameter (RArgs)
typedef struct {
    tf_ds_exl3_rlayer l[TF_DS_EXL3_GMAX];
    int32_t n;
    int32_t M;
} tf_ds_exl3_rargs;
TF_DS_ASSERT(sizeof(tf_ds_exl3_rargs) == 392, "tf_ds_exl3_rargs size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rargs, n) == 384, "n");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rargs, M) == 388, "M");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rargs, l) == 0, "l");

#endif

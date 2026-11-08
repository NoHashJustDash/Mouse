// SPDX-License-Identifier: GPL-3.0-or-later
// TensorRT plugin "MouseMoeExperts" v1: the routed experts of a Mouse-MoE token-routed
// layer as one TensorRT layer instead of the ~60 nodes of the portable ONNX dispatch.
//
// Inputs (TensorRT tensors, all linear):
//   0 x       [T, D] half   token activations (T = boards x 64 squares)
//   1 router  [E, D] float  router weight (nn.Linear(D, E).weight)
//   2 bias    [E]    float  balance bias added for selection only
//   3 w1      [E, D, H] half
//   4 b1      [E, H] half
//   5 w2      [E, H, D] half
//   6 b2      [E, D] float
// Output 0   [T, D] half: sum over the top_k chosen experts of gate * expert(x).
//
// Semantics match the network's exported routed layer: softmax router scores in
// float, selection by scores + bias with ties to the lowest expert id, gates renormalised
// over the chosen experts (raw score when top_k = 1), each expert mish(x W1 + b1) W2 + b2.
// The shared expert stays in the TensorRT graph.
//
// Launches: clear counts; route (half a warp per token, a lane per expert); prepare (one
// thread: per-expert row offsets and the grouped-GEMM problem descriptions); gather (each
// expert's rows made contiguous); two CUTLASS grouped GEMMs whose sizes are read on the
// device (no host synchronisation, CUDA-graph safe), the first with bias + Mish fused;
// combine (per token, b2 and gates in float, in choice order, so results are
// deterministic). CUTLASS 3.9.2 (BSD-3-Clause) supplies the GEMMs.
//
// The dynamic-shape overrides below hide IPluginV2's legacy overloads, which TensorRT
// itself makes private and final in IPluginV2DynamicExt (nvcc diagnostic 997).
#pragma nv_diag_suppress 997
#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

namespace {
using nvinfer1::DataType;
using nvinfer1::DimsExprs;
using nvinfer1::DynamicPluginTensorDesc;
using nvinfer1::IExprBuilder;
using nvinfer1::IPluginV2DynamicExt;
using nvinfer1::PluginField;
using nvinfer1::PluginFieldCollection;
using nvinfer1::PluginFieldType;
using nvinfer1::PluginTensorDesc;
using nvinfer1::TensorFormat;

constexpr char const* kName = "MouseMoeExperts";
constexpr char const* kVersion = "1";
constexpr int kMaxExperts = 16;  // one lane of a half warp per expert
constexpr int kMaxTopK = 4;
constexpr int kRouteWarps = 8;
constexpr int kRouteBlocks = 96;  // grid-stride: each block stages the router weights once
constexpr std::size_t kAlign = 256;

std::size_t aligned(std::size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

__device__ __forceinline__ float mish(float v) {
    const float softplus = fmaxf(v, 0.0F) + log1pf(expf(-fabsf(v)));
    return v * tanhf(softplus);
}

// Mish for CUTLASS's generic epilogue, computed in float.
template <typename T>
struct Mish {
    static const bool kIsHeavy = true;
    CUTLASS_HOST_DEVICE T operator()(T const& value) const {
#if defined(__CUDA_ARCH__)
        return T(mish(static_cast<float>(value)));
#else
        return value;
#endif
    }
};
template <typename T, int N>
struct Mish<cutlass::Array<T, N>> {
    static const bool kIsHeavy = true;
    CUTLASS_HOST_DEVICE cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& value) const {
        cutlass::Array<T, N> result;
        Mish<T> op;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) result[i] = op(value[i]);
        return result;
    }
};

using Half = cutlass::half_t;
using Row = cutlass::layout::RowMajor;
// out = mish(acc + C) with C the bias row broadcast through a zero stride.
using FirstEpilogue = cutlass::epilogue::thread::LinearCombinationGeneric<Mish, Half, 8, float, float>;
using SecondEpilogue = cutlass::epilogue::thread::LinearCombination<Half, 8, float, float>;
template <typename Epilogue>
using GroupedGemm = cutlass::gemm::device::GemmGrouped<
    typename cutlass::gemm::kernel::DefaultGemmGrouped<
        Half, Row, cutlass::ComplexTransform::kNone, 8, Half, Row,
        cutlass::ComplexTransform::kNone, 8, Half, Row, float, cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 32>,
        cutlass::gemm::GemmShape<32, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 4,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel>;
using FirstGemm = GroupedGemm<FirstEpilogue>;
using SecondGemm = GroupedGemm<SecondEpilogue>;

// Half a warp per token, lane e of the half computes expert e's float logit from float4
// shared-memory loads. Softmax, then top-k by biased score with ties to the lowest expert
// id, gates, and each assignment's position in its expert's list.
__global__ void route_kernel(__half const* __restrict__ x, float const* __restrict__ router,
                             float const* __restrict__ bias, int tokens, int width, int experts,
                             int top_k, int* __restrict__ counts, int* __restrict__ expert_of,
                             int* __restrict__ position_of, float* __restrict__ gates) {
    extern __shared__ float route_shared[];
    const int ld = width + 4;  // padded rows: lanes read different rows without bank conflicts
    float* weights = route_shared;                // [experts, ld]
    float* staged = route_shared + experts * ld;  // [2 * kRouteWarps, width]
    for (int i = threadIdx.x; i < experts * width / 4; i += blockDim.x) {
        const int e = i / (width / 4), d = (i - e * (width / 4)) * 4;
        *reinterpret_cast<float4*>(weights + e * ld + d) =
            *reinterpret_cast<float4 const*>(router + static_cast<std::size_t>(e) * width + d);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int half = lane >> 4, expert = lane & 15;
    float* row_stage = staged + (warp * 2 + half) * width;
    const bool active = expert < experts;
    const int halves = gridDim.x * kRouteWarps * 2;
    for (int base = (blockIdx.x * kRouteWarps + warp) * 2; base < tokens; base += halves) {
        const int token = base + half;
        const bool valid = token < tokens;
        if (valid)
            for (int d = expert; d < width; d += 16)
                row_stage[d] = __half2float(x[static_cast<std::size_t>(token) * width + d]);
        __syncwarp();
        float logit = -INFINITY;
        if (valid && active) {
            float4 partial = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            float4 const* xv = reinterpret_cast<float4 const*>(row_stage);
            float4 const* wv = reinterpret_cast<float4 const*>(weights + expert * ld);
#pragma unroll 8
            for (int d = 0; d < width / 4; ++d) {
                const float4 a = xv[d], b = wv[d];
                partial.x += a.x * b.x;
                partial.y += a.y * b.y;
                partial.z += a.z * b.z;
                partial.w += a.w * b.w;
            }
            logit = (partial.x + partial.y) + (partial.z + partial.w);
        }
        __syncwarp();
        // Reductions stay within each 16-lane half (offsets below 16).
        float maximum = logit;
        for (int offset = 8; offset > 0; offset >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, offset));
        const float exponent = (valid && active) ? expf(logit - maximum) : 0.0F;
        float sum = exponent;
        for (int offset = 8; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffU, sum, offset);
        const float score = sum > 0.0F ? exponent / sum : 0.0F;
        float biased = (valid && active) ? score + bias[expert] : -INFINITY;
        int chosen[kMaxTopK];
        float chosen_score[kMaxTopK];
        float chosen_sum = 0.0F;
        for (int k = 0; k < top_k; ++k) {
            float value = biased;
            int index = active ? expert : experts;
            for (int offset = 8; offset > 0; offset >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffU, value, offset);
                const int other_index = __shfl_xor_sync(0xffffffffU, index, offset);
                if (other > value || (other == value && other_index < index)) {
                    value = other;
                    index = other_index;
                }
            }
            chosen[k] = index;
            chosen_score[k] = __shfl_sync(0xffffffffU, score, (lane & 16) + index);
            chosen_sum += chosen_score[k];
            if (expert == index) biased = -INFINITY;
        }
        if (valid && expert == 0)
            for (int k = 0; k < top_k; ++k) {
                const int a = token * top_k + k;
                const int e = chosen[k];
                gates[a] = top_k > 1 ? chosen_score[k] / chosen_sum : chosen_score[k];
                expert_of[a] = e;
                position_of[a] = atomicAdd(counts + e, 1);
            }
    }
}

// Per-expert problem descriptions for both grouped GEMMs, read on the device.
struct Problems {
    cutlass::gemm::GemmCoord* sizes;  // [2][experts]
    Half** a;                         // [2][experts], likewise b, c, d
    Half** b;
    Half** c;
    Half** d;
    int64_t* lda;  // [2][experts], likewise ldb, ldc, ldd
    int64_t* ldb;
    int64_t* ldc;
    int64_t* ldd;
};

__global__ void prepare_kernel(int const* __restrict__ counts, int experts, int width, int hidden,
                               Half const* w1, Half const* b1, Half const* w2, Half* rows,
                               Half* hidden_rows, Half* y, int* __restrict__ starts, Problems p) {
    if (threadIdx.x != 0) return;
    int start = 0;
    for (int e = 0; e < experts; ++e) {
        const int m = counts[e];
        starts[e] = start;
        const int first = e, second = experts + e;
        p.sizes[first] = cutlass::gemm::GemmCoord(m, hidden, width);
        p.a[first] = rows + static_cast<std::size_t>(start) * width;
        p.b[first] = const_cast<Half*>(w1) + static_cast<std::size_t>(e) * width * hidden;
        p.c[first] = const_cast<Half*>(b1) + static_cast<std::size_t>(e) * hidden;
        p.d[first] = hidden_rows + static_cast<std::size_t>(start) * hidden;
        p.lda[first] = width;
        p.ldb[first] = hidden;
        p.ldc[first] = 0;  // the bias row, broadcast to every output row
        p.ldd[first] = hidden;
        p.sizes[second] = cutlass::gemm::GemmCoord(m, width, hidden);
        p.a[second] = hidden_rows + static_cast<std::size_t>(start) * hidden;
        p.b[second] = const_cast<Half*>(w2) + static_cast<std::size_t>(e) * hidden * width;
        p.c[second] = y + static_cast<std::size_t>(start) * width;  // unused: beta = 0
        p.d[second] = y + static_cast<std::size_t>(start) * width;
        p.lda[second] = hidden;
        p.ldb[second] = width;
        p.ldc[second] = width;
        p.ldd[second] = width;
        start += m;
    }
}

// Copies each assignment's token row to its expert's contiguous block (8 halves a thread).
__global__ void gather_kernel(__half const* __restrict__ x, int const* __restrict__ expert_of,
                              int const* __restrict__ position_of, int const* __restrict__ starts,
                              int assignments, int width, int top_k, __half* __restrict__ rows) {
    const std::size_t i = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 8;
    if (i >= static_cast<std::size_t>(assignments) * width) return;
    const int a = static_cast<int>(i / width), d = static_cast<int>(i - static_cast<std::size_t>(a) * width);
    const std::size_t row = static_cast<std::size_t>(starts[expert_of[a]] + position_of[a]);
    *reinterpret_cast<int4*>(rows + row * width + d) =
        *reinterpret_cast<int4 const*>(x + static_cast<std::size_t>(a / top_k) * width + d);
}

// out[t] = sum_k gate[t, k] * (y[row of (t, k)] + b2[expert]), summed in choice order.
__global__ void combine_kernel(__half const* __restrict__ y, float const* __restrict__ b2,
                               int const* __restrict__ starts, int const* __restrict__ expert_of,
                               int const* __restrict__ position_of, float const* __restrict__ gates,
                               int tokens, int width, int top_k, __half* __restrict__ out) {
    const std::size_t i = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 8;
    if (i >= static_cast<std::size_t>(tokens) * width) return;
    const int token = static_cast<int>(i / width);
    const int d = static_cast<int>(i - static_cast<std::size_t>(token) * width);
    float total[8] = {};
    for (int k = 0; k < top_k; ++k) {
        const int a = token * top_k + k;
        const int expert = expert_of[a];
        const float gate = gates[a];
        const std::size_t row = static_cast<std::size_t>(starts[expert] + position_of[a]);
        const int4 packed = *reinterpret_cast<int4 const*>(y + row * width + d);
        __half2 const* yv = reinterpret_cast<__half2 const*>(&packed);
        float const* bias = b2 + static_cast<std::size_t>(expert) * width + d;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 v = __half22float2(yv[j]);
            total[2 * j] += gate * (v.x + bias[2 * j]);
            total[2 * j + 1] += gate * (v.y + bias[2 * j + 1]);
        }
    }
    __half2 result[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) result[j] = __floats2half2_rn(total[2 * j], total[2 * j + 1]);
    *reinterpret_cast<int4*>(out + i) = *reinterpret_cast<int4 const*>(result);
}

class MoePlugin final : public IPluginV2DynamicExt {
   public:
    explicit MoePlugin(int top_k) : top_k_(top_k) {}

    char const* getPluginType() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override {
        first_blocks_ = FirstGemm::sufficient();
        second_blocks_ = SecondGemm::sufficient();
        return first_blocks_ > 0 && second_blocks_ > 0 ? 0 : 1;
    }
    void terminate() noexcept override {}
    std::size_t getSerializationSize() const noexcept override { return sizeof(int); }
    void serialize(void* buffer) const noexcept override { std::memcpy(buffer, &top_k_, sizeof(int)); }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* name) noexcept override { namespace_ = name ? name : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
    IPluginV2DynamicExt* clone() const noexcept override {
        auto* copy = new (std::nothrow) MoePlugin(top_k_);
        if (!copy) return nullptr;
        copy->setPluginNamespace(namespace_.c_str());
        copy->first_blocks_ = first_blocks_;
        copy->second_blocks_ = second_blocks_;
        return copy;
    }
    DataType getOutputDataType(int32_t, DataType const* types, int32_t) const noexcept override {
        return types[0];
    }
    DimsExprs getOutputDimensions(int32_t, DimsExprs const* inputs, int32_t,
                                  IExprBuilder&) noexcept override {
        return inputs[0];
    }
    bool supportsFormatCombination(int32_t pos, PluginTensorDesc const* io, int32_t,
                                   int32_t) noexcept override {
        if (io[pos].format != TensorFormat::kLINEAR) return false;
        switch (pos) {
            case 0:
            case 3:
            case 4:
            case 5:
            case 7:
                return io[pos].type == DataType::kHALF;
            default:
                return io[pos].type == DataType::kFLOAT;
        }
    }
    void configurePlugin(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*,
                         int32_t) noexcept override {}
    std::size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t, PluginTensorDesc const*,
                                 int32_t) const noexcept override {
        const auto tokens = static_cast<std::size_t>(inputs[0].dims.d[0]);
        const auto width = static_cast<std::size_t>(inputs[0].dims.d[1]);
        const auto experts = static_cast<std::size_t>(inputs[1].dims.d[0]);
        const auto hidden = static_cast<std::size_t>(inputs[3].dims.d[2]);
        return layout(experts, tokens * top_k_, width, hidden).total;
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const*, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        const int tokens = static_cast<int>(in[0].dims.d[0]);
        const int width = static_cast<int>(in[0].dims.d[1]);
        const int experts = static_cast<int>(in[1].dims.d[0]);
        const int hidden = static_cast<int>(in[3].dims.d[2]);
        if (tokens <= 0) return 0;
        if (experts > kMaxExperts || top_k_ < 1 || top_k_ > kMaxTopK || top_k_ > experts ||
            width % 8 != 0 || hidden % 8 != 0 || in[1].dims.d[1] != width ||
            in[3].dims.d[1] != width || in[5].dims.d[1] != hidden || in[5].dims.d[2] != width)
            return 1;
        const int assignments = tokens * top_k_;
        const auto l = layout(experts, assignments, width, hidden);
        auto* base = static_cast<unsigned char*>(workspace);
        auto* counts = reinterpret_cast<int*>(base + l.counts);
        auto* starts = reinterpret_cast<int*>(base + l.starts);
        auto* expert_of = reinterpret_cast<int*>(base + l.expert_of);
        auto* position_of = reinterpret_cast<int*>(base + l.position_of);
        auto* gates = reinterpret_cast<float*>(base + l.gates);
        auto* rows = reinterpret_cast<Half*>(base + l.rows);
        auto* hidden_rows = reinterpret_cast<Half*>(base + l.hidden_rows);
        auto* y = reinterpret_cast<Half*>(base + l.y);
        Problems problems{reinterpret_cast<cutlass::gemm::GemmCoord*>(base + l.sizes),
                          reinterpret_cast<Half**>(base + l.a),
                          reinterpret_cast<Half**>(base + l.b),
                          reinterpret_cast<Half**>(base + l.c),
                          reinterpret_cast<Half**>(base + l.d),
                          reinterpret_cast<int64_t*>(base + l.lda),
                          reinterpret_cast<int64_t*>(base + l.ldb),
                          reinterpret_cast<int64_t*>(base + l.ldc),
                          reinterpret_cast<int64_t*>(base + l.ldd)};
        const auto* x = static_cast<__half const*>(inputs[0]);
        const auto* router = static_cast<float const*>(inputs[1]);
        const auto* bias = static_cast<float const*>(inputs[2]);
        const auto* w1 = static_cast<Half const*>(inputs[3]);
        const auto* b1 = static_cast<Half const*>(inputs[4]);
        const auto* w2 = static_cast<Half const*>(inputs[5]);
        const auto* b2 = static_cast<float const*>(inputs[6]);

        if (cudaMemsetAsync(counts, 0, experts * sizeof(int), stream) != cudaSuccess) return 1;
        const std::size_t route_shared =
            (static_cast<std::size_t>(experts) * (width + 4) + 2 * kRouteWarps * width) * sizeof(float);
        if (route_shared > 48 * 1024 &&
            cudaFuncSetAttribute(route_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(route_shared)) != cudaSuccess)
            return 1;
        const dim3 route_grid(std::min((tokens + 2 * kRouteWarps - 1) / (2 * kRouteWarps), kRouteBlocks));
        route_kernel<<<route_grid, kRouteWarps * 32, route_shared, stream>>>(
            x, router, bias, tokens, width, experts, top_k_, counts, expert_of, position_of, gates);
        prepare_kernel<<<1, 32, 0, stream>>>(counts, experts, width, hidden, w1, b1, w2, rows,
                                              hidden_rows, y, starts, problems);
        const std::size_t gather_threads = static_cast<std::size_t>(assignments) * width / 8;
        gather_kernel<<<static_cast<unsigned>((gather_threads + 255) / 256), 256, 0, stream>>>(
            x, expert_of, position_of, starts, assignments, width, top_k_,
            reinterpret_cast<__half*>(rows));

        typename FirstGemm::Arguments first_args(
            problems.sizes, experts, first_blocks_, {1.0F, 1.0F}, problems.a, problems.b,
            problems.c, problems.d, problems.lda, problems.ldb, problems.ldc, problems.ldd);
        FirstGemm first;
        if (first.initialize(first_args, nullptr, stream) != cutlass::Status::kSuccess ||
            first.run(stream) != cutlass::Status::kSuccess)
            return 1;
        typename SecondGemm::Arguments second_args(
            problems.sizes + experts, experts, second_blocks_, {1.0F, 0.0F},
            problems.a + experts, problems.b + experts, problems.c + experts,
            problems.d + experts, problems.lda + experts, problems.ldb + experts,
            problems.ldc + experts, problems.ldd + experts);
        SecondGemm second;
        if (second.initialize(second_args, nullptr, stream) != cutlass::Status::kSuccess ||
            second.run(stream) != cutlass::Status::kSuccess)
            return 1;

        const std::size_t combine_threads = static_cast<std::size_t>(tokens) * width / 8;
        combine_kernel<<<static_cast<unsigned>((combine_threads + 255) / 256), 256, 0, stream>>>(
            reinterpret_cast<__half const*>(y), b2, starts, expert_of, position_of, gates, tokens,
            width, top_k_, static_cast<__half*>(outputs[0]));
        return cudaPeekAtLastError() == cudaSuccess ? 0 : 1;
    }

   private:
    struct Layout {
        std::size_t counts, starts, expert_of, position_of, gates, sizes, a, b, c, d, lda, ldb,
            ldc, ldd, rows, hidden_rows, y, total;
    };
    static Layout layout(std::size_t experts, std::size_t assignments, std::size_t width,
                         std::size_t hidden) {
        Layout l{};
        std::size_t offset = 0;
        auto take = [&offset](std::size_t bytes) {
            const auto at = offset;
            offset += aligned(bytes);
            return at;
        };
        l.counts = take(experts * sizeof(int));
        l.starts = take(experts * sizeof(int));
        l.expert_of = take(assignments * sizeof(int));
        l.position_of = take(assignments * sizeof(int));
        l.gates = take(assignments * sizeof(float));
        l.sizes = take(2 * experts * sizeof(cutlass::gemm::GemmCoord));
        l.a = take(2 * experts * sizeof(Half*));
        l.b = take(2 * experts * sizeof(Half*));
        l.c = take(2 * experts * sizeof(Half*));
        l.d = take(2 * experts * sizeof(Half*));
        l.lda = take(2 * experts * sizeof(int64_t));
        l.ldb = take(2 * experts * sizeof(int64_t));
        l.ldc = take(2 * experts * sizeof(int64_t));
        l.ldd = take(2 * experts * sizeof(int64_t));
        l.rows = take(assignments * width * sizeof(Half));
        l.hidden_rows = take(assignments * hidden * sizeof(Half));
        l.y = take(assignments * width * sizeof(Half));
        l.total = offset;
        return l;
    }
    int top_k_ = 2;
    int first_blocks_ = 0;
    int second_blocks_ = 0;
    std::string namespace_;
};

class MoePluginCreator final : public nvinfer1::IPluginCreator {
   public:
    MoePluginCreator() {
        fields_.emplace_back("top_k", nullptr, PluginFieldType::kINT32, 1);
        collection_.nbFields = static_cast<int32_t>(fields_.size());
        collection_.fields = fields_.data();
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &collection_; }
    nvinfer1::IPluginV2* createPlugin(char const*, PluginFieldCollection const* fc) noexcept override {
        int top_k = 2;
        for (int i = 0; fc && i < fc->nbFields; ++i)
            if (std::strcmp(fc->fields[i].name, "top_k") == 0 &&
                fc->fields[i].type == PluginFieldType::kINT32 && fc->fields[i].length == 1)
                top_k = *static_cast<int const*>(fc->fields[i].data);
        return make(top_k);
    }
    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           std::size_t length) noexcept override {
        if (length != sizeof(int)) return nullptr;
        int top_k = 0;
        std::memcpy(&top_k, data, sizeof(int));
        return make(top_k);
    }
    void setPluginNamespace(char const* name) noexcept override { namespace_ = name ? name : ""; }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

   private:
    nvinfer1::IPluginV2* make(int top_k) const noexcept {
        if (top_k < 1 || top_k > kMaxTopK) return nullptr;
        auto* plugin = new (std::nothrow) MoePlugin(top_k);
        if (plugin) plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
    std::string namespace_;
};
}  // namespace

// Registers the creator when the library loads (ONNX Runtime trt_extra_plugin_lib_paths
// and ctypes both rely on this); getPluginCreators serves IPluginRegistry::loadLibrary.
REGISTER_TENSORRT_PLUGIN(MoePluginCreator);

extern "C" __declspec(dllexport) nvinfer1::IPluginCreator* const* getPluginCreators(
    int32_t& count) {
    static MoePluginCreator creator;
    static nvinfer1::IPluginCreator* const creators[] = {&creator};
    count = 1;
    return creators;
}
extern "C" __declspec(dllexport) void setLoggerFinder(nvinfer1::ILoggerFinder*) {}

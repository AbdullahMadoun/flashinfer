# SM120 sparse MLA decode scheduling

`SparseMLASm120Wrapper.run` and `run_mega` accept
`impl="auto"`, `"low-latency"`, or `"high-throughput"`. The default is `auto`.
Explicit choices require inline FP8 KV with UE8M0 scales per 32 features,
FP8 QK arithmetic, flat cache rows, contiguous sparse indices, D=512 and
H in {8, 16, 32, 64}. Other formats retain their existing automatic dispatch.
Saved calibration profiles remain applicable to `auto`.
An explicit `prefill_impl` override cannot be combined with an explicit `impl`.

The high-throughput choice directly launches the CuTe C++ FP8 prefill kernel.
It does not copy or fork that implementation. One CTA handles one query and
up to 32 heads, with independent K/V producers, QK consumers and softmax/PV
consumers. The low-latency choice uses one KV producer warp and compute warps
that each handle eight heads. Each compute warp performs QK, online softmax
and PV in order. Independent CTAs process selected-KV chunks when the head-task
grid is too small to occupy the GPU. This is local split-KV, not multi-device
context parallelism.

For split execution, a CTA cluster owns one query/head task. CTAs publish
normalized BF16 partial contexts and base-2 LSE, then synchronize. All CTAs
share the final softmax merge and output work within the same kernel launch.
The scratch tensors are per invocation; no last-CTA counters or initialization
kernel are required. Empty chunks contribute zero context and negative-infinite
LSE. Negative indices and length tails are masked before cache reads.

`run_mega` fuses adjacent-pair RoPE on the final 64 Q features, attention,
normalization with the optional sink, inverse output RoPE, and FP8 output
conversion. Q RoPE and the normalized context retain the BF16 rounding
boundaries of the separate operations. Learned projections, Q normalization,
KV packing, and sparse-index construction remain outside this launch.
`run` uses the same attention implementation with BF16 input/output boundaries.

Implementation locations:

- `kernels/cute_decode/decode_fp8.cuh`: low-latency compute and cluster finalization.
- `kernels/cute_prefill/prefill_fp8.cuh`: shared high-throughput backbone.
- `csrc/sparse_mla_sm120/cute_decode_dispatch.cu`: native planning and launches.
- `flashinfer/mla/_sparse_mla_sm120/_prepared.py`: validation and cached preparation.

The kernel paths above are relative to
`include/flashinfer/attention/sparse_mla_sm120/`. Prepare the wrapper with a
warm-up call before CUDA graph capture. Shape/device/implementation choices
are cached; live lengths, sparse indices and positions are read on every replay.

## Automatic selection

Let `P = T * ceil(H / 32)` be the prefill backbone's unsplit CTA count.
Auto admits high throughput when `P / num_SMs >= 0.5`, then compares whole
SM waves. Prefill has one resident CTA per SM; low latency has two, except
for its four-compute-warp specialization, which has one. Using a measured
unsplit decode/prefill wave-cost ratio of approximately 4/3 avoids selecting
prefill at its second-wave boundary while decode still fits in one wave.
This ratio is an SM120 heuristic, not a hardware-independent performance
promise. For H=64, three compute warps create 24/24/16-head tasks;
smaller head counts use one, two or four compute warps. A partial head task
still participates fully in the pipelines and barriers, but bounds checks
prevent out-of-range tensor accesses.

Low latency picks the largest power-of-two split count up to eight whose
CTA count fits in the resident CTA capacity, capped by the number of selected 32-entry
tiles. This is a portable cluster bound. The common eight-split, three-warp
case specializes split count at compile time and preloads merge operands.
These rules use device capacity rather than hard-coded token thresholds;
explicit `impl` choices remain available for application-level measurements.

## Verification and measurement

Run `pytest tests/attention/test_sparse_mla_sm120_cute_decode.py`. Tests compare
against independent high-precision attention, check Q quantization's LSE
separately, compare fused and separate operations, and replay graphs with
mutated sparse inputs including empty rows. Very short ragged rows amplify
FP8 Q quantization error; their LSE is also checked against an independent
FP8-quantized-query reference, without weakening the dense-reference gate.

From the repository root:

```sh
python -m benchmarks.bench_sparse_mla_sm120_decode --output decode.json
```

The canonical sweep is B=1..20 requests, seven verification queries per
request (T=7..140), H=64, D=512, 128 sliding-window candidates plus 512 sparse
compressed candidates. Requests have disjoint cache pools and a 32,768-token
history with compression ratio two. Packing, index selection, allocation and
reference evaluation are outside timing. Results include raw samples,
medians and selected plans. Ten calls per CUDA graph and rotated measurement
order reduce launch noise and order bias. Compare whole fused calls with the
sum of the separate prologue, attention and epilogue calls; NCU replay time is
not a benchmark measurement.

## Measured acceptance run

On RTX PRO 6000 Blackwell Server Edition (188 SMs), CUDA 13.4, graph medians
in microseconds were:

| Queries | Previous mega | Auto |
|---:|---:|---:|
| 7 | 14.66 | 14.45 |
| 14 | 34.11 | 18.76 |
| 49 | 71.80 | 39.24 |
| 98 | 123.41 | 53.76 |
| 119 | 142.86 | 54.39 |
| 140 | 129.35 | 80.41 |

All 20 canonical sizes passed against both the archived, tuned pre-rewrite
mega implementation and its separate-stage pipeline. The low-token margin
is small; the larger gains come from using the prefill backbone and avoiding
its second-wave boundary. These are single-GPU measurements, not universal
latency guarantees. The committed benchmark reproduces the new choices and
their separate-stage pipelines; the old ABI requires its archived adapter.

NCU's complete 41-pass report for T=7 showed no local-memory spills. Excessive
shared-memory wavefronts fell from 258,048 in the initial three-warp candidate
to 56,448 after transposing KV scale staging. Vectorized Q/partial accesses,
native MXFP8 QK, split-load prefetching and skipping the first tile's redundant
accumulator rescale provided the remaining improvements. NCU replay duration
was not used for the acceptance numbers above.

FP8 probability quantization and split boundaries differ from legacy footer
cache decode, so outputs are not promised to be bitwise equal to that route.
Independent-reference accuracy gates and fused/separate equality are tested.

The acceptance run passed 584 tests across decode, prefill integration, cache
layouts, execution/precision policies, calibration, bindings and LSE handling.
Targeted Compute Sanitizer memcheck, racecheck and synccheck runs reported
zero errors or hazards for split/unsplit, partial-head, ragged and empty cases.

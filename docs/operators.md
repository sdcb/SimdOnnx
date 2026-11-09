# Operator coverage

Generated from [`tests/Sdcb.SimdOnnx.Tests/Conformance/node-expected.tsv`](../tests/Sdcb.SimdOnnx.Tests/Conformance/node-expected.tsv), the ONNX backend node tests of onnx v1.22.0 run on the CPU (`NodeTests`). Do not edit by hand: `SIMDONNX_WRITE_MANIFEST=1 dotnet test tests/Sdcb.SimdOnnx.Tests -f net10.0 --filter FullyQualifiedName~NodeTests` rewrites both files.

How to read it:

- A test **passes** when every float output is within ONNX's backend-test tolerance (|a − b| ≤ 1e-7 + 1e-3·|b|). Integer inputs (shapes, indices) are baked in as constants, because the engine evaluates integer tensors at compile time; `pass-const` means the test also needed its other float inputs (e.g. weights, Clip bounds) as constants.
- **Skipped** tests have non-float outputs or inputs the engine does not take (bool, string, fp16, int8, sequences, …). They are not counted in the per-op totals below.
- NCHW and NHWC are the two CPU plan layouts (`CompileOptions.Nhwc`).

| | tests | pass (NCHW) | pass (NHWC) |
|---|---:|---:|---:|
| all | 1098 | 153 | 152 |
| skipped (dtype) | 667 | | |

## Single-op tests

| op | tests | pass NCHW | pass NHWC | most common reason for the rest |
|---|---:|---:|---:|---|
| Abs | 1 | 0 | 0 | unsupported Op 'Abs' not supported (node ''). |
| Acos | 2 | 0 | 0 | unsupported Op 'Acos' not supported (node ''). |
| Acosh | 2 | 0 | 0 | unsupported Op 'Acosh' not supported (node ''). |
| Adagrad | 2 | 0 | 0 | unsupported Op 'Adagrad' not supported (node ''). |
| Adam | 2 | 0 | 0 | unsupported Op 'Adam' not supported (node ''). |
| Add | 2 | 2 | 2 |  |
| AffineGrid | 4 | 0 | 0 | unsupported Op 'AffineGrid' not supported (node ''). |
| ArrayFeatureExtractor | 1 | 0 | 0 | unsupported Op 'ArrayFeatureExtractor' not supported (node ''). |
| Asin | 2 | 0 | 0 | unsupported Op 'Asin' not supported (node ''). |
| Asinh | 2 | 0 | 0 | unsupported Op 'Asinh' not supported (node ''). |
| Atan | 2 | 0 | 0 | unsupported Op 'Atan' not supported (node ''). |
| Atanh | 2 | 0 | 0 | unsupported Op 'Atanh' not supported (node ''). |
| Attention | 60 | 0 | 0 | unsupported Op 'Attention' not supported (node ''). |
| AveragePool | 20 | 10 | 10 | unsupported AveragePool '': rank-4 only. |
| BatchNormalization | 4 | 2 | 2 | unsupported BatchNormalization '': non-constant input 1. |
| Bernoulli | 1 | 0 | 0 | unsupported Op 'Bernoulli' not supported (node ''). |
| Binarizer | 1 | 0 | 0 | unsupported Op 'Binarizer' not supported (node ''). |
| CausalConvWithState | 11 | 0 | 0 | unsupported Op 'CausalConvWithState' not supported (node ''). |
| Ceil | 2 | 0 | 0 | unsupported Op 'Ceil' not supported (node ''). |
| Celu | 1 | 0 | 0 | unsupported Op 'Celu' not supported (node ''). |
| CenterCropPad | 6 | 0 | 0 | unsupported Op 'CenterCropPad' not supported (node ''). |
| Clip | 9 | 9 | 9 |  |
| Col2Im | 5 | 0 | 0 | unsupported Op 'Col2Im' not supported (node ''). |
| Concat | 12 | 12 | 12 |  |
| Conv | 6 | 6 | 6 |  |
| ConvTranspose | 11 | 0 | 0 | unsupported ConvTranspose '': only group=1, stride==kernel, no pad. |
| Cos | 2 | 0 | 0 | unsupported Op 'Cos' not supported (node ''). |
| Cosh | 2 | 0 | 0 | unsupported Op 'Cosh' not supported (node ''). |
| DFT | 10 | 0 | 0 | unsupported Op 'DFT' not supported (node ''). |
| DeformConv | 4 | 0 | 0 | unsupported Op 'DeformConv' not supported (node ''). |
| DepthToSpace | 2 | 2 | 2 |  |
| Det | 2 | 0 | 0 | unsupported Op 'Det' not supported (node ''). |
| Div | 3 | 3 | 3 |  |
| Dropout | 4 | 0 | 0 | unsupported Op 'Dropout' not supported (node ''). |
| Elu | 3 | 0 | 0 | unsupported Op 'Elu' not supported (node ''). |
| Erf | 1 | 1 | 1 |  |
| Exp | 2 | 0 | 0 | unsupported Op 'Exp' not supported (node ''). |
| Expand | 2 | 0 | 0 | unsupported Op 'Expand' not supported (node ''). |
| Flatten | 9 | 0 | 0 | unsupported Op 'Flatten' not supported (node ''). |
| FlexAttention | 9 | 0 | 0 | unsupported Op 'FlexAttention' not supported (node ''). |
| Floor | 2 | 0 | 0 | unsupported Op 'Floor' not supported (node ''). |
| GRU | 4 | 0 | 0 | unsupported Op 'GRU' not supported (node ''). |
| Gather | 4 | 0 | 0 | unsupported Op 'Gather' not supported (node ''). |
| GatherElements | 3 | 0 | 0 | unsupported Op 'GatherElements' not supported (node ''). |
| GatherND | 1 | 0 | 0 | unsupported Op 'GatherND' not supported (node ''). |
| Gelu | 4 | 0 | 0 | unsupported Op 'Gelu' not supported (node ''). |
| Gemm | 11 | 8 | 8 | unsupported Gemm '': alpha != 1. |
| GlobalAveragePool | 2 | 2 | 2 |  |
| GlobalMaxPool | 2 | 0 | 0 | unsupported Op 'GlobalMaxPool' not supported (node ''). |
| GridSample | 18 | 0 | 0 | unsupported Op 'GridSample' not supported (node ''). |
| GroupNormalization | 2 | 0 | 0 | unsupported Op 'GroupNormalization' not supported (node ''). |
| HardSigmoid | 3 | 3 | 3 |  |
| HardSwish | 1 | 0 | 0 | unsupported Op 'HardSwish' not supported (node ''). |
| Hardmax | 7 | 0 | 0 | unsupported Op 'Hardmax' not supported (node ''). |
| Identity | 3 | 2 | 2 | invalid Tensor '' has no float-readable data. |
| InstanceNormalization | 2 | 0 | 0 | unsupported Op 'InstanceNormalization' not supported (node ''). |
| LRN | 2 | 0 | 0 | unsupported Op 'LRN' not supported (node ''). |
| LSTM | 4 | 0 | 0 | unsupported Op 'LSTM' not supported (node ''). |
| LayerNormalization | 19 | 0 | 0 | unsupported LayerNorm only over last dim. |
| LeakyRelu | 3 | 3 | 3 |  |
| LinearAttention | 13 | 0 | 0 | unsupported Op 'LinearAttention' not supported (node ''). |
| Log | 2 | 0 | 0 | unsupported Op 'Log' not supported (node ''). |
| LogSoftmax | 7 | 0 | 0 | unsupported Op 'LogSoftmax' not supported (node ''). |
| LpNormalization | 6 | 0 | 0 | unsupported Op 'LpNormalization' not supported (node ''). |
| LpPool | 8 | 0 | 0 | unsupported Op 'LpPool' not supported (node ''). |
| MatMul | 7 | 4 | 4 | unsupported MatMul '': rank < 2. |
| Max | 4 | 0 | 0 | unsupported Op 'Max' not supported (node ''). |
| MaxPool | 16 | 8 | 8 | unsupported MaxPool '': rank-4 only. |
| MaxUnpool | 2 | 0 | 0 | unsupported Op 'MaxUnpool' not supported (node ''). |
| Mean | 3 | 0 | 0 | unsupported Op 'Mean' not supported (node ''). |
| MeanVarianceNormalization | 1 | 0 | 0 | unsupported Op 'MeanVarianceNormalization' not supported (node ''). |
| Min | 4 | 0 | 0 | unsupported Op 'Min' not supported (node ''). |
| Mish | 1 | 0 | 0 | unsupported Op 'Mish' not supported (node ''). |
| Mod | 1 | 0 | 0 | unsupported Op 'Mod' not supported (node ''). |
| Momentum | 3 | 0 | 0 | unsupported Op 'Momentum' not supported (node ''). |
| Mul | 3 | 3 | 3 |  |
| Neg | 2 | 0 | 0 | unsupported Op 'Neg' not supported (node ''). |
| NegativeLogLikelihoodLoss | 18 | 0 | 0 | unsupported Op 'NegativeLogLikelihoodLoss' not supported (node ''). |
| OneHot | 2 | 0 | 0 | unsupported Op 'OneHot' not supported (node ''). |
| OptionalGetElement | 3 | 0 | 0 | unsupported Op 'OptionalGetElement' not supported (node ''). |
| PRelu | 2 | 0 | 0 | unsupported Op 'PRelu' not supported (node ''). |
| Pad | 3 | 1 | 1 | unsupported Pad '': axes input. |
| Pow | 6 | 1 | 1 | unsupported Pow '': only a constant exponent of 2. |
| RMSNormalization | 19 | 0 | 0 | unsupported Op 'RMSNormalization' not supported (node ''). |
| RNN | 4 | 0 | 0 | unsupported Op 'RNN' not supported (node ''). |
| Range | 1 | 0 | 0 | unsupported Op 'Range' not supported (node ''). |
| Reciprocal | 2 | 0 | 0 | unsupported Op 'Reciprocal' not supported (node ''). |
| ReduceL1 | 9 | 0 | 0 | unsupported Op 'ReduceL1' not supported (node ''). |
| ReduceL2 | 9 | 0 | 0 | unsupported Op 'ReduceL2' not supported (node ''). |
| ReduceLogSum | 5 | 0 | 0 | unsupported Op 'ReduceLogSum' not supported (node ''). |
| ReduceLogSumExp | 1 | 0 | 0 | unsupported Op 'ReduceLogSumExp' not supported (node ''). |
| ReduceMax | 9 | 0 | 0 | unsupported Op 'ReduceMax' not supported (node ''). |
| ReduceMean | 8 | 0 | 0 | unsupported ReduceMean '': axes [1] (rank 3). |
| ReduceMin | 9 | 0 | 0 | unsupported Op 'ReduceMin' not supported (node ''). |
| ReduceProd | 9 | 0 | 0 | unsupported Op 'ReduceProd' not supported (node ''). |
| ReduceSum | 12 | 0 | 0 | unsupported Op 'ReduceSum' not supported (node ''). |
| ReduceSumSquare | 9 | 0 | 0 | unsupported Op 'ReduceSumSquare' not supported (node ''). |
| Relu | 1 | 1 | 1 |  |
| Reshape | 10 | 10 | 9 |  |
| Resize | 39 | 2 | 2 | unsupported Resize '': antialias / axes. |
| ReverseSequence | 2 | 0 | 0 | unsupported Op 'ReverseSequence' not supported (node ''). |
| RoiAlign | 3 | 0 | 0 | unsupported Op 'RoiAlign' not supported (node ''). |
| RotaryEmbedding | 8 | 0 | 0 | unsupported Op 'RotaryEmbedding' not supported (node ''). |
| Round | 1 | 0 | 0 | unsupported Op 'Round' not supported (node ''). |
| STFT | 2 | 0 | 0 | unsupported Op 'STFT' not supported (node ''). |
| Scan | 4 | 0 | 0 | unsupported Op 'Scan' not supported (node ''). |
| Scatter | 2 | 0 | 0 | unsupported Op 'Scatter' not supported (node ''). |
| ScatterElements | 7 | 0 | 0 | unsupported Op 'ScatterElements' not supported (node ''). |
| ScatterND | 5 | 0 | 0 | unsupported Op 'ScatterND' not supported (node ''). |
| Selu | 3 | 0 | 0 | unsupported Op 'Selu' not supported (node ''). |
| SequenceInsert | 2 | 0 | 0 | unsupported Op 'SequenceInsert' not supported (node ''). |
| SequenceMap | 6 | 0 | 0 | unsupported Op 'SequenceMap' not supported (node ''). |
| Shrink | 2 | 0 | 0 | unsupported Op 'Shrink' not supported (node ''). |
| Sigmoid | 2 | 2 | 2 |  |
| Sign | 1 | 0 | 0 | unsupported Op 'Sign' not supported (node ''). |
| Sin | 2 | 0 | 0 | unsupported Op 'Sin' not supported (node ''). |
| Sinh | 2 | 0 | 0 | unsupported Op 'Sinh' not supported (node ''). |
| Slice | 8 | 8 | 8 |  |
| Softmax | 7 | 5 | 5 | unsupported Softmax '': axis 0 is not the last. |
| SoftmaxCrossEntropyLoss | 34 | 0 | 0 | unsupported Op 'SoftmaxCrossEntropyLoss' not supported (node ''). |
| Softplus | 2 | 0 | 0 | unsupported Op 'Softplus' not supported (node ''). |
| Softsign | 2 | 0 | 0 | unsupported Op 'Softsign' not supported (node ''). |
| SpaceToDepth | 2 | 0 | 0 | unsupported Op 'SpaceToDepth' not supported (node ''). |
| Split | 16 | 16 | 16 |  |
| SplitToSequence | 3 | 0 | 0 | unsupported Op 'SplitToSequence' not supported (node ''). |
| Sqrt | 2 | 2 | 2 |  |
| Squeeze | 2 | 2 | 2 |  |
| Sub | 3 | 3 | 3 |  |
| Sum | 3 | 0 | 0 | unsupported Op 'Sum' not supported (node ''). |
| Swish | 1 | 0 | 0 | unsupported Op 'Swish' not supported (node ''). |
| Tan | 2 | 0 | 0 | unsupported Op 'Tan' not supported (node ''). |
| Tanh | 2 | 2 | 2 |  |
| TensorScatter | 3 | 0 | 0 | unsupported Op 'TensorScatter' not supported (node ''). |
| ThresholdedRelu | 3 | 0 | 0 | unsupported Op 'ThresholdedRelu' not supported (node ''). |
| Tile | 2 | 0 | 0 | unsupported Op 'Tile' not supported (node ''). |
| Transpose | 7 | 7 | 7 |  |
| TreeEnsemble | 1 | 0 | 0 | unsupported Op 'TreeEnsemble' not supported (node ''). |
| Unsqueeze | 7 | 7 | 7 |  |
| Upsample | 1 | 0 | 0 | unsupported Op 'Upsample' not supported (node ''). |

## Function-expanded and multi-node tests

333 tests; 4 pass on NCHW, 4 on NHWC.

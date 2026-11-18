#if !NETSTANDARD2_0
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Wasm;
using System.Threading.Tasks;
using SimdMath = Sdcb.SimdOnnx.Kernels.SimdMath;

namespace Sdcb.SimdOnnx.Decoding.Kernels;

// WebAssembly SIMD128 (PackedSimd) MatMul / fused MatMul+ArgMax over the
// [col/16][k][16] packed weights. The Vector<float> kernels keep 16
// accumulators (4 rows x 16 columns) live, or a stackalloc'd accumulator
// array for ArgMax; V8 maps v128 values onto at most 15 allocatable XMM
// registers on x64, so both spill. These run 4 rows x 8 columns per pass
// (8 accumulators), twice per 16-column tile. Accumulation is zero-initialised,
// k ascending, separate multiply then add: lane-for-lane the arithmetic of the
// Vector kernels.
internal static partial class MatMul
{
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void TryPackedSimd(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns, float[]? packedWeights)
    {
        if (packedWeights is null || rows < 4 || inner < 64 || columns < 1024)
        {
            TryVector(input, weights, output, batch, rows, inner, columns, packedWeights);
            return;
        }
        fixed (float* inputPtr = input, weightsPtr = weights, packedPtr = packedWeights, outputPtr = output)
        {
            for (int b = 0; b < batch; b++)
            {
                int row = 0;
                for (; row <= rows - 4; row += 4)
                {
                    float* in0 = inputPtr + (b * rows + row) * inner;
                    float* out0 = outputPtr + (b * rows + row) * columns;
                    int col = 0;
                    for (; col <= columns - 16; col += 16)
                    {
                        float* tile = packedPtr + (col / 16) * inner * 16;
                        Rows4x8PackedSimd(in0, inner, tile, out0 + col, columns);
                        Rows4x8PackedSimd(in0, inner, tile + 8, out0 + col + 8, columns);
                    }
                    for (int r = 0; r < 4; r++)
                        for (int c = col; c < columns; c++)
                        {
                            float sum = 0;
                            for (int k = 0; k < inner; k++)
                                sum += in0[r * inner + k] * weightsPtr[k * columns + c];
                            out0[r * columns + c] = sum;
                        }
                }
                if (row < rows)
                    MatMulRows1Vector(
                        input.Slice(b * rows * inner, rows * inner), weights,
                        output.Slice(b * rows * columns, rows * columns),
                        1, row, rows, inner, columns);
            }
        }
    }

    // 4 rows x 8 columns of a packed 16-column tile (tile already offset to
    // the column half), zero-initialised, stored to output rows.
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void Rows4x8PackedSimd(float* input, int inner, float* tile, float* output, int columns)
    {
        float* p0 = input, p1 = input + inner, p2 = input + 2 * inner, p3 = input + 3 * inner;
        Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
        Vector128<float> b0 = a0, b1 = a0, b2 = a0, b3 = a0;
        float* wk = tile;
        for (int k = 0; k < inner; k++, wk += 16)
        {
            Vector128<float> w0 = PackedSimd.LoadVector128(wk);
            Vector128<float> w1 = PackedSimd.LoadVector128(wk + 4);
            Vector128<float> v = PackedSimd.LoadScalarAndSplatVector128(p0 + k);
            a0 = PackedSimd.Add(a0, PackedSimd.Multiply(w0, v)); b0 = PackedSimd.Add(b0, PackedSimd.Multiply(w1, v));
            v = PackedSimd.LoadScalarAndSplatVector128(p1 + k);
            a1 = PackedSimd.Add(a1, PackedSimd.Multiply(w0, v)); b1 = PackedSimd.Add(b1, PackedSimd.Multiply(w1, v));
            v = PackedSimd.LoadScalarAndSplatVector128(p2 + k);
            a2 = PackedSimd.Add(a2, PackedSimd.Multiply(w0, v)); b2 = PackedSimd.Add(b2, PackedSimd.Multiply(w1, v));
            v = PackedSimd.LoadScalarAndSplatVector128(p3 + k);
            a3 = PackedSimd.Add(a3, PackedSimd.Multiply(w0, v)); b3 = PackedSimd.Add(b3, PackedSimd.Multiply(w1, v));
        }
        PackedSimd.Store(output, a0); PackedSimd.Store(output + 4, b0);
        PackedSimd.Store(output + columns, a1); PackedSimd.Store(output + columns + 4, b1);
        PackedSimd.Store(output + columns * 2, a2); PackedSimd.Store(output + columns * 2 + 4, b2);
        PackedSimd.Store(output + columns * 3, a3); PackedSimd.Store(output + columns * 3 + 4, b3);
    }

    // ------------------------------------------------------------------ ArgMax

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxPackedPackedSimd(ReadOnlySpan<float> input,
        ReadOnlySpan<float> weights, ReadOnlySpan<float> packedWeights,
        ReadOnlySpan<float> bias, Span<int> indices, Span<float> scores,
        int batch, int rows, int inner, int columns, int threads)
    {
        int full4 = rows / 4, tail = rows - full4 * 4;
        int rowJobs = full4 + tail;
        int jobs = checked(batch * rowJobs);
        int workers = Math.Min(ArgMaxWorkers(batch, rows, inner, columns, threads), jobs);
        bool hasBias = !bias.IsEmpty;
        fixed (float* inputPin = input, weightsPin = weights,
            packedPin = packedWeights, biasPin = bias)
        fixed (int* indicesPin = indices)
        fixed (float* scoresPin = scores)
        {
            float* inputPtr = inputPin, weightsPtr = weightsPin,
                packedPtr = packedPin, biasPtr = biasPin;
            int* indicesPtr = indicesPin;
            float* scoresPtr = scoresPin;
            void Worker(int w)
            {
                for (int job = w; job < jobs; job += workers)
                {
                    int b = job / rowJobs, j = job - b * rowJobs;
                    if (j < full4)
                        MatMulArgMaxRows4PackedSimd(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b, j * 4,
                            rows, inner, columns);
                    else
                        MatMulArgMaxPackedVectorRows1(inputPtr, weightsPtr, packedPtr,
                            biasPtr, hasBias, indicesPtr, scoresPtr, b,
                            full4 * 4 + j - full4, rows, inner, columns);
                }
            }
            if (workers > 1) Parallel.For(0, workers, Worker);
            else Worker(0);
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static unsafe void MatMulArgMaxRows4PackedSimd(float* input,
        float* weights, float* packed, float* bias, bool hasBias,
        int* indices, float* scores, int batch, int row,
        int rows, int inner, int columns)
    {
        // Running per-row lane maxima / column indices / softmax denominators,
        // updated once per 8-column half in ascending column order (the
        // Vector kernel's order).
        Vector128<float>* state = stackalloc Vector128<float>[12];
        Vector128<float> negInf = Vector128.Create(float.NegativeInfinity);
        for (int r = 0; r < 4; r++)
        {
            state[r] = negInf;
            state[4 + r] = Vector128<float>.Zero;
            state[8 + r] = Vector128<float>.Zero;
        }
        float* p0 = input + (batch * rows + row) * inner;
        float* p1 = p0 + inner, p2 = p1 + inner, p3 = p2 + inner;
        Vector128<float> lanes = Vector128.Create(0f, 1f, 2f, 3f);
        Vector128<float>* tileOut = stackalloc Vector128<float>[8];
        int col = 0;
        for (; col <= columns - 16; col += 16)
        {
            float* tile = packed + (col / 16) * inner * 16;
            for (int half = 0; half < 16; half += 8)
            {
                Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                Vector128<float> b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                float* wk = tile + half;
                for (int k = 0; k < inner; k++, wk += 16)
                {
                    Vector128<float> w0 = PackedSimd.LoadVector128(wk);
                    Vector128<float> w1 = PackedSimd.LoadVector128(wk + 4);
                    Vector128<float> v = PackedSimd.LoadScalarAndSplatVector128(p0 + k);
                    a0 = PackedSimd.Add(a0, PackedSimd.Multiply(w0, v)); b0 = PackedSimd.Add(b0, PackedSimd.Multiply(w1, v));
                    v = PackedSimd.LoadScalarAndSplatVector128(p1 + k);
                    a1 = PackedSimd.Add(a1, PackedSimd.Multiply(w0, v)); b1 = PackedSimd.Add(b1, PackedSimd.Multiply(w1, v));
                    v = PackedSimd.LoadScalarAndSplatVector128(p2 + k);
                    a2 = PackedSimd.Add(a2, PackedSimd.Multiply(w0, v)); b2 = PackedSimd.Add(b2, PackedSimd.Multiply(w1, v));
                    v = PackedSimd.LoadScalarAndSplatVector128(p3 + k);
                    a3 = PackedSimd.Add(a3, PackedSimd.Multiply(w0, v)); b3 = PackedSimd.Add(b3, PackedSimd.Multiply(w1, v));
                }
                int column = col + half;
                if (hasBias)
                {
                    Vector128<float> bl = PackedSimd.LoadVector128(bias + column);
                    Vector128<float> bh = PackedSimd.LoadVector128(bias + column + 4);
                    a0 = PackedSimd.Add(a0, bl); a1 = PackedSimd.Add(a1, bl); a2 = PackedSimd.Add(a2, bl); a3 = PackedSimd.Add(a3, bl);
                    b0 = PackedSimd.Add(b0, bh); b1 = PackedSimd.Add(b1, bh); b2 = PackedSimd.Add(b2, bh); b3 = PackedSimd.Add(b3, bh);
                }
                tileOut[0] = a0; tileOut[1] = a1; tileOut[2] = a2; tileOut[3] = a3;
                tileOut[4] = b0; tileOut[5] = b1; tileOut[6] = b2; tileOut[7] = b3;
                UpdateHalfPackedSimd(tileOut, PackedSimd.Add(PackedSimd.Splat((float)column), lanes), state);
            }
        }
        Span<Vector<float>> vectorMaxima = stackalloc Vector<float>[4];
        Span<Vector<float>> vectorIndices = stackalloc Vector<float>[4];
        Span<Vector<float>> vectorSums = stackalloc Vector<float>[4];
        for (int r = 0; r < 4; r++)
        {
            vectorMaxima[r] = state[r].AsVector();
            vectorIndices[r] = state[4 + r].AsVector();
            vectorSums[r] = state[8 + r].AsVector();
        }
        float* maxima = stackalloc float[4];
        int* best = stackalloc int[4];
        float* sums = stackalloc float[4];
        ReduceVector(vectorMaxima, vectorIndices, vectorSums, maxima, best, sums);
        FinishScalarTail(input, weights, bias, hasBias, indices, scores,
            batch, row, 4, rows, inner, columns, col, maxima, best, sums);
    }

    // The per-tile epilogue (argmax + online softmax, an exp per logit) stays out of line: inlined
    // into the MAC loop's method it costs the 5800X ~4x its arithmetic (AVX2: +65% instead of +15%
    // on a 120 x 18385 head), while one call per register tile is noise next to inner MACs.
    /// <summary>tile: rows 0-3 of columns [column, column + 4), then rows 0-3 of [column + 4, column + 8).</summary>
    [MethodImpl(MethodImplOptions.NoInlining | MethodImplCompat.AggressiveOptimization)]
    private static unsafe void UpdateHalfPackedSimd(Vector128<float>* tile, Vector128<float> low, Vector128<float>* state)
    {
        Vector128<float> high = PackedSimd.Add(low, Vector128.Create(4f));
        for (int r = 0; r < 4; r++)
            UpdatePackedSimd(tile[r], low, state, r);
        for (int r = 0; r < 4; r++)
            UpdatePackedSimd(tile[4 + r], high, state, r);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void UpdatePackedSimd(Vector128<float> values, Vector128<float> candidate,
        Vector128<float>* state, int row)
    {
        Vector128<float> previous = state[row];
        Vector128<float> replace = PackedSimd.CompareGreaterThan(values, previous);
        // online softmax denominator, see UpdateArgMax
        Vector128<float> e = SimdMath.ExpPk(PackedSimd.Negate(PackedSimd.Abs(PackedSimd.Subtract(values, previous))));
        Vector128<float> s = state[8 + row];
        state[8 + row] = PackedSimd.BitwiseSelect(PackedSimd.Add(PackedSimd.Multiply(s, e), Vector128<float>.One),
            PackedSimd.Add(s, e), replace);
        state[row] = PackedSimd.Max(previous, values);
        state[4 + row] = PackedSimd.BitwiseSelect(candidate, state[4 + row], replace);
    }
}
#endif

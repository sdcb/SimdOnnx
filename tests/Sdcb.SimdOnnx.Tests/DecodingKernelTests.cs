using Sdcb.SimdOnnx.Decoding;
using Sdcb.SimdOnnx.Decoding.Kernels;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>CTC projection + argmax kernels (moved from SimdPaddleOCR with the kernels) and the public decoding API.</summary>
public class DecodingKernelTests
{
    [Theory]
    [InlineData(5, 32, 40)]
    [InlineData(4, 64, 1040)]
    [InlineData(5, 64, 1040)]
    [InlineData(4, 32, 39)]
    [InlineData(6, 32, 37)]
    [InlineData(7, 96, 1037)]
    public void MatMul_VectorMatchesReference(int rows, int inner, int columns)
    {
        const int batch = 2;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, 0.02f);
        float[]? packed = columns >= 16 ? PackMatMul(weights, inner, columns) : null;

        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);

        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);

        float[] vectorActual = new float[expected.Length];
        Assert.True(MatMul.TryVector(input, weights, vectorActual, batch, rows, inner, columns, packed));
        AssertClose(expected, vectorActual, rtol: 5e-5f, atol: 5e-5f);
    }

    [Fact]
    public void MatMul_Rows8PackedHandlesPartialColumnTile()
    {
        const int batch = 1, rows = 8, inner = 64, columns = 1050;
        float[] input = Ramp(batch * rows * inner, 0.01f);
        float[] weights = Ramp(inner * columns, -0.02f);
        float[] packed = PackMatMul(weights, inner, columns);
        float[] expected = new float[batch * rows * columns];
        MatMulRef(input, weights, expected, batch, rows, inner, columns);
        float[] actual = new float[expected.Length];
        Assert.True(MatMul.Try(input, weights, actual, batch, rows, inner, columns, packed));
        AssertClose(expected, actual, rtol: 5e-5f, atol: 5e-5f);
    }

    // rows pick the tier's row blocks: 16/8 (AVX-512), 4 (AVX2, Vector, NEON, WASM), 1 (tails);
    // 1037 / 1030 columns leave a scalar column tail; packed: false runs the scalar kernel.
    [Theory]
    [InlineData(4, 64, 1040, true)]
    [InlineData(5, 96, 1037, true)]
    [InlineData(3, 64, 1030, true)]
    [InlineData(7, 96, 6624, true)]
    [InlineData(8, 64, 1040, true)]
    [InlineData(16, 96, 1037, true)]
    [InlineData(24, 64, 1030, true)]
    [InlineData(5, 64, 1037, false)]
    [InlineData(8, 96, 1040, false)]
    public void MatMul_ArgMaxMatchesReference(int rows, int inner, int columns, bool packed)
    {
        // Random data: a ramp over the inner dim produces near-tied column scores,
        // and argmax then flips between rounding orders (FMA vs mul+add).
        const int batch = 2;
        float[] input = RandomFill(batch * rows * inner, 1f, seed: 1234);
        float[] weights = RandomFill(inner * columns, 0.25f, seed: 5678);
        float[] bias = RandomFill(columns, 0.5f, seed: 9012);
        var (expectedIndex, expectedProbability) = SoftmaxArgMaxRef(input, weights, bias, batch * rows, inner, columns);

        int[] indices = new int[batch * rows];
        float[] scores = new float[batch * rows];
        Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
            batch, rows, inner, columns, packed ? PackMatMul(weights, inner, columns) : null, threads: 3));
        Assert.Equal(expectedIndex, indices);
        AssertProbabilities(expectedProbability, scores);
    }

    // Streamed REC batches: units back to back. With threads > 1 and every rows[u] % 4 == 0 the
    // AVX-512 tier splits the column tiles across workers and merges their lane states; otherwise
    // units sharing a kernel are merged into one row-blocked call.
    [Theory]
    [InlineData(new[] { 1, 2, 1 }, new[] { 8, 12, 16 }, 4)]
    [InlineData(new[] { 3, 1 }, new[] { 4, 20 }, 3)]
    [InlineData(new[] { 2, 1, 2 }, new[] { 8, 5, 16 }, 4)]
    [InlineData(new[] { 1, 2 }, new[] { 16, 8 }, 1)]
    public void MatMul_ArgMaxUnitsMatchesReference(int[] batches, int[] rows, int threads)
    {
        const int inner = 96, columns = 1037;
        int totalRows = 0;
        for (int u = 0; u < batches.Length; u++) totalRows += batches[u] * rows[u];
        float[] input = RandomFill(totalRows * inner, 1f, seed: 77);
        float[] weights = RandomFill(inner * columns, 0.25f, seed: 78);
        float[] bias = RandomFill(columns, 0.5f, seed: 79);
        var (expectedIndex, expectedProbability) = SoftmaxArgMaxRef(input, weights, bias, totalRows, inner, columns);

        int[] indices = new int[totalRows];
        float[] scores = new float[totalRows];
        Assert.True(MatMul.TryArgMaxUnits(input, weights, bias, indices, scores, batches, rows,
            inner, columns, PackMatMul(weights, inner, columns), threads));
        Assert.Equal(expectedIndex, indices);
        AssertProbabilities(expectedProbability, scores);
    }

    // Softmax is shift-invariant: the online denominator must not overflow or flush to zero when
    // every logit sits far from 0, and a dominant class must come out at (almost exactly) 1.
    [Theory]
    [InlineData(-1000f, 4, true)]
    [InlineData(1000f, 4, true)]
    [InlineData(-1000f, 16, true)]
    [InlineData(1000f, 16, true)]
    [InlineData(1000f, 5, false)]
    public void MatMul_ArgMaxProbabilityIsShiftInvariant(float offset, int rows, bool packed)
    {
        const int batch = 1, inner = 64, columns = 1040;
        float[] input = RandomFill(batch * rows * inner, 1f, seed: 11);
        float[] weights = RandomFill(inner * columns, 0.25f, seed: 12);
        float[] bias = RandomFill(columns, 0.5f, seed: 13);
        float[]? pack = packed ? PackMatMul(weights, inner, columns) : null;
        int[] i0 = new int[rows], i1 = new int[rows];
        float[] p0 = new float[rows], p1 = new float[rows];
        Assert.True(MatMul.TryArgMax(input, weights, bias, i0, p0, batch, rows, inner, columns, pack));
        float[] shifted = bias.Select(b => b + offset).ToArray();
        Assert.True(MatMul.TryArgMax(input, weights, shifted, i1, p1, batch, rows, inner, columns, pack));
        Assert.Equal(i0, i1);
        // logits near +-1000 keep ~6e-5 of absolute precision: compare loosely
        AssertClose(p0, p1, rtol: 2e-3f, atol: 1e-6f);

        // one class 60 above the rest: probability rounds to 1, and never above it
        shifted[777] += 60f;
        Assert.True(MatMul.TryArgMax(input, weights, shifted, i1, p1, batch, rows, inner, columns, pack));
        Assert.All(i1, value => Assert.Equal(777, value));
        Assert.All(p1, value => Assert.InRange(value, 0.99999f, 1f));
    }

    // From SimdPaddleOCR PR #39 (YDLuo-1): only the blank and the last class (the scalar column tail
    // of the packed kernels) have weights, so each row's probability has a closed form; nine rows
    // leave a row tail in every tier's row blocking.
    [Theory]
    [InlineData(false, true)]
    [InlineData(true, true)]
    [InlineData(false, false)]
    [InlineData(true, false)]
    public void MatMul_ArgMaxProbabilityAcrossRowAndColumnTails(bool withBias, bool packed)
    {
        const int rows = 9, inner = 72, columns = 1025;
        float[] weights = new float[inner * columns];
        weights[(inner - 1) * columns] = -1;
        weights[^1] = 1;
        float[] bias = withBias ? new float[columns] : [];
        if (withBias) bias[^1] = 3.5f;
        float[] input = new float[rows * inner];
        for (int r = 0; r < rows; r++) input[(r + 1) * inner - 1] = r - 4;
        int[] indices = new int[rows];
        float[] scores = new float[rows];
        Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores, 1, rows, inner, columns,
            packed ? PackMatMul(weights, inner, columns) : null, threads: 1));

        for (int r = 0; r < rows; r++)
        {
            double blank = 4 - r, last = r - 4 + (withBias ? 3.5 : 0);
            int winner = last > blank ? columns - 1 : 0;
            double top = Math.Max(blank, last);
            double probability = 1 / (Math.Exp(blank - top) + Math.Exp(last - top) + (columns - 2) * Math.Exp(-top));
            Assert.Equal(winner, indices[r]);
            Assert.InRange(Math.Abs(scores[r] - probability), 0, 1e-6);
        }
    }

    [Theory]
    [InlineData(float.NaN, true)]
    [InlineData(float.NegativeInfinity, true)]
    [InlineData(float.PositiveInfinity, true)]
    [InlineData(float.NaN, false)]
    [InlineData(float.NegativeInfinity, false)]
    [InlineData(float.PositiveInfinity, false)]
    public void MatMul_ArgMaxRejectsNonFiniteLogit(float invalid, bool packed)
    {
        const int rows = 4, inner = 64, columns = 1025;
        float[] weights = new float[inner * columns];
        float[] bias = new float[columns];
        bias[^1] = invalid;
        Assert.Throws<InvalidDataException>(() => MatMul.TryArgMax(new float[rows * inner], weights, bias,
            new int[rows], new float[rows], 1, rows, inner, columns,
            packed ? PackMatMul(weights, inner, columns) : null, threads: 1));
    }

    /// <summary>Per row: the float argmax (first index on ties) and its softmax probability in double.</summary>
    internal static (int[] Index, float[] Probability) SoftmaxArgMaxRef(float[] input, float[] weights,
        float[] bias, int totalRows, int inner, int columns)
    {
        float[] logits = new float[totalRows * columns];
        MatMulRef(input, weights, logits, 1, totalRows, inner, columns);
        int[] index = new int[totalRows];
        float[] probability = new float[totalRows];
        for (int r = 0; r < totalRows; r++)
        {
            int best = 0;
            float max = logits[r * columns] + bias[0];
            for (int c = 1; c < columns; c++)
            {
                float value = logits[r * columns + c] + bias[c];
                if (value > max) { max = value; best = c; }
            }
            double sum = 0;
            for (int c = 0; c < columns; c++)
                sum += Math.Exp((double)(logits[r * columns + c] + bias[c]) - max);
            index[r] = best;
            probability[r] = (float)(1 / sum);
        }
        return (index, probability);
    }

    internal static void AssertProbabilities(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
    {
        foreach (float value in actual) Assert.InRange(value, float.Epsilon, 1f);
        // logits differ from the reference by FMA rounding (~1e-6 abs), exp turns that relative
        AssertClose(expected, actual, rtol: 1e-4f, atol: 1e-7f);
    }

    internal static float[] PackMatMul(ReadOnlySpan<float> weights, int inner, int columns)
    {
        int fullTiles = columns / 16;
        float[] packed = new float[fullTiles * inner * 16];
        for (int tile = 0; tile < fullTiles; tile++)
            for (int k = 0; k < inner; k++)
                weights.Slice(k * columns + tile * 16, 16).CopyTo(packed.AsSpan((tile * inner + k) * 16, 16));
        return packed;
    }

    internal static void MatMulRef(ReadOnlySpan<float> input, ReadOnlySpan<float> weights,
        Span<float> output, int batch, int rows, int inner, int columns)
    {
        for (int b = 0; b < batch; b++)
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < columns; col++)
                {
                    float sum = 0;
                    for (int k = 0; k < inner; k++)
                        sum += input[(b * rows + row) * inner + k] * weights[k * columns + col];
                    output[(b * rows + row) * columns + col] = sum;
                }
    }

    internal static float[] Ramp(int length, float scale)
    {
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (i % 17 - 8) * scale;
        return values;
    }

    internal static float[] RandomFill(int length, float amplitude, int seed)
    {
        var rng = new Random(seed);
        float[] values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = (float)(rng.NextDouble() * 2 - 1) * amplitude;
        return values;
    }

    internal static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual,
        float rtol = 2e-5f, float atol = 2e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float tol = atol + rtol * Math.Max(Math.Abs(e), Math.Abs(a));
            if (Math.Abs(e - a) > tol)
                Assert.Fail($"index {i}: expected {e}, actual {a}, tol {tol}");
        }
    }

    [Theory]
    [InlineData(4, 1)]
    [InlineData(256, 4)]
    [InlineData(512, 4)]
    [InlineData(516, 4)]
    [InlineData(1024, 4)]
    [InlineData(2048, 1)]
    public void ReusedWorkspaceResetsMaximaAndTieIndices(int rows, int threads)
    {
        const int batch = 2, inner = 64, columns = 1040;
        float[] input = new float[batch * rows * inner];
        float[] weights = new float[inner * columns];
        float[] packed = PackMatMul(weights, inner, columns);
        float[] bias = new float[columns];
        int[] indices = new int[batch * rows];
        float[] scores = new float[indices.Length];
        for (int iteration = 0; iteration < 8; iteration++)
        {
            bool tied = (iteration & 1) != 0;
            Array.Fill(bias, -20f);
            int expected = tied ? 0 : columns - 1 - iteration;
            if (!tied) bias[expected] = 5f;
            Assert.True(MatMul.TryArgMax(input, weights, bias, indices, scores,
                batch, rows, inner, columns, packed, threads));
            Assert.All(indices, value => Assert.Equal(expected, value));
            // all logits are the bias: 1 / columns when tied, else 1 / (1 + (columns - 1) e^-25)
            float probability = tied ? 1f / columns : (float)(1 / (1 + (columns - 1) * Math.Exp(-25)));
            Assert.All(scores, value => Assert.Equal(probability, value, 1e-6f));
        }
    }

    [Fact]
    public void CtcProjection_MatchesKernel()
    {
        const int batch = 2, rows = 8, inner = 64, columns = 1040;
        float[] input = RandomFill(batch * rows * inner, 1f, 3), weights = RandomFill(inner * columns, 0.2f, 4), bias = RandomFill(columns, 0.5f, 5);
        var head = new CtcProjection(weights, bias, inner, columns);
        Assert.True(head.IsFused);
        int[] a = new int[batch * rows], b = new int[a.Length];
        float[] sa = new float[a.Length], sb = new float[a.Length];
        Assert.True(head.ArgMax(input, batch, rows, a, sa, 4));
        float[] logits = new float[batch * rows * columns];
        MatMulRef(input, weights, logits, batch, rows, inner, columns);
        for (int r = 0; r < a.Length; r++)
        {
            for (int c = 0; c < columns; c++) logits[r * columns + c] += bias[c];
            Span<float> row = logits.AsSpan(r * columns, columns);
            b[r] = CtcGreedyDecoder.ArgMax(row, out float max);
            double sum = 0;
            foreach (float value in row) sum += Math.Exp((double)value - max);
            sb[r] = (float)(1 / sum);
        }
        Assert.Equal(b, a);
        AssertProbabilities(sb, sa);
    }

    [Fact]
    public void Collapse_MergesRepeatsAndDropsBlanks()
    {
        int[] idx = [0, 3, 3, 0, 3, 5, 5, 5];
        float[] sc = [.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f];
        var tokens = new CtcToken[idx.Length];
        int n = CtcGreedyDecoder.Collapse(idx, sc, tokens);
        Assert.Equal(3, n);
        Assert.Equal(new CtcToken(3, .2f, 1, 3), tokens[0]);
        Assert.Equal(new CtcToken(3, .5f, 4, 5), tokens[1]);
        Assert.Equal(new CtcToken(5, .6f, 5, 8), tokens[2]);
    }
}

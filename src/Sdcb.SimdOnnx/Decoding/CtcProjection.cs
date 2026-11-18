using Sdcb.SimdOnnx.Decoding.Kernels;

namespace Sdcb.SimdOnnx.Decoding;

/// <summary>
/// The class projection of a CTC head (features [rows, inner] x W [inner, columns] + bias) fused
/// with the per-row argmax, so the [rows, columns] logits (columns can be thousands of classes) are
/// never stored. Wide heads (inner &gt;= 64, columns &gt;= 1024) run register-blocked SIMD kernels over
/// weights packed once into 16-column tiles (AVX-512 / AVX2 / NEON / WASM SIMD / Vector&lt;T&gt;).
/// Immutable and thread-safe: share one instance across sessions.
/// </summary>
internal sealed class CtcProjection
{
    /// <param name="weights">Row-major [inner, columns] (an ONNX MatMul's B).</param>
    /// <param name="bias">Per-class bias ([columns]) or null.</param>
    /// <param name="inner">rows of <paramref name="weights"/> (the projected feature size).</param>
    /// <param name="columns">columns of <paramref name="weights"/> (the class count).</param>
    public CtcProjection(float[] weights, float[]? bias, int inner, int columns)
    {
        if (inner <= 0 || columns <= 0) throw new ArgumentOutOfRangeException(nameof(inner));
        if (weights.Length != (long)inner * columns) throw new ArgumentException("weights must hold inner x columns floats.", nameof(weights));
        if (bias is not null && bias.Length != columns) throw new ArgumentException("bias must hold columns floats.", nameof(bias));
        Weights = weights;
        Bias = bias;
        Inner = inner;
        Columns = columns;
        Packed = Pack(weights, inner, columns);
    }

    /// <summary>Feature width (the projection's K).</summary>
    public int Inner { get; }
    /// <summary>Class count, blank included.</summary>
    public int Columns { get; }
    /// <summary>The row-major [inner, columns] weights.</summary>
    public float[] Weights { get; }
    /// <summary>Per-class bias, or null.</summary>
    public float[]? Bias { get; }
    /// <summary>[columns/16][inner][16] tiles (null when the head is too narrow for the packed kernels).</summary>
    internal float[]? Packed { get; }

    /// <summary>True when the fused kernels serve this head (narrow heads: project, then <see cref="CtcGreedyDecoder.ArgMax"/>).</summary>
    public bool IsFused => MatMul.CanFuseArgMax(1, Inner, Columns, Packed);

    /// <summary>
    /// Per row of <paramref name="features"/> ([batch * rows, inner]): the index of the largest logit
    /// and that logit (ties keep the first index). Throws on non-finite logits.
    /// </summary>
    /// <returns>False when the head is not <see cref="IsFused"/> (nothing written).</returns>
    public bool ArgMax(ReadOnlySpan<float> features, int batch, int rows, Span<int> indices, Span<float> maxLogits, int threads = 1)
    {
        if (features.Length != (long)batch * rows * Inner) throw new ArgumentException("features must hold batch x rows x inner floats.", nameof(features));
        return MatMul.TryArgMax(features, Weights, Bias ?? [], indices, maxLogits, batch, rows, Inner, Columns, Packed, threads);
    }

    /// <summary>
    /// <see cref="ArgMax"/> over consecutive units ([batches[u], rows[u]] each) stored back to back:
    /// runs of units the same packed kernel serves go out as one call (better weight reuse); every
    /// row's arithmetic is that of its standalone call.
    /// </summary>
    public bool ArgMaxUnits(ReadOnlySpan<float> features, ReadOnlySpan<int> batches, ReadOnlySpan<int> rows,
        Span<int> indices, Span<float> maxLogits, int threads = 1)
    {
        if (batches.Length != rows.Length) throw new ArgumentException("batches and rows differ in length.", nameof(rows));
        return MatMul.TryArgMaxUnits(features, Weights, Bias ?? [], indices, maxLogits, batches, rows, Inner, Columns, Packed, threads);
    }

    private static float[]? Pack(float[] w, int inner, int columns)
    {
        if (inner < 64 || columns < 1024) return null;
        int tiles = columns / 16;
        float[] packed = new float[checked(tiles * inner * 16)];
        for (int tile = 0; tile < tiles; tile++)
            for (int k = 0; k < inner; k++)
                Array.Copy(w, k * columns + tile * 16, packed, (tile * inner + k) * 16, 16);
        return packed;
    }
}

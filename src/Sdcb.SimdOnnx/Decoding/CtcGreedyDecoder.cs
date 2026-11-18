using Sdcb.SimdOnnx.Decoding.Kernels;

namespace Sdcb.SimdOnnx.Decoding;

/// <summary>One emitted CTC label: its class, its frame score and the half-open frame run it covers.</summary>
/// <param name="Class">Class index (never the blank).</param>
/// <param name="Score">Score of the run's first frame (a max logit or a probability, as given).</param>
/// <param name="Start">First frame of the run.</param>
/// <param name="End">One past the run's last frame of this class (blank frames excluded); the frame count when it reaches the end.</param>
internal readonly record struct CtcToken(int Class, float Score, int Start, int End);

/// <summary>
/// Greedy (best-path) CTC decoding: a label is emitted where the per-frame argmax is not the blank
/// and differs from the previous frame's argmax (blank frames included), as PaddleOCR decodes.
/// </summary>
internal static class CtcGreedyDecoder
{
    /// <summary>
    /// Collapses per-frame argmax <paramref name="indices"/> (with their <paramref name="scores"/>)
    /// into <paramref name="tokens"/> (at least as long as the frame count).
    /// </summary>
    /// <returns>The number of tokens written.</returns>
    public static int Collapse(ReadOnlySpan<int> indices, ReadOnlySpan<float> scores, Span<CtcToken> tokens, int blank = 0)
    {
        if (scores.Length != indices.Length) throw new ArgumentException("scores and indices differ in length.", nameof(scores));
        if (tokens.Length < indices.Length) throw new ArgumentException("tokens must hold one entry per frame.", nameof(tokens));
        int previous = blank, emitted = 0, open = -1, steps = indices.Length;
        for (int step = 0; step < steps; step++)
        {
            int best = indices[step];
            if (best != blank && best != previous)
            {
                if (open >= 0) tokens[open] = tokens[open] with { End = step };
                tokens[emitted] = new CtcToken(best, scores[step], step, steps);
                open = emitted++;
            }
            else if (open >= 0 && best != previous)
            {
                // a blank ends the run (repeated frames of the class stayed inside)
                tokens[open] = tokens[open] with { End = step };
                open = -1;
            }
            previous = best;
        }
        return emitted;
    }

    /// <summary>Index of the largest value of <paramref name="row"/> (first on ties) and that value; throws on non-finite values.</summary>
    public static int ArgMax(ReadOnlySpan<float> row, out float max)
    {
        if (row.IsEmpty) throw new ArgumentException("empty row", nameof(row));
        return Kernels.ArgMax.Find(row, out max);
    }

    /// <summary>softmax(<paramref name="logits"/>) at the row maximum <paramref name="max"/>: 1 / Σ exp(x − max).</summary>
    public static float MaxProbability(ReadOnlySpan<float> logits, float max) => SimdKernels.SoftmaxMaximumProbability(logits, max);
}

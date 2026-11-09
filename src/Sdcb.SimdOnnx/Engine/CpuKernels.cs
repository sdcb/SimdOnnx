using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Kernels;

namespace Sdcb.SimdOnnx.Engine;

/// <summary>
/// CPU-private packed weights (conv panels + Winograd transforms, Gemm panels,
/// UpHead lanes) stored in the plan's op params. Per-size plans of one model
/// carry identical weights, so a <see cref="Cache"/> shared across sizes packs
/// each layer once: entries are keyed by node identity and only reused when the
/// weights (and bias) are element-for-element equal.
/// </summary>
public static class CpuKernels
{
    /// <summary>Packed-weight cache shared by the per-shape CPU plans of one model.</summary>
    public sealed class Cache
    {
        private readonly Dictionary<(ExecOp op, string name, int src), (float[] w, float[]? b, Lazy<object> kp)> _d = new();

        /// <summary>The packed weights of <paramref name="n"/>, packed once: workers compiling the
        /// same layer concurrently (new shapes on several threads) share one pack instead of each
        /// keeping its own copy.</summary>
        [MethodImpl(MethodImplCompat.AggressiveOptimization)]
        public object GetOrPack(ExecNode n, float[] w, float[]? b, Func<object> pack)
        {
            var key = (n.Op, n.Name, n.SrcNode);
            Lazy<object> kp;
            lock (_d)
            {
                if (!_d.TryGetValue(key, out var e) || !Same(e.w, w) || !Same(e.b, b))
                    _d[key] = e = (w, b, new Lazy<object>(pack, LazyThreadSafetyMode.ExecutionAndPublication));
                kp = e.kp;
            }
            return kp.Value;
        }

        [MethodImpl(MethodImplCompat.AggressiveOptimization)]
        private static bool Same(float[]? a, float[]? b) =>
            ReferenceEquals(a, b) || (a is not null && b is not null && a.AsSpan().SequenceEqual(b));
    }

    /// <summary>Fills missing kernel plans of an NHWC plan (idempotent; plans may be shared).</summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static void Pack(ExecPlan plan, Cache? cache)
    {
        if (!plan.Nhwc) return;
        CValue[] V = plan.Values;
        foreach (ExecNode n in plan.Nodes)
        {
            n.Ext?.PackCpu(plan, n, cache);
            switch (n.P)
            {
                case GemmP gp when gp.KernelPlan is null && V[n.In[1]].Floats is { } bw:
                {
                    int[] bs = V[n.In[1]].Shape;
                    int kk = gp.TransB ? bs[1] : bs[0], nn = gp.TransB ? bs[0] : bs[1];
                    float[]? gb = gp.BiasValue >= 0 ? V[gp.BiasValue].Floats : null;
                    object kp = Get(cache, n, bw, gb, () => Nhwc.PackGemm(bw, kk, nn, gp.TransB, gb));
                    Interlocked.CompareExchange(ref gp.KernelPlan, kp, null);
                    break;
                }
                case ConvP cp when cp.KernelPlan is null:
                {
                    CValue w = V[n.In[1]];
                    if (w.Floats is null)
                        throw new NotSupportedException($"Conv '{n.Name}': run-time weights in an NHWC plan (weights must be constant).");
                    float[]? b = cp.BiasValue >= 0 ? V[cp.BiasValue].Floats : null;
                    // from the weights ([oc, ic/group, kh, kw]): the activation's Shape may be the
                    // physical NHWC one (e.g. a graph input behind the boundary relayout)
                    int ic = w.Shape[1] * cp.Group;
                    object kp = Get(cache, n, w.Floats!, b, () =>
                    {
                        NhwcConvPlan p = Nhwc.Pack(w.Floats!, w.Shape, b, cp.Group, ic);
                        if (Nhwc.WinogradEligible(cp, w.Shape, ic)) p.WinoU = Nhwc.PackWinograd(w.Floats!, w.Shape[0], ic);
                        return p;
                    });
                    Interlocked.CompareExchange(ref cp.KernelPlan, kp, null);
                    break;
                }
                case ConvTP tp when tp.KernelPlan is null:
                {
                    CValue w = V[n.In[1]];
                    float[]? b = tp.BiasValue >= 0 ? V[tp.BiasValue].Floats : null;
                    object kp = Get(cache, n, w.Floats!, b, () => Nhwc.PackConvT(w.Floats!, w.Shape, b));
                    Interlocked.CompareExchange(ref tp.KernelPlan, kp, null);
                    break;
                }
            }
        }
    }

    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    internal static object Get(Cache? cache, ExecNode n, float[] w, float[]? b, Func<object> pack) =>
        cache is null ? pack() : cache.GetOrPack(n, w, b, pack);
}

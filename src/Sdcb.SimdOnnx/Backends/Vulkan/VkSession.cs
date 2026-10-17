using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Grow-only device arena + host-visible staging + host arrays shared by the sessions of one
/// owner (one shape-varying OCR stage): sessions of different shapes borrow them, so only the
/// largest shape's memory is ever held. Sessions borrowing it must run one at a time.
/// </summary>
internal sealed unsafe class VkSessionBuffers : IDisposable
{
    private readonly VkDevice _dev;
    internal VkBuffer? Arena, StageIn, StageOut;
    internal float* InMap, OutMap;
    internal ulong ArenaBytes, InBytes, OutBytes;
    /// <summary>Bumped whenever a device buffer is replaced (sessions re-record against the new one).</summary>
    internal int Generation;

    internal VkSessionBuffers(VkDevice dev) => _dev = dev;

    internal void Ensure(ulong arena, ulong inBytes, ulong outBytes, int inFloats, int outFloats)
    {
        if (arena > ArenaBytes)
        {
            Arena?.Free();
            ArenaBytes = Math.Max(arena, ArenaBytes + ArenaBytes / 2);
            Arena = _dev.NewStorageBuffer(ArenaBytes, hostVisible: false);
            Generation++;
        }
        if (inBytes > InBytes)
        {
            if (StageIn is not null) { StageIn.Unmap(); StageIn.Free(); }
            InBytes = Math.Max(inBytes, InBytes + InBytes / 2);
            StageIn = _dev.NewStorageBuffer(InBytes, hostVisible: true, preferHost: Env.Flag("VK_STAGE_WC") == "0");
            InMap = (float*)StageIn.Map();
            Generation++;
        }
        if (outBytes > OutBytes)
        {
            if (StageOut is not null) { StageOut.Unmap(); StageOut.Free(); }
            OutBytes = Math.Max(outBytes, OutBytes + OutBytes / 2);
            StageOut = _dev.NewStorageBuffer(OutBytes, hostVisible: true, preferHost: true);
            OutMap = (float*)StageOut.Map();
            Generation++;
        }
    }

    public void Dispose()
    {
        Arena?.Free();
        if (StageIn is not null) { StageIn.Unmap(); StageIn.Free(); }
        if (StageOut is not null) { StageOut.Unmap(); StageOut.Free(); }
        Arena = StageIn = StageOut = null;
    }
}

/// <summary>
/// GPU counterpart of <see cref="InferenceSession"/>: one activation arena in
/// device-local memory, persistent host-visible staging buffers, and a single
/// pre-recorded command buffer (staging copy-in → all dispatches → copy-out).
/// Not thread-safe; one per thread or serialize externally.
/// </summary>
public sealed unsafe class VkInferenceSession : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkGraphModel _graph;
    private readonly ExecPlan _plan;

    internal readonly VkBuffer Arena;     // device-local activation arena
    private readonly VkBuffer _stageIn;   // host-visible float input staging
    private readonly VkBuffer _stageOut;  // host-visible float output staging
    private float* _stageInMap, _stageOutMap;
    private readonly VkSessionBuffers? _shared;   // borrowed buffers (not owned / not freed)
    private readonly int _sharedGen;
    private readonly IntPtr _cmd, _fence;
    private readonly IntPtr[] _descSets;  // per-dispatch sets when !PushDescriptors
    private readonly ulong _inOff, _outOff;
    private readonly int _inFloats, _outFloats;
    private readonly long[] _inSub;       // each input's element offset into the staged region
    private readonly int _es;             // bytes per stored element (2 = fp16 storage)
    internal bool F16 => _es == 2;

    internal VkInferenceSession(VkDevice dev, ExecPlan plan, VkGraphModel graph, VkSessionBuffers? shared = null)
    {
        _dev = dev;
        _plan = plan;
        _graph = graph;
        _graph.Acquire();
        _es = graph.Es;

        // every graph input is staged as one region (inputs are planned back to back)
        (long inFirst, long inCount, _inSub) = plan.InputRegion();
        _inFloats = checked((int)inCount);
        _outFloats = checked((int)plan.Values[plan.OutputValue].ElementCount);
        _inOff = (ulong)inFirst * (ulong)_es;
        _outOff = (ulong)plan.BufOffset[plan.Values[plan.OutputValue].Buf] * (ulong)_es;
        if (shared is not null)
        {
            shared.Ensure(graph.ArenaBytes + 64, StageInBytes, StageOutBytes, _inFloats, _outFloats);
            _shared = shared;
            _sharedGen = shared.Generation;
            Arena = shared.Arena!;
            _stageIn = shared.StageIn!;
            _stageOut = shared.StageOut!;
            _stageInMap = shared.InMap;
            _stageOutMap = shared.OutMap;
        }
        else
        {
            Arena = dev.NewStorageBuffer(graph.ArenaBytes + 64, hostVisible: false);
            _stageIn = dev.NewStorageBuffer(StageInBytes, hostVisible: true, preferHost: Env.Flag("VK_STAGE_WC") == "0");
            _stageOut = dev.NewStorageBuffer(StageOutBytes, hostVisible: true, preferHost: true);
            _stageInMap = (float*)_stageIn.Map();
            _stageOutMap = (float*)_stageOut.Map();
        }
        _fence = dev.NewFence();

        // command pool / descriptor pool are externally synchronized (sessions are built concurrently)
        Monitor.Enter(dev.Sync);
        try
        {
        if (!dev.PushDescriptors)
        {
            _descSets = new IntPtr[_graph.Dispatches.Length];
            for (int i = 0; i < _graph.Dispatches.Length; i++)
            {
                var d = _graph.Dispatches[i];
                IntPtr set = dev.NewDescriptorSet(d.Pipe.SetLayout);
                for (uint b = 0; b < d.Binds.Length; b++)
                {
                    (VkBuffer buf, _, _) = Resolve(d.Binds[b]);
                    dev.BindBuffer(set, b, buf, d.Binds[b].Offset, d.Binds[b].Size);
                }
                _descSets[i] = set;
            }
        }
        else _descSets = [];

        _cmd = dev.NewCommandBuffer();
        Record();
        }
        finally { Monitor.Exit(dev.Sync); }
    }

    internal VkBuffer ResolveBuf(VkBind b) => Resolve(b).buf;
    internal ulong ResolveOff(VkBind b) => b.Offset;

    internal (VkBuffer buf, ulong off, ulong size) Resolve(VkBind b) => b.Slot switch
    {
        VkSlot.Arena => (Arena, b.Offset, b.Size),
        VkSlot.Consts => (_graph.Consts, b.Offset, b.Size),
        VkSlot.StageIn => (_stageIn, b.Offset, b.Size),
        VkSlot.StageOut => (_stageOut, b.Offset, b.Size),
        VkSlot.Direct => (b.Buf!, b.Offset, b.Size),
        _ => (_graph.Dummy, b.Offset, b.Size),
    };

    /// <summary>Input tensor shape (e.g. [1,3,H,W]).</summary>
    public int[] InputShape => _plan.Values[_plan.InputValue].Shape;
    public int[] OutputShape => _plan.Values[_plan.OutputValue].Shape;

    /// <summary>The writable input window: the mapped fp32 staging buffer itself (no host copy).
    /// Do not write it between <see cref="Submit"/> and <see cref="Wait"/>.</summary>
    public Span<float> Input => GetInput(0);
    /// <summary>Number of graph inputs.</summary>
    public int InputCount => _inSub.Length;
    /// <summary>The writable staging window of graph input <paramref name="index"/> (graph order).</summary>
    public Span<float> GetInput(int index) =>
        new(_stageInMap + _inSub[index], checked((int)_plan.Values[_plan.Inputs[index]].ElementCount));
    /// <summary>Logical shape of graph input <paramref name="index"/>.</summary>
    public int[] GetInputShape(int index) => _plan.Values[_plan.Inputs[index]].Shape;
    /// <summary>The output tensor after <see cref="Run"/> (the mapped staging buffer; valid until the next run).</summary>
    public ReadOnlySpan<float> Output => new(_stageOutMap, _outFloats);

    /// <summary>False once the borrowed buffers were regrown by another session (re-create this one).</summary>
    internal bool BuffersCurrent => _shared is null || _shared.Generation == _sharedGen;

    internal ulong InputOffset => _inOff;
    internal ulong OutputOffset => _outOff;
    internal ulong InputBytes => (ulong)_inFloats * (ulong)_es;
    internal ulong OutputBytes => (ulong)_outFloats * (ulong)_es;
    // host staging is always fp32: fp16 sessions convert on the device (f32to16 / f16to32 dispatches)
    private ulong StageInBytes => (ulong)_inFloats * 4UL;
    private ulong StageOutBytes => (ulong)_outFloats * 4UL;
    internal int ElemBytes => _es;

    /// <summary>Host float -> staged storage elements (fp16 narrowing when F16).</summary>
    internal static void ToStorage(ReadOnlySpan<float> src, void* dst, bool f16)
    {
        if (!f16) { src.CopyTo(new Span<float>(dst, src.Length)); return; }
        var d = new Span<Half>(dst, src.Length);
        for (int i = 0; i < src.Length; i++) d[i] = (Half)src[i];
    }

    /// <summary>Stored elements -> host float (fp16 widening when F16).</summary>
    internal static void FromStorage(void* src, Span<float> dst, bool f16)
    {
        if (!f16) { new ReadOnlySpan<float>(src, dst.Length).CopyTo(dst); return; }
        var sp = new ReadOnlySpan<Half>(src, dst.Length);
        for (int i = 0; i < dst.Length; i++) dst[i] = (float)sp[i];
    }

    /// <summary>Downloads dst.Length stored elements at byte offset <paramref name="off"/>.</summary>
    internal void DownloadElems(VkBuffer buf, ulong off, Span<float> dst)
    {
        ulong bytes = (ulong)dst.Length * (ulong)_es;
        byte[] tmp = new byte[bytes];
        fixed (byte* t = tmp)
        {
            _dev.Download(buf, t, bytes, off);
            FromStorage(t, dst, F16);
        }
    }

    /// <summary>Record one dispatch (push-descriptor path only; caller emits barriers).</summary>
    internal static void RecordOne(VkDevice dev, IntPtr cmd, VkPipeline pipe,
        (VkBuffer buf, ulong off, ulong size)[] binds, byte[] pc, uint gx, uint gy, uint gz = 1)
    {
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        for (uint b = 0; b < binds.Length; b++)
        {
            infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = binds[b].size };
            writes[b] = new Vk.VkWriteDescriptorSet
            {
                SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                PBufferInfo = infos + b,
            };
        }
        dev.CmdPushDescriptors(cmd, pipe.Layout, writes, (uint)binds.Length);
        if (pc.Length > 0)
            fixed (byte* p = pc)
                Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, (uint)pc.Length, p);
        Vk.vkCmdDispatch(cmd, gx, gy, gz);
    }

    /// <summary>Record all dispatches with barriers into an open command buffer.</summary>
    internal void RecordCompute(IntPtr cmd)
    {
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        bool first = true;
        for (int i = 0; i < _graph.Dispatches.Length; i++)
        {
            var d = _graph.Dispatches[i];
            if (!first) _dev.CmdComputeBarrier(cmd);
            first = false;
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            if (_dev.PushDescriptors)
            {
                for (uint b = 0; b < d.Binds.Length; b++)
                {
                    (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                    infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                    writes[b] = new Vk.VkWriteDescriptorSet
                    {
                        SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                        DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                        PBufferInfo = infos + b,
                    };
                }
                _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            }
            else
            {
                IntPtr set = _descSets[i];
                Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, d.Pipe.Layout, 0, 1, &set, 0, null);
            }
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
    }

    private void Record()
    {
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");
        if (F16)
        {
            // staging(fp32) -> arena(fp16 input) on the device
            RecordCvt(_cmd, "f32to16", _stageIn, 0, StageInBytes, Arena, _inOff, InputBytes, (uint)_inFloats);
            _dev.CmdComputeBarrier(_cmd);
        }
        else
        {
            // staging -> arena(input)
            Vk.VkBufferCopy cin = new() { Size = InputBytes, DstOffset = _inOff };
            Vk.vkCmdCopyBuffer(_cmd, _stageIn.Buffer, Arena.Buffer, 1, &cin);
            // barrier: transfer-write -> shader-read
            Vk.VkMemoryBarrier mb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
                1, &mb, 0, null, 0, null);
        }
        RecordCompute(_cmd);
        if (F16)
        {
            // arena(fp16 output) -> staging(fp32)
            _dev.CmdComputeBarrier(_cmd);
            RecordCvt(_cmd, "f16to32", Arena, _outOff, OutputBytes, _stageOut, 0, StageOutBytes, (uint)_outFloats);
            Vk.VkMemoryBarrier hb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessHostRead,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageHost, 0,
                1, &hb, 0, null, 0, null);
            Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
            return;
        }
        // barrier: shader-write -> transfer-read
        Vk.VkMemoryBarrier mb2 = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead,
        };
        Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0,
            1, &mb2, 0, null, 0, null);
        Vk.VkBufferCopy cout = new() { Size = OutputBytes, SrcOffset = _outOff };
        Vk.vkCmdCopyBuffer(_cmd, Arena.Buffer, _stageOut.Buffer, 1, &cout);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    private readonly List<IntPtr> _cvtSets = [];

    /// <summary>One element-conversion dispatch (staging &lt;-&gt; arena) into an open command buffer.</summary>
    private void RecordCvt(IntPtr cmd, string shader, VkBuffer src, ulong srcOff, ulong srcBytes,
        VkBuffer dst, ulong dstOff, ulong dstBytes, uint n)
    {
        VkPipeline pipe = _dev.GetPipeline(shader, 2, 4);
        uint groups = (n + 255) / 256, gx = Math.Min(groups, 65535u), gy = (groups + gx - 1) / gx;
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
        if (_dev.PushDescriptors)
        {
            var infos = stackalloc Vk.VkDescriptorBufferInfo[2];
            var writes = stackalloc Vk.VkWriteDescriptorSet[2];
            infos[0] = new Vk.VkDescriptorBufferInfo { Buffer = src.Buffer, Offset = srcOff, Range = Math.Max(srcBytes, 4) };
            infos[1] = new Vk.VkDescriptorBufferInfo { Buffer = dst.Buffer, Offset = dstOff, Range = Math.Max(dstBytes, 4) };
            for (uint b = 0; b < 2; b++)
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer, PBufferInfo = infos + b,
                };
            _dev.CmdPushDescriptors(cmd, pipe.Layout, writes, 2);
        }
        else
        {
            IntPtr set = _dev.NewDescriptorSet(pipe.SetLayout);
            _dev.BindBuffer(set, 0, src, srcOff, Math.Max(srcBytes, 4));
            _dev.BindBuffer(set, 1, dst, dstOff, Math.Max(dstBytes, 4));
            _cvtSets.Add(set);
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        }
        Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, 4, &n);
        Vk.vkCmdDispatch(cmd, gx, gy, 1);
    }

    /// <summary>Debug/test: record a single dispatch list slice and run it.</summary>
    internal void DebugRunRange(int dStart, int dCount)
    {
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        for (int i = dStart; i < dStart + dCount; i++)
        {
            var d = _graph.Dispatches[i];
            if (i > dStart) _dev.CmdComputeBarrier(cmd);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            for (uint b = 0; b < d.Binds.Length; b++)
            {
                (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
    }

    /// <summary>
    /// Debug/test: time each dispatch by submitting it alone (barrier + dispatch) and
    /// wall-timing the fenced submission. Includes ~20-40us launch overhead per dispatch;
    /// accurate enough to rank ms-scale kernels. Returns name→best-of-2 ms.
    /// </summary>
    internal Dictionary<string, double> DebugProfile()
    {
        int n = _graph.Dispatches.Length;
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var writes = stackalloc Vk.VkWriteDescriptorSet[8];
        var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
        var best = new double[n];
        Array.Fill(best, double.MaxValue);
        void RecordDispatch(VkDispatch d)
        {
            _dev.CmdComputeBarrier(cmd);
            Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, d.Pipe.Pipeline);
            for (uint b = 0; b < d.Binds.Length; b++)
            {
                (VkBuffer buf, ulong off, ulong size) = Resolve(d.Binds[b]);
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = buf.Buffer, Offset = off, Range = size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(cmd, d.Pipe.Layout, writes, (uint)d.Binds.Length);
            if (d.Pc.Length > 0)
                fixed (byte* pc = d.Pc)
                    Vk.vkCmdPushConstants(cmd, d.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)d.Pc.Length, pc);
            Vk.vkCmdDispatch(cmd, d.Gx, d.Gy, d.Gz);
        }
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        // warm pass to keep state identical to production (input already in arena)
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        foreach (var d in _graph.Dispatches) RecordDispatch(d);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        // timed passes: GPU timestamps after every dispatch of one production-like submit, min of 5
        var qci = new Vk.VkQueryPoolCreateInfo { SType = VkConst.StQueryPoolCreateInfo, QueryType = VkConst.QueryTypeTimestamp, QueryCount = (uint)n + 1 };
        Vk.Check(Vk.vkCreateQueryPool(_dev.Device, &qci, null, out IntPtr pool), "vkCreateQueryPool");
        var ts = new ulong[n + 1];
        for (int rep = 0; rep < 5; rep++)
        {
            Vk.vkResetCommandBuffer(cmd, 0);
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
            Vk.vkCmdResetQueryPool(cmd, pool, 0, (uint)n + 1);
            _dev.CmdComputeBarrier(cmd);
            Vk.vkCmdWriteTimestamp(cmd, 0x2000u /* BOTTOM_OF_PIPE */, pool, 0);
            for (int i = 0; i < n; i++)
            {
                RecordDispatch(_graph.Dispatches[i]);
                Vk.vkCmdWriteTimestamp(cmd, 0x2000u, pool, (uint)i + 1);
            }
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            _dev.Submit(cmd, fence);
            _dev.WaitFence(fence);
            fixed (ulong* t = ts)
                Vk.Check(Vk.vkGetQueryPoolResults(_dev.Device, pool, 0, (uint)n + 1, (nuint)(ts.Length * 8), t, 8, 1u | 2u /* 64BIT | WAIT */), "vkGetQueryPoolResults");
            for (int i = 0; i < n; i++)
            {
                double ms = (ts[i + 1] - ts[i]) * _dev.TimestampPeriodNs * 1e-6;
                if (ms < best[i]) best[i] = ms;
            }
        }
        Vk.vkDestroyQueryPool(_dev.Device, pool, null);
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
        var agg = new Dictionary<string, double>();
        PerDispatch = new (string, double)[n];
        for (int i = 0; i < n; i++)
        {
            string name = _graph.Dispatches[i].Name ?? $"d{i}";
            PerDispatch[i] = (name, best[i]);
            name = name[..(name.IndexOf('(') is int cut && cut > 0 ? cut : name.Length)];
            agg[name] = agg.TryGetValue(name, out double v) ? v + best[i] : best[i];
        }
        return agg;
    }

    /// <summary>Per-dispatch timings from the last <see cref="DebugProfile"/> call.</summary>
    internal (string Name, double Ms)[]? PerDispatch;

    /// <summary>Debug: the compiled plan backing this session.</summary>
    internal ExecPlan DebugPlan => _plan;

    /// <summary>Debug: download the whole arena (ArenaFloats floats) into dst.</summary>
    internal void DebugArena(Span<float> dst)
    {
        DownloadElems(Arena, 0, dst.Slice(0, _plan.ArenaFloats));
    }

    /// <summary>Debug: graph consts buffer + the session's arena buffer.</summary>
    internal (VkBuffer consts, VkBuffer arena) DebugBuffers => (_graph.Consts, Arena);

    /// <summary>Debug/test: download an activation value's region into dst.</summary>
    internal void DebugDownload(int valueId, float[] dst)
    {
        CValue v = _plan.Values[valueId];
        DownloadElems(Arena, (ulong)_plan.BufOffset[v.Buf] * (ulong)_es, dst.AsSpan(0, checked((int)v.ElementCount)));
    }

    /// <summary>Debug/test: copy host input into the arena's input region.</summary>
    internal void DebugUploadInput()
    {
        float[] host = new Span<float>(_stageInMap, _inFloats).ToArray();
        ToStorage(host, _stageInMap, F16);
        _stageIn.Flush(0, InputBytes);
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        Vk.VkBufferCopy cin = new() { Size = InputBytes, DstOffset = _inOff };
        Vk.vkCmdCopyBuffer(cmd, _stageIn.Buffer, Arena.Buffer, 1, &cin);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        _dev.DestroyFence(fence);
        _dev.FreeCommandBuffer(cmd);
        host.CopyTo(new Span<float>(_stageInMap, _inFloats));
    }

    /// <summary>Copy host input into staging and run the recorded pipeline.</summary>
    public void Run()
    {
        Submit();
        Wait();
    }

    /// <summary>Stage the input and submit without waiting (pair with <see cref="Wait"/>); the host
    /// may meanwhile work on other data, but not touch this session's (or its shared) buffers.</summary>
    internal void Submit()
    {
        if (!BuffersCurrent) throw new InvalidOperationException("Shared Vulkan buffers were regrown; recreate the session.");
        // staging is fp32 in both modes (fp16 sessions convert on the device)
        _stageIn.Flush(0, StageInBytes);
        _dev.Submit(_cmd, _fence);
    }

    /// <summary>Wait for the submitted run and read the output back.</summary>
    internal void Wait()
    {
        _dev.WaitFence(_fence);
        _stageOut.Invalidate(0, StageOutBytes);
    }

    public void Dispose()
    {
        _dev.DestroyFence(_fence);
        lock (_dev.Sync) _dev.FreeCommandBuffer(_cmd);
        // descriptor sets live in the device pool; pool teardown frees them
        if (_shared is null)
        {
            if (_stageInMap != null) { _stageIn.Unmap(); _stageInMap = null; }
            if (_stageOutMap != null) { _stageOut.Unmap(); _stageOutMap = null; }
            Arena.Free(); _stageIn.Free(); _stageOut.Free();
        }
        _graph.Release();
    }
}


using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Runs several graph units (e.g. OCR line batches of different widths) per call, re-recorded
/// every run instead of one pre-recorded session per shape: each unit owns a region of a
/// grow-only arena and of the fp32 host staging buffers, and the units of a wave are recorded
/// level-interleaved (dispatch k of every unit, then one barrier) into one command buffer, so
/// small units fill the GPU together instead of paying a submit + fence round trip each (after
/// SimdPaddleOCR's GpuRunner). Waves are submitted back to back; the caller can consume a
/// finished wave while later ones still run. Not thread-safe.
/// </summary>
internal sealed unsafe class VkUnitRunner : IDisposable
{
    private readonly VkDevice _dev;
    private readonly IntPtr _pool;                 // private command pool: recording happens on the caller's thread
    private readonly List<IntPtr> _cmds = [], _fences = [];
    private IntPtr _descPool;                      // per-run descriptor sets when push descriptors are missing
    private uint _descCap;
    private VkBuffer? _arena, _stIn, _stOut;
    private ulong _arenaBytes, _inBytes, _outBytes;
    private float* _inMap, _outMap;
    private readonly List<Unit> _units = [];
    private readonly List<int> _waveEnd = [];      // exclusive unit index ending each wave
    private bool _laidOut, _inFlight;
    private int _waited;

    private sealed class Unit
    {
        public required VkGraphModel G;
        public ulong ArenaOff, InOff, OutOff;      // bytes into the shared buffers
        public int InFloats, OutFloats;
    }

    internal VkUnitRunner(VkDevice dev)
    {
        _dev = dev;
        _pool = dev.NewCommandPool();
    }

    internal int Count => _units.Count;
    internal int Waves => _waveEnd.Count;

    /// <summary>Drops the previous run's units (it must have been waited for).</summary>
    internal void Clear()
    {
        if (_inFlight) throw new InvalidOperationException("Previous run still in flight.");
        foreach (Unit u in _units) u.G.Release();
        _units.Clear();
        _waveEnd.Clear();
        _laidOut = false;
    }

    /// <summary>Adds a unit over <paramref name="graph"/> (the runner takes its own reference).</summary>
    internal int Add(VkGraphModel graph)
    {
        if (_laidOut) throw new InvalidOperationException("Add units before writing inputs.");
        graph.Acquire();
        ExecPlan plan = graph.Plan;
        _units.Add(new Unit
        {
            G = graph,
            InFloats = checked((int)plan.Values[plan.InputValue].ElementCount),
            OutFloats = checked((int)plan.Values[plan.OutputValue].ElementCount),
        });
        return _units.Count - 1;
    }

    /// <summary>Ends the current wave (the units added so far that are not in a wave yet).</summary>
    internal void Cut()
    {
        int last = _waveEnd.Count == 0 ? 0 : _waveEnd[^1];
        if (_units.Count > last) _waveEnd.Add(_units.Count);
    }

    /// <summary>One past the last unit of wave <paramref name="wave"/>.</summary>
    internal int WaveEnd(int wave) => _waveEnd[wave];

    internal int WaveOf(int unit)
    {
        for (int w = 0; w < _waveEnd.Count; w++)
            if (unit < _waveEnd[w]) return w;
        throw new ArgumentOutOfRangeException(nameof(unit));
    }

    /// <summary>The unit's input window: the mapped fp32 staging itself (no host copy).</summary>
    internal Span<float> Input(int unit)
    {
        LayOut();
        Unit u = _units[unit];
        return new Span<float>((byte*)_inMap + u.InOff, u.InFloats);
    }

    /// <summary>The unit's output after its wave was waited for (valid until the next run).</summary>
    internal ReadOnlySpan<float> Output(int unit)
    {
        Unit u = _units[unit];
        return new ReadOnlySpan<float>((byte*)_outMap + u.OutOff, u.OutFloats);
    }

    internal int[] OutputShape(int unit)
    {
        ExecPlan plan = _units[unit].G.Plan;
        return plan.Values[plan.OutputValue].Shape;
    }

    private void LayOut()
    {
        if (_laidOut) return;
        Cut();
        ulong arena = 0, inB = 0, outB = 0;
        foreach (Unit u in _units)
        {
            u.ArenaOff = arena;
            arena = (arena + u.G.ArenaBytes + 64 + 255) & ~255UL;
            u.InOff = inB;
            inB = (inB + (ulong)u.InFloats * 4 + 255) & ~255UL;
            u.OutOff = outB;
            outB = (outB + (ulong)u.OutFloats * 4 + 255) & ~255UL;
        }
        // grow-only (nothing is in flight here); 1.5x slack keeps varying runs from regrowing often
        if (arena > _arenaBytes)
        {
            _arena?.Free();
            _arenaBytes = Math.Max(arena, _arenaBytes + _arenaBytes / 2);
            _arena = _dev.NewStorageBuffer(_arenaBytes, hostVisible: false);
        }
        if (inB > _inBytes)
        {
            if (_stIn is not null) { _stIn.Unmap(); _stIn.Free(); }
            _inBytes = Math.Max(inB, _inBytes + _inBytes / 2);
            _stIn = _dev.NewStorageBuffer(_inBytes, hostVisible: true, preferHost: Env.Flag("VK_STAGE_WC") == "0");
            _inMap = (float*)_stIn.Map();
        }
        if (outB > _outBytes)
        {
            if (_stOut is not null) { _stOut.Unmap(); _stOut.Free(); }
            _outBytes = Math.Max(outB, _outBytes + _outBytes / 2);
            _stOut = _dev.NewStorageBuffer(_outBytes, hostVisible: true, preferHost: true);
            _outMap = (float*)_stOut.Map();
        }
        _laidOut = true;
    }

    /// <summary>Records and submits every wave (inputs must be written).</summary>
    internal void Submit()
    {
        LayOut();
        if (_units.Count == 0) return;
        while (_cmds.Count < _waveEnd.Count)
        {
            _cmds.Add(_dev.AllocateCommandBuffer(_pool));
            _fences.Add(_dev.NewFence());
        }
        if (!_dev.PushDescriptors) ResetDescriptors();
        _stIn!.Flush(0, _inBytes);
        _waited = 0;
        int first = 0;
        for (int w = 0; w < _waveEnd.Count; w++)
        {
            Record(_cmds[w], first, _waveEnd[w]);
            first = _waveEnd[w];
        }
        _inFlight = true;
        for (int w = 0; w < _waveEnd.Count; w++) _dev.Submit(_cmds[w], _fences[w]);
    }

    /// <summary>Waits for wave <paramref name="wave"/> (and every earlier one).</summary>
    internal void Wait(int wave)
    {
        if (!_inFlight) return;
        for (; _waited <= wave && _waited < _waveEnd.Count; _waited++)
            _dev.WaitFence(_fences[_waited]);
        _stOut!.Invalidate(0, _outBytes);
        if (_waited >= _waveEnd.Count) _inFlight = false;
    }

    internal void WaitAll() => Wait(_waveEnd.Count - 1);

    private void ResetDescriptors()
    {
        uint need = 0;
        foreach (Unit u in _units) need += (uint)u.G.Dispatches.Length + 2;
        if (need > _descCap || _descPool == IntPtr.Zero)
        {
            if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
            _descCap = Math.Max(need, _descCap + _descCap / 2);
            _descPool = _dev.NewDescriptorPool(_descCap);
        }
        else Vk.vkResetDescriptorPool(_dev.Device, _descPool, 0);
    }

    private (VkBuffer buf, ulong off, ulong size) Resolve(Unit u, VkBind b) => b.Slot switch
    {
        VkSlot.Arena => (_arena!, u.ArenaOff + b.Offset, b.Size),
        VkSlot.Consts => (u.G.Consts, b.Offset, b.Size),
        VkSlot.StageIn => (_stIn!, u.InOff + b.Offset, b.Size),
        VkSlot.StageOut => (_stOut!, u.OutOff + b.Offset, b.Size),
        VkSlot.Direct => (b.Buf!, b.Offset, b.Size),
        _ => (u.G.Dummy, b.Offset, b.Size),
    };

    private void Bind(IntPtr cmd, VkPipeline pipe, ReadOnlySpan<(VkBuffer buf, ulong off, ulong size)> binds)
    {
        var infos = stackalloc Vk.VkDescriptorBufferInfo[binds.Length];
        var writes = stackalloc Vk.VkWriteDescriptorSet[binds.Length];
        IntPtr set = _dev.PushDescriptors ? IntPtr.Zero : _dev.AllocateDescriptorSet(_descPool, pipe.SetLayout);
        for (int b = 0; b < binds.Length; b++)
        {
            infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = Math.Max(binds[b].size, 4) };
            writes[b] = new Vk.VkWriteDescriptorSet
            {
                SType = VkConst.StWriteDescriptorSet, DstSet = set, DstBinding = (uint)b,
                DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer, PBufferInfo = infos + b,
            };
        }
        if (_dev.PushDescriptors) _dev.CmdPushDescriptors(cmd, pipe.Layout, writes, (uint)binds.Length);
        else
        {
            Vk.vkUpdateDescriptorSets(_dev.Device, (uint)binds.Length, writes, 0, null);
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        }
    }

    private void Record(IntPtr cmd, int first, int end)
    {
        Vk.vkResetCommandBuffer(cmd, 0);
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo, Flags = 1 /* ONE_TIME_SUBMIT */ };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        var binds = new (VkBuffer, ulong, ulong)[8];

        // staging (fp32) -> each unit's input region (fp16 narrowing on the device)
        bool anyCopy = false;
        for (int i = first; i < end; i++)
        {
            Unit u = _units[i];
            ExecPlan plan = u.G.Plan;
            ulong inOff = u.ArenaOff + (ulong)plan.BufOffset[plan.Values[plan.InputValue].Buf] * (ulong)u.G.Es;
            if (u.G.F16) Convert(cmd, "f32to16", _stIn!, u.InOff, (ulong)u.InFloats * 4, _arena!, inOff, (ulong)u.InFloats * 2, (uint)u.InFloats);
            else
            {
                Vk.VkBufferCopy c = new() { SrcOffset = u.InOff, DstOffset = inOff, Size = (ulong)u.InFloats * 4 };
                Vk.vkCmdCopyBuffer(cmd, _stIn!.Buffer, _arena!.Buffer, 1, &c);
                anyCopy = true;
            }
        }
        Barrier(cmd, anyCopy);

        int levels = 0;
        for (int i = first; i < end; i++) levels = Math.Max(levels, _units[i].G.Dispatches.Length);
        for (int d = 0; d < levels; d++)
        {
            for (int i = first; i < end; i++)
            {
                Unit u = _units[i];
                if (d >= u.G.Dispatches.Length) continue;
                VkDispatch x = u.G.Dispatches[d];
                if (x.Gx == 0 || x.Gy == 0 || x.Gz == 0) continue;
                Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, x.Pipe.Pipeline);
                if (binds.Length < x.Binds.Length) binds = new (VkBuffer, ulong, ulong)[x.Binds.Length];
                for (int b = 0; b < x.Binds.Length; b++) binds[b] = Resolve(u, x.Binds[b]);
                Bind(cmd, x.Pipe, binds.AsSpan(0, x.Binds.Length));
                if (x.Pc.Length > 0)
                    fixed (byte* pc = x.Pc)
                        Vk.vkCmdPushConstants(cmd, x.Pipe.Layout, VkConst.StageComputeShader, 0, (uint)x.Pc.Length, pc);
                Vk.vkCmdDispatch(cmd, x.Gx, x.Gy, x.Gz);
            }
            _dev.CmdComputeBarrier(cmd);
        }

        // each unit's output region -> staging (fp32)
        anyCopy = false;
        for (int i = first; i < end; i++)
        {
            Unit u = _units[i];
            ExecPlan plan = u.G.Plan;
            ulong outOff = u.ArenaOff + (ulong)plan.BufOffset[plan.Values[plan.OutputValue].Buf] * (ulong)u.G.Es;
            if (u.G.F16) Convert(cmd, "f16to32", _arena!, outOff, (ulong)u.OutFloats * 2, _stOut!, u.OutOff, (ulong)u.OutFloats * 4, (uint)u.OutFloats);
            else
            {
                if (!anyCopy)
                {
                    Vk.VkMemoryBarrier tb = new() { SType = VkConst.StMemoryBarrier, SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead };
                    Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0, 1, &tb, 0, null, 0, null);
                }
                Vk.VkBufferCopy c = new() { SrcOffset = outOff, DstOffset = u.OutOff, Size = (ulong)u.OutFloats * 4 };
                Vk.vkCmdCopyBuffer(cmd, _arena!.Buffer, _stOut!.Buffer, 1, &c);
                anyCopy = true;
            }
        }
        Vk.VkMemoryBarrier hb = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite | VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessHostRead,
        };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader | VkConst.PipelineStageTransfer, VkConst.PipelineStageHost, 0,
            1, &hb, 0, null, 0, null);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
    }

    private void Barrier(IntPtr cmd, bool afterCopy)
    {
        if (!afterCopy) { _dev.CmdComputeBarrier(cmd); return; }
        Vk.VkMemoryBarrier mb = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = VkConst.AccessShaderWrite | VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead,
        };
        Vk.vkCmdPipelineBarrier(cmd, VkConst.PipelineStageComputeShader | VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
            1, &mb, 0, null, 0, null);
    }

    private void Convert(IntPtr cmd, string shader, VkBuffer src, ulong srcOff, ulong srcBytes, VkBuffer dst, ulong dstOff, ulong dstBytes, uint n)
    {
        VkPipeline pipe = _dev.GetPipeline(shader, 2, 4);
        uint groups = (n + 255) / 256, gx = Math.Min(groups, 65535u), gy = (groups + gx - 1) / gx;
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);
        Bind(cmd, pipe, [(src, srcOff, srcBytes), (dst, dstOff, dstBytes)]);
        Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, 4, &n);
        if (gx > 0) Vk.vkCmdDispatch(cmd, gx, gy, 1);
    }

    public void Dispose()
    {
        if (_inFlight)
            try { WaitAll(); } catch { /* lost device: abandon */ }
        _inFlight = false;
        Clear();
        foreach (IntPtr f in _fences) _dev.DestroyFence(f);
        _dev.DestroyCommandPool(_pool);
        if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
        _arena?.Free();
        if (_stIn is not null) { _stIn.Unmap(); _stIn.Free(); }
        if (_stOut is not null) { _stOut.Unmap(); _stOut.Free(); }
    }
}

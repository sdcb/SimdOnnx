using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Vulkan;

/// <summary>
/// Image-to-image inference on the GPU: interleaved 8-bit pixels in, the same format out, with the
/// pixel conversion (normalization, padding, cropping, alpha pass-through) done by shaders in the same
/// command buffer as the model, so a frame is one upload of the bytes, one submit and one download.
/// The model maps a [1, 3, H, W] input to a [1, 3, H, W] output (see <see cref="ImageSessionOptions"/>).
/// </summary>
/// <remarks>
/// Any image size: per-size state is only the cached graph. The session owns grow-only device buffers
/// (within <see cref="ArenaBudget"/>) and re-records its command buffer when the size or format
/// changes. Not thread-safe: one run at a time; pool sessions for concurrency. Preview API.
/// </remarks>
public sealed unsafe class VkImageSession : IDisposable
{
    private readonly VkBackend _be;
    private readonly VkDevice _dev;
    private readonly OnnxModel _model;
    private readonly ImageSessionOptions _opt;
    private readonly CompileOptions _compile;
    private readonly IntPtr _cmdPool, _cmd, _fence;
    private IntPtr _descPool;              // !PushDescriptors: rebuilt on each record
    private VkBuffer? _arena, _src, _dst;
    private readonly VkBuffer _lut;
    private byte* _srcMap, _dstMap;
    private VkGraphModel? _g;              // graph of the bound tensor size (reference held)
    private int _h, _w, _ih, _iw, _pt, _pl;
    private PixelFormat _fmt;
    private bool _recorded, _disposed;
    private readonly bool _f16;
    private readonly VkPipeline _pre, _post;

    internal VkImageSession(VkBackend be, OnnxModel model, ImageSessionOptions? options)
    {
        _be = be;
        _dev = be.Dev;
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _opt = options ?? new ImageSessionOptions();
        _ = _opt.Pads(1, 1);   // validates PadMultiple
        _compile = (_opt.Compile ?? VkBackend.PipeOptions) with { Target = PlanTarget.Vulkan, Nhwc = true, PhysicalIo = true, LogicalOutput = false };
        _f16 = be.F16;
        string sfx = _f16 ? "_h" : "";
        _pre = _dev.GetPipeline("img2nhwc" + sfx, 3, 9 * 4);
        _post = _dev.GetPipeline("nhwc2img" + sfx, 3, 10 * 4);
        float[] table = _opt.Normalization.Table;
        _lut = _dev.NewStorageBuffer((ulong)table.Length * 4, hostVisible: true);
        try
        {
            fixed (float* t = table) _dev.Upload(_lut, t, (ulong)table.Length * 4);
            _cmdPool = _dev.NewCommandPool();
            try
            {
                _cmd = _dev.AllocateCommandBuffer(_cmdPool);
                _fence = _dev.NewFence();
            }
            catch
            {
                _dev.DestroyCommandPool(_cmdPool);
                throw;
            }
        }
        catch
        {
            _lut.Free();
            throw;
        }
    }

    /// <summary>The model this session runs.</summary>
    public OnnxModel Model => _model;

    /// <summary>
    /// Device bytes the session's arena may grow to and keep (default: 3 GB; on unified-memory GPUs at
    /// most a quarter of the shared heap). An image that needs more still runs; its arena is replaced by
    /// an exact-size one as soon as a smaller image comes along.
    /// </summary>
    public ulong ArenaBudget => BudgetFor(_be.Device);

    internal static ulong BudgetFor(GpuDeviceInfo d) =>
        d.UnifiedMemory && d.DeviceLocalBytes > 0 ? Math.Min(3UL << 30, d.DeviceLocalBytes / 4) : 3UL << 30;

    /// <summary>Device bytes held now (arena and pixel staging).</summary>
    public long DeviceBytes => (long)((_arena?.Size ?? 0) + (_src?.Size ?? 0) + (_dst?.Size ?? 0) + _lut.Size);

    private int Es => _f16 ? 2 : 4;
    private int Bpp => ImageTensor.BytesPerPixel(_fmt);

    private ulong Off(int value)
    {
        ExecPlan p = _g!.Plan;
        return (ulong)p.BufOffset[p.Values[value].Buf] * (ulong)Es;
    }

    private ulong Bytes(int value) => (ulong)_g!.Plan.Values[value].ElementCount * (ulong)Es;

    /// <summary>Binds an image size and format: the (cached) graph of its tensor size, grown buffers,
    /// a re-recorded command buffer when anything changed.</summary>
    private void Bind(int ih, int iw, PixelFormat fmt)
    {
        if (_recorded && ih == _ih && iw == _iw && fmt == _fmt) return;
        _recorded = false;   // stays false if anything below throws (buffers may be gone)
        var (pt, pl, h, w) = _opt.Pads(ih, iw);
        if (_g is null || h != _h || w != _w)
        {
            VkGraphModel g = _be.AcquireGraph(_model, [1, 3, h, w], _compile);
            ExecPlan p = g.Plan;
            int[] os = p.Values[p.OutputValue].Shape;
            if (p.Inputs.Length != 1 || os.Length != 4 || os[0] != 1 || os[1] != 3 || os[2] != h || os[3] != w)
            {
                g.Release();
                throw new NotSupportedException($"An image session needs a single-input model with a [1, 3, {h}, {w}] output; got [{string.Join(", ", os)}].");
            }
            _g?.Release();
            _g = g;
            _h = h; _w = w;
        }
        _ih = ih; _iw = iw; _pt = pt; _pl = pl; _fmt = fmt;

        ulong need = _g.ArenaBytes + 64;
        ulong budget = ArenaBudget;
        // within the budget the arena only grows; an over-budget one (a huge image) is dropped to the
        // exact size as soon as a size that fits comes along. Headroom (within budget): 1/8 on the
        // first allocation so neighbouring sizes fit, 1/4 on regrowth (a fresh multi-GB arena costs
        // tens of ms to allocate and first-touch)
        bool shrink = _arena is not null && _arena.Size > budget && need <= budget;
        if (_arena is null || _arena.Size < need || shrink)
        {
            ulong div = _arena is null ? 8UL : 4UL;
            ulong size = !shrink && need / div * (div + 1) <= budget ? need / div * (div + 1) : need;
            _arena?.Free();
            _arena = null;
            _arena = _dev.NewStorageBuffer(size, hostVisible: false);
        }
        ulong bytes = (ulong)ih * (ulong)iw * (ulong)Bpp + 4;
        Staging(ref _src, ref _srcMap, bytes, preferHost: false);
        Staging(ref _dst, ref _dstMap, bytes, preferHost: true);
        Record();
        _recorded = true;
    }

    /// <summary>Grow-only mapped staging buffer; the map pointer is cleared with the buffer it points into.</summary>
    private void Staging(ref VkBuffer? buf, ref byte* map, ulong need, bool preferHost)
    {
        if (buf is not null && buf.Size < need)
        {
            if (map != null) buf.Unmap();
            map = null;
            buf.Free();
            buf = null;
        }
        buf ??= _dev.NewStorageBuffer(need + need / 4, hostVisible: true, preferHost);
        if (map == null) map = (byte*)buf.Map();
    }

    private void Record()
    {
        VkGraphModel g = _g!;
        ExecPlan plan = g.Plan;
        if (!_dev.PushDescriptors)
        {
            if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
            _descPool = IntPtr.Zero;
            _descPool = _dev.NewDescriptorPool((uint)(g.Dispatches.Length + 2));
        }
        uint h = (uint)_h, w = (uint)_w, bpp = (uint)Bpp, swap = _opt.Swap(_fmt);
        Vk.Check(Vk.vkResetCommandBuffer(_cmd, 0), "vkResetCommandBuffer");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");

        // pixels -> tensor input
        Dispatch(_pre,
            [(_src!, 0UL, _src!.Size), (_arena!, Off(plan.InputValue), Bytes(plan.InputValue)), (_lut, 0UL, _lut.Size)],
            VkGraphModel.PcOf(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, (uint)_iw * bpp, bpp, swap),
            (w + 255) / 256, h, 1);
        var binds = new (VkBuffer buf, ulong off, ulong size)[8];
        for (int i = 0; i < g.Dispatches.Length; i++)
        {
            VkDispatch d = g.Dispatches[i];
            _dev.CmdComputeBarrier(_cmd);
            if (binds.Length < d.Binds.Length) binds = new (VkBuffer, ulong, ulong)[d.Binds.Length];
            for (int b = 0; b < d.Binds.Length; b++)
            {
                VkBind vb = d.Binds[b];
                VkBuffer buf = vb.Slot switch
                {
                    VkSlot.Arena => _arena!,
                    VkSlot.Consts => g.Consts,
                    VkSlot.Dummy => g.Dummy,
                    VkSlot.Direct => vb.Buf!,
                    _ => throw new InvalidOperationException($"physical-I/O graph dispatch binds {vb.Slot}"),
                };
                binds[b] = (buf, vb.Offset, vb.Size);
            }
            Dispatch(d.Pipe, binds.AsSpan(0, d.Binds.Length), d.Pc, d.Gx, d.Gy, d.Gz);
        }
        // tensor output -> pixels
        _dev.CmdComputeBarrier(_cmd);
        uint nwords = bpp == 4 ? (uint)_ih * (uint)_iw : ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        Dispatch(_post,
            [(_arena!, Off(plan.OutputValue), Bytes(plan.OutputValue)), (_dst!, 0UL, _dst!.Size), (_src!, 0UL, _src!.Size)],
            VkGraphModel.PcOfF([h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords, bpp, swap], _opt.OutputScale),
            (nwords + 255) / 256, 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    private void Dispatch(VkPipeline pipe, ReadOnlySpan<(VkBuffer buf, ulong off, ulong size)> binds,
        byte[] pc, uint gx, uint gy, uint gz)
    {
        Vk.vkCmdBindPipeline(_cmd, VkConst.BindPointCompute, pipe.Pipeline);
        if (_dev.PushDescriptors)
        {
            var writes = stackalloc Vk.VkWriteDescriptorSet[8];
            var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
            for (int b = 0; b < binds.Length; b++)
            {
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = binds[b].size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = (uint)b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(_cmd, pipe.Layout, writes, (uint)binds.Length);
        }
        else
        {
            IntPtr set = _dev.AllocateDescriptorSet(_descPool, pipe.SetLayout);
            for (int b = 0; b < binds.Length; b++)
                _dev.BindBuffer(set, (uint)b, binds[b].buf, binds[b].off, binds[b].size);
            Vk.vkCmdBindDescriptorSets(_cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        }
        if (pc.Length > 0)
            fixed (byte* p = pc)
                Vk.vkCmdPushConstants(_cmd, pipe.Layout, VkConst.StageComputeShader, 0, (uint)pc.Length, p);
        Vk.vkCmdDispatch(_cmd, gx, gy, gz);
    }

    /// <summary>
    /// Runs the model on a <paramref name="width"/> x <paramref name="height"/> image in
    /// <paramref name="format"/> (rows <paramref name="srcStride"/> bytes apart) and writes the result,
    /// same format and size, to <paramref name="dst"/> (rows <paramref name="dstStride"/> bytes apart).
    /// 4-byte formats keep each pixel's alpha. <paramref name="dst"/> may be <paramref name="src"/>.
    /// </summary>
    /// <exception cref="VkDeviceLostException">The device was lost; dispose this session (the backend
    /// re-creates its device on next use).</exception>
    public void Run(ReadOnlySpan<byte> src, int width, int height, int srcStride, Span<byte> dst, int dstStride,
        PixelFormat format = PixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkImageSession));
        int bpp = ImageTensor.BytesPerPixel(format);
        if (width < 1 || height < 1) throw new ArgumentOutOfRangeException(width < 1 ? nameof(width) : nameof(height));
        if (srcStride < width * bpp) throw new ArgumentOutOfRangeException(nameof(srcStride));
        if (dstStride < width * bpp) throw new ArgumentOutOfRangeException(nameof(dstStride));
        if (src.Length < (long)(height - 1) * srcStride + width * bpp)
            throw new ArgumentException($"src holds {src.Length} bytes, {height} rows of stride {srcStride} need more.", nameof(src));
        if (dst.Length < (long)(height - 1) * dstStride + width * bpp)
            throw new ArgumentException($"dst holds {dst.Length} bytes, {height} rows of stride {dstStride} need more.", nameof(dst));
        Bind(height, width, format);

        int row = width * bpp;
        ulong bytes = (ulong)height * (ulong)row;
        if (srcStride == row) src.Slice(0, (int)bytes).CopyTo(new Span<byte>(_srcMap, (int)bytes));
        else
            for (int y = 0; y < height; y++)
                src.Slice(y * srcStride, row).CopyTo(new Span<byte>(_srcMap + (long)y * row, row));
        _src!.Flush(0, bytes);

        _dev.Submit(_cmd, _fence);
        _dev.WaitFence(_fence);
        _dst!.Invalidate(0, bytes);

        if (dstStride == row) new ReadOnlySpan<byte>(_dstMap, (int)bytes).CopyTo(dst);
        else
            for (int y = 0; y < height; y++)
                new ReadOnlySpan<byte>(_dstMap + (long)y * row, row).CopyTo(dst.Slice(y * dstStride));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _dev.DestroyCommandPool(_cmdPool);
        if (_descPool != IntPtr.Zero) { Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null); _descPool = IntPtr.Zero; }
        _dev.DestroyFence(_fence);
        if (_src is not null) { if (_srcMap != null) _src.Unmap(); _src.Free(); _src = null; _srcMap = null; }
        if (_dst is not null) { if (_dstMap != null) _dst.Unmap(); _dst.Free(); _dst = null; _dstMap = null; }
        _arena?.Free(); _arena = null;
        _lut.Free();
        _g?.Release(); _g = null;
        _recorded = false;
    }
}

using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdOnnx.Backends.Metal;

/// <summary>
/// Image-to-image inference on Metal, the counterpart of <see cref="Vulkan.VkImageSession"/>: pixels
/// in, the same format out, with the pixel conversion done by kernels in the same command buffer as the
/// model. All buffers are shared memory, so staging is a copy into a mapped buffer. Any image size;
/// grow-only buffers within <see cref="ArenaBudget"/>. Not thread-safe. Preview API.
/// </summary>
public sealed unsafe class MtlImageSession : IDisposable
{
    private const int Es = 2;              // fp16 storage
    private readonly MetalBackend _be;
    private readonly MtlDevice _dev;
    private readonly OnnxModel _model;
    private readonly ImageSessionOptions _opt;
    private readonly CompileOptions _compile;
    private MtlBuffer? _arena, _src, _dst;
    private readonly MtlBuffer _lut;
    private MtlGraphModel? _g;
    private int _h, _w, _ih, _iw, _pt, _pl;
    private PixelFormat _fmt;
    private bool _bound, _disposed;

    internal MtlImageSession(MetalBackend be, OnnxModel model, ImageSessionOptions? options)
    {
        _be = be;
        _dev = be.Dev;
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _opt = options ?? new ImageSessionOptions();
        _ = _opt.Pads(1, 1);   // validates PadMultiple
        _compile = (_opt.Compile ?? MetalBackend.PipeOptions) with { Target = PlanTarget.Metal, Nhwc = true, PhysicalIo = true, LogicalOutput = false };
        float[] table = _opt.Normalization.Table;
        _lut = _dev.NewBuffer((nuint)(table.Length * 4));
        table.AsSpan().CopyTo(new Span<float>((void*)_lut.Contents, table.Length));
    }

    /// <summary>The model this session runs.</summary>
    public OnnxModel Model => _model;

    /// <summary>Device bytes the arena may grow to and keep (3 GB); a larger image still runs, and its
    /// arena is replaced by an exact-size one when a smaller image comes along.</summary>
    public ulong ArenaBudget => 3UL << 30;

    /// <summary>Device bytes held now (arena and pixel staging).</summary>
    public long DeviceBytes => (long)((_arena?.Length ?? 0) + (_src?.Length ?? 0) + (_dst?.Length ?? 0) + _lut.Length);

    private int Bpp => ImageTensor.BytesPerPixel(_fmt);

    private ulong Off(int value)
    {
        ExecPlan p = _g!.Plan;
        return (ulong)p.BufOffset[p.Values[value].Buf] * Es;
    }

    private void Bind(int ih, int iw, PixelFormat fmt)
    {
        if (_bound && ih == _ih && iw == _iw && fmt == _fmt) return;
        _bound = false;
        var (pt, pl, h, w) = _opt.Pads(ih, iw);
        if (_g is null || h != _h || w != _w)
        {
            MtlGraphModel g = _be.GraphFor(_model, [1, 3, h, w], _compile);
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

        ulong need = (ulong)_g.Plan.ArenaFloats * Es + 64, budget = ArenaBudget;
        bool shrink = _arena is not null && _arena.Length > budget && need <= budget;
        if (_arena is null || _arena.Length < need || shrink)
        {
            ulong div = _arena is null ? 8UL : 4UL;
            ulong size = !shrink && need / div * (div + 1) <= budget ? need / div * (div + 1) : need;
            _arena?.Dispose();
            _arena = null;
            _arena = _dev.NewBuffer((nuint)size);
        }
        ulong bytes = (ulong)ih * (ulong)iw * (ulong)Bpp + 4;
        if (_src is null || _src.Length < bytes) { _src?.Dispose(); _src = null; _src = _dev.NewBuffer((nuint)(bytes + bytes / 4)); }
        if (_dst is null || _dst.Length < bytes) { _dst?.Dispose(); _dst = null; _dst = _dev.NewBuffer((nuint)(bytes + bytes / 4)); }
        _bound = true;
    }

    private void Record(MtlEncoder enc)
    {
        MtlGraphModel g = _g!;
        ExecPlan plan = g.Plan;
        uint h = (uint)_h, w = (uint)_w, bpp = (uint)Bpp, swap = _opt.Swap(_fmt);

        enc.SetPipeline(_dev.Pipe("img2nhwc"));
        enc.SetBuffer(_src!, 0, 0);
        enc.SetBuffer(_arena!, Off(plan.InputValue), 1);
        enc.SetBuffer(_lut, 0, 2);
        var pc = stackalloc uint[9] { h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, (uint)_iw * bpp, bpp, swap };
        enc.SetPc(pc, 9 * 4, 3);
        enc.Dispatch((w + 255) / 256, h, 1, 256, 1, 1);

        foreach (var d in g.Dispatches)
            MtlInferenceSession.RecordOne(enc, g, d, _arena!);

        uint nwords = bpp == 4 ? (uint)_ih * (uint)_iw : ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        enc.SetPipeline(_dev.Pipe("nhwc2img"));
        enc.SetBuffer(_arena!, Off(plan.OutputValue), 0);
        enc.SetBuffer(_dst!, 0, 1);
        enc.SetBuffer(_src!, 0, 2);
        var pc2 = stackalloc uint[10] { h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords, bpp, swap, 0 };
        *(float*)(pc2 + 9) = _opt.OutputScale;
        enc.SetPc(pc2, 10 * 4, 3);
        enc.Dispatch((nwords + 255) / 256, 1, 1, 256, 1, 1);
    }

    /// <summary>Runs the model on an image and writes the result in the same format and size; see
    /// <see cref="Vulkan.VkImageSession.Run"/>.</summary>
    public void Run(ReadOnlySpan<byte> src, int width, int height, int srcStride, Span<byte> dst, int dstStride,
        PixelFormat format = PixelFormat.Bgr24)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(MtlImageSession));
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
        byte* srcMap = (byte*)_src!.Contents, dstMap = (byte*)_dst!.Contents;
        if (srcStride == row) src.Slice(0, height * row).CopyTo(new Span<byte>(srcMap, height * row));
        else
            for (int y = 0; y < height; y++)
                src.Slice(y * srcStride, row).CopyTo(new Span<byte>(srcMap + (long)y * row, row));

        using (AutoReleasePool.Create())
        {
            var enc = _dev.BeginCommands();
            Record(enc);
            enc.End();
            enc.CommitAndWait();
        }

        if (dstStride == row) new ReadOnlySpan<byte>(dstMap, height * row).CopyTo(dst);
        else
            for (int y = 0; y < height; y++)
                new ReadOnlySpan<byte>(dstMap + (long)y * row, row).CopyTo(dst.Slice(y * dstStride));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _arena?.Dispose(); _src?.Dispose(); _dst?.Dispose(); _lut.Dispose();
        _arena = _src = _dst = null;
        _g?.Release(); _g = null;
    }
}

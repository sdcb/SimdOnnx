using System.Buffers;
using System.Runtime.CompilerServices;
using Sdcb.SimdOnnx.Imaging.Kernels;

namespace Sdcb.SimdOnnx.Imaging;

/// <summary>Interleaved 8-bit source / destination pixel layout.</summary>
internal enum PixelFormat
{
    /// <summary>3 bytes, B,G,R (OpenCV / GDI+ 24bpp).</summary>
    Bgr24 = 0,
    /// <summary>3 bytes, R,G,B.</summary>
    Rgb24 = 1,
    /// <summary>4 bytes, B,G,R,A (Skia / GDI+ 32bpp); alpha ignored on input.</summary>
    Bgra32 = 2,
    /// <summary>4 bytes, R,G,B,A (ImageSharp Rgba32); alpha ignored on input.</summary>
    Rgba32 = 3,
}

/// <summary>Memory order of a 3-channel image tensor.</summary>
internal enum TensorLayout
{
    /// <summary>[3][h][w] planes.</summary>
    Nchw = 0,
    /// <summary>[h][w][3] interleaved.</summary>
    Nhwc = 1,
}

/// <summary>Order of the three tensor channels.</summary>
internal enum ChannelOrder
{
    /// <summary>Channel 0 = blue (OpenCV-trained models, PaddleOCR).</summary>
    Bgr = 0,
    /// <summary>Channel 0 = red.</summary>
    Rgb = 1,
}

/// <summary>
/// The per-channel byte → float map of a preprocessing recipe as a 3 x 256 table, so any formula
/// (and its exact floating-point rounding) costs one lookup per element. Immutable and shareable.
/// </summary>
internal sealed class PixelNormalization
{
    private PixelNormalization(float[] table, ChannelOrder order)
    {
        Table = table;
        Order = order;
    }

    /// <summary>[3][256]: value of byte <c>v</c> in tensor channel <c>c</c> at <c>c * 256 + v</c>.</summary>
    internal float[] Table { get; }

    /// <summary>Tensor channel order the table is indexed by.</summary>
    public ChannelOrder Order { get; }

    /// <summary>The float a byte <paramref name="value"/> becomes in tensor channel <paramref name="channel"/>.</summary>
    public float Map(int channel, int value) => Table[checked(channel * 256 + value)];

    /// <summary>Table from any per-channel function of the byte value (its first argument is
    /// the tensor channel, in <paramref name="order"/>; the second the byte value).</summary>
    public static PixelNormalization FromFunction(Func<int, int, float> map, ChannelOrder order = ChannelOrder.Bgr)
    {
        if (map is null) throw new ArgumentNullException(nameof(map));
        float[] t = new float[3 * 256];
        for (int c = 0; c < 3; c++)
            for (int v = 0; v < 256; v++)
                t[c * 256 + v] = map(c, v);
        return new PixelNormalization(t, order);
    }

    /// <summary><c>(v / 255 - mean[c]) / std[c]</c> evaluated in double, then rounded to float
    /// (mean / std in tensor channel order).</summary>
    public static PixelNormalization MeanStd(ReadOnlySpan<double> mean, ReadOnlySpan<double> std, ChannelOrder order = ChannelOrder.Bgr)
    {
        if (mean.Length != 3 || std.Length != 3) throw new ArgumentException("mean and std need 3 values.");
        double[] m = mean.ToArray(), inv = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];
        return FromFunction((c, v) => (float)((v / 255.0 - m[c]) * inv[c]), order);
    }

    /// <summary><c>v / 255f</c> (single-precision division), the [0, 1] scaling.</summary>
    public static PixelNormalization Unit(ChannelOrder order = ChannelOrder.Bgr) => FromFunction((_, v) => v / 255f, order);
}

/// <summary>
/// Image ↔ tensor conversions for model pre- and post-processing: interleaved 8-bit pixels in any
/// <see cref="PixelFormat"/> to NCHW / NHWC floats (optionally resized with OpenCV's exact 8-bit
/// INTER_LINEAR arithmetic, or replicate-padded), and tensors back to pixels.
/// </summary>
internal static unsafe class ImageTensor
{
    /// <summary>Bytes per pixel of <paramref name="format"/>.</summary>
    public static int BytesPerPixel(PixelFormat format) => format switch
    {
        PixelFormat.Bgr24 or PixelFormat.Rgb24 => 3,
        PixelFormat.Bgra32 or PixelFormat.Rgba32 => 4,
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };

    private static bool RedFirst(PixelFormat format) => format is PixelFormat.Rgb24 or PixelFormat.Rgba32;

    /// <summary>
    /// Resizes the image to <paramref name="resizedWidth"/> x <paramref name="resizedHeight"/> exactly
    /// as OpenCV's <c>cv2.resize(..., INTER_LINEAR)</c> does on 8-bit images (11-bit fixed-point
    /// coefficients, x64 SIMD vertical rounding), then maps every byte through
    /// <paramref name="normalization"/> into <paramref name="destination"/> — no intermediate image.
    /// The tensor is <paramref name="tensorWidth"/> wide (default: the resized width); columns past
    /// the resized width are not written (pad them first). Rows split across
    /// <paramref name="threads"/> workers give bit-identical output.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void ResizeLinear(ReadOnlySpan<byte> source, int width, int height, int stride, PixelFormat format,
        int resizedWidth, int resizedHeight, Span<float> destination, TensorLayout layout, PixelNormalization normalization,
        int tensorWidth = 0, int threads = 1)
    {
        Validate(source, width, height, stride, format);
        if (resizedWidth <= 0) throw new ArgumentOutOfRangeException(nameof(resizedWidth));
        if (resizedHeight <= 0) throw new ArgumentOutOfRangeException(nameof(resizedHeight));
        if (tensorWidth == 0) tensorWidth = resizedWidth;
        if (tensorWidth < resizedWidth) throw new ArgumentOutOfRangeException(nameof(tensorWidth));
        if (destination.Length < 3L * resizedHeight * tensorWidth) throw new ArgumentException("destination is too small.", nameof(destination));
        if (normalization is null) throw new ArgumentNullException(nameof(normalization));

        int[] xOffsets = ArrayPool<int>.Shared.Rent(resizedWidth);
        short[] xCoefficients = ArrayPool<short>.Shared.Rent(resizedWidth * 2);
        try
        {
            LinearCoefficients(width, resizedWidth, xOffsets, xCoefficients);
            int workers = Math.Max(1, Math.Min(threads, resizedHeight / MinRowsPerWorker));
            fixed (byte* src = source)
            fixed (float* dst = destination)
            fixed (float* lut = normalization.Table)
            {
                var job = new ResizeJob
                {
                    Source = src, Stride = stride, Width = width, Height = height, Format = format,
                    ResizedWidth = resizedWidth, ResizedHeight = resizedHeight, TensorWidth = tensorWidth,
                    Destination = dst, Lut = lut, Nhwc = layout == TensorLayout.Nhwc,
                    SwapOut = normalization.Order == ChannelOrder.Rgb, XOffsets = xOffsets, XCoefficients = xCoefficients,
                };
                if (workers <= 1) job.Rows(0, 1);
                else
                {
                    // rows interleaved across workers: each row's arithmetic is the serial one
                    ResizeJob shared = job;
                    Parallel.For(0, workers, w => shared.Rows(w, workers));
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(xOffsets);
            ArrayPool<short>.Shared.Return(xCoefficients);
        }
    }

    // below this many rows per worker the split costs more than it saves
    private const int MinRowsPerWorker = 24;

    private struct ResizeJob
    {
        public byte* Source;
        public int Stride, Width, Height, ResizedWidth, ResizedHeight, TensorWidth;
        public PixelFormat Format;
        public float* Destination, Lut;
        public bool Nhwc, SwapOut;
        public int[] XOffsets;
        public short[] XCoefficients;

        [MethodImpl(MethodImplCompat.AggressiveOptimization)]
        public readonly void Rows(int first, int step)
        {
            int[] row0 = ArrayPool<int>.Shared.Rent(ResizedWidth * 3), row1 = ArrayPool<int>.Shared.Rent(ResizedWidth * 3);
            try
            {
                int plane = ResizedHeight * TensorWidth;
                // accumulators are B,G,R; tensor channel c reads accumulator c (BGR) or 2 - c (RGB)
                int a0 = SwapOut ? 2 : 0, a2 = 2 - a0;
                float* l0 = Lut, l1 = Lut + 256, l2 = Lut + 512;
                for (int oy = first; oy < ResizedHeight; oy += step)
                {
                    LinearCoordinate(oy, Height, ResizedHeight, out int sy, out short beta0, out short beta1);
                    int sy0 = Math.Min(Math.Max(sy, 0), Height - 1), sy1 = Math.Min(Math.Max(sy + 1, 0), Height - 1);
                    HorizontalRow(Source + (long)sy0 * Stride, Width, ResizedWidth, XOffsets, XCoefficients, row0, Format);
                    HorizontalRow(Source + (long)sy1 * Stride, Width, ResizedWidth, XOffsets, XCoefficients, row1, Format);
                    int rowStart = oy * TensorWidth;
                    for (int ox = 0; ox < ResizedWidth; ox++)
                    {
                        int o = ox * 3;
                        float v0 = l0[Vertical(row0[o + a0], row1[o + a0], beta0, beta1)];
                        float v1 = l1[Vertical(row0[o + 1], row1[o + 1], beta0, beta1)];
                        float v2 = l2[Vertical(row0[o + a2], row1[o + a2], beta0, beta1)];
                        if (Nhwc)
                        {
                            float* p = Destination + (long)(rowStart + ox) * 3;
                            p[0] = v0; p[1] = v1; p[2] = v2;
                        }
                        else
                        {
                            float* p = Destination + rowStart + ox;
                            p[0] = v0; p[plane] = v1; p[2 * plane] = v2;
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(row0);
                ArrayPool<int>.Shared.Return(row1);
            }
        }
    }

    // VResizeLinearVec_32s8u (OpenCV resize.cpp): the x64 SIMD vertical pass, clamped to a byte
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Vertical(int h0, int h1, short beta0, short beta1)
    {
        int v = (((h0 >> 4) * beta0 >> 16) + ((h1 >> 4) * beta1 >> 16) + 2) >> 2;
        return v < 0 ? 0 : v > 255 ? 255 : v;
    }

    private static void LinearCoefficients(int sourceSize, int destinationSize, int[] offsets, short[] coefficients)
    {
        const int scale = 1 << 11;
        for (int d = 0; d < destinationSize; d++)
        {
            // OpenCV computes the coordinate in softdouble: float can move a coefficient by one at ties
            double coordinate = ((double)d + 0.5) * sourceSize / destinationSize - 0.5;
            int source = checked((int)Math.Floor(coordinate));
            double fraction = coordinate - source;
            if (source < 0) { source = 0; fraction = 0; }
            if (source >= sourceSize - 1) { source = sourceSize - 1; fraction = 0; }
            offsets[d] = source;
            coefficients[d * 2] = CvRound((1 - fraction) * scale);
            coefficients[d * 2 + 1] = CvRound(fraction * scale);
        }
    }

    private static void LinearCoordinate(int destination, int sourceSize, int destinationSize,
        out int source, out short coefficient0, out short coefficient1)
    {
        const int scale = 1 << 11;
        double coordinate = ((double)destination + 0.5) * sourceSize / destinationSize - 0.5;
        source = checked((int)Math.Floor(coordinate));
        double fraction = coordinate - source;
        // vertical edges keep the raw fraction: OpenCV clips the source rows, not the coefficients
        coefficient0 = CvRound((1 - fraction) * scale);
        coefficient1 = CvRound(fraction * scale);
    }

    private static short CvRound(double value) => checked((short)Math.Round(value, MidpointRounding.ToEven));

    // horizontal pass: B,G,R accumulators (value x coefficient, 11-bit) per output pixel
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    private static void HorizontalRow(byte* row, int sourceWidth, int destinationWidth, int[] offsets, short[] coefficients,
        int[] destination, PixelFormat format)
    {
        switch (format)
        {
            case PixelFormat.Bgr24:
                for (int x = 0; x < destinationWidth; x++)
                {
                    int sx = offsets[x], sx1 = Math.Min(sx + 1, sourceWidth - 1);
                    short c0 = coefficients[x * 2], c1 = coefficients[x * 2 + 1];
                    int s0 = sx * 3, s1 = sx1 * 3, d = x * 3;
                    destination[d] = row[s0] * c0 + row[s1] * c1;
                    destination[d + 1] = row[s0 + 1] * c0 + row[s1 + 1] * c1;
                    destination[d + 2] = row[s0 + 2] * c0 + row[s1 + 2] * c1;
                }
                break;
            case PixelFormat.Rgb24:
                PixelRow.GatherRgb24(row, sourceWidth, destinationWidth, offsets, coefficients, destination);
                break;
            default:
                PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, destination, format == PixelFormat.Rgba32);
                break;
        }
    }

    /// <summary>
    /// Pixels → a <paramref name="tensorWidth"/> x <paramref name="tensorHeight"/> tensor at the
    /// image's own scale: the image sits at (<paramref name="padLeft"/>, <paramref name="padTop"/>) and
    /// every other position replicates the nearest edge pixel. Each byte goes through
    /// <paramref name="normalization"/>. Row bands run on up to <paramref name="threads"/> workers.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void FromPixels(ReadOnlySpan<byte> source, int width, int height, int stride, PixelFormat format,
        Span<float> destination, TensorLayout layout, PixelNormalization normalization,
        int tensorWidth = 0, int tensorHeight = 0, int padLeft = 0, int padTop = 0, int threads = 1)
    {
        Validate(source, width, height, stride, format);
        if (tensorWidth == 0) tensorWidth = width;
        if (tensorHeight == 0) tensorHeight = height;
        if (tensorWidth <= 0 || tensorHeight <= 0) throw new ArgumentOutOfRangeException(nameof(tensorWidth));
        if (destination.Length < 3L * tensorWidth * tensorHeight) throw new ArgumentException("destination is too small.", nameof(destination));
        if (normalization is null) throw new ArgumentNullException(nameof(normalization));
        int bpp = BytesPerPixel(format);
        // byte offsets of tensor channels 0 / 2 inside a pixel
        bool swap = RedFirst(format) != (normalization.Order == ChannelOrder.Rgb);
        int o0 = swap ? 2 : 0, o2 = 2 - o0;
        int tw = tensorWidth, th = tensorHeight, plane = tw * th;
        bool nhwc = layout == TensorLayout.Nhwc;
        fixed (byte* sp0 = source)
        fixed (float* dp0 = destination, lp = normalization.Table)
        {
            byte* sp = sp0;
            float* dp = dp0, l0 = lp, l1 = lp + 256, l2 = lp + 512;
            Bands(th, (long)tw * 3, threads, (y0, y1) =>
            {
                for (int y = y0; y < y1; y++)
                {
                    byte* row = sp + (long)Math.Min(Math.Max(y - padTop, 0), height - 1) * stride;
                    for (int x = 0; x < tw; x++)
                    {
                        byte* px = row + Math.Min(Math.Max(x - padLeft, 0), width - 1) * bpp;
                        float v0 = l0[px[o0]], v1 = l1[px[1]], v2 = l2[px[o2]];
                        if (nhwc)
                        {
                            float* p = dp + ((long)y * tw + x) * 3;
                            p[0] = v0; p[1] = v1; p[2] = v2;
                        }
                        else
                        {
                            float* p = dp + (long)y * tw + x;
                            p[0] = v0; p[plane] = v1; p[2 * plane] = v2;
                        }
                    }
                }
            });
        }
    }

    /// <summary>
    /// A tensor back to pixels: the <paramref name="width"/> x <paramref name="height"/> window at
    /// (<paramref name="cropLeft"/>, <paramref name="cropTop"/>) of a <paramref name="tensorWidth"/>
    /// x <paramref name="tensorHeight"/> tensor, each value as <c>clamp((int)(v * scale + 0.5f), 0, 255)</c>.
    /// 4-byte formats take each pixel's alpha from <paramref name="alphaSource"/> (same format and size,
    /// row stride <paramref name="alphaStride"/>; may alias <paramref name="destination"/>), or 255.
    /// </summary>
    [MethodImpl(MethodImplCompat.AggressiveOptimization)]
    public static void ToPixels(ReadOnlySpan<float> source, TensorLayout layout, int tensorWidth, int tensorHeight,
        Span<byte> destination, int width, int height, int stride, PixelFormat format,
        ChannelOrder order = ChannelOrder.Bgr, int cropLeft = 0, int cropTop = 0, float scale = 255f,
        ReadOnlySpan<byte> alphaSource = default, int alphaStride = 0, int threads = 1)
    {
        if (width <= 0 || height <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (cropLeft < 0 || cropTop < 0 || cropLeft + width > tensorWidth || cropTop + height > tensorHeight)
            throw new ArgumentOutOfRangeException(nameof(cropLeft));
        int bpp = BytesPerPixel(format);
        if (source.Length < 3L * tensorWidth * tensorHeight) throw new ArgumentException("source is too small.", nameof(source));
        if (stride < width * bpp || destination.Length < (long)(height - 1) * stride + width * bpp)
            throw new ArgumentException("destination is too small.", nameof(destination));
        bool alpha = bpp == 4 && !alphaSource.IsEmpty;
        if (alpha && alphaSource.Length < (long)(height - 1) * alphaStride + width * 4)
            throw new ArgumentException("alphaSource is too small.", nameof(alphaSource));
        bool swap = RedFirst(format) != (order == ChannelOrder.Rgb);
        int o0 = swap ? 2 : 0, o2 = 2 - o0;
        int tw = tensorWidth, plane = tensorWidth * tensorHeight;
        bool nhwc = layout == TensorLayout.Nhwc;
        fixed (float* sp0 = source)
        fixed (byte* dp0 = destination, ap0 = alphaSource)
        {
            float* sp = sp0;
            byte* dp = dp0, ap = ap0;
            Bands(height, (long)width * 3, threads, (y0, y1) =>
            {
                for (int y = y0; y < y1; y++)
                {
                    byte* drow = dp + (long)y * stride;
                    long t = (long)(y + cropTop) * tw + cropLeft;
                    if (nhwc && bpp == 3 && !swap)
                    {
                        float* r = sp + t * 3;
                        for (int i = 0; i < width * 3; i++) drow[i] = ToByte(r[i], scale);
                        continue;
                    }
                    byte* arow = alpha ? ap + (long)y * alphaStride : null;
                    for (int x = 0; x < width; x++)
                    {
                        float v0, v1, v2;
                        if (nhwc) { float* r = sp + (t + x) * 3; v0 = r[0]; v1 = r[1]; v2 = r[2]; }
                        else { float* r = sp + t + x; v0 = r[0]; v1 = r[plane]; v2 = r[2 * plane]; }
                        byte* d = drow + x * bpp;
                        d[o0] = ToByte(v0, scale);
                        d[1] = ToByte(v1, scale);
                        d[o2] = ToByte(v2, scale);
                        if (bpp == 4) d[3] = alpha ? arow[x * 4 + 3] : (byte)255;
                    }
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte ToByte(float v, float scale)
    {
        int i = (int)(v * scale + 0.5f);
        return (byte)(i < 0 ? 0 : i > 255 ? 255 : i);
    }

    // body(r0, r1) over [0, rows) in bands of at least ~64K elements
    private static void Bands(int rows, long elementsPerRow, int threads, Action<int, int> body)
    {
        int bands = (int)Math.Min(Math.Max(rows * elementsPerRow / 65536, 1), Math.Min(rows, 4L * Math.Max(threads, 1)));
        if (threads <= 1 || bands <= 1) { body(0, rows); return; }
        Parallel.For(0, bands, new ParallelOptions { MaxDegreeOfParallelism = threads },
            b => body((int)((long)rows * b / bands), (int)((long)rows * (b + 1) / bands)));
    }

    private static void Validate(ReadOnlySpan<byte> source, int width, int height, int stride, PixelFormat format)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        int bpp = BytesPerPixel(format);
        if (stride < (long)width * bpp) throw new ArgumentException("stride is too small.", nameof(stride));
        if ((long)(height - 1) * stride + (long)width * bpp > source.Length) throw new ArgumentException("source is too small.", nameof(source));
    }
}

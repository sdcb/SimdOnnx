using Sdcb.SimdOnnx.Imaging;

namespace Sdcb.SimdOnnx.Tests;

/// <summary>
/// <see cref="ImageTensor"/> against straightforward per-pixel references (OpenCV's 8-bit
/// INTER_LINEAR fixed-point arithmetic, replicate padding, rounding back to bytes). Every SIMD
/// gather tier runs through the same tests in CI's ISA matrix.
/// </summary>
public class ImageTensorTests
{
    private static readonly PixelNormalization Unit = PixelNormalization.Unit();

    [Theory]
    [InlineData(PixelFormat.Bgr24)]
    [InlineData(PixelFormat.Rgb24)]
    [InlineData(PixelFormat.Bgra32)]
    [InlineData(PixelFormat.Rgba32)]
    public void ResizeLinear_MatchesReference(object formatArg)
    {
        var format = (PixelFormat)formatArg;
        var rng = new Random((int)format + 1);
        int bpp = ImageTensor.BytesPerPixel(format);
        var norm = PixelNormalization.MeanStd([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]);
        for (int it = 0; it < 25; it++)
        {
            int w = rng.Next(1, 300), h = rng.Next(1, 200), stride = w * bpp + rng.Next(0, 9);
            byte[] src = new byte[stride * h];
            rng.NextBytes(src);
            int rw = rng.Next(1, 400), rh = rng.Next(1, 300), tw = rw + rng.Next(0, 17);
            float[] want = Reference(src, w, h, stride, format, rw, rh, tw, norm);
            foreach (TensorLayout layout in new[] { TensorLayout.Nchw, TensorLayout.Nhwc })
                foreach (int threads in new[] { 1, 4 })
                {
                    float[] got = new float[want.Length];
                    ImageTensor.ResizeLinear(src, w, h, stride, format, rw, rh, got, layout, norm, tw, threads);
                    if (layout == TensorLayout.Nhwc) got = NhwcToNchw(got, rh, tw);
                    for (int i = 0; i < want.Length; i++)
                        if (want[i] != got[i])
                            Assert.Fail($"{format} {w}x{h} -> {rw}x{rh} (tensor {tw}) {layout} t{threads}: [{i}] {got[i]} != {want[i]}");
                }
        }
    }

    [Fact]
    public void ResizeLinear_IdentityKeepsBytes_RgbOrderSwaps()
    {
        // same size: every coefficient is (2048, 0), so the bytes pass through unchanged
        byte[] bgr = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        float[] t = new float[12];
        ImageTensor.ResizeLinear(bgr, 2, 2, 6, PixelFormat.Bgr24, 2, 2, t, TensorLayout.Nhwc,
            PixelNormalization.FromFunction((_, v) => v));
        Assert.Equal(bgr.Select(b => (float)b), t);
        ImageTensor.ResizeLinear(bgr, 2, 2, 6, PixelFormat.Bgr24, 2, 2, t, TensorLayout.Nhwc,
            PixelNormalization.FromFunction((_, v) => v, ChannelOrder.Rgb));
        Assert.Equal(new float[] { 30, 20, 10, 60, 50, 40, 90, 80, 70, 120, 110, 100 }, t);
    }

    [Theory]
    [InlineData(PixelFormat.Bgr24, ChannelOrder.Bgr)]
    [InlineData(PixelFormat.Rgba32, ChannelOrder.Bgr)]
    [InlineData(PixelFormat.Bgra32, ChannelOrder.Rgb)]
    [InlineData(PixelFormat.Rgb24, ChannelOrder.Rgb)]
    public void FromPixels_ReplicatePads(object formatArg, object orderArg)
    {
        var format = (PixelFormat)formatArg;
        var order = (ChannelOrder)orderArg;
        var rng = new Random(5);
        int bpp = ImageTensor.BytesPerPixel(format), w = 37, h = 23, stride = w * bpp + 3;
        byte[] src = new byte[stride * h];
        rng.NextBytes(src);
        int tw = 64, th = 32, left = 9, top = 4;
        var norm = PixelNormalization.Unit(order);
        float[] nchw = new float[3 * tw * th], nhwc = new float[nchw.Length];
        ImageTensor.FromPixels(src, w, h, stride, format, nchw, TensorLayout.Nchw, norm, tw, th, left, top, threads: 4);
        ImageTensor.FromPixels(src, w, h, stride, format, nhwc, TensorLayout.Nhwc, norm, tw, th, left, top);
        bool redFirst = format is PixelFormat.Rgb24 or PixelFormat.Rgba32;
        for (int y = 0; y < th; y++)
            for (int x = 0; x < tw; x++)
            {
                int sy = Math.Clamp(y - top, 0, h - 1), sx = Math.Clamp(x - left, 0, w - 1);
                for (int c = 0; c < 3; c++)
                {
                    // tensor channel c holds red when (RGB order, c == 0) or (BGR order, c == 2)
                    bool red = c != 1 && (order == ChannelOrder.Rgb) == (c == 0);
                    int byteIndex = c == 1 ? 1 : red == redFirst ? 0 : 2;
                    float want = src[sy * stride + sx * bpp + byteIndex] / 255f;
                    Assert.Equal(want, nchw[(c * th + y) * tw + x]);
                    Assert.Equal(want, nhwc[(y * tw + x) * 3 + c]);
                }
            }
    }

    [Theory]
    [InlineData(PixelFormat.Bgr24)]
    [InlineData(PixelFormat.Rgb24)]
    [InlineData(PixelFormat.Bgra32)]
    [InlineData(PixelFormat.Rgba32)]
    public void ToPixels_RoundTripsAndCopiesAlpha(object formatArg)
    {
        var format = (PixelFormat)formatArg;
        var rng = new Random(9);
        int bpp = ImageTensor.BytesPerPixel(format), w = 41, h = 19, stride = w * bpp + 5;
        byte[] src = new byte[stride * h];
        rng.NextBytes(src);
        int tw = 48, th = 32, left = 3, top = 6;
        foreach (TensorLayout layout in new[] { TensorLayout.Nchw, TensorLayout.Nhwc })
        {
            float[] t = new float[3 * tw * th];
            ImageTensor.FromPixels(src, w, h, stride, format, t, layout, Unit, tw, th, left, top);
            byte[] back = new byte[src.Length];
            ImageTensor.ToPixels(t, layout, tw, th, back, w, h, stride, format, cropLeft: left, cropTop: top,
                alphaSource: bpp == 4 ? src : default, alphaStride: stride, threads: 4);
            for (int y = 0; y < h; y++)
                Assert.True(src.AsSpan(y * stride, w * bpp).SequenceEqual(back.AsSpan(y * stride, w * bpp)), $"{layout} row {y}");
        }
        // clamping, rounding, alpha 255 without a source
        float[] edge = [-0.5f, 0.4f / 255f, 0.6f / 255f, 2f, 254.4f / 255f, 254.6f / 255f];
        byte[] px = new byte[2 * bpp];
        ImageTensor.ToPixels(edge, TensorLayout.Nhwc, 2, 1, px, 2, 1, 2 * bpp, format);
        byte[] want = [0, 0, 1, 255, 254, 255];
        bool swap = format is PixelFormat.Rgb24 or PixelFormat.Rgba32;
        for (int x = 0; x < 2; x++)
        {
            Assert.Equal(want[x * 3 + (swap ? 2 : 0)], px[x * bpp]);
            Assert.Equal(want[x * 3 + 1], px[x * bpp + 1]);
            Assert.Equal(want[x * 3 + (swap ? 0 : 2)], px[x * bpp + 2]);
            if (bpp == 4) Assert.Equal(255, px[x * bpp + 3]);
        }
    }

    [Fact]
    public void Gather32_IsaMatchesScalar()
    {
        int sourceWidth = 37, destinationWidth = 64;
        byte[] row = new byte[sourceWidth * 4];
        int[] offsets = new int[destinationWidth];
        short[] coefficients = new short[destinationWidth * 2];
        var rng = new Random(7);
        for (int i = 0; i < row.Length; i++) row[i] = (byte)rng.Next(256);
        for (int x = 0; x < destinationWidth; x++)
        {
            offsets[x] = rng.Next(0, sourceWidth);
            short c0 = (short)rng.Next(0, 2049);
            coefficients[x * 2] = c0;
            coefficients[x * 2 + 1] = (short)(2048 - c0);
        }
        int[] scalar = new int[destinationWidth * 3];
        int[] dispatched = new int[destinationWidth * 3];
        Sdcb.SimdOnnx.Imaging.Kernels.PixelRow.Gather32Scalar(row, sourceWidth, destinationWidth, offsets, coefficients, scalar, rgba: true);
        Sdcb.SimdOnnx.Imaging.Kernels.PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, dispatched, rgba: true);
        Assert.Equal(scalar, dispatched);
        Sdcb.SimdOnnx.Imaging.Kernels.PixelRow.Gather32Scalar(row, sourceWidth, destinationWidth, offsets, coefficients, scalar, rgba: false);
        Sdcb.SimdOnnx.Imaging.Kernels.PixelRow.Gather32(row, sourceWidth, destinationWidth, offsets, coefficients, dispatched, rgba: false);
        Assert.Equal(scalar, dispatched);
    }

    // OpenCV cv2.resize INTER_LINEAR on 8-bit images, per output pixel (resize.cpp: 11-bit
    // coefficients; x64 SIMD vertical pass VResizeLinearVec_32s8u), then the table; NCHW, B,G,R
    private static float[] Reference(byte[] src, int w, int h, int stride, PixelFormat format, int rw, int rh, int tw,
        PixelNormalization norm)
    {
        int bpp = ImageTensor.BytesPerPixel(format);
        bool redFirst = format is PixelFormat.Rgb24 or PixelFormat.Rgba32;
        float[] o = new float[3 * rh * tw];
        for (int oy = 0; oy < rh; oy++)
        {
            double fy = (oy + 0.5) * h / rh - 0.5;
            int sy = (int)Math.Floor(fy);
            double ay = fy - sy;
            short b0 = (short)Math.Round((1 - ay) * 2048, MidpointRounding.ToEven), b1 = (short)Math.Round(ay * 2048, MidpointRounding.ToEven);
            int y0 = Math.Clamp(sy, 0, h - 1), y1 = Math.Clamp(sy + 1, 0, h - 1);
            for (int ox = 0; ox < rw; ox++)
            {
                double fx = (ox + 0.5) * w / rw - 0.5;
                int sx = (int)Math.Floor(fx);
                double ax = fx - sx;
                if (sx < 0) { sx = 0; ax = 0; }
                if (sx >= w - 1) { sx = w - 1; ax = 0; }
                short a0 = (short)Math.Round((1 - ax) * 2048, MidpointRounding.ToEven), a1 = (short)Math.Round(ax * 2048, MidpointRounding.ToEven);
                int x1 = Math.Min(sx + 1, w - 1);
                for (int c = 0; c < 3; c++)   // c: 0 = blue, 2 = red
                {
                    int bi = c == 1 ? 1 : (c == 0) != redFirst ? 0 : 2;
                    int H(int y) => src[y * stride + sx * bpp + bi] * a0 + src[y * stride + x1 * bpp + bi] * a1;
                    int v = (((H(y0) >> 4) * b0 >> 16) + ((H(y1) >> 4) * b1 >> 16) + 2) >> 2;
                    v = Math.Clamp(v, 0, 255);
                    int tc = norm.Order == ChannelOrder.Rgb ? 2 - c : c;
                    o[(tc * rh + oy) * tw + ox] = norm.Map(tc, v);
                }
            }
        }
        return o;
    }

    private static float[] NhwcToNchw(float[] t, int h, int w)
    {
        float[] o = new float[t.Length];
        for (int i = 0; i < h * w; i++)
            for (int c = 0; c < 3; c++) o[c * h * w + i] = t[i * 3 + c];
        return o;
    }
}

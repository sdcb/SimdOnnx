using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdOnnx.Tests;

namespace Sdcb.SimdOnnx.PublicApiTests;

/// <summary><see cref="VkImageSession"/> / <see cref="MtlImageSession"/> against the CPU pixel path (<see cref="ImageTensor"/>).</summary>
public class ImageSessionTests
{
    // y[c] = x[c] * k[c]: tells the channels apart, so a channel-order mix-up shows
    private static readonly float[] K = [1f, 0.5f, 0.25f];
    private static OnnxModel ScaleModel() => OnnxModel.Load(new OnnxBuilder().Opset(18).Input("x", 1, 3, -1, -1)
        .Init("k", K, 1, 3, 1, 1).Node("Mul", ["x", "k"], ["y"]).Output("y").Build());

    private static byte[] Reference(byte[] src, int w, int h, int stride, PixelFormat f, ImageSessionOptions o)
    {
        var (pt, pl, th, tw) = PadsOf(o, h, w);
        float[] t = new float[3 * th * tw];
        ImageTensor.FromPixels(src, w, h, stride, f, t, TensorLayout.Nhwc, o.Normalization, tw, th, pl, pt);
        for (int i = 0; i < t.Length; i++) t[i] *= K[i % 3];
        byte[] dst = new byte[src.Length];
        ImageTensor.ToPixels(t, TensorLayout.Nhwc, tw, th, dst, w, h, stride, f, o.Normalization.Order,
            cropLeft: pl, cropTop: pt, scale: o.OutputScale, alphaSource: src, alphaStride: stride);
        return dst;
    }

    private static (int pt, int pl, int h, int w) PadsOf(ImageSessionOptions o, int h, int w)
    {
        int m = o.PadMultiple, ph = (m - h % m) % m, pw = (m - w % m) % m;
        return o.Padding == ImagePadding.TopLeft ? (ph, pw, h + ph, w + pw) : (0, 0, h + ph, w + pw);
    }

    [SkippableTheory]
    [InlineData(PixelFormat.Bgr24, ChannelOrder.Bgr, ImagePadding.TopLeft, 32)]
    [InlineData(PixelFormat.Rgb24, ChannelOrder.Bgr, ImagePadding.BottomRight, 8)]
    [InlineData(PixelFormat.Bgra32, ChannelOrder.Rgb, ImagePadding.TopLeft, 32)]
    [InlineData(PixelFormat.Rgba32, ChannelOrder.Rgb, ImagePadding.BottomRight, 1)]
    public void Run_MatchesTheCpuPixelPath(PixelFormat format, ChannelOrder order, ImagePadding padding, int multiple)
    {
        Skip.IfNot(VkBackend.TryGetDefault(out VkBackend? vk), "no Vulkan device");
        var o = new ImageSessionOptions { Normalization = PixelNormalization.Unit(order), Padding = padding, PadMultiple = multiple };
        using VkImageSession s = vk!.CreateImageSession(ScaleModel(), o);
        Check((src, w, h, stride, dst) => s.Run(src, w, h, stride, dst, stride, format), format, o);
        Assert.True(s.DeviceBytes > 0);
    }

    [SkippableTheory]
    [InlineData(PixelFormat.Bgr24, ChannelOrder.Bgr, ImagePadding.TopLeft, 32)]
    [InlineData(PixelFormat.Rgb24, ChannelOrder.Bgr, ImagePadding.BottomRight, 8)]
    [InlineData(PixelFormat.Bgra32, ChannelOrder.Rgb, ImagePadding.TopLeft, 32)]
    [InlineData(PixelFormat.Rgba32, ChannelOrder.Rgb, ImagePadding.BottomRight, 1)]
    public void Metal_Run_MatchesTheCpuPixelPath(PixelFormat format, ChannelOrder order, ImagePadding padding, int multiple)
    {
        Skip.IfNot(MetalBackend.TryGetDefault(out MetalBackend? mb), "no Metal device");
        var o = new ImageSessionOptions { Normalization = PixelNormalization.Unit(order), Padding = padding, PadMultiple = multiple };
        using MtlImageSession s = mb!.CreateImageSession(ScaleModel(), o);
        Check((src, w, h, stride, dst) => s.Run(src, w, h, stride, dst, stride, format), format, o);
        Assert.True(s.DeviceBytes > 0);
    }

    private delegate void RunImage(byte[] src, int w, int h, int stride, byte[] dst);

    private static void Check(RunImage run, PixelFormat format, ImageSessionOptions o)
    {
        int multiple = o.PadMultiple;
        int bpp = ImageTensor.BytesPerPixel(format);
        var rng = new Random((int)format * 7 + multiple);
        // odd sizes, padded strides, and a size change (re-record) on the same session
        foreach (var (w, h) in new[] { (37, 21), (64, 32), (5, 3), (37, 21) })
        {
            int stride = w * bpp + 7;
            byte[] src = new byte[stride * h];
            rng.NextBytes(src);
            byte[] got = new byte[src.Length];
            run(src, w, h, stride, got);
            byte[] want = Reference(src, w, h, stride, format, o);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w * bpp; x++)
                {
                    int i = y * stride + x;
                    // fp16 storage: one level at most
                    Assert.True(Math.Abs(got[i] - want[i]) <= 1, $"{w}x{h} ({x / bpp},{y}) byte {x % bpp}: {got[i]} vs {want[i]}");
                }
        }
    }

    [SkippableFact]
    public void Run_RejectsAModelThatIsNotImageToImage()
    {
        Skip.IfNot(VkBackend.TryGetDefault(out VkBackend? vk), "no Vulkan device");
        OnnxModel pool = OnnxModel.Load(new OnnxBuilder().Opset(18).Input("x", 1, 3, -1, -1)
            .Node("GlobalAveragePool", ["x"], ["y"]).Output("y").Build());
        using VkImageSession s = vk!.CreateImageSession(pool);
        byte[] img = new byte[8 * 8 * 3];
        Assert.Throws<NotSupportedException>(() => s.Run(img, 8, 8, 24, img, 24));
    }
}

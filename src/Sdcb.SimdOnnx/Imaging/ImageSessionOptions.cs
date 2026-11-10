using Sdcb.SimdOnnx.Engine;

namespace Sdcb.SimdOnnx.Imaging;

/// <summary>Where the padding of a GPU image session goes when the image is rounded up to
/// <see cref="ImageSessionOptions.PadMultiple"/>.</summary>
public enum ImagePadding
{
    /// <summary>The image sits at the bottom right; the rows above and columns left of it repeat its edge.</summary>
    TopLeft,
    /// <summary>The image sits at the top left; the rows below and columns right of it repeat its edge.</summary>
    BottomRight,
}

/// <summary>
/// How a GPU image session (Vulkan <c>VkImageSession</c>, Metal <c>MtlImageSession</c>) turns pixels into
/// the model's [1, 3, H, W] input and its [1, 3, H, W] output back into pixels. Preview API.
/// </summary>
public sealed record ImageSessionOptions
{
    /// <summary>Byte → float map of the input, and the tensor channel order of input and output
    /// (default: <see cref="PixelNormalization.Unit"/>, B,G,R in [0, 1]).</summary>
    public PixelNormalization Normalization { get; init; } = PixelNormalization.Unit();

    /// <summary>Output value → byte: <c>clamp((int)(v * OutputScale + 0.5f), 0, 255)</c>.</summary>
    public float OutputScale { get; init; } = 255f;

    /// <summary>The tensor's height and width are the image's rounded up to a multiple of this.</summary>
    public int PadMultiple { get; init; } = 1;

    /// <summary>Where the padding goes.</summary>
    public ImagePadding Padding { get; init; } = ImagePadding.BottomRight;

    /// <summary>Compile options of the graph (target, NHWC and physical I/O are forced); null = the backend's default.</summary>
    public CompileOptions? Compile { get; init; }

    /// <summary>Validated (pad top, pad left, tensor height, tensor width) of an image.</summary>
    internal (int pt, int pl, int h, int w) Pads(int height, int width)
    {
        if (PadMultiple < 1) throw new ArgumentOutOfRangeException(nameof(PadMultiple));
        int ph = (PadMultiple - height % PadMultiple) % PadMultiple, pw = (PadMultiple - width % PadMultiple) % PadMultiple;
        return Padding == ImagePadding.TopLeft ? (ph, pw, height + ph, width + pw) : (0, 0, height + ph, width + pw);
    }

    /// <summary>1 when tensor channel 0 is byte 2 of a pixel in <paramref name="format"/>.</summary>
    internal uint Swap(PixelFormat format)
    {
        bool rgbBytes = format is PixelFormat.Rgb24 or PixelFormat.Rgba32;
        return rgbBytes != (Normalization.Order == ChannelOrder.Rgb) ? 1u : 0u;
    }
}

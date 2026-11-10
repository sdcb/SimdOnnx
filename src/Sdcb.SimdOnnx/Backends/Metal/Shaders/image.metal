// ---- image session I/O: interleaved u8 (3 / 4 bytes per pixel) <-> NHWC [h][w][3] ----
struct PcImgIn { uint oh, ow, ih, iw, pt, pl, stride, bpp, swap; };
static inline uint imgbyte(device const uint* s, uint i) { return (s[i >> 2] >> ((i & 3u) * 8u)) & 0xFFu; }
// u8 -> tensor through the normalization table lut[c * 256 + v], replicate padding (image at (pt, pl))
kernel void img2nhwc(device const uint* src [[buffer(0)]],
                     device half* o [[buffer(1)]],
                     device const float* lut [[buffer(2)]],
                     constant PcImgIn& p [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]])
{
    uint ox = gid.x;
    if (ox >= p.ow) return;
    uint oy = gid.y;
    uint sy = uint(clamp(int(oy) - int(p.pt), 0, int(p.ih) - 1));
    uint sx = uint(clamp(int(ox) - int(p.pl), 0, int(p.iw) - 1));
    uint s = sy * p.stride + sx * p.bpp, d = (oy * p.ow + ox) * 3u;
    uint ob = p.swap * 2u;
    o[d] = half(lut[imgbyte(src, s + ob)]);
    o[d + 1u] = half(lut[256u + imgbyte(src, s + 1u)]);
    o[d + 2u] = half(lut[512u + imgbyte(src, s + 2u - ob)]);
}

struct PcImgOut { uint h, w, ih, iw, pt, pl, nwords, bpp, swap; float scale; };
static inline uint imgu8(device const half* x, uint i, float scale)
{
    return uint(clamp(float(x[i]) * scale + 0.5f, 0.0f, 255.0f));
}
// tensor -> tightly packed u8, the ih x iw window at (pt, pl); one thread packs 4 bytes; 4 bpp takes
// alpha from the (tightly packed, same format) source upload
kernel void nhwc2img(device const half* x [[buffer(0)]],
                     device uint* o [[buffer(1)]],
                     device const uint* a [[buffer(2)]],
                     constant PcImgOut& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nwords) return;
    uint ob = p.swap * 2u;
    if (p.bpp == 4u)
    {
        uint dy = gid / p.iw, dx = gid - dy * p.iw;
        uint i = ((dy + p.pt) * p.w + dx + p.pl) * 3u;
        o[gid] = imgu8(x, i + ob, p.scale) | (imgu8(x, i + 1u, p.scale) << 8u)
               | (imgu8(x, i + 2u - ob, p.scale) << 16u) | (a[gid] & 0xFF000000u);
        return;
    }
    uint byteIdx = gid * 4u;
    uint byteTotal = p.ih * p.iw * 3u;
    uint px = byteIdx / 3u, c = byteIdx - px * 3u;
    uint dy = px / p.iw, dx = px - dy * p.iw;
    uint packed = 0u;
    for (uint j = 0u; j < 4u; j++)
    {
        uint val = 0u;
        if (byteIdx < byteTotal)
        {
            uint ch = c == 1u ? 1u : (c == 0u ? ob : 2u - ob);
            val = imgu8(x, ((dy + p.pt) * p.w + dx + p.pl) * 3u + ch, p.scale);
        }
        packed |= val << (j * 8u);
        byteIdx++;
        if (++c == 3u) { c = 0u; if (++dx == p.iw) { dx = 0u; dy++; } }
    }
    o[gid] = packed;
}

// Elementwise / shape kernels. All activation storage is fp16
// (half/half4); math fp32 — see common.metal.

// ---- unary: mode 0 relu, 1 sigmoid, 2 erf, 3 gelu, 4 copy, 5 hardswish, 6 swish ----
struct PcUnary { uint n, mode; };
kernel void unary(device const half* x [[buffer(0)]],
                  device half* o [[buffer(1)]],
                  constant PcUnary& p [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float v = float(x[gid]);
    float r;
    if (p.mode == 0u) r = max(v, 0.0f);
    else if (p.mode == 1u) r = 1.0f / (1.0f + fexp1(-v));
    else if (p.mode == 2u) r = ferf1(v);
    else if (p.mode == 3u) r = 0.5f * v * (1.0f + ferf1(v * 0.7071067811865475f));
    else if (p.mode == 5u) r = v * clamp(v + 3.0f, 0.0f, 6.0f) * (1.0f / 6.0f);
    else if (p.mode == 6u) r = v / (1.0f + fexp1(-v));
    else r = v;
    o[gid] = half(r);
}

// ---- ew2: elementwise binary + act, rank<=4 broadcast ----
struct PcEw2 {
    uint n, s0, s1, s2;
    uint as0, as1, as2, as3;
    uint bs0, bs1, bs2, bs3;
    uint op, act;
};
kernel void ew2(device const half* a [[buffer(0)]],
                device const half* b [[buffer(1)]],
                device half* o [[buffer(2)]],
                constant PcEw2& p [[buffer(3)]],
                uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.s0, r0 = gid % p.s0;
    uint d1 = r0 / p.s1, r1 = r0 % p.s1;
    uint d2 = r1 / p.s2, d3 = r1 % p.s2;
    float va = float(a[d0 * p.as0 + d1 * p.as1 + d2 * p.as2 + d3 * p.as3]);
    float vb = float(b[d0 * p.bs0 + d1 * p.bs1 + d2 * p.bs2 + d3 * p.bs3]);
    float v = p.op == 0u ? va + vb : p.op == 1u ? va - vb : p.op == 2u ? va * vb : va / vb;
    if (p.act == 1u) v = max(v, 0.0f);
    else if (p.act == 2u) v = 1.0f / (1.0f + exp(-v));
    o[gid] = half(v);
}

// ---- ew2v: same-shape vec4 fast path ----
struct PcEw2v { uint nv4, op, act; };
kernel void ew2v(device const half4* a [[buffer(0)]],
                 device const half4* b [[buffer(1)]],
                 device half4* o [[buffer(2)]],
                 constant PcEw2v& p [[buffer(3)]],
                 uint gid [[thread_position_in_grid]])
{
    if (gid >= p.nv4) return;
    float4 va = float4(a[gid]), vb = float4(b[gid]), v;
    if (p.op == 0u) v = va + vb;
    else if (p.op == 1u) v = va - vb;
    else if (p.op == 2u) v = va * vb;
    else v = va / vb;
    if (p.act == 1u) v = max(v, float4(0.0f));
    else if (p.act == 2u) v = 1.0f / (1.0f + exp(-v));
    o[gid] = half4(v);
}

// ---- transpose: rank<=4 permute ----
struct PcTranspose { uint n, os0, os1, os2, ps0, ps1, ps2, ps3; };
kernel void transpose(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcTranspose& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.os0, r0 = gid % p.os0;
    uint d1 = r0 / p.os1, r1 = r0 % p.os1;
    uint d2 = r1 / p.os2, d3 = r1 % p.os2;
    o[gid] = x[d0 * p.ps0 + d1 * p.ps1 + d2 * p.ps2 + d3 * p.ps3];
}

// ---- transpose6: rank 5-6 permute (pixel (un)shuffle relayouts) ----
struct PcTranspose6 { uint n, os0, os1, os2, os3, os4, ps0, ps1, ps2, ps3, ps4, ps5; };
kernel void transpose6(device const half* x [[buffer(0)]],
                       device half* o [[buffer(1)]],
                       constant PcTranspose6& p [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.os0, r = gid % p.os0;
    uint d1 = r / p.os1; r %= p.os1;
    uint d2 = r / p.os2; r %= p.os2;
    uint d3 = r / p.os3; r %= p.os3;
    uint d4 = r / p.os4, d5 = r % p.os4;
    o[gid] = x[d0 * p.ps0 + d1 * p.ps1 + d2 * p.ps2 + d3 * p.ps3 + d4 * p.ps4 + d5 * p.ps5];
}

// ---- unary2: mode 0 leaky relu (x >= 0 ? x : a x), 1 tanh, 2 clip(x, a, b) ----
struct PcUnary2 { uint n, mode; float a, b; };
kernel void unary2(device const half* x [[buffer(0)]],
                   device half* o [[buffer(1)]],
                   constant PcUnary2& p [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float v = float(x[gid]);
    float r;
    if (p.mode == 0u) r = v >= 0.0f ? v : v * p.a;
    else if (p.mode == 1u) r = precise::tanh(clamp(v, -15.0f, 15.0f));
    else r = min(max(v, p.a), p.b);
    o[gid] = half(r);
}

// ---- d2s: NHWC DepthToSpace (CRD: ch = c b^2 + (y % b) b + x % b; DCR: ((y % b) b + x % b) oc + c) ----
struct PcD2s { uint total, oc, oh, ow, ih, iw, b, crd; };
kernel void d2s(device const half* x [[buffer(0)]],
                device half* o [[buffer(1)]],
                constant PcD2s& p [[buffer(2)]],
                uint gid [[thread_position_in_grid]])
{
    if (gid >= p.total) return;
    uint c = gid % p.oc, px = gid / p.oc;
    uint xo = px % p.ow, r = px / p.ow, y = r % p.oh, n = r / p.oh;
    uint sub = (y % p.b) * p.b + xo % p.b;
    uint ch = p.crd != 0u ? c * p.b * p.b + sub : sub * p.oc + c;
    uint ic = p.oc * p.b * p.b;
    o[gid] = x[((n * p.ih + y / p.b) * p.iw + xo / p.b) * ic + ch];
}

// ---- slice: rank<=4 strided slice ----
struct PcSlice {
    uint n, os0, os1, os2;
    uint st0, st1, st2, st3;
    uint sp0, sp1, sp2, sp3;
    uint is0, is1, is2, is3;
};
kernel void slice(device const half* x [[buffer(0)]],
                  device half* o [[buffer(1)]],
                  constant PcSlice& p [[buffer(2)]],
                  uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint d0 = gid / p.os0, r0 = gid % p.os0;
    uint d1 = r0 / p.os1, r1 = r0 % p.os1;
    uint d2 = r1 / p.os2, d3 = r1 % p.os2;
    o[gid] = x[(p.st0 + d0 * p.sp0) * p.is0 + (p.st1 + d1 * p.sp1) * p.is1
             + (p.st2 + d2 * p.sp2) * p.is2 + (p.st3 + d3 * p.sp3) * p.is3];
}

// ---- concat_ax: copy one input into dst at axis offset ----
struct PcConcat { uint n, inner, srcAxis, dstAxis, axisOff; };
kernel void concat_ax(device const half* x [[buffer(0)]],
                      device half* o [[buffer(1)]],
                      constant PcConcat& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint ii = gid % p.inner;
    uint a = (gid / p.inner) % p.srcAxis;
    uint oi = gid / (p.inner * p.srcAxis);
    o[(oi * p.dstAxis + a + p.axisOff) * p.inner + ii] = x[gid];
}

// ---- pad: rank-4 all-dims pad, constant or reflect ----
struct PcPad {
    uint id0, id1, id2, id3;
    uint od0, od1, od2, od3;
    uint pb0, pb1, pb2, pb3;
    uint mode;          // 0 const 1 reflect
    float value;
};
static inline int padcoord(uint oidx, uint pb, uint idim, uint mode, thread bool& miss)
{
    int s = int(oidx) - int(pb);
    if (uint(s) >= idim)
    {
        if (mode == 0u) { miss = true; return 0; }
        s = mirror1(s, int(idim));
    }
    miss = false; return s;
}
kernel void pad(device const half* x [[buffer(0)]],
                device half* o [[buffer(1)]],
                constant PcPad& p [[buffer(2)]],
                uint gid [[thread_position_in_grid]])
{
    uint n = p.od0 * p.od1 * p.od2 * p.od3;
    if (gid >= n) return;
    uint s2 = p.od3, s1 = p.od2 * s2, s0 = p.od1 * s1;
    uint d0 = gid / s0, r0 = gid % s0;
    uint d1 = r0 / s1, r1 = r0 % s1;
    uint d2 = r1 / s2, d3 = r1 % s2;
    bool m0, m1, m2, m3;
    int c0 = padcoord(d0, p.pb0, p.id0, p.mode, m0);
    int c1 = padcoord(d1, p.pb1, p.id1, p.mode, m1);
    int c2 = padcoord(d2, p.pb2, p.id2, p.mode, m2);
    int c3 = padcoord(d3, p.pb3, p.id3, p.mode, m3);
    if (m0 || m1 || m2 || m3) { o[gid] = half(p.value); return; }
    uint si2 = p.id3, si1 = p.id2 * si2, si0 = p.id1 * si1;
    o[gid] = x[uint(c0) * si0 + uint(c1) * si1 + uint(c2) * si2 + uint(c3)];
}

// ---- bilinear / bilinear4: NHWC half-pixel bilinear + fused add ----
struct PcBilinear { uint c, ih, iw, oh, ow, flags; float shy, shx; };
kernel void bilinear(device const half* x [[buffer(0)]],
                     device const half* a [[buffer(1)]],
                     device half* o [[buffer(2)]],
                     constant PcBilinear& p [[buffer(3)]],
                     uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    float fy = (float(oy) + 0.5f) / p.shy - 0.5f;
    float fx = (float(ox) + 0.5f) / p.shx - 0.5f;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    float v00 = float(x[(y0 * p.iw + x0) * p.c + ch]);
    float v01 = float(x[(y0 * p.iw + x1) * p.c + ch]);
    float v10 = float(x[(y1 * p.iw + x0) * p.c + ch]);
    float v11 = float(x[(y1 * p.iw + x1) * p.c + ch]);
    float v = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
    if ((p.flags & 1u) != 0u) v = max(v, 0.0f);
    if ((p.flags & 2u) != 0u)
    {
        float s = float(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, 0.0f) : s;
    }
    o[gid] = half(v);
}
kernel void bilinear4(device const half4* x [[buffer(0)]],
                      device const half4* a [[buffer(1)]],
                      device half4* o [[buffer(2)]],
                      constant PcBilinear& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    float fy = (float(oy) + 0.5f) / p.shy - 0.5f;
    float fx = (float(ox) + 0.5f) / p.shx - 0.5f;
    int fy0 = int(floor(fy)), fx0 = int(floor(fx));
    float dy = fy - float(fy0), dx = fx - float(fx0);
    uint y0 = uint(clamp(fy0, 0, int(p.ih) - 1)), x0 = uint(clamp(fx0, 0, int(p.iw) - 1));
    uint y1 = uint(clamp(fy0 + 1, 0, int(p.ih) - 1)), x1 = uint(clamp(fx0 + 1, 0, int(p.iw) - 1));
    float4 v00 = float4(x[(y0 * p.iw + x0) * c4 + ch]);
    float4 v01 = float4(x[(y0 * p.iw + x1) * c4 + ch]);
    float4 v10 = float4(x[(y1 * p.iw + x0) * c4 + ch]);
    float4 v11 = float4(x[(y1 * p.iw + x1) * c4 + ch]);
    float4 v = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
    if ((p.flags & 1u) != 0u) v = max(v, float4(0.0f));
    if ((p.flags & 2u) != 0u)
    {
        float4 s = float4(a[gid]);
        v += (p.flags & 4u) != 0u ? max(s, float4(0.0f)) : s;
    }
    o[gid] = half4(v);
}

// ---- maxpool / maxpool4: NHWC max pool + optional relu ----
struct PcPool {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl, relu;
};
kernel void maxpool(device const half* x [[buffer(0)]],
                    device half* o [[buffer(1)]],
                    constant PcPool& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oy = px / p.ow;
    float m = -3.4e38f;
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            m = max(m, float(x[(uint(sy) * p.iw + uint(sx)) * p.c + ch]));
        }
    }
    if (m == -3.4e38f) m = 0.0f;
    if (p.relu != 0u && m < 0.0f) m = 0.0f;
    o[gid] = half(m);
}
kernel void maxpool4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcPool& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oy = px / p.ow;
    float4 m = float4(-3.4e38f);
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            m = max(m, float4(x[(uint(sy) * p.iw + uint(sx)) * c4 + ch]));
        }
    }
    if (m.x == -3.4e38f) m = float4(0.0f);
    if (p.relu != 0u) m = select(m, float4(0.0f), m < float4(0.0f));
    o[gid] = half4(m);
}

// ---- avgpool / avgpool4: NHWC average pool over n images (zero padding; flags bit0 = count_include_pad) ----
struct PcAvg {
    uint c, ih, iw, oh, ow;
    uint kh, kw, sh, sw, pt, pl, flags, n;
};
kernel void avgpool(device const half* x [[buffer(0)]],
                    device half* o [[buffer(1)]],
                    constant PcAvg& p [[buffer(2)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.c * p.oh * p.ow * max(p.n, 1u)) return;
    uint ch = gid % p.c, px = gid / p.c;
    uint ox = px % p.ow, oyg = px / p.ow;
    uint img = oyg / p.oh, oy = oyg - img * p.oh, ibase = img * p.ih;
    float s = 0.0f;
    uint cnt = 0u;
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            s += float(x[((ibase + uint(sy)) * p.iw + uint(sx)) * p.c + ch]);
            cnt++;
        }
    }
    float div = (p.flags & 1u) != 0u ? float(p.kh * p.kw) : float(max(cnt, 1u));
    o[gid] = half(s / div);
}
kernel void avgpool4(device const half4* x [[buffer(0)]],
                     device half4* o [[buffer(1)]],
                     constant PcAvg& p [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint c4 = p.c >> 2;
    if (gid >= c4 * p.oh * p.ow * max(p.n, 1u)) return;
    uint ch = gid % c4, px = gid / c4;
    uint ox = px % p.ow, oyg = px / p.ow;
    uint img = oyg / p.oh, oy = oyg - img * p.oh, ibase = img * p.ih;
    float4 s = float4(0.0f);
    uint cnt = 0u;
    for (uint ky = 0u; ky < p.kh; ky++)
    {
        int sy = int(oy * p.sh + ky) - int(p.pt);
        if (uint(sy) >= p.ih) continue;
        for (uint kx = 0u; kx < p.kw; kx++)
        {
            int sx = int(ox * p.sw + kx) - int(p.pl);
            if (uint(sx) >= p.iw) continue;
            s += float4(x[((ibase + uint(sy)) * p.iw + uint(sx)) * c4 + ch]);
            cnt++;
        }
    }
    float div = (p.flags & 1u) != 0u ? float(p.kh * p.kw) : float(max(cnt, 1u));
    o[gid] = half4(s / div);
}

// ---- softmax: rows of `len` (the last axis); one simdgroup per row, 8 rows per threadgroup ----
struct PcSoftmax { uint rows, len; };
kernel void softmax(device const half* x [[buffer(0)]],
                    device half* o [[buffer(1)]],
                    constant PcSoftmax& p [[buffer(2)]],
                    uint wg [[threadgroup_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]])
{
    uint lane = tid & 31u;
    uint row = wg * 8u + (tid >> 5);
    if (row >= p.rows) return;
    uint base = row * p.len;
    float m = -3.4e38f;
    for (uint i = lane; i < p.len; i += 32u) m = max(m, float(x[base + i]));
    m = simd_max(m);
    float sum = 0.0f;
    for (uint i = lane; i < p.len; i += 32u) sum += exp(float(x[base + i]) - m);
    float inv = 1.0f / simd_sum(sum);
    for (uint i = lane; i < p.len; i += 32u) o[base + i] = half(exp(float(x[base + i]) - m) * inv);
}

// ---- layernorm: one workgroup (256 threads) per row ----
struct PcLn { uint len; uint rows; float eps; };
kernel void layernorm(device const half* x [[buffer(0)]],
                      device const float* w [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      device half* o [[buffer(3)]],
                      constant PcLn& p [[buffer(4)]],
                      uint wg [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]])
{
    // one simdgroup (32 lanes) per row, 8 rows per threadgroup: lane-strided
    // accumulate + simd_sum only, no threadgroup barrier at all.
    uint lane = tid & 31u;
    uint row = wg * 8u + (tid >> 5);
    if (row >= p.rows) return;
    uint base = row * p.len;
    float s = 0.0f, s2 = 0.0f;
    for (uint i = lane; i < p.len; i += 32u)
    {
        float v = float(x[base + i]);
        s += v; s2 += v * v;
    }
    s = simd_sum(s); s2 = simd_sum(s2);
    float mean = s / float(p.len);
    float inv = rsqrt(s2 / float(p.len) - mean * mean + p.eps);
    for (uint i = lane; i < p.len; i += 32u)
        o[base + i] = half((float(x[base + i]) - mean) * inv * w[i] + b[i]);
}

// ---- attention: o[n,h,t,:] = sum_s softmax_s(scale * q[n,h,t,:] . k[n,h,s,:]) v[n,h,s,:] ----
// q / k / v / o are strided views (element strides, any layout); one thread per query row (online
// softmax, fp32), 256 queries of one (n, h) per threadgroup, k / v rows staged in threadgroup tiles.
struct PcAttn {
    uint T, S, D, DV, H;
    uint qb, qn, qh, qt, qd;
    uint kb, kn, kh, ks, kd;
    uint vb, vn, vh, vs, vd;
    uint ob, on, oh, ot, od;
    float scale;
};
template <uint DMAX, uint TILE>
static inline void attn_body(device const half* q, device const half* k, device const half* v, device half* o,
                             constant PcAttn& p, uint2 wg, uint tid, threadgroup float* kt, threadgroup float* vt)
{
    uint t = wg.x * 256u + tid;
    uint n = wg.y / p.H, h = wg.y % p.H;
    bool valid = t < p.T;
    float qr[DMAX], acc[DMAX];
    uint qo = p.qb + n * p.qn + h * p.qh + (valid ? t : 0u) * p.qt;
    for (uint d = 0u; d < DMAX; d++)
    {
        qr[d] = valid && d < p.D ? float(q[qo + d * p.qd]) * p.scale : 0.0f;
        acc[d] = 0.0f;
    }
    uint ko = p.kb + n * p.kn + h * p.kh, vo = p.vb + n * p.vn + h * p.vh;
    float m = -3.402823e38f, l = 0.0f;
    for (uint s0 = 0u; s0 < p.S; s0 += TILE)
    {
        uint cnt = min(TILE, p.S - s0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < TILE * DMAX; e += 256u)
        {
            uint j = e / DMAX, d = e % DMAX;
            bool in_ = j < cnt;
            kt[e] = in_ && d < p.D ? float(k[ko + (s0 + j) * p.ks + d * p.kd]) : 0.0f;
            vt[e] = in_ && d < p.DV ? float(v[vo + (s0 + j) * p.vs + d * p.vd]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0u; j < cnt; j++)
        {
            float sc = 0.0f;
            for (uint d = 0u; d < DMAX; d++) sc += qr[d] * kt[j * DMAX + d];
            float mn = max(m, sc);
            float corr = exp(m - mn), e = exp(sc - mn);
            l = l * corr + e;
            for (uint d = 0u; d < DMAX; d++) acc[d] = acc[d] * corr + e * vt[j * DMAX + d];
            m = mn;
        }
    }
    if (!valid) return;
    float il = 1.0f / l;
    uint oo = p.ob + n * p.on + h * p.oh + t * p.ot;
    for (uint d = 0u; d < DMAX; d++) if (d < p.DV) o[oo + d * p.od] = half(acc[d] * il);
}
kernel void attn(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                 device const half* v [[buffer(2)]], device half* o [[buffer(3)]],
                 constant PcAttn& p [[buffer(4)]],
                 uint2 wg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float kt[64 * 32], vt[64 * 32];
    uint tid = lid.x;
    attn_body<32, 64>(q, k, v, o, p, wg, tid, kt, vt);
}
kernel void attn_d64(device const half* q [[buffer(0)]], device const half* k [[buffer(1)]],
                     device const half* v [[buffer(2)]], device half* o [[buffer(3)]],
                     constant PcAttn& p [[buffer(4)]],
                     uint2 wg [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float kt[32 * 64], vt[32 * 64];
    uint tid = lid.x;
    attn_body<64, 32>(q, k, v, o, p, wg, tid, kt, vt);
}

// ---- safediv: o = clamp(a / max(b, eps), lo, hi), fp32 math over fp16 storage ----
struct PcSafeDiv { uint n; float eps, lo, hi; };
kernel void safediv(device const half* a [[buffer(0)]],
                    device const half* b [[buffer(1)]],
                    device half* o [[buffer(2)]],
                    constant PcSafeDiv& p [[buffer(3)]],
                    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    o[gid] = half(clamp(float(a[gid]) / max(float(b[gid]), p.eps), p.lo, p.hi));
}

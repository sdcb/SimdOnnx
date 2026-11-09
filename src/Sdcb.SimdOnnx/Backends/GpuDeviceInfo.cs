namespace Sdcb.SimdOnnx.Backends;

/// <summary>Capabilities of the GPU a backend runs on (for choosing between GPU and CPU).</summary>
/// <param name="Name">Driver-reported device name.</param>
/// <param name="VendorId">PCI vendor id (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel, 0x5143 Qualcomm; 0 on Metal).</param>
/// <param name="Fp16Storage">Activations and weights can be stored as fp16.</param>
/// <param name="CooperativeMatrixFp16">fp16 cooperative-matrix (tensor core / XMX) conv and GEMM kernels are used.</param>
/// <param name="SubgroupSizeMin">Smallest subgroup (SIMD) width the device may run.</param>
/// <param name="SubgroupSizeMax">Largest subgroup width.</param>
/// <param name="UnifiedMemory">Device memory is system memory (integrated / mobile GPU).</param>
/// <param name="DeviceLocalBytes">Size of the device-local heap.</param>
public sealed record GpuDeviceInfo(
    string Name,
    uint VendorId,
    bool Fp16Storage,
    bool CooperativeMatrixFp16,
    int SubgroupSizeMin,
    int SubgroupSizeMax,
    bool UnifiedMemory,
    ulong DeviceLocalBytes);

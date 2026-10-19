# 未来架构设想：PerformanceHint（延时 / 吞吐模式）

状态：**设想，暂不实施**（2026-10-19）。本文记录动机、原理和大致方案，供以后真正动手时参考。

## 1. 动机

### 1.1 现在的旋钮对用户不友好

SimdPaddleOCR 目前暴露的并行参数是实现细节，而不是用户意图：

| 参数 | 含义 | 问题 |
|---|---|---|
| `LineWorkerCount`（bench 里的 `1w/4w/16w`） | **同一张图内部** CLS/REC 文本行的并发 worker 数 | 用户无法推断该设多少；16w 在 5950X 和 5800X 上都不比 4w 快，medium 16w 反而慢 25–50% |
| `DetIntraOpThreads` | 检测网络算子内部线程数 | 和上面的参数互相影响，默认值靠经验写死 |
| `PreprocessWorkerCount`、`GpuCtcIntraOpThreads` | 裁剪/预处理、GPU 识别后 CTC 投影的线程数 | 同上 |

默认值 `LineWorkerCount = min(ProcessorCount, 4)` 是在 8 核 Zen 3 上调出来的，并不随机器"自动变快"。

### 1.2 现在的 benchmark 只测延时

`docs/perf.md` 的所有 Paddle 数字都是**一张接一张串行处理**的单图延时（见 SimdPaddleOCR
`test/Sdcb.SimdPaddleOCR.Tests/Program.cs` 的主循环），`--workers` 只是单图内部的行并发。

这解释了为什么 16 核的 5950X 相比 8 核的 5800X 几乎没有优势（16w / t=16 只快 10–25%，1w/4w 持平）：

- 单图 det 只有 ~20 ms，开到 16 线程后被每个算子的同步屏障、串行的前后处理（轮廓、NMS、裁剪）卡住（Amdahl）；
- 两颗 CPU 都是双通道 DDR4，5950X 每核可用带宽只有一半；
- 5950X 是双 CCD，细粒度 parallel-for 的跨 CCD 同步比单 CCD 的 5800X 贵；
- 16w 比 4w 慢说明单图并发早已饱和（带宽 / 缓存争抢 / 线程过度订阅），加核没用。

而图与图之间完全独立，是天然的并行层。真正需要"跑满整台机器"的用户（批量文档处理、服务端）
今天只能自己并发调用 `PaddleOcrAll.Run`（它是线程安全的），但每次 `Run` 仍按单图独占整机来分配线程
（det 最多 16、REC 用 `cpu / workers`），并发后会严重过度订阅。

## 2. 目标

1. 用户只需要表达**意图**：要单张图最快（延时），还是单位时间处理最多张图（吞吐）。
2. **默认值**按机器自动得出一个相对较快的配置，而不是固定的 4w。
3. 原有的细粒度参数保留为高级覆盖项，显式设置时优先于 hint。
4. 遵守项目底线：任何模式都**不比旧版慢、不比旧版多占内存**（吞吐模式下与旧引擎在相同并发度下对比）。

## 3. 原理

参考 OpenVINO 的 `ov::hint::performance_mode`（`LATENCY` / `THROUGHPUT`）。

### 3.1 延时模式（默认）

- 一次只处理一张图，整机资源都给这一张图：det 高 intra-op，行阶段多 worker。
- 优化目标：单图 p50/p99 延时。这就是今天的行为，只是线程数改为按机器推导（见 4.3）。

### 3.2 吞吐模式

- 引擎内部维护 **N 个流（stream）**，每个流同时处理一张图，每个流只用 `≈ 物理核 / N` 个线程。
- 好处：
  - 串行部分（前后处理、det 后处理、裁剪、rec 小 batch）在图级别并行起来，不再是 Amdahl 瓶颈；
  - 每流线程少 → 每个算子的屏障更便宜，跨 CCD 的同步消失（若按 CCD 绑定，L3 命中也更好）；
  - 多核机器真正能吃满。预估 5950X 吞吐比延时模式高 1.5–2.5 倍，相对 5800X 的优势拉开到
    1.4–1.7 倍左右（**待实测**）。
- 代价：
  - 单图延时变长（每张图只拿到部分核）；
  - 内存上升：权重 / 常量包 / plan 已在 worker 间共享，但每个流有自己的激活 arena 和工作区，
    内存大致随 N 线性增长。N 的自动选择必须考虑这一点，并允许用户设上限。
- GPU（Vulkan/Metal）：吞吐模式意味着多个请求在飞，让 CPU 前后处理和 GPU 计算重叠
  （类似今天 Vulkan 4w 部分做到的事），流数一般 2–4 即可。

## 4. 方案草图

### 4.1 API（SimdPaddleOCR，GcdrNet 类似）

```csharp
public enum PerformanceHint { Latency, Throughput }

public sealed class PaddleOcrOptions
{
    // 新增
    public PerformanceHint PerformanceHint { get; init; } = PerformanceHint.Latency;
    public int NumStreams { get; init; }        // 0 = 自动；仅 Throughput 生效
    public long MaxMemoryBytes { get; init; }   // 0 = 不限制；自动选 N 时作为上限（可选）

    // 保留：显式设置（非 0）时覆盖 hint 推导出的值
    public int LineWorkerCount { get; init; }
    public int DetIntraOpThreads { get; init; }
    public int PreprocessWorkerCount { get; init; }
    public int GpuCtcIntraOpThreads { get; init; }
}
```

使用方式：

- 延时：照旧 `ocr.Run(image)`。
- 吞吐：可并发调用 `ocr.Run`，引擎内部用 N 个流排队执行（超过 N 的调用等待），
  或提供批量入口 `RunMany(IEnumerable<image>)` / `IAsyncEnumerable` 管线。
  两种入口都应保证结果与逐张串行处理逐位一致。

### 4.2 SimdOnnx 引擎层需要的改动

- **每流线程组**：目前 kernel 用 .NET 全局线程池（`Parallel.For`），多个流会互相抢线程、互相拖慢屏障。
  需要一个可注入的执行器（`Session.Threads` 之外再加线程组/调度器对象），每个流有固定的 worker 集合。
- **亲和性（第二步）**：在 Zen 多 CCD、Intel 大小核、多 NUMA 机器上把流绑到一个 CCD / 核簇。
  Windows 需 P/Invoke `SetThreadGroupAffinity` / `GetLogicalProcessorInformationEx`，
  Linux 用 `sched_setaffinity` + `/sys/devices/system/cpu` 拓扑。不支持的平台退化为不绑定。
- **每流 arena**：`Session` 已有独立 arena，plan/常量共享；需确认多个流同时 `Run` 时没有隐藏的共享可变状态。

### 4.3 自动推导（默认值）

启动时探测一次拓扑：物理核数 P、逻辑核数 L、L3 域（CCD/核簇）数 C、可用内存。

| 模式 | 推导（初版，需按实测调整） |
|---|---|
| Latency | det intra-op = min(L, 16)；行阶段 worker 与 REC intra-op 按 P 推导，**不再固定 4**，按实测扩展曲线在 P 上取拐点 |
| Throughput（CPU） | N = max(C, P / 4)（每流约 4 个物理核，且不跨 L3 域），再受内存上限约束；每流 intra-op = P / N |
| Throughput（GPU） | N = 2–4（按设备队列 / 显存），CPU 前后处理线程 = P / N |

推导规则写成纯函数（输入拓扑，输出配置），便于单元测试，并在日志里打印最终配置，
就像现在 Vulkan 的 `[vk] device=...` 面包屑一样。

## 5. 验证计划

1. **先做低成本实验，再定 API**：在 benchmark 加 `--concurrency K`（同时 K 张图，每张 intra-op = cpu/K），
   在 5950X、5800X、13900、8 Gen 3 上跑 K = 1/2/4/8，记录 images/s 与峰值内存，确定 N 的推导规则。
2. 同时在 GcdrNet 上扫 t = 1/2/4/8/16/32 的扩展曲线，确认单图延时模式的拐点。
3. 新增 bench 指标：吞吐模式报 images/s 和峰值 WS；与旧引擎在**同并发度**下 interleaved A/B，
   延时模式继续用现有 `tools/ab-paddle.ps1` 口径，保证默认模式不回退。
4. 正确性：吞吐模式输出必须与串行逐位一致（exact 行数、hash）。

## 6. 风险与未决问题

- 吞吐模式的内存增长与"不比旧版多占内存"的底线如何定义对比基线（建议：旧引擎由用户自行并发 N 路时的内存）。
- .NET 线程池之外自建线程组的成本（启动、空转自旋策略、与用户自己的线程池共存）。
- 亲和性在虚拟机 / Hyper-V 宿主（如 5950X 测试机开了 Hyper-V 角色）下可能失效或适得其反，需要开关。
- 大小核（Intel 12 代+、移动 SoC）上流的均衡：慢核上的流会拖尾，可能需要按核类型分配或工作窃取。
- 是否把 hint 下沉到 SimdOnnx 的通用 `Session` 层，让所有消费者（Paddle、GcdrNet、DocGainNet）共享同一套推导。

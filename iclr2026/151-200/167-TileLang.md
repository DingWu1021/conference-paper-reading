![Figure 1](https://arxiv.org/html/2504.17577/x1.png)

# TileLang: Bridge Programmability and Performance in Modern Neural Kernels

> **Venue**: ICLR 2026 Oral | **ArXiv**: 2504.17577 | **TileLang**

现代人工智能工作负载对高性能计算核函数（kernel）的需求日益迫切。无论是大语言模型的训练还是推理，核心的矩阵乘法、注意力机制乃至各类融合算子，其执行效率直接决定了整个系统的吞吐量与延迟。然而，手工编写这些算子需要程序员对GPU硬件架构有极为深入的理解：共享内存的分配与访问模式、线程束（warp）的调度方式、张量核心（Tensor Core）的调用接口、软件流水线（software pipelining）的设计，以及针对不同硬件代际的定制优化——这些知识的积累往往需要数年实践。另一方面，以Triton为代表的领域特定编译器虽然大幅降低了编写GPU kernel的门槛，但其高度自动化的调度策略也剥夺了程序员对关键优化细节的精细控制权，导致在某些复杂场景（如需要warp专用化或寄存器压力精细管理的算子）下性能仍存在明显差距。TileLang正是在这一背景下被提出的：来自微软亚洲研究院的研究团队希望在"可编程性"与"高性能"之间架起一座桥梁，让开发者能够以接近自然语言的Python代码描述kernel的数据流逻辑，同时保留对内存层次结构、线程绑定和硬件加速指令的显式控制能力。

TileLang的核心设计理念是以"Tile"（切片）为第一类编程对象，将整个核函数的开发抽象为对切片数据的操作序列。一个Tile是一块具有明确形状的数据——例如 $\text{block\_M} \times \text{block\_K}$ 大小的矩阵块——由一个线程块（thread block）、一个线程束（warp）或单个线程共同拥有和操作。这一抽象使得程序员可以在更高的层次上描述算法逻辑，而不必深陷于每个线程具体访问哪个内存地址的细节之中。TileLang提供了一套丰富的Tile算子：计算类的 `T.gemm()`（矩阵乘法，对应底层MMA/WGMMA/TCGEN5MMA或AMD MFMA指令）、`T.reduce_max()`、`T.reduce_sum()`；内存操作类的 `T.alloc_shared()`（显式分配共享内存）、`T.alloc_fragment()`（分配寄存器片段）、`T.copy()`（数据搬移，在软件流水线环境中自动转化为异步拷贝指令 `cp.async`）；调度控制类的 `T.Pipelined()`（声明软件流水线区域及缓冲级数，如三级缓冲）、`T.use_swizzle()`（优化L2缓存局部性的地址交错）以及 `T.annotate_layout()`（手动指定特定buffer的内存布局）。

与Triton不同，TileLang要求程序员显式地将数据声明在共享内存还是寄存器中，而不是由编译器自动推断。这种设计牺牲了部分便利性，但换来了对内存层次结构的精确掌控——尤其在需要精确控制寄存器使用量（以避免寄存器溢出）或需要在不同warp之间通过共享内存交换数据时，这种控制能力不可或缺。TileLang的编程模型可以概括为：程序员负责描述kernel的数据流本身（即操作什么数据、在哪里存储、以什么顺序计算），而把大多数其他优化工作——线程绑定、内存布局、张量化（tensorization）、流水线化——交给编译器自动完成。

TileLang的两大技术创新分别是**Tile推断（Tile Inference）**和**Tile推荐（Tile Recommendation）**。Tile推断机制将tile程序建模为融合计算图，从程序员的部分注解中自动推导出完整的tile配置，具体而言是通过布局推断（Layout Inference）系统来确定每个tile算子在内存和循环层面的布局方式。该系统为每个tile算子实现了两个关键接口：`Lower`接口将高层算子降级为中间表示（IR），`InferLayout`接口则根据算子的计算语义确定关联的内存布局。编译器维护一个层次化的 `LayoutMap`，记录每个buffer的布局信息，并在整个计算图中传播约束，以保证相邻算子之间的数据布局兼容性，从而避免不必要的显式转置操作。Tile推荐机制则从硬件配置文件和启发式规则出发，为给定的kernel自动建议高效的tile尺寸组合（如block_M、block_N、block_K的取值），减少手动调参的负担。两者共同构成了TileLang"高可编程性"的基础：程序员只需关注核心的算法结构，大量繁琐的低层决策可以由系统自动处理或半自动推导。

TileLang的布局推断系统将内存布局表达为线性化地址表达式 $\sum_{i} y_{i} s_{i}$，其中 $y_i$ 代表各维度的逻辑索引，$s_i$ 代表对应的步长（stride）。通过在计算图上传播这些布局约束，编译器能够自动确定哪些中间buffer需要在共享内存和寄存器之间进行数据重排，以及如何高效地完成这些重排操作。对于动态形状的支持，TileLang提供了 `T.dynamic()` 接口，使得同一kernel在不同输入尺寸下无需重新编译，降低了实际部署中的启动开销。

在实验设计上，TileLang选取了现代AI系统中最具代表性的核心kernel类型进行评估：多头注意力（Multi-Head Attention, MHA）、线性注意力（Linear Attention）、广义矩阵乘法（GEMM）、反量化矩阵乘法（Dequantize GEMM）以及DeepSeek提出的多层潜在注意力（Multi-head Latent Attention, MLA）。评估硬件覆盖了NVIDIA RTX 4090、A100、H100以及AMD Instinct MI300X，与工业界主流部署环境高度吻合。对比基线包括FlashAttention-2/3、Triton、cuBLAS等经过高度优化的手工实现或编译框架。

实验结果方面，TileLang展现出令人瞩目的性能表现。在GEMM任务上，TileLang在4090、A100、H100和MI300X上分别达到Triton的 $1.08\times$、$1.03\times$、$1.13\times$ 和 $1.25\times$ 加速比；与厂商优化库（cuBLAS等）相比，则分别达到 $1.10\times$、$0.97\times$、$1.00\times$ 和 $1.04\times$，基本持平或超越。在注意力kernel方面，TileLang实现的FlashAttention在H100上相比FlashAttention-3和Triton基线分别实现了 $1.36\times$ 至 $1.70\times$ 的加速；线性注意力kernel则相比Triton取得了 $1.77\times$ 至 $2.10\times$ 的提升。最具戏剧性的是MLA解码性能：TileLang实现的MLA kernel在H100上相比PyTorch基线达到了高达 $1075.9\times$ 的加速，展示了在复杂融合算子场景下手工调优与通用框架之间巨大的性能鸿沟，以及TileLang填补这一鸿沟的能力。反量化GEMM方面，TileLang相比cuBLAS-WFP16AFP16实现了 $7.65\times$ 的加速。在代码简洁性方面，TileLang实现的融合注意力kernel仅需不到80行Python代码，相比手工CUDA/CUTLASS实现减少了高达90%的代码量；而AMD GPU上相比Triton的最高 $6\times$ 加速比更说明TileLang在跨平台适配上的优势。

TileLang目前已支持CUDA（NVIDIA GPU）、HIP（AMD GPU）、CPU、WebGPU等多种后端，并集成了对NVIDIA CuTe DSL和华为昇腾（AscendC）的实验性支持，体现了其面向多硬件生态的设计定位。在工程化方面，TileLang基于TVM编译器基础设施构建，支持与NVIDIA的CuTe库以及AMD的Composable Kernel库集成，复用成熟的底层tile库实现。该框架已在工业界得到实际采用，例如BitBLAS项目就采用TileLang来实现高性能的反量化GEMM kernel。

从更宏观的视角来看，TileLang代表了AI基础设施领域一个重要的设计哲学转变：在"全自动编译器"与"完全手工实现"之间寻求一个精心设计的中间地带。Triton倾向于自动化，牺牲了对细节的控制；CUTLASS提供了完整控制，但学习曲线极为陡峭；TileLang则试图以Tile这一直觉性的抽象层次，让程序员用接近算法描述的代码表达kernel逻辑，同时通过编译器的布局推断和自动调优填补从高层描述到硬件指令之间的gap。这一工作对AI系统研究者、编译器研究者以及需要为新硬件快速实现高性能算子的工程师而言都具有重要参考价值。随着大语言模型结构日趋复杂（稀疏注意力、MoE、MLA等各类变体层出不穷），能够快速、高效地实现定制化kernel的工具将在AI系统栈中扮演越来越关键的角色，TileLang的探索为这一方向提供了颇具说服力的方案。

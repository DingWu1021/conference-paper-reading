![Figure 1](https://arxiv.org/html/2512.13255v1/x1.png)

# BezierFlow: Learning Bezier Stochastic Interpolant Schedulers for Few-Step Generation

> **Venue**: ICLR 2026 | **ArXiv**: 2512.13255 | **BezierFlow**

扩散模型与流模型在图像生成、三维点云生成等领域取得了显著进展，但其采样过程通常需要数百步网络前向传播，推理成本极高。为此，学界提出了各种加速方法，包括蒸馏（distillation）、一致性训练（consistency training）以及更高效的ODE求解器设计。近年来，一类轻量级训练方法受到关注：通过在预训练模型基础上学习最优时间步离散化（timestep discretization），以极少的训练开销实现少步采样的显著加速。然而，这类方法的优化空间仅局限于ODE求解器的离散时间步，忽视了更丰富的连续轨迹空间，限制了方法的表达能力与泛化范围。BézierFlow正是在这一背景下被提出，旨在将优化范围从离散时间步扩展到连续采样轨迹，通过学习随机插值（stochastic interpolant, SI）调度器，在极低训练代价下实现更强的少步生成性能。

BézierFlow的核心思路建立在随机插值框架之上。在该框架中，生成过程通过插值系数对数据与噪声进行混合，中间状态可表示为 $x(t) = \alpha(t) x_1 + \sigma(t) x_0$，其中 $x_1$ 为数据样本，$x_0$ 为噪声，$\alpha(t)$ 与 $\sigma(t)$ 分别为信号与噪声的调度函数。传统方法固定预训练模型的调度器，仅在离散时间步层面进行优化；而BézierFlow则提出对整个调度器进行重参数化，将其替换为贝塞尔函数（Bézier function）表示的新调度器，从而在连续轨迹空间中搜索最优采样路径。

具体而言，BézierFlow将调度系数 $\bar{\alpha}(s)$ 与 $\bar{\sigma}(s)$ 参数化为 $n$ 阶贝塞尔函数：

$$\bar{\alpha}(s) = (\alpha_1 - \alpha_0) \sum_{i=0}^{n} b_{i,n}(s) C_i^{(\alpha)} + \alpha_0$$

$$\bar{\sigma}(s) = (\sigma_1 - \sigma_0) \sum_{i=0}^{n} b_{i,n}(s) C_i^{(\sigma)} + \sigma_0$$

其中 $b_{i,n}(s)$ 为伯恩斯坦基多项式（Bernstein basis polynomial）。贝塞尔参数化的关键优势在于其控制点天然满足边界条件（boundary conditions）：通过固定首尾控制点，可以确保调度函数在 $s=0$ 和 $s=1$ 处取预定值，无需额外约束。内部控制点则通过累积softmax（cumulative softmax）进行排序，以保证信噪比（SNR）的单调性——这是扩散模型调度器的关键数学性质。这样，整个优化问题从复杂的约束优化转化为在时间范围内学习一组有序控制点，大幅简化了参数化难度。

在训练目标方面，BézierFlow采用教师强制KL散度（teacher-forcing KL divergence）作为学习信号，通过一个可处理的代理损失来最小化原始调度器与新调度器所生成轨迹之间的分布差异：

$$\mathcal{L}(\theta) = \mathbb{E}\left[d\left(\xi(x_0, \{t_i\}, S_\phi),\, \bar{\xi}_\theta(x_0, \{s_i\}, S_\phi)\right)\right]$$

其中 $d$ 采用LPIPS感知距离度量，$\xi$ 与 $\bar{\xi}_\theta$ 分别表示原始调度器与学习调度器下的采样轨迹。整个训练过程在单卡GPU上仅需约15分钟，相较于蒸馏方法通常需要数天训练，BézierFlow的训练代价不足其0.2%。

在实验设计上，BézierFlow在扩散模型与流模型两类代表性框架下进行了全面评估。对于扩散模型，采用EDM预训练模型在CIFAR-10、FFHQ、AFHQv2三个数据集上进行测试；对于流模型，选取ReFlow（CIFAR-10）、FlowDCN（ImageNet）以及Stable Diffusion v3.5（大规模文本到图像生成）作为评测基准。评价指标主要为FID（Fréchet Inception Distance），采样步数NFE（Number of Function Evaluations）限制在10步以内，重点考察少步采样场景下的生成质量。

实验结果充分验证了BézierFlow的有效性。在CIFAR-10 EDM模型上，NFE=4时，BézierFlow取得FID为9.55，显著优于当时最先进的轻量级训练方法LD3（FID=12.04）。在CIFAR-10 ReFlow上，NFE=4时，BézierFlow的FID为20.64，而LD3为38.95，提升幅度接近一倍。在ImageNet FlowDCN和Stable Diffusion v3.5上，BézierFlow同样在多数NFE设置下取得最优结果，展示出在大规模、高分辨率模型上的良好迁移能力。消融实验（ablation study）表明，提升贝塞尔函数的阶数 $n$ 有助于改善生成质量，在 $n=32$ 附近趋于收敛。此外，作者还验证了BézierFlow的跨NFE泛化能力：使用NFE=10训练的模型在推理时可以直接泛化到NFE=6和NFE=8，且性能优于在对应NFE下直接训练的离散方法——这是连续贝塞尔参数化相比离散时间步方法的独特优势，体现了轨迹级优化的本质好处。在与Bespoke Solver（另一基于学习的求解器设计方法）的对比中，使用相同训练目标时，BézierFlow同样表现更优，进一步证明了贝塞尔连续参数化相较于离散逐步变量的优越性。

在方法扩展性验证上，BézierFlow还成功应用于三维点云生成（ShapeNet数据集，基于PVD模型）和布局生成任务，展现出跨模态、跨任务的泛化能力，表明其方法论并不局限于图像合成领域，而是适用于更广泛的生成建模场景。

综合来看，BézierFlow提出了一种优雅而高效的少步生成加速框架，其核心贡献在于将随机插值调度器的优化空间从离散时间步拓展到连续贝塞尔轨迹，同时利用贝塞尔函数的内在数学性质（边界条件自动满足、单调性易于保证、高度可微）大幅降低了优化难度。该方法与预训练模型解耦，无需修改模型权重，仅需极少训练资源即可在广泛的扩散和流模型上取得显著的少步生成质量提升。这一工作不仅为生成模型加速提供了新的技术路径，也为理解采样轨迹优化的本质提供了富有洞见的视角，对于推动扩散/流模型在实际应用中的高效部署具有重要的实践价值。

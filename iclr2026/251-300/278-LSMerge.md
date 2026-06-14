# LS-Merge: Merging Language Models in Latent Space

> **Venue**: ICLR 2026 | **ArXiv**: N/A | **LSMerge**

模型合并（model merging）是复用大型语言模型（LLM）预训练知识的一种高效范式：通过在参数空间中直接对多个模型的权重进行线性组合或算术运算，可以在无需额外预训练的前提下将不同能力整合到单一模型中。然而，现有方法几乎无一例外地依赖于"架构同质性"假设——即参与合并的各模型必须具有完全相同的层结构、隐藏维度和参数形状。这一严苛约束极大地限制了模型合并技术的应用范围：一旦两个模型的宽度（hidden size）或深度（层数）存在差异，直接的权重插值就会变得脆弱甚至完全不可行。此外，大多数合并方法还要求至少存在两个独立的预训练检查点，这使得"自合并"（self-merging，对单一模型进行内部增强）场景无法被覆盖。来自KAIST、牛津大学和DeepAuto.ai的研究团队在ICLR 2026上提出了LS-Merge框架，将模型合并从原始的权重空间迁移到一个经过学习的平滑隐空间（latent space），从根本上绕开了架构异质性的障碍，同时在同质合并场景中也取得了优于传统方法的性能。

LS-Merge的设计起点是一项细致的权重统计分析。作者对Gemma-3（1B和4B指令微调版本）以及LLaMA-3.2（3B指令微调版本）的各transformer子模块权重进行了前四阶矩统计，包括均值（mean）、方差（variance）、偏度（skewness）和峰度（kurtosis）。分析结果揭示了LLM权重的两个显著特性：其一，权重均值接近零、方差极小，呈现出高度集中的分布；其二，峰度（尤其是自注意力层的早期层）普遍偏高，部分层的超额峰度（excess kurtosis）甚至超过15，表明权重分布具有明显的重尾特性（heavy-tailed）——即存在少量幅值极大的参数，这些参数对模型功能至关重要。这一发现否定了先前工作中将LLM权重近似为高斯分布的假设，并直接指导了编码网络的选型：编码器必须能够保留稀有的高幅值参数，而非将其过度正则化至狭窄的高斯先验。与此同时，作者通过主成分分析（PCA）表明，各层权重矩阵的方差几乎全部集中在少数主成分上（$\sum_{i=1}^{r}\lambda_i / \sum_{i=1}^{D}\lambda_i \approx 1$，其中 $r \ll \min(n, m)$），由Eckart–Young定理可知权重集合近似位于一个低维光滑流形附近，从而从理论上论证了使用VAE对权重进行压缩编码的可行性。

在方法层面，LS-Merge的核心是一个基于Transformer架构的变分自编码器（VAE）。对于每一层的权重张量，首先将其展平为一维向量 $w \in \mathbb{R}^{L}$，再进行零填充后划分为 $n = \lfloor L_p / c \rfloor$ 个不重叠的块（chunks），每块大小为 $c$，批量输入形如 $X \in \mathbb{R}^{B \times n \times c}$。每个块经过线性嵌入后送入Transformer编码器 $E_\theta$（具有可选的token下采样），输出隐向量 $z$；解码器 $D_\varphi$ 则从隐向量重建分块权重。整体优化目标采用 $\beta$-VAE损失：

$$\mathcal{L} = -\mathbb{E}_{q_\varphi(z|w)}\left[\log p_\theta(w|z)\right] + \beta \cdot \mathrm{KL}\left(q_\varphi(z|w) \| p(z)\right)$$

其中先验 $p(z)$ 为标准高斯分布，$\beta$ 为固定超参数。为解决重尾权重导致训练早期易发生坍塌（collapse）的问题，作者设计了两阶段课程训练策略：第一阶段关闭KL项，先将确定性自编码器训练至收敛；第二阶段开启KL项进行微调，使隐空间形成结构化的平滑分布，同时不牺牲重建保真度。

对于同质合并（homogeneous merging）和自合并（self-merging），流程直接而优雅：对两个模型的权重分别编码得到隐向量 $z_a = E(W_a)$ 和 $z_b = E(W_b)$，在隐空间进行线性插值 $z_\lambda = (1-\lambda)z_a + \lambda z_b$，再解码得到合并模型权重 $\tilde{W}_\alpha = D(z_\lambda)$。自合并则通过对单个模型后验分布采样多个隐向量后再取均值来实现。作者指出，隐空间中的域内插值比直接权重空间平均更能保留功能一致性，传统合并算子（如Model Soup、Task Arithmetic）均可直接迁移至隐向量操作。

对于异质合并（heterogeneous merging），LS-Merge提出了两个关键机制。第一，维度匹配投影（dimensionality-matching projection）：设源模型有 $n_s$ 层、每层参数大小为 $M$，目标模型有 $n_t$ 层、大小为 $N$，则通过比例映射系数 $r = n_t N / (n_s M)$ 将源模型的隐向量规整到与目标模型相同的维度，使得两者的总容量匹配。第二，最优传输对齐（Optimal Transport alignment，OT alignment）：对于来自不同模型家族的检查点（如Gemma与LLaMA），它们的隐向量分布位于不相交的流形上，简单插值会产生落在目标解码器有效流形之外的低质量权重。为此，作者将异质合并建模为流形配准（manifold registration）问题，通过求解2-Wasserstein距离下的Monge最优传输问题：

$$T^* = \arg\min_T \int \|z - T(z)\|_2^2 \, d\mu_\text{src}(z), \quad \text{s.t.} \quad T_\# \mu_\text{src} = \mu_\text{tgt}$$

在高斯近似下，最优传输映射具有闭合形式的仿射解：$\tilde{z}_\text{src} = T^*(z_\text{src}) = \mu_t + A(z_\text{src} - \mu_s)$，其中 $A = \Sigma_s^{-1/2}(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2})^{1/2}\Sigma_s^{-1/2}$ 为将源分布的均值和协方差同时对齐到目标分布的线性变换。对齐后的隐向量 $\tilde{Z}_\text{src}$ 与目标隐向量 $Z_\text{tgt}$ 共享相同的支撑，插值结果 $Z_\lambda^\text{OT} = (1-\lambda)Z_\text{tgt} + \lambda\tilde{Z}_\text{src}$ 能够保持在目标解码器的有效密度区域内，从而产生稳定、功能完整的合并权重。对于多个LoRA专家的融合，LS-Merge通过凸重心插值 $z_\text{merged} = \sum_{i=1}^N \lambda_i z_i^{(m)}$ 将框架推广至任意数量的模型。

实验设计涵盖四个评估场景，并在一系列标准NLP基准上进行测试，包括语言理解（MMLU、MMLU-Pro）、常识推理（HellaSwag、WinoGrande、ARC-Challenge）、数学（GSM8k）以及知识密集型任务（TruthfulQA、NLGraph、Knowledge Crosswords、AbstainQA）。自合并实验中，使用单一Transformer-VAE（6层编码/解码块，压缩率为2）联合训练于Gemma-3-1B-it和Gemma-3-4B-it的权重快照，LS-Merge相对于原始基模型和VAE单点重建均取得了约4%的平均性能提升，且对参数量较小的模型（1B）提升更为显著。专家合并实验中，LS-Merge在10个LoRA专家融合任务上全面优于包括Uniform Soup、Greedy Soup、SLERP和DARE-TIES在内的所有基线：MMLU从50.8%提升至56.0%，HellaSwag从54.6%提升至60.1%。与表示合并方法的对比实验（Table 4）显示，LS-Merge在Llama-2-13B上的MMLU（55.07 vs. 54.18）和IFEval（36.41 vs. 35.67）上与激活感知合并方法AIM持平甚至略优，并大幅领先于Task Arithmetic，表明仅操作权重隐空间即可达到需要访问模型激活的方法的性能水平。

跨架构合并实验进一步验证了LS-Merge的核心价值。在族内异质合并（Gemma-3-4B-it → Gemma-3-1B-it）中，对齐后的隐空间插值在MMLU val/test上均优于未对齐基线，且在较小混合系数（$\lambda \in [0.05, 0.20]$）下效果最好。在跨族合并（LLaMA-3.2-1B-instruct → Gemma-3-1B-it）中，不加对齐的参数或隐向量直接混合导致性能下降，而OT对齐后的方法不仅恢复了性能，还在WinoGrande（57.75 vs. 56.83）和ARC-Challenge（43.34 vs. 42.78）上超越了基模型。消融实验从多个角度验证了各组件的贡献：仅合并MLP层可获得适度性能提升，仅合并注意力层反而降低性能，而同时合并两者才能达到最优；VAE泛化实验表明，在低压缩率（$r=1.6$）下，训练于Gemma-3-4B-it的VAE可直接泛化到未见过的Gemma-3-1B-it和LLaMA-3.2-1B-it上，但高压缩率（$r=4$）会因权重分布高度集中于零附近而导致后验坍缩；线性vs非线性的对比消融（Table 8）则揭示了一个重要发现：PCA在任何压缩率下均使模型退化至接近随机猜测（MMLU约25.5%），而VAE在 $r=1.6$ 至 $r=4.0$ 全区间内均维持接近原始模型的性能，这有力地证明了LLM权重所在的流形是非线性的，线性投影无法保留其功能结构，非线性的隐空间学习是几何上的必要选择而非风格偏好。

LS-Merge的意义在于为模型合并领域开辟了全新的研究维度。其核心洞察——即将权重视为一种可以被生成模型建模的数据模态，并在该模态的光滑隐空间中进行操作——不仅解决了异质架构合并的难题，更从理论上阐明了直接权重空间操作的几何局限性。通过将最优传输引入隐向量对齐，LS-Merge提供了一种有原则、可解释的跨架构知识融合机制。尽管目前在高压缩率下的后验坍缩问题仍是一个局限，但作者指出过完备的隐空间（overcomplete latent space）可以有效缓解这一问题。未来工作方向包括开发更高效的属性感知权重编码器、通过迭代隐空间重心插值实现更复杂的单模型组合，以及将框架推广至多模态生成等领域。代码已计划开源于 https://github.com/sorobedio/ls-merge/，为后续研究提供了坚实基础。

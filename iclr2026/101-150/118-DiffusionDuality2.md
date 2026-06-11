![Figure 1](https://arxiv.org/html/2602.21185/x1.png)

# The Diffusion Duality, Chapter II: Psi-Samplers and Efficient Curriculum

> **Venue**: ICLR 2026 | **ArXiv**: 2602.21185 | **DiffusionDuality2**

离散扩散模型近年来在语言建模和图像生成领域受到越来越多的关注，尤其是以均匀状态（uniform-state）扩散和掩码（masked）扩散为代表的两类主流范式。前者因能够在少步生成和引导采样中表现出色而被认为具备自我纠错能力，但其采样质量在使用祖先采样器（ancestral sampler）时往往随步数增加而趋于饱和，难以持续提升。掩码扩散模型则因在语言生成基准上的突出表现长期被视为主导方法。本文（Diffusion Duality系列的第二章）由Justin Deschenaux、Caglar Gulcehre和Subham Sekhar Sahoo撰写，正是针对上述瓶颈提出系统性突破——通过引入一族新型预测-校正（Predictor-Corrector，PC）采样器 $\Psi$-samplers，以及一套面向高斯松弛训练的高效课程学习算法，全面提升了均匀状态离散扩散模型的推理质量与训练效率。

本文的核心方法论建立在所谓 $\Psi$-后验（$\Psi$-posterior）框架之上。作者构造了一类非马尔可夫（non-Markovian）的叠加后验，使其边际分布与标准马尔可夫前向过程完全一致，从而在不改变训练目标的前提下，实现对采样轨迹的灵活控制。形式上，$\Psi$-后验由以下混合形式定义：

$$\Psi_{s|t}(\cdot|\mathbf{x}^\ell, \mathbf{z}_t^\ell) = \kappa_t\, q_{s|t}(\cdot|\mathbf{z}_t^\ell, \mathbf{x}^\ell) + (1 - \kappa_t)\, q_s(\cdot|\mathbf{x}^\ell)$$

其中 $\kappa_t \in [0,1]$ 是时刻 $t$ 处的混合系数，用于在"预测器"分量 $q_{s|t}$（即经典祖先采样方向）与"校正器"分量 $q_s$（独立于当前噪声状态、仅依赖干净数据的边际分布）之间进行插值。当 $\kappa_t = 1$ 时，恢复为普通祖先采样；当 $\kappa_t < 1$ 时，校正器分量向轨迹注入受控噪声，实现对累积误差的主动纠正，同时严格保持与前向过程一致的边际分布。

在实际的反向过程中，由于真实的干净数据 $\mathbf{x}$ 不可观测，需用神经网络估计量 $\mathbf{x}_\theta^\ell$ 替代，得到参数化的 $\Psi$-采样器：

$$[\Psi^\theta_{s|t}(\cdot|\mathbf{z}_t)]^\ell = \kappa_t\, q_{s|t}(\cdot|\mathbf{z}_t^\ell, \mathbf{x}_\theta^\ell) + (1-\kappa_t)\bigl[\alpha_s\, q_{0|t}(\cdot|\mathbf{z}_t^\ell, \mathbf{x}_\theta^\ell) + (1-\alpha_s)\,\boldsymbol{\pi}\bigr]$$

其中偏移项 $(1-\kappa_t)(1-\alpha_s)\boldsymbol{\pi}$ 具有重要物理含义：对于掩码扩散，该项允许已解码的词元重新回到掩码状态，从而实现纠错；对于均匀状态模型，该项保证即便去噪器对某些词元赋予接近零的概率，采样时仍具有非零概率，有效防止解码陷入退化。这一设计统一并推广了Campbell等人（2022）和Wang等人（2025）的先前PC方法，使其可适用于任意噪声先验，而不局限于特定的前向过程形式。

在课程学习算法方面，本文针对大词汇表（词汇量 $K$ 可超过100,000）下高斯松弛训练的内存瓶颈提出了高效解决方案。传统做法需要显式构造完整的 $K$ 维高斯向量，在低softmax温度 $\tau = 10^{-3}$ 下对内存和计算均造成极大压力。作者观察到在低温极限下softmax分布高度稀疏，因此提出仅采样top-$k$（$k$ 可低至2）个关键分量的两步近似策略。第一步利用均匀随机变量的次序统计量递推采样 $K-1$ 个零均值高斯分量，并独立抽取对应真实词元的"特殊"分量，通过比较判断真实词元是否落入top-$k$；第二步对softmax归一化因子进行近似：

$$\tilde{Z} \approx \sum_{i \in \text{top-}k} \exp\!\bigl(\mathcal{K}_i/\tau\bigr) + \delta\,\exp\!\bigl(\tilde{w}/\tau\bigr) + (K - k - \delta)\exp\!\Bigl[\sigma_t^2/(2\tau^2) - \log\Phi(\mathcal{K}_k/\sigma_t) + \log\Phi(\cdots)\Bigr]$$

这一三部分分解分别对应top-$k$项、干净词元项（若被选中）以及未采样的零均值项（以闭合形式期望近似），整体上实现了在极低内存占用下对高斯松弛训练目标的精确近似。

在实验设计上，作者在语言建模和图像生成两个任务上对 $\Psi$-samplers 进行了系统评测。语言建模实验在OpenWebText数据集上进行，以生成困惑度（generative perplexity）和一元统计熵（unigram entropy）作为评估指标，与掩码扩散基线MDLM和其对应的PC采样方法ReMDM对比。结果表明，$\Psi$-samplers 在相同一元熵条件下取得了更低的生成困惑度，且随着NFE（网络前向评估次数）的增加持续改善，突破了祖先采样在序列长度（$L=1024$）处的饱和瓶颈。推荐的最优配置为：以 $\eta=0.05$ 的比例对 $\kappa_t$ 进行重缩放，并结合核采样（nucleus sampling，$p=0.9$）。图像生成实验在CIFAR-10数据集上进行，评估FID和Inception Score，$\Psi$-samplers 在两项指标上均优于祖先采样和ReMDM基线，推荐使用余弦型 $\kappa_t$ 时间表，设 $\kappa_t = 0.95$，并在时刻 $t_{\text{on}} \in \{0.5, 0.6\}$ 处开启校正器。训练效率方面，将新的高效课程算法整合到Duo++模型中，与Duo基线相比，峰值内存占用降低33%，训练速度提升25%，且在LM1B和OpenWebText上的困惑度基本持平，在多项选择题QA下游任务上也保持了相当的表现。

综合来看，本文的研究意义体现在多个层面。首先，$\Psi$-samplers 框架从理论上揭示了离散扩散采样中预测器与校正器的统一结构，为后续设计更灵活、更强大的采样策略提供了坚实基础。其次，文章通过严格的实验对比，挑战了"掩码扩散在语言建模上天然优于均匀状态扩散"的主流认知，表明通过改进采样器，均匀状态模型同样具备与掩码扩散相当甚至更优的生成质量，为该类模型的进一步发展提供了新的信心。第三，高效课程学习算法大幅降低了大词汇表场景下高斯松弛训练的计算门槛，使研究者和工程师在有限资源下也能训练高质量的离散扩散语言模型。整体而言，本文是Diffusion Duality系列工作的重要续篇，系统推进了离散扩散模型在推理效率与训练可扩展性两个核心维度上的技术边界，对生成式大语言模型的研究社区具有较强的参考价值。

![Figure 1](https://arxiv.org/html/2509.25170/x1.png)

# GLASS Flows: Efficient Inference for Reward Alignment of Flow and Diffusion Models

> **Venue**: ICLR 2026 | **ArXiv**: 2509.25170 | **GLASSFlows**

生成模型的推理时奖励对齐（reward alignment）是近年来深度生成模型研究的重要方向。对于扩散模型（Diffusion Models）和流匹配模型（Flow Matching Models），现有的推理时对齐算法往往依赖随机微分方程（SDE）采样，而 SDE 采样在计算效率上相比确定性的常微分方程（ODE）采样存在明显劣势，这一瓶颈严重制约了对齐算法在大规模模型上的实际应用。GLASS Flows 正是在这一背景下被提出的，其核心动机是在保持随机马尔可夫转移所带来的探索性的同时，引入 ODE 的计算效率，从而在推理时实现更高质量的奖励对齐。

现有的基于序列蒙特卡洛（Sequential Monte Carlo，SMC）的对齐方法（如 Feynman-Kac Steering，简称 FKS）需要在生成过程的每一步对粒子进行重采样和分支，这就要求采样过程具有随机性，即必须依赖 SDE 采样来产生马尔可夫转移。然而，SDE 采样相较于 ODE 采样往往需要更多的神经网络函数评估（NFE）才能保持生成质量，而在对齐场景下，模型需要同时维护多个粒子，这使得计算开销成倍增加。如何在不牺牲随机探索能力的前提下提升采样效率，是该领域亟待解决的核心问题。

GLASS Flows 的技术贡献在于提出了一种全新的"流匹配中的流匹配"（flow matching within a flow matching）范式，利用充分统计量（sufficient statistics）理论，将对马尔可夫转移分布 $p_{t'|t}(x_{t'} \mid x_t)$ 的采样问题重新表述为一个内层流匹配问题（inner flow matching problem），并且该内层模型可以直接从已有的预训练权重中提取，无需任何额外训练。

在理论构建上，作者定义了一族基于联合分布 $(X_t, X_{t'})$ 的 GLASS 转移，通过可控的相关参数 $\rho \in [-1, 1]$ 来调节两个时间步之间的随机依赖程度。命题一（Proposition 1）证明了经典的 DDPM 转移是 GLASS 转移在特定 $\rho$ 取值下的特例，这为方法的通用性提供了理论保障。核心技术创新在于引入充分统计量变换：

$$S(x) = \frac{\mu^T \Sigma^{-1} x}{\mu^T \Sigma^{-1} \mu}$$

该变换将对两个时间步 $x_t$ 和 $x_{t'}$ 的联合观测压缩为单个有效测量值，从而可以利用预训练的去噪器（denoiser）来计算后验期望。命题二（Proposition 2）进一步确立了 GLASS 去噪器的计算公式：

$$D_{\mu, \Sigma}(x) = D_{t^*}(\alpha_{t^*} S(x))$$

其中 $t^*$ 由调度器（scheduler）的逆运算得出，整个计算过程只需一次预训练模型的前向推理。定理一（Theorem 1）在此基础上构造了 GLASS 速度场：

$$u_s(\bar{x}_s \mid x_t, t) = w_1(s)\bar{x}_s + w_2(s) D_{\mu(s), \Sigma(s)} + w_3(s) x_t$$

其中 $w_1, w_2, w_3$ 是由调度器导数确定的时间相关权重系数。对该 ODE 进行数值积分即可从条件分布 $p_{t'|t}$ 中高效采样，且采样过程完全基于 ODE 而非 SDE，从而实现了计算效率的大幅提升。

在实际算法（Algorithm 1）中，GLASS Flows 采用 $K$ 次顺序转移、每次转移 $M$ 步的结构，总计需要 $K \cdot M$ 次神经网络函数评估。当 $K=1$ 时退化为标准流匹配；当 $M=1$ 时退化为 DDIM；而中间情形则允许用户在效率与随机探索之间灵活折中。该方法与分类器自由引导（classifier-free guidance）完全兼容，只需将引导调整后的速度场代入内层 ODE 即可。

实验评估覆盖了从受控条件到大规模文本生成图像（text-to-image）的多个层级。首先，作者在 ImageNet 256 数据集上使用 DiT/SiT 模型验证了后验采样的质量，通过 Fréchet Inception Distance（FID）衡量，GLASS Flows 在相同采样步数（$M$ 从 2 到 50）下均显著优于 DDPM SDE 采样，同时值函数估计（value function estimation）与真实奖励的相关性也大幅提升。其次，在 SiT（ImageNet）和 FLUX（文本生成图像）模型上，GLASS Flows 在总计 50 次神经网络函数评估的预算下，以 $N=6$ 个等间距转移节点，同时在 FID 和 GenEval 基准上与 ODE 采样性能相当，而传统 DDPM 采样在相同预算下表现明显更差。

最关键的实验是将 GLASS Flows 集成到 Feynman-Kac Steering（FKS）框架中，在 FLUX 大模型上进行文本生成图像的奖励对齐评估。实验设置为 8 个粒子、共 400 次神经网络函数评估，评估指标涵盖四个独立奖励模型：CLIP、PickScore、HPSv2 和 ImageReward，并通过 GenEval 基准防止奖励过拟合（reward hacking）。实验结果（表一）显示，使用 DDPM SDE 的传统 FKS 甚至弱于简单的 Best-of-N 基线；而 FKS-GLASS（使用 DDPM 转移）在四个奖励模型上均有提升且 GenEval 不下降；进一步将相关参数设置为 $\rho = 0.4$ 的 FKS-GLASS 取得更优结果，例如 CLIP 得分达到 39.8，GenEval-Pick 达到 74.3。结合奖励梯度引导后（表二），ImageReward 从 1.45 进一步提升至 1.52，GenEval 维持在约 73%，充分证明了方法的可扩展性与鲁棒性。

GLASS Flows 的核心贡献在于从理论上打通了"ODE 效率"与"SDE 随机性"之间长期存在的对立关系，为推理时扩展（inference-time scaling）提供了一个无需重训练的即插即用解决方案。该方法的普适性体现在：它适用于任何基于调度器的扩散或流匹配模型，与现有对齐框架（如 SMC、奖励引导）无缝集成，并且理论基础扎实、实现简洁。从更广泛的研究意义来看，GLASS Flows 为生成模型推理时计算资源的高效利用提供了新范式，有望推动奖励对齐、可控生成和推理时搜索等研究方向的进一步发展，尤其是在大规模工业级生成模型的实际部署场景中具有重要应用价值。

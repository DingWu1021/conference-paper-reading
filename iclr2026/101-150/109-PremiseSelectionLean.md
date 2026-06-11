![Figure 1](https://arxiv.org/html/2506.07477/x1.png)

# Premise Selection for a Lean Hammer

> **Venue**: ICLR 2026 | **ArXiv**: 2506.07477 | **PremiseSelectionLean**

形式化数学验证（formal mathematics verification）是利用计算机辅助证明助理（proof assistant）来确保数学定理和软件逻辑正确性的重要方向。Lean 作为近年来受到广泛关注的交互式定理证明器，其庞大的数学库 Mathlib 已收录数十万条引理与定理，为形式化数学奠定了坚实基础。然而，在使用 Lean 进行证明时，用户往往需要手动从数十万条前提（premise）中挑选出与当前证明目标相关的引理，这一过程极为繁琐且耗时。为此，研究者们引入了"hammer"工具的概念——该工具自动整合前提选择（premise selection）、逻辑翻译（translation to external automated theorem provers）与证明重建（proof reconstruction）三大模块，以期在大幅减轻用户负担的同时提升形式化验证的自动化程度。然而，针对 Lean 的端到端、领域通用型 hammer 工具此前尚属空白，现有方法或仅关注前提选择的某一子问题，或无法有效适应用户自定义的本地定义与库外前提，亟待系统性突破。

本文的核心贡献在于提出了 LeanPremise——一个专为在依赖类型论（dependent type theory）框架下与 hammer 协同工作而训练的神经前提选择系统，并在此基础上结合现有的翻译与证明重建组件，构建出 LeanHammer，即首个面向 Lean 的端到端领域通用 hammer 工具。在方法设计上，LeanPremise 采用编码器-only 的 Transformer 架构，利用余弦相似度进行前提检索：

$$\text{select\_premises}(s, k, \mathcal{P}_s) = \text{top-}k_{p \in \mathcal{P}_s}\ \text{sim}(E(s), E(p))$$

其中 $\text{sim}(u, v) = u^\top v / \|u\|_2 \|v\|_2$，$E(\cdot)$ 为编码器，$s$ 为当前证明状态，$\mathcal{P}_s$ 为可用前提集合。训练阶段采用改进的 InfoNCE 对比学习损失，针对正样本稀疏问题引入负样本采样与掩码机制：

$$\mathcal{L}(E) = \frac{1}{B}\sum_{i=1}^{B} \frac{\exp(\text{sim}(E(s_i), E(p_i^+))/\tau)}{\exp(\text{sim}(E(s_i), E(p_i^+))/\tau) + \sum_{p_i^- \in \mathcal{N}_i} \exp(\text{sim}(E(s_i), E(p_i^-))/\tau)}$$

该掩码损失有效避免了将出现在多个真实前提集中的样本误标记为负样本，从而提升了训练稳定性与召回率。

在数据提取方面，LeanPremise 设计了专门的签名提取（signature extraction）流程：通过禁用符号美化打印、使用完全限定名称（fully qualified names），确保前提的表示仅依赖于逻辑结构而非语法变体，从而实现更鲁棒的相似度计算。在状态与前提提取方面，系统同时支持项式证明（term-style proof）与策略式证明（tactic-style proof），并收集显式与隐式前提（包括 `simp` 等自动化策略调用的前提），尤其是在依赖类型论中至关重要的定义等式（definitional equalities）。此外，训练数据仅使用能够关闭整个目标所需的前提，而非仅修改目标的中间步骤，从而使前提选择更直接面向最终证明目标。

LeanPremise 还特别设计了动态适应用户上下文的机制：系统通过缓存稳定库版本的嵌入向量，在用户定义新前提时仅对增量部分重新计算嵌入，并通过服务器-客户端架构实现从 Lean 端直接调用，单次前提选择延迟约为一秒（CPU 环境下）。这一设计使 LeanPremise 能够有效推荐用户本地定义的引理以及训练数据之外的库中前提，极大拓展了其实用范围。LeanHammer 的完整推理流水线为：首先调用 Aesop 进行快速证明搜索，随后通过 Lean-auto 将目标翻译为高阶逻辑，交由外部自动定理证明器 Zipperposition 求解，最终由 Duper 完成证明重建。

在实验设计上，训练数据包含来自 206,005 个定理证明的 469,965 个状态，共形成 5,817,740 个状态-前提对。作者对三种规模的预训练模型进行微调：小型（23M 参数）、中型（33M 参数）和大型（82M 参数），学习率为 $2 \times 10^{-4}$，批大小 $B=256$，负样本数 $B^-=3$。评估基准包括 Mathlib 测试集（500 条定理）以及分布外泛化测试集 miniCTX-v2。

实验结果充分证明了 LeanPremise 的有效性。在 Mathlib 测试集上，大型模型取得了 recall@32 达 72.7% 的表现，完整流水线下证明率为 30.1%（累积配置下达 33.3%），而以真实前提作为上界的证明率为 43%。与现有最强基线 ReProver（参数量高达 218M）相比，LeanHammer 在完整设置下证明的定理数量是使用 ReProver 作为前提选择器时的 2.5 倍（即多 150%）。与纯符号方法 MePo 相比，LeanPremise 的 recall@32 高出 73%。在分布外泛化评估（miniCTX-v2）中，大型模型取得平均 20.7% 的证明率（真实前提上界为 26.1%），表明系统在训练域之外同样保持了良好的泛化能力。消融实验进一步验证了各核心设计的必要性：去除专为 hammer 设计的数据提取流程会显著降低性能；禁用负样本采样使 recall@16 从 61.1% 下降至 51.8%；损失掩码机制也对最终性能有可量化的正向贡献。此外，神经方法与符号方法（MePo）能够证明不同的定理集合，二者的组合进一步提升了整体证明率，表明二者具有良好的互补性。

本文的研究意义是多方面的。首先，LeanHammer 作为首个面向 Lean 的端到端领域通用 hammer 工具，填补了神经检索与符号推理之间的重要鸿沟，为形式化验证领域提供了切实可用的自动化工具。其次，本文提出的专为 hammer 设计的数据提取与训练方法，为依赖类型论框架下的前提选择研究提供了新的范式参考。第三，动态适应用户上下文的设计使系统能够处理库外前提与本地定义，极大提升了实用性与可扩展性。最后，实验结果表明，即便参数量远小于 ReProver（大型模型仅 82M），LeanPremise 依然能够大幅超越后者，揭示了针对特定任务设计训练数据与损失函数的重要性远胜于单纯追求模型规模。本工作为未来研究提供了丰富的启示：神经方法与符号方法的集成、基于强化学习的端到端优化、以及将 hammer 范式推广至更多证明助理（如 Coq、Isabelle）等方向，均值得深入探索。

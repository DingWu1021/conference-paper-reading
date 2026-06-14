![Figure 1](https://arxiv.org/html/2604.21232/x1.png)

# ReCAPA: Hierarchical Predictive Correction to Mitigate Cascading Failures

> **Venue**: ICLR 2026 | **ArXiv**: 2604.21232 | **ReCAPA**

视觉-语言-动作（Vision-Language-Action，VLA）系统是当前具身智能领域的核心范式，其目标是让智能体依据自然语言指令在多模态环境中执行由多个步骤组成的复杂任务。然而，这类系统在长时序任务执行过程中面临一个根本性挑战：一旦某个中间步骤产生误差，该局部错误便会沿着后续步骤逐级传播并不断累积，最终演变为整体性的级联失败（cascading failures）。现有的主流解决思路大多依赖事后修正机制（post-hoc correction），或在固定的任务分解框架与对齐方案下运行，缺乏对执行过程中动态偏差的预判能力。实验数据显示，在部分基准测试中，单个子目标的失误就可能导致后续步骤的成功率骤降超过60%，足见级联失败问题的严重性。此外，现有评估指标大多仅衡量最终任务是否完成，无法刻画错误在执行过程中的传播与消退动态，这为系统诊断与改进带来了困难。针对上述局限，来自工业界与学术界的研究人员提出了ReCAPA（Predictive Alignment and Planning Architecture），一个通过分层预测性校正来遏制级联失败的新框架。

ReCAPA的核心思想是在动作（action）、子目标（subgoal）和轨迹（trajectory）三个抽象层次上同时施加预测性校正与语义对齐约束，从而实现从底层细粒度操作到顶层全局意图的跨层监督。框架的主体模块称为层次化预测校正控制（Hierarchical Predictive Correction and Control，HPCC）。在动作层，系统处理抓取、移动、擦拭等原子操作，将其组合为子目标序列；在子目标层，系统维护中层任务的因果顺序，并预测轨迹的最终走向；在轨迹层，系统编码整体任务意图与语义结构，为底层动作提供全局约束信号。具体地，在每个层次 $l$ 上，模型将当前片段集合 $\mathcal{T}^l$ 编码为表示 $\mathbf{z}^l$，再经过基于Transformer的预测器生成下一层预测表示 $\hat{\mathbf{z}}^{l+1}$，并与真实目标 $\mathbf{z}^{l+1}$ 进行对比学习，损失函数采用InfoNCE对比损失：

$$\mathcal{L}_{\text{pred}}^l = -\log\frac{\exp(\text{sim}(\hat{\mathbf{z}}^{l+1}, \mathbf{z}^{l+1}))}{\exp(\text{sim}(\hat{\mathbf{z}}^{l+1}, \mathbf{z}^{l+1})) + \sum_j \exp(\text{sim}(\hat{\mathbf{z}}^{l+1}, \mathbf{z}^{l+1}_{\text{neg},j}))}$$

其中负样本由大语言模型生成，以提高对比学习的难度和判别性。

在语义对齐方面，ReCAPA引入了两个互补的对齐模块。第一个是基于Sinkhorn算法的最优传输模块（Sinkhorn-based Module），其将整条轨迹的分布与任务提示的分布在潜在空间中进行全局性对齐，损失形式为：

$$\mathcal{L}_{\text{sinkhorn}}(\mu, \nu) = \text{OT}_\varepsilon(\mu, \nu) - \frac{1}{2}\text{OT}_\varepsilon(\mu, \mu) - \frac{1}{2}\text{OT}_\varepsilon(\nu, \nu)$$

该模块无需逐词匹配即可捕获全局一致性，有效防止语义漂移。第二个是得分场模块（Score-field Module），采用去噪得分匹配（denoising score matching）学习局部修正向量场：

$$\mathcal{L}_{\text{score}} = \mathbb{E}\left[\left(s_\psi(\mathbf{z}^l + \xi, \mathbf{p}) - (-\xi/\sigma^2)\right)^2\right]$$

该模块能为每个状态嵌入提供指向提示所定义高密度区域的局部梯度修正，与Sinkhorn模块的全局指导形成互补。两者的综合作用使得动作生成器在训练过程中能够同时接受来自细粒度对比信号和全局分布约束的双重监督，从而在推理时具备将细粒度步骤持续对齐到整体任务意图的能力。整体训练目标为：

$$\mathcal{L}_{\text{total}} = \sum_{l \in \{\text{action}, \text{subgoal}\}} \left(\lambda_{\text{pred}}^l \mathcal{L}_{\text{pred}}^l + \lambda_{\text{score}}^l \mathcal{L}_{\text{score}}^l\right) + \lambda_{\text{sinkhorn}} \mathcal{L}_{\text{sinkhorn}}$$

其中超参数设置为 $\lambda_{\text{pred}}=0.5$，$\lambda_{\text{score}}=0.2$，$\lambda_{\text{sinkhorn}}=0.1$。

除方法框架外，论文还引入了两个全新的评估指标，以填补现有评估体系仅关注终态成功率的不足。第一个指标是错误传播率（Error Propagation Rate，EPR），定义为：

$$\text{EPR}_k = \Pr(e_{t_0+k}=1 \mid e_{t_0}=1) - \Pr(e_{t_0+k}=1 \mid e_{t_0}=0)$$

EPR量化了在初始时刻 $t_0$ 发生错误后，滞后 $k$ 步时出现后续错误的超额概率，取值范围为 $[-1, 1]$，接近零表示错误被有效遏制。例如 $\text{EPR}_3 = 0.4$ 意味着在发生初始错误后第三步再次出错的概率比无错误基线高出40%。第二个指标是传播衰减系数（Propagation Attenuation Coefficient，PAC），通过对后误差风险的对数线性拟合斜率取反来度量错误消退速率，PAC越高表明系统恢复越迅速，错误影响消散越快。EPR侧重边际风险量化，PAC侧重恢复速度，两者共同构成对长时序任务中错误动态的全面诊断工具。

在实验评估上，研究团队在VisualAgentBench、MineDojo和AI2-THOR三个具身智能基准上进行了全面测试，参照对象涵盖GPT-4o mini、Gemini 2.5 Flash、Claude Sonnet等强力专有模型基线以及多个开源LLM基线。在VisualAgentBench上，ReCAPA取得58.65的平均分，超越所有对比基线；在OmniGibson子任务上得分为50.6（对比GPT-4o mini的46.7），在Minecraft子任务上得分为66.7（对比Gemini 2.5 Flash的62.1）。在MineDojo基准上，ReCAPA在10个任务中的8个优于基线，在取奶桶任务上成功率高达95%，建炉任务73%，制作木镐80%。在AI2-THOR上，ReCAPA取得0.75的成功率（优于LLaMAR基线的0.68），运输率0.93，协调平衡指标0.93为所有方法最优。值得注意的是，覆盖率指标（0.95）略低于GPT-4V（0.97），论文认为这反映了ReCAPA在结构一致性约束下对探索广度的保守取向，是稳定性与探索性之间固有权衡的体现。在错误动态指标上，ReCAPA在OmniGibson环境中滞后10步的EPR仅为0.082，而GPT-4o-mini超过0.30、Claude Sonnet超过0.45，充分证明了层次化预测校正在错误遏制方面的优越性。

消融实验进一步验证了框架各组件的必要性。当去掉完整的HPCC模块时，Behavior任务上的成功率从72.2降至59.3，下降幅度超过12个百分点；仅保留动作+子目标两层校正时得63.6，仅保留动作+轨迹时得65.1，仅保留子目标+轨迹时得66.3，三层全保留才能达到最优的72.2，说明每个层次的贡献不可或缺且相互增强。在对齐策略消融中，去掉全部对齐模块降至65.8，仅用Sinkhorn为66.1，仅用Score-field为64.4，将KL散度替换Sinkhorn与Score-field组合得70.3，完整对齐策略才能达到72.2，印证了两个对齐模块互补协同的设计哲学。训练策略消融则表明，单独自底向上（62.1）、并行独立（63.4）、自顶向下（66.7）以及冻结轨迹层（68.9）均不及联合优化（72.2），验证了层次化模块必须联合训练方可实现样本高效的语义对齐行为这一关键结论。

ReCAPA的研究意义体现在多个维度。在技术层面，其将最优传输理论与得分匹配方法引入VLA系统的训练对齐，开辟了具身智能领域的新技术方向；分层预测校正架构为跨时序尺度的错误遏制提供了系统化解决方案，区别于以往单一层次的修正思路。在评估体系层面，EPR和PAC两个指标填补了长时序任务评估的空白，有助于研究社区建立更全面的性能衡量标准，推动对系统鲁棒性和容错能力的深入理解。从更宏观的视角看，本工作将具身智能中的级联失败问题与序列决策的误差累积理论相结合，为构建能够在开放世界中稳定执行长时序任务的通用具身智能体奠定了重要的方法论基础。

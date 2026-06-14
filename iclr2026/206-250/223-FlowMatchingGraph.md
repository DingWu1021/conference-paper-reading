# Bures-Wasserstein Flow Matching for Graph Generation

> **Venue**: ICLR 2026 | **ArXiv**: 2506.14020 | **FlowMatchingGraph**

图生成任务在药物发现、分子设计、电路布局等科学与工程领域中具有极为重要的应用价值。近年来，扩散模型与流匹配模型（flow-based models）在图生成领域取得了显著进展，其核心思想是构造一条连接参考分布与数据分布的概率路径，并通过学习这条路径上的速度场来实现生成。然而，这些方法普遍存在一个根本性的局限：它们将图中的节点特征与边结构独立处理，并在节点与边所处的解耦空间中分别进行线性插值来构造概率路径。这种解耦式插值方式破坏了图结构固有的整体关联模式——图并非欧氏空间中相互独立的元素集合，而是节点与边相互依存的复杂系统。在非欧几里得流形上强行使用线性插值，会使插值轨迹偏离真实的数据流形，导致概率路径变得不规则、不光滑，进而引发训练动态不佳和采样收敛失败等问题。本文作者 Keyue Jiang、Jiahao Cui、Xiaowen Dong 和 Laura Toni 对这一基础性问题进行了系统的理论分析，并提出了一套完整的解决方案。

本文的核心贡献在于：首先在理论层面建立了图生成模型中概率路径构造的规范化框架，然后基于此框架提出了名为 BWFlow 的流匹配算法。技术方法的出发点是将图表示为由马尔可夫随机场（Markov Random Field, MRF）参数化的连通系统，从而从根本上捕捉节点与边之间的联合演化关系。具体而言，论文的 Definition 2 将图的概率分布定义为：节点特征服从有色高斯分布 $\mathbf{X} \sim \mathcal{N}(\mathbf{X}, \mathbf{\Lambda}^\dagger)$，边结构服从 Dirac 分布 $\mathcal{E} \sim \delta(\mathbf{W})$，节点与边之间的依赖关系通过图 Laplacian 矩阵的结构显式捕捉。这一建模框架使得对整个图系统的联合演化分析成为可能，而非对各组成部分独立处理。

在此 MRF 表示框架的基础上，论文进一步引入 Bures-Wasserstein 距离来度量两个图分布之间的最优传输代价。Proposition 1 给出了图分布之间 Bures-Wasserstein 距离的闭合解析式：

$$d_{\mathrm{BW}}(\mathbf{G}_0, \mathbf{G}_1) = \|\mathbf{X}_0 - \mathbf{X}_1\|_F^2 + \beta \cdot \mathrm{trace}\left(\text{涉及 Laplacian 伪逆的项}\right)$$

这一距离度量尊重图空间的非欧几何结构，将最优传输理论引入到图概率路径的设计之中。基于此距离，Proposition 2 推导出了最优传输意义下的插值路径（OT-optimal interpolant）：节点特征部分仍采用线性插值 $\mathbf{X}_t = (1-t)\mathbf{X}_0 + t\mathbf{X}_1$，而边结构部分则采用本质不同的非线性插值方案：

$$\mathbf{L}^\dagger_t = \mathbf{L}_0^{1/2} \left[ (1-t)\mathbf{L}_0^\dagger + t \left( \mathbf{L}_0^{\dagger/2} \mathbf{L}_1^\dagger \mathbf{L}_0^{\dagger/2} \right)^{1/2} \right]^2 \mathbf{L}_0^{1/2}$$

这里 $\mathbf{L}$ 为图的 Laplacian 矩阵，$\mathbf{L}^\dagger$ 为其伪逆。边结构上的非线性插值与现有方法的线性插值有本质区别，能够构造出更光滑、更规则的概率路径，从而避免在 $t \approx 0.8$ 附近出现的速度估计失准和采样收敛失败问题。

对于速度场的计算，Proposition 3 给出了节点特征与边结构各自的闭合解析解。节点特征的条件速度场为 $v_t(\mathbf{X}_t | \mathbf{G}_0, \mathbf{G}_1) = \frac{1}{1-t}(\mathbf{X}_1 - \mathbf{X}_t)$，边结构的条件速度场为 $v_t(\mathcal{E}_t | \mathbf{G}_0, \mathbf{G}_1) = \dot{\mathbf{W}}_t = \mathrm{diag}(\dot{\mathbf{L}}_t) - \dot{\mathbf{L}}_t$。这些速度场均具有闭合解析式，无需仿真即可高效计算，从而使得训练和采样算法既稳定又高效。对于包含离散节点类型和 Bernoulli 边分布的情形，论文也进行了相应的离散扩展：离散条件速度场为 $v_t(\mathcal{E}_t | \mathbf{G}_1, \mathbf{G}_0) = (1 - 2\mathcal{E}_t) \cdot \dot{\mathbf{W}}_t / (\mathbf{W}_t \circ (1 - \mathbf{W}_t))$，在离散设定下同样保留了非线性插值带来的优势。最终，BWFlow 框架将上述推导出的最优概率路径应用于流匹配的训练与采样算法设计，可适配于连续和离散两类流匹配算法。

在实验设计方面，论文在普通图生成和分子图生成两大任务上对 BWFlow 进行了全面评估。普通图生成实验使用了三个标准基准数据集：平面图（Planar）、树图（Tree）以及随机块模型图（SBM），采用 V.U.N.（Valid, Unique, Novel）指标衡量生成质量。BWFlow 在平面图任务上达到 $97.5 \pm 2.5$，与最强基线 DeFoG 的 $99.5 \pm 1.0$ 相当；在 SBM 数据集上达到 $90.5 \pm 4.0$，与 DeFoG 的 $90.0 \pm 5.1$ 相比具有可比性甚至略优。在分子图生成任务上，论文在 GUACAMOL 和 MOSES 两个基准数据集上进行了评测。在 GUACAMOL 基准上，BWFlow 的有效性（Validity）达到 98.8%、唯一性（Uniqueness）达到 98.9%、新颖性（Novelty）达到 97.4%，均与 DeFoG 的 99.0%/99.0%/97.9% 处于同一量级。在 MOSES 基准上，BWFlow 的 FCD（Fréchet ChemNet Distance）评分为 1.32，显著优于 DeFoG 的 1.95，其中 FCD 值越低表示生成质量越好。消融实验（Figure 3）从三个维度验证了方法的有效性：Bures-Wasserstein 插值在整个采样过程中始终保持与真实数据分布更近的距离；BW 插值方案在独立比较中优于线性插值基线；BWFlow 的训练收敛曲线比线性基线更稳定，表现出更好的训练动态。论文特别指出，BWFlow 在无需大量针对路径操控技巧进行超参数搜索的条件下即可取得有竞争力的性能，体现出方法的鲁棒性。

从更宏观的视角来看，本文的研究意义在于从理论层面揭示了现有图生成方法的结构性缺陷，并提供了原则性的修正方向。将图表示为 MRF 并利用 Bures-Wasserstein 最优传输构造概率路径，不仅在数学上具有坚实的理论基础，也在实践中带来了切实的性能提升。这一框架将最优传输理论与图的非欧几何结构有机结合，为流匹配模型在结构化数据生成领域的应用拓展提供了重要参考。值得注意的是，论文在 Remark 2 中诚实地承认 GraphMRF 表示并非普适模型，存在一定的约束条件（要求 Laplacian 矩阵恰好有一个零特征值且满足半正定性质），但这些约束在平面图、随机块模型图和分子图等主流图生成任务中均能满足。总体而言，BWFlow 代表了图生成领域中将几何意识引入概率路径设计的一次重要尝试，其在分子生成等高价值应用场景中的潜力尤为值得期待。

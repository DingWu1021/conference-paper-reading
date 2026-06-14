# Understanding the Learning Phases in Self-Supervised Learning via Critical Periods

> **Venue**: ICLR 2026 | **ArXiv**: N/A | **SSLLearningPhases**

自监督学习（Self-Supervised Learning，SSL）已成为从无标注数据中学习可迁移表征的强力预训练范式，被广泛应用于视觉识别、遥感图像分析、医学影像等下游任务中。然而，一个基础但长期被忽视的问题是：SSL模型究竟应该预训练多长时间？主流实践默认"预训练越久表征越好"，并将最终检查点作为下游迁移的唯一选择。本文作者JangHyeon Lee、Philipe Ambrozio Dias、Yao-Yi Chiang与Dalton Lunga（分别来自明尼苏达大学与橡树岭国家实验室）对这一直觉提出了系统性挑战，揭示出SSL预训练过程中存在一个被忽视的**可迁移性权衡（transferability trade-off）**现象：在多种不同的SSL设置下，中间检查点在域外（Out-of-Domain，OOD）泛化上往往优于最终检查点，而额外的预训练主要有益于域内（In-Domain，ID）性能的提升。这一现象促使作者借鉴神经科学与监督学习中的"关键期（Critical Periods，CP）"概念，对SSL学习阶段进行深入分析，并提出了两种实用干预策略以充分利用这一洞见。

在研究背景上，关键期理论最初源于神经生物学，指生物神经系统发育过程中存在对外部刺激特别敏感的时间窗口，在此窗口之外的刺激缺失对最终发育影响较小。Achille等人（2018）的先驱工作将这一概念迁移到了深度神经网络的监督训练场景：模型在训练早期会经历高可塑性（plasticity）阶段，此时输入数据的扰动会对最终表征产生持久影响；随着训练推进，模型进入巩固阶段（consolidation phase），可塑性下降，同等扰动的影响减弱。Fisher信息（Fisher Information，FI）被用作可塑性的代理指标——训练早期FI急剧上升后逐渐下降，其动态曲线与模型对外部干预的敏感窗口高度吻合。然而，上述分析框架均针对监督学习场景，其中标注标签直接定义了训练目标与下游评估。SSL预训练采用代理任务（pretext task）从无标注数据中提取自监督信号，与下游任务的关系更为间接，这使得直接套用传统关键期分析存在根本性困难。本文正是在此背景下，首次将关键期理论系统延伸至SSL领域，并深入探讨其与下游迁移性之间的关系。

在核心方法与技术贡献上，论文从两个互补视角对SSL中的学习阶段进行刻画。**第一个探针：预训练数据的缺陷注入（Deficit Injection）**。作者将传统关键期研究中的扰动实验重新设计为适配SSL的形式。设模型参数为 $\theta$，在干净数据分布 $\mathcal{D}$ 上以自监督损失 $\mathcal{L}_{\text{SSL}}(\theta(x))$ 进行训练。缺陷注入将从扰动分布 $\tilde{\mathcal{D}}$（例如高斯噪声替代真实图像）采样的数据注入预训练过程，注入窗口起始于 epoch $t$，持续 $\Delta$ 个 epoch，之后恢复干净数据训练至总 epoch $T$。定义敏感度分数为

$$S(t) = \Phi(\tilde{\theta}) - \Phi(\hat{\theta})$$

其中 $\Phi$ 为下游任务评估指标（如分类精度），$\hat{\theta}$ 为干净基线模型，$\tilde{\theta}$ 为注入缺陷后的模型。若早期注入导致敏感度显著高于后期注入，则说明SSL训练存在关键期。实验沿用了将输入替换为高斯噪声的设定，缺陷窗口长度分别取5、30、50个epoch，注入时机分别为训练早期（epoch 0）、中期（epoch 450）与晚期（epoch 750）。**第二个探针：追踪代理任务上的Fisher信息**。在SSL中，代理任务定义了自监督的监督信号。作者将Fisher信息矩阵（FIM）定义在代理任务的条件分布 $p_\theta(y|x)$ 上：

$$F = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{p_\theta(y|x)} \left[ \nabla_\theta \log p_\theta(y|x) \nabla_\theta \log p_\theta(y|x)^\top \right]$$

追踪FI的迹（trace of FIM）随预训练的演变可揭示参数对代理任务信号的敏感性随时间的变化规律。FI高时模型处于高可塑性阶段；FI稳定后关键期关闭（CP closure），模型进入巩固或过度专业化阶段。作者进一步将**关键期**定义为FI稳定之前的训练阶段，而**过度专业化阶段（overspecialization phase）**定义为CP关闭后ID性能持续提升但OOD性能开始下降的阶段。基于上述分析，论文提出两种CP引导的干预策略。**CP-CS（CP-guided Checkpoint Selection）**：通过监测预训练过程中FI迹的变化，识别FI曲线稳定的epoch区间，并将该区间附近的检查点作为下游迁移的起点。该策略无需额外计算成本，不需要任何标注数据，为OOD迁移提供了一个基于原则的检查点选择规则。**CP-SD（CP-guided Self-Distillation）**：为同时兼顾OOD和ID性能，CP-SD在下游微调阶段将CP检查点（teacher）的早期层表征蒸馏到最终检查点（student）的对应层中。总体优化目标为

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{distill}}^{\text{early layers}}$$

其中 $\lambda$ 为超参数，蒸馏损失仅施加于早期层，后期层仅由任务损失优化。选择早期层的依据在于：层次探测（layer-wise probing）实验表明，CP检查点在早期层的OOD性能增益最为显著，而后期层则在最终检查点中保留了更多与预训练分布对齐的ID专用信息。

在实验设计与主要结果上，作者对判别式SSL方法（SimCLR、VICReg、DINO、DINOv2）与生成式SSL方法（MAE）进行了系统评估，预训练数据集以fMoW（Functional Map of the World）遥感图像为主，并在ImageNet上做了验证。下游评估涵盖域内数据集（fMoW-val）与多个域外数据集（fMoW-WILDS、EuroSAT、EuroSAT-Spatial），以线性探测（linear probing）精度为主要评估指标。**可迁移性权衡的普遍性**：在所有测试的SSL方法中，作者均观察到OOD性能在中间检查点达到峰值后下降，同时ID性能随预训练持续提升的权衡现象（图2）。不同方法的权衡出现时机有所差异，例如SimCLR的OOD峰值出现较晚，与其FI曲线稳定时间吻合。**关键期的存在性**：缺陷注入实验（图3）表明，相同扰动在训练早期注入导致的下游性能下降显著大于在训练晚期注入的影响，证实了SSL中关键期的存在。FI轨迹分析（图4）进一步显示FI在训练初期急速上升随后下降并趋于稳定，与监督学习中观测到的动态曲线相似。**CP动态与可迁移性的对齐**：图5展示了FI稳定（CP关闭，灰色阴影）与OOD性能峰值高度吻合的规律，提示CP关闭是广泛迁移性的最优时间节点。**CP-CS的效果**：选取FI稳定点附近的检查点作为下游迁移起点，在多种SSL方法和评估数据集上一致提升了OOD性能，同时相较于最终检查点对ID性能的损失较小。**CP-SD的量化效果**：以VICReg-RN50-fMoW为例（表1），最终检查点在fMoW-val（ID）上达到62.1%，在fMoW-WILDS（OOD）上仅34.1%；CP检查点在OOD上大幅提升至43.0%，但ID略有下降至61.0%；CP-SD（早期层蒸馏）实现了最佳综合性能，ID为61.7%、fMoW-WILDS OOD提升至44.5%、EuroSAT OOD达94.4%、EuroSAT-Spatial OOD达92.5%，均优于单独使用CP或最终检查点。消融实验（附录F）还验证了该现象在不同学习率策略（固定vs余弦）、不同minibatch重用程度及不同骨干架构（ViT-S vs ViT-B）下的稳健性。此外，表征漂移（representation drift）与多模态SatCLIP实验进一步支持了早期预训练阶段具有更高表征可塑性的结论。

从总结与研究意义来看，本文的核心贡献在于打破了SSL领域"预训练越久越好"的固有假设，从理论与实验两个维度揭示了SSL预训练中学习阶段的内在结构。借助关键期框架，研究者首次在SSL语境下系统证明了训练初期高可塑性阶段与中期OOD峰值、晚期过度专业化阶段之间的动态演变关系。Fisher信息作为无监督的可塑性量化工具，为识别最优迁移检查点提供了无需标注的实用指引。CP-CS与CP-SD两种干预策略轻量且有效，前者仅需在预训练时监测FI轨迹，后者在微调阶段通过蒸馏少量层的表征即可同时提升OOD和ID性能。这对实际应用具有重要价值，尤其是在遥感、医学影像等数据域差异显著的场景中，合理的检查点选择可以显著改善模型的跨域泛化能力。未来方向包括将分析推广至语言、时序与音频等其他模态，探索iBOT、JEPA等更多SSL方法，以及结合无监督表征质量指标（如RankMe、LiDAR）构建更全面的预训练动态监测体系。本文为理解大规模自监督预训练的内在机制提供了新的理论视角，也为迁移学习实践中的检查点管理策略提供了有据可依的原则性指导。

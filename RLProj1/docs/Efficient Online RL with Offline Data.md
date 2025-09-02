## Abstract
为了解决Sample Efficiency与Exploration的问题，我们将offline data引入到online RL的问题当中。
我们的核心思路：
- 将off-policy methods也用来学习offline data
- 我们需要进行一系列比较小的改动来优化off-policy RL的性能。
不论是什么样的数据集都可以应用，无论是大量数据还是小量数据都可以。
# 1. Introduction
Online RL的强大性能一般都是基于与真实环境之间的online interaction的
真实环境中还可能会面临Sparse-Reward等问题
解决方案：使用Previous Policy生成的data或者human expert生成的数据
- 这样可以提供一个initial dataset, to "kick-start" 这个learning process
我们的方法需要
- 能够take advantage of offline data
- 能够减轻distribution shift可能带来的问题
我们的思路
- 没有offline pre-training与imitation terms
- 直接使用off-policy methods，用来学习offline data
我们实现方法能够成功的关键
- 创建 a minimal set of key design choices
- 首先提出一个symmetric sampling
- 防止value function的over-extrapolation
- 我们使用Layer Normalization来在一定程度上防止over-extrapolation
- Large Ensembles同样是非常重要的
RLPD(RL with Prior Data)
- 可以比先前的方法表现更好
- 仍然保留了online algorithms的主要优势
- 我们的方法同样具有Generality，在不同的offline datasets上表现良好
我们的方法证明了online off-policy方法在offline data上学习的强大性能
- 但是需要sampling方法
- normalizing
- large ensembles
这证明了我们每个individual ingredients的重要性。
且我们的模型在不同的offline data上（不管是expert还是sub-optimal）都有良好的表现。
# 2. Related Work
Offline RL pre-training
- 之前的方法同样参考使用了Large Ensembles以及Multiple Gradient Step的方法来增强数据的利用率
- 我们的方法介绍了Additional Hyperparameters
- 我们的方法==没有== offline pre-training
Constraining to prior data
- 有一种方法是让agent生成接近offline data分布的数据
- 我们也并没有使用BC Term限制我们的policy
- 我们的方法对**Dataset质量没有要求**
Unconstrained Methods with prior data
- 有一些方法在初始化replay buffer的时候使用了offline data
- 有一些方法平衡了online与offline数据的使用
- Balanced Sampling是非常重要的
# 3. Preliminaries
我们的方法是have access to offline datasets的
# 4. Online RL with Offline Data
我们的方法最好是对Pre-collected data的质量与数量agnostic的
我们的方法没有explicit constraints，也没有pre-training
基于SAC打造
我们的方法是minimally invasive的
## 4.1 Design Choice 1：How to incorporate offline data
我们的方法叫**Symmetric Sampling**
每一个batch
- 一半来自于Replay Buffer
- 另一半来自ofline data buffer
直接应用到canonical off-policy methods的时候，可能性能会不够好
## 4.2 Design Choice 2: Layer Normalization Mitigates Catastrophic Overestimation
![[Pasted image 20250902195316.png]]
传统的off-policy RL算法会对OOD actions不熟悉，出现overestimation，可能还会导致training instabilities and possible divergence
原本的方法可能是explicitly discourages OOD actions，可以被看做是anti-exploration
***我们的方法是，确保我们的funcion不要extrapolate***就可以了。
我们的Layer Normalization方法可以有效地Bound the Q-Values
LayerNorm可以做到限制Q值范数，防止发散的作用。
![[Pasted image 20250902195250.png]]
可以看出这里确实有效地抑制了Overestimation
## 4.3 Design Choice 3: Sample Efficient RL

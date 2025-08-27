# Cal-QL (Calibrated Conservative Q-Learning)

## 📌 背景
- **Offline RL** 面临两个核心问题：
  1. **Overestimation**：OOD (out-of-distribution) 动作的 Q 值可能被高估。
  2. **Over-pessimism**：保守方法 (如 CQL) 虽能防高估，但可能过度悲观，导致策略退步 (unlearning)。

- **Cal-QL 目标**：在保证不高估的同时，避免过度悲观。
- 如果我们过于悲观，那么在online learning中学习到的一些实际上不好的动作的Q-value会表现出来更高，

---

## ⚙️ 核心思想
- 引入一个 **参考策略 μ (reference policy)**，通常选用行为策略 (behavior policy)。
- 希望学到的 Q 值满足：
  $$
  V^\mu(s) \leq \mathbb{E}_{a \sim \pi}[Q_\theta(s,a)] \leq Q^\pi_{\text{true}}(s,a)
  $$
  即：**下界是参考策略，上界是真实策略**。

- **关键机制**：
  - 当 $Q_\theta(s,a) > V^\mu(s)$：应用 **Conservative Term** 压低 OOD 动作 → 防止高估。
  - 当 $Q_\theta(s,a) \leq V^\mu(s)$：不再压低，保持下界 → ==防止过度悲观。==

---

## 📝 Calibrated Regularizer
- 标准 CQL 正则项：
  $$
  R(\theta) = \mathbb{E}_{s\sim D, a \sim \pi}[Q_\theta(s,a)] - \mathbb{E}_{s,a \sim D}[Q_\theta(s,a)]
  $$

- **Cal-QL 修改后**：
  $$
  R(\theta) = \mathbb{E}_{s\sim D, a \sim \pi}[\max(Q_\theta(s,a), V^\mu(s))] - \mathbb{E}_{s,a \sim D}[Q_\theta(s,a)]
  $$

- 直观理解：
  - `max` 操作在 $Q < V^\mu$ 时切断梯度，不再往下压。
  - 只在 Q 高于参考策略时才发挥 CQL 保守作用。
**存在正则项**保证了我们可以防止Overestimation
**正则项中**存在截断机制，保证了我们不会过度压低这里的Q值导致Unlearning现象。
选定一个Reference Policy，使得我们不会低于这个Reference Policy对应的Q值。

---

## 📈 理论分析 (Section 6)
- **Cumulative Regret** 可分解为两部分：
  1. Miscalibration (Q 低于最优 → 校准控制)
  2. Overestimation (Q 高于真实 → 保守控制)

- **定理结果**：
  - 如果参考策略 μ 接近最优 → Cal-QL 的 regret bound 明显优于 Hybrid RL。
  - 如果 μ 很差 → Cal-QL 至少不比 Hybrid RL 差。

---

## 🔬 实验 (Section 7)
- **高 update-to-data ratio (UTD)** 场景：
  - 普通 Q-learning → OOD 发散。
  - CQL → 过度悲观，性能退化。
  - Cal-QL → 稳定在参考值和真实值之间，性能更好。
- 实验证明 Cal-QL 在极端训练条件下仍然鲁棒。

---

## ✅ 总结
- **TD Error**：保证 Q 收敛到真实值。
- **Conservative Term**：限制高估。
- **Calibration**：提供参考策略下界，避免过度悲观。
- **最终效果**：学到的 Q 始终夹在参考策略和真实策略之间 → 稳定高效。

# Moment-based DRO 基于矩的分布鲁棒优化

## 从投资组合优化问题开始

考虑一个单周期投资组合优化问题。假设共有 $n$ 种资产， $x_i$ 为第 $i$ 种资产的回报（是随机的）， $w_i$ 为分配到第 $i$ 种资产的投资权重，则投资总汇报为 $w^Tx$ 。

一个非常著名的衡量投资风险的度量是在险价值（Value-at-Risk，VaR），其定义为：

$$VaR(w)=inf\{r|P(r\leq-w^Tx)\leq \epsilon\}$$

即：使得投资损失超过 $r$ 的概率不超过 $\epsilon$ 的最小 $r$ 。

![img](assets/Landing_page_2-6.jpg)

我们不知道 $x$ 的准确分布，但是我们可以知道一些信息，比如均值 $\bar{x}$ 和协方差 $\Gamma$ 。
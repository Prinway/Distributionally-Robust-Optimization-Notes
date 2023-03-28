# Wasserstein-based DRO 基于 Wasserstein 距离的分布鲁棒优化

## 衡量分布间的距离

对于许多问题，我们知道随机变量的经验分布。即使并非如此，如果我们对于随机变量 $\xi$ 有 $N$ 个独立同分布的样本 $\xi_i$ ，也可以构造出经验分布。一种最方便的构造方式是：

$$\hat{\mathbb{P}_N}=\frac{1}{N}\sum_{i=1}^N \delta_{\hat{\xi_i}}$$

其中， $\delta_{\hat{\xi_i}}$ 是第 $i$ 个样本 $\hat{\xi_i}$ 的狄拉克（Dirac）点质量。

在构建模糊集的方式上，除了利用矩信息之外，另一种思路是衡量真实分布与经验分布之间的距离。在这种情况下，我们以经验分布为中心，将与经验分布不超过某一距离的所有分布纳入模糊集中。于是，如何定义两个概率分布间的距离成了关键，这一度量不仅需要有统计学意义，还应尽量让相应的分布鲁棒优化模型可处理。

## Wasserstein 度量

目前很受欢迎的一种度量方式是 Wasserstein 距离。设 $\mathbb{P}$ 和 $\mathbb{Q}$ 为可测空间 $(\Xi,\mathcal{B})$ 上的两个概率测度（分布），则它们之间的 Wasserstein 距离定义为：

$$W(\mathbb{P},\mathbb{Q})=\inf_{\Pi}\{\int_{\Xi \times \Xi} \mathbb{E}_{\Pi}[d(\xi,\xi')] \Pi(d\xi,d\xi'): \Pi(\Xi,d\xi')=\mathbb{P}(d\xi'), \Pi(d\xi,\Xi)=\mathbb{Q}(d\xi)\}$$

其中 $\Pi$ 为联合分布。如果把两个分布 $\mathbb{P}$ 和 $\mathbb{Q}$ 看作两个土堆，则 Wasserstein 距离可以视作为把一个分布搬运到另一个分布的成本（所以它也称作 earth mover's distance，动土距离），而式中的 $d(x,y)$ 则对应着成本函数。

![img](assets/symmetry_1d.png)

特别地，在离散的情况下，Wasserstein 距离是把商品从一堆生产者运输到一堆需求者的最优传输距离。设 $\mathbb{P}=\sum_{i=1}^N p_i \delta_{\hat{\xi_i}}$ ， $\mathbb{Q}=\sum_{j=1}^M q_j \delta_{\hat{{\xi'}_j}}$ ， $\gamma_{ij}$ 为从 $i$ 到 $j$ 的运输计划， $c(i,j)$ 为从 $i$ 到 $j$ 的运输成本，则：

$$W(\mathbb{P},\mathbb{Q})=\inf_{\gamma_{ij}>0} \{ \sum_{i,j} \gamma_{ij} · c(\xi_i,{\xi'}_j): \sum_j \gamma_{ij}=p_i, \forall i, \sum_i \gamma_{ij}=q_j, \forall j \}$$

![img](assets/earth_move_1.png)

有了 Wasserstein 距离，我们就可以定义一个 Wasserstein 球：

$$\mathbb{B}_\epsilon(\mathbb{P})= \{\mathbb{Q}:W(\mathbb{Q},\mathbb{P}) \leq \epsilon\}$$

## 分布鲁棒优化与机器学习

Wasserstein 距离近年来非常受欢迎的原因之一是其与机器学习中的许多问题存在着联系。

以逻辑回归问题为例，设 $x\in R^n$ 为输入向量， $y\in\{1,-1\}$ 为标签， $\beta$ 为回归参数，给定条件概率 $P(y \vert x)=(1+e^{-y·\beta^Tx})^{-1}$，若训练集中有 $N$ 个样本 $\xi_i=(x_i,y_i)$ ，则 $\beta$ 的极大似然估计为：

$$\hat{\beta}=\arg\min_\beta \frac{1}{N} \sum_{i=1}^N l_\beta(\xi_i)=\arg\min_\beta \mathbb{E}_{\hat{P_N}}[l_\beta(\xi)]$$

其中 $l_\beta(x,y)=\log (1+e^{-y·\beta^Tx})$ 为损失函数， $\hat{\mathbb{P}_N}$ 为经验分布。

不过，使用这种方式估计 $\beta$ 很容易出现过拟合的现象，导致 $\beta$ 的泛化能力较差，原因在于极大似然估计中所使用的经验分布 $\hat{\mathbb{P}_N}$ 来源于训练集，其与真实分布 $\mathbb{P}^*$ 之间存在差距。

在机器学习中，常用通过添加正则项的方式来提升泛化能力：

$$\hat{\beta}=\arg \min_\beta \mathbb{E}_{\hat{P_N}}[l_\beta(\xi)]+\epsilon R(\beta)$$

添加正则项之后， $\beta$ 的泛化能力在实验中确实得到了提升，但这一方法的理论解释性不强。本质上来说，我们希望 $\beta$ 在样本外数据上也表现得很好，即从经验分布 $\hat{\mathbb{P}_N}$ 出发，以某种方式将真实分布 $\mathbb{P}^*$ 也纳入考虑。恰好，以经验分布 $\hat{\mathbb{P}_N}$ 为中心的 Wasserstein 球满足了我们的想法！于是，得到以下问题：

$$\inf_{\beta} \sup_{\mathbb{P}\in \mathbb{B}_\epsilon(\hat{\mathbb{P}_N})} \mathbb{E}_{\mathbb{P}}[l_\beta(\xi)]$$

这又是一个分布鲁棒线性规划。

## 分布鲁棒线性规划

在逻辑回归问题中，如果我们有两个不同的样本 $\xi=(x,y)$ 和 $\xi'=(x',y')$ ，则它们之间的运输成本可被定义为：

$$d(\xi,\xi')=\Vert x-x' \Vert+\frac{K}{2} \vert y-y'\vert$$

其中， $K$ 是自己定义的一个权重，用来调节特征和标签的不确定性。例如，当 $K=+\infty$ 时，说明不允许对同一个特征出现两种不同的标签。

定义了运输成本（即定义了 Wasserstein 球之后），就可以完整地写出刚才那个分布鲁棒线性规划问题：

$$\inf_{\beta \in R^n} \sup_{\mathbb{Q}\in \mathbb{B}_\epsilon(\hat{\mathbb{P}_N})} \mathbb{E}_{(x,y) \sim \mathbb{Q}}[l_\beta(x,y)]$$

其中：

<!-- $$
\begin{array}{rl}
l_\beta(x,y)=&\log (1+e^{-y·\beta^Tx}) \\
\mathbb{B}_\epsilon(\hat{\mathbb{P}_N})=&\{\mathbb{Q}:W(\mathbb{Q},\hat{\mathbb{P}_N}) \leq \epsilon\} \\
\hat{\mathbb{P}_N}(\xi)=&\frac{1}{N}\sum_{i=1}^N \delta_{(\hat{x_i},\hat{y_i})}(\xi)
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\Zk87cPQYBz.svg"></div> 

$$W(\mathbb{P},\mathbb{Q})=\inf_{\Pi}\{\int_{\Xi \times \Xi} d(\xi,\xi') \Pi(d\xi,d\xi'): \Pi(\Xi,d\xi')=\mathbb{P}(d\xi'), \Pi(d\xi,\Xi)=\mathbb{Q}(d\xi)\}$$

<!-- $$
\begin{array}{rl}
\Xi=&R^n \times \{-1,+1\} \\
\xi=&(x,y) \\
d(\xi,\xi')=&\Vert x-x' \Vert+\frac{K}{2} \vert y-y'\vert
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\Yv82tCGCHV.svg"></div>

和基于矩的分布鲁棒线性规划问题类似，先把内层问题显式地写出来：

<!-- $$
\begin{array}{ll}
&\sup_{\mathbb{Q}\in \mathbb{B}_\epsilon(\hat{\mathbb{P}_N})} \mathbb{E}_{(x,y) \sim \mathbb{Q}}[l_\beta(x,y)]\\
=& \sup_{\Pi} \int_\Xi l_\beta(x,y)\mathbb{Q}(d\xi) \\
s.t & \int_{\Xi \times \Xi} d(\xi,\xi') \Pi(d\xi,d\xi') \leq \epsilon \\
& \Pi(\Xi,d\xi')=\hat{\mathbb{P}_N}(d\xi') \\
& \mathbb{Q}(d\xi)=\Pi(d\xi,\Xi)
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\0kufYePF9i.svg"></div>

在本问题中， $\hat{\mathbb{P}_N}$ 是离散分布，所以：

<!-- $$
\begin{array}{ll}
\mathbb{Q}(d\xi)&=\Pi(d\xi,\Xi)\\
&=\int_{\xi'\in \Xi} \Pi(d\xi,d\xi')\\
&=\sum_{i=1}^N\Pi(d\xi\vert\xi'=(\hat{x_i},\hat{y_i}))·\hat{\mathbb{P}_N}(\hat{x_i},\hat{y_i})\\
&=\frac{1}{N}\sum_{i=1}^N\Pi(d\xi\vert\xi'=(\hat{x_i},\hat{y_i}))
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\OJNr4LZTop.svg"></div>

把 $\Pi(d\xi\vert\xi'=(\hat{x_i},\hat{y_i}))$ 写为 $\mathbb{Q}^i(d\xi)$ ，则内层问题变为具有 $N$ 个决策变量的问题：

<!-- $$
\begin{array}{ll}
&\sup_{\mathbb{Q}^i\geq 0} \frac{1}{N}\sum_{i=1}^N \int_{\Xi} l_\beta(\xi)\mathbb{Q}^i(d\xi)\\
s.t & \frac{1}{N}\sum_{i=1}^N\int_{\Xi} d(\xi,(\hat{x_i},\hat{y_i})) \mathbb{Q}^i(d\xi) \leq \epsilon \\
& \int_\Xi \mathbb{Q}^i(d\xi)=1
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\DF9pzGUrqF.svg"></div>

由于我们的标签 $y$ 只有两个，可以对 $\mathbb{Q}^i$ 进行分解：

$$\mathbb{Q}^i=\mathbb{Q}^i(dx,y=1)+\mathbb{Q}^i(dx,y=-1)=\mathbb{Q}_{+1}^i(dx)+\mathbb{Q}_{-1}^i(dx)$$

相应地，内层问题被写为：

<!-- $$
\begin{array}{ll}
&\sup_{\mathbb{Q}_{±1}^i\geq 0} \frac{1}{N}\sum_{i=1}^N
\int_{R^n}l_\beta(x,1)\mathbb{Q}_{+1}^i(dx)+l_\beta(x,-1)\mathbb{Q}_{-1}^i(dx) \\
s.t &\frac{1}{N}\sum_{i=1}^N[\int_{R^n} d((x,1)),(\hat{x_i},\hat{y_i})) \mathbb{Q}_{+1}^i(dx)+\int_{R^n} d((x,-1)),(\hat{x_i},\hat{y_i})) \mathbb{Q}_{-1}^i(dx)] \leq \epsilon \\
& \int_\Xi \mathbb{Q}_{+1}^i(dx)+\mathbb{Q}_{-1}^i(dx)=1
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\RhqqpYTOCR.svg"></div>

代入成本函数并对内层问题进行整理：

<!-- $$
\begin{array}{ll}
&\sup_{\mathbb{Q}_{±1}^i\geq 0} \frac{1}{N}\sum_{i=1}^N
\int_{R^n}l_\beta(x,1)\mathbb{Q}_{+1}^i(dx)+l_\beta(x,-1)\mathbb{Q}_{-1}^i(dx) \\
s.t &\frac{1}{N}\sum_{i=1}^N[\sum_{i:\hat{y_i}=1} \int_{R^n}\Vert x-\hat{x_i} \Vert \mathbb{Q}_{+1}^i(dx)+(\Vert x-\hat{x_i} \Vert+K)\mathbb{Q}_{-1}^i(dx)\\
& \quad \sum_{i:\hat{y_i}=-1} \int_{R^n}(\Vert x-\hat{x_i} \Vert+K) \mathbb{Q}_{+1}^i(dx)+\Vert x-\hat{x_i} \Vert\mathbb{Q}_{-1}^i(dx)] \leq \epsilon \\
& \int_\Xi \mathbb{Q}_{+1}^i(dx)+\mathbb{Q}_{-1}^i(dx)=1 \\
=&\sup_{\mathbb{Q}_{±1}^i\geq 0} \frac{1}{N}\sum_{i=1}^N
\int_{R^n}l_\beta(x,1)\mathbb{Q}_{+1}^i(dx)+l_\beta(x,-1)\mathbb{Q}_{-1}^i(dx) \\
s.t &\frac{1}{N} [\int_{R^n} K \sum_{i:\hat{y_i}=1}\mathbb{Q}_{-1}^i(dx)+ K \sum_{i:\hat{y_i}=-1}\mathbb{Q}_{+1}^i(dx) + \sum_{i=1}^N \Vert x-\hat{x_i} \Vert (\mathbb{Q}_{+1}^i(dx)+\mathbb{Q}_{-1}^i(dx))] \leq \epsilon \\
& \int_\Xi \mathbb{Q}_{+1}^i(dx)+\mathbb{Q}_{-1}^i(dx)=1 \\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\yx9bOQM3Gx.svg"></div>

对约束分别引入拉格朗日乘子，写对偶：
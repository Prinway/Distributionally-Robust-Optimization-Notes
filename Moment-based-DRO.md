# Moment-based DRO 基于矩的分布鲁棒优化

## 从投资组合优化问题开始

考虑一个单周期投资组合优化问题。假设共有 $n$ 种资产， $x_i$ 为第 $i$ 种资产的回报（是随机的）， $w_i$ 为分配到第 $i$ 种资产的投资权重，则投资总汇报为 $w^Tx$ 。

一个非常著名的衡量投资风险的度量是在险价值（Value-at-Risk，VaR），其定义为：

$$VaR(w)=\inf\{r:P(r\leq-w^Tx)\leq \epsilon\}$$

即：使得投资损失超过 $r$ 的概率不超过 $\epsilon$ 的最小 $r$ 。

![img](assets/Landing_page_2-6.jpg)

## 基于矩的模糊集

我们不知道 $x$ 的准确分布，但是我们可以得知关于准确分布的一些信息，尤其是一阶和二阶矩信息，比如均值 $\bar{x}$ 和协方差 $\Gamma$ 。引入基于矩的模糊集 $\mathcal{P}$ ，考虑VaR的分布鲁棒模型DR-VaR：

$$VaR_{DR}=\inf\quad r$$

$$s.t \quad\sup_{\mathbb{P} \in \mathcal{P}} P_{\mathbb{P}}(r\leq-w^Tx)\leq \epsilon$$

其中， $\mathbb{P}$ 是 $x$ 的分布， $\mathcal{P}$ 是包含所有均值为 $\bar{x}\in R^n$ 且方差为 $\Gamma \in S_n^{++}$ 的分布的集合。

## 分布鲁棒机会约束

上述DR-VaR问题中的约束是一个分布鲁棒机会约束（distributionally robust chance constrain, DRCC），含义是：即使在最差的分布 $\mathbb{P}$ 下，投资损失超过 $r$ 的概率也不超过 $\epsilon$ 。这一约束无法直接求解，需要进一步处理：

$$\sup_{\mathbb{P} \in \mathcal{P}} P_{\mathbb{P}}(r\leq-w^Tx)$$

$$=\quad \sup \int_{R^n} \mathbb{I}_{\{r\leq-w^Tx\}}(x)d\mathbb{P}(x)$$

$$s.t \quad \int_{R^n}d\mathbb{P}(x)=1,$$

$$\quad \quad \int_{R^n}xd\mathbb{P}(x)=\bar{x},$$

$$\quad \quad \int_{R^n}(x-\bar{x})(x-\bar{x})^Td\mathbb{P}(x)=\Gamma,$$

$$\quad \quad \mathbb{P}(x)\geq 0.$$

其中 $\mathbb{I}_{\{A\}}$ 是指示函数，当事件A发生是取值为1，否则为0； $\mathbb{P}(x)$ 是决策变量。这个DRCC是关于 $\mathbb{P}(x)$ 的线性规划问题，前3个约束条件又可以合并写成：

<!-- $$
\quad \quad \int_{R^n}\begin{bmatrix}x \\ 1\end{bmatrix}\begin{bmatrix}x \\ 1\end{bmatrix}^Td\mathbb{P}(x)=\begin{bmatrix}\Gamma+\bar{x}\bar{x}^T & \bar{x} \\ \bar{x}^T& 1\end{bmatrix}=\Sigma\in S_{++}^{n+1}
$$ --> 

<div align="center"><img style="background: white;" src="svg\qaEnGo2z0S.svg"></div>

## 分布鲁棒机会约束的对偶问题

此时，这个DRCC依然难以求解。遇事不决，写对偶！对矩阵 $\Sigma$ 引入拉格朗日乘子 $M\in S^{n+1}$ ，考虑DRCC的拉格朗日函数：

<!-- $$
L(\mathbb{P},M)=\int_{R^n} \mathbb{I}_{\{r\leq-w^Tx\}}(x)d\mathbb{P}(x)+\langle M,\Sigma-\int_{R^n}\begin{bmatrix}x \\ 1\end{bmatrix}\begin{bmatrix}x \\ 1\end{bmatrix}^Td\mathbb{P}(x) \rangle
$$ --> 

<div align="center"><img style="background: white;" src="svg\dSCWgn65FL.svg"></div>

<!-- $$
\sup_{\mathbb{P}\geq 0} L(\mathbb{P},M)=\langle M,\Sigma \rangle+\sup_{\mathbb{P}\geq0}\int_{R^n} [\mathbb{I}_{\{r\leq-w^Tx\}}(x)-\begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix}]d\mathbb{P}(x)
$$ --> 

<div align="center"><img style="background: white;" src="svg\Y5INpBlCUt.svg"></div>

$$=\quad\sup_{\mathbb{P}\geq 0}\langle M,\Sigma \rangle$$

<!-- $$
s.t\quad\mathbb{I}_{\{r\leq-w^Tx\}}(x)\leq \begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix} \quad \forall x
$$ --> 

<div align="center"><img style="background: white;" src="svg\erum1kZq9z.svg"></div>

由于 $\forall x$ 的存在，此问题仍然包含了无穷多个约束。进一步地，对指示函数进行拆解，可得到以下两个条件：

<!-- $$
\begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix} \geq 0 \quad \forall x
$$ --> 

<div align="center"><img style="background: white;" src="svg\WzeN9sG9se.svg"></div>

<!-- $$
\begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix} \geq 1 \quad \forall x:r\leq-w^Tx
$$ --> 

<div align="center"><img style="background: white;" src="svg\7BWRkO8NIH.svg"></div>

第一个条件直接等价于 $M \in S_+^{n+1}$ 。第二个条件可以等价地写作：

<!-- $$
1 \leq \inf \begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix}
$$ --> 

<div align="center"><img style="background: white;" src="svg\7OQeAhkzsI.svg"></div>

$$\quad s.t \quad r+w^Tx \leq 0$$

对不等式约束引入拉格朗日乘子 $\tau$ ，则第二个条件等价于：

$$\exists \tau \geq 0$$

<!-- $$
s.t \quad \begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix}+\tau(r+w^Tx) \geq 1 \quad \forall x \in R^n
$$ --> 

<div align="center"><img style="background: white;" src="svg\TUJ2jFXhn5.svg"></div>

即：

$$\exists \tau \geq 0$$

<!-- $$
s.t \quad \begin{bmatrix}x \\ 1\end{bmatrix}^T(M+\begin{bmatrix}0 & \frac{1}{2} \tau w \\ \frac{1}{2} \tau w ^T & \tau r-1\end{bmatrix})\begin{bmatrix}x \\ 1\end{bmatrix} \geq 0 \quad \forall x \in R^n
$$ --> 

<div align="center"><img style="background: white;" src="svg\H1BzPJuMVW.svg"></div>

所以，DRCC的对偶问题为：

$$\inf_{M\in S^{n+1}} \sup_{\mathbb{P}\geq 0} L(\mathbb{P},M)$$

$$=\inf\langle M,\Sigma \rangle$$

<!-- $$
s.t \quad M+\begin{bmatrix}0 & \frac{1}{2} \tau w \\ \frac{1}{2} \tau w ^T & \tau r-1\end{bmatrix} \in S_+^{n+1}
$$ --> 

<div align="center"><img style="background: white;" src="svg\EK4vhNjHCE.svg"></div>

$$\quad \quad M\in S_+^{n+1}$$

$$\quad \quad \tau \geq 0$$

## 引入统计量的估计

## 分布鲁棒线性规划

## 分布鲁棒线性规划的对偶问题






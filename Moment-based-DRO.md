# Moment-based DRO 基于矩的分布鲁棒优化

## 从投资组合优化问题开始

考虑一个单周期投资组合优化问题。假设共有 $n$ 种资产， $x_i$ 为第 $i$ 种资产的回报（是随机的）， $w_i$ 为分配到第 $i$ 种资产的投资权重，则投资总回报为 $w^Tx$ 。

一个非常著名的衡量投资风险的度量是在险价值（Value-at-Risk，VaR），其定义为：

$$VaR(w)=\inf\{r:P(r\leq-w^Tx)\leq \epsilon\}$$

即：使得投资损失超过 $r$ 的概率不超过 $\epsilon$ 的最小 $r$ 。

![img](assets/Landing_page_2-6.jpg)

## 基于矩的模糊集

我们不知道 $x$ 的准确分布，但是我们可以得知关于准确分布的一些信息，尤其是一阶和二阶矩信息，比如均值 $\mu$ 和协方差 $\Gamma$ 。引入基于矩的模糊集 $\mathcal{P}$ ，考虑VaR的分布鲁棒模型DR-VaR：

<!-- $$
\begin{array}{ll}
\inf & r \\
s.t & \sup_{\mathbb{P} \in \mathcal{P}} P_{\mathbb{P}}(r\leq-w^Tx)\leq \epsilon
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\oJRGiGcJyi.svg"></div>

其中， $\mathbb{P}$ 是 $x$ 的分布， $\mathcal{P}$ 是包含所有均值为 $\mu\in R^n$ 且方差为 $\Gamma \in S_n^{++}$ 的分布的集合。

## 分布鲁棒机会约束

上述DR-VaR问题中的约束是一个分布鲁棒机会约束（distributionally robust chance constrain, DRCC），含义是：即使在最差的分布 $\mathbb{P}$ 下，投资损失超过 $r$ 的概率也不超过 $\epsilon$ 。这一约束无法直接求解，需要进一步处理：

<!-- $$
\begin{array}{ll}
& \sup_{\mathbb{P} \in \mathcal{P}} \quad P_{\mathbb{P}}(r\leq-w^Tx) \\
= & \sup \quad \int_{R^n} \mathbb{I}_{\{r\leq-w^Tx\}}(x)d\mathbb{P}(x) \\
s.t & \int_{R^n}d\mathbb{P}(x)=1, \\
& \int_{R^n}xd\mathbb{P}(x)=\mu,\\
& \int_{R^n}(x-\mu)(x-\mu)^Td\mathbb{P}(x)=\Gamma, \\
& \mathbb{P}(x)\geq 0. \\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\h0GiDlAjdh.svg"></div>

其中 $\mathbb{I}_{\{A\}}$ 是指示函数，当事件A发生是取值为1，否则为0； $\mathbb{P}(x)$ 是决策变量。这个DRCC是关于 $\mathbb{P}(x)$ 的线性规划问题，前3个约束条件又可以合并写成：

<!-- $$
\quad \quad \int_{R^n}\begin{bmatrix}x \\ 1\end{bmatrix}\begin{bmatrix}x \\ 1\end{bmatrix}^Td\mathbb{P}(x)=\begin{bmatrix}\Gamma+\mu\mu^T & \mu \\ \mu^T& 1\end{bmatrix}=\Sigma\in S_{++}^{n+1}
$$ --> 

<div align="center"><img style="background: white;" src="svg\kqmx4J6TCf.svg"></div>

## 分布鲁棒机会约束的对偶问题

此时，这个DRCC依然难以求解。遇事不决，写对偶！对矩阵 $\Sigma$ 引入拉格朗日乘子 $M\in S^{n+1}$ ，考虑DRCC的拉格朗日函数：

<!-- $$
L(\mathbb{P},M)=\int_{R^n} \mathbb{I}_{\{r\leq-w^Tx\}}(x)d\mathbb{P}(x)+\langle M,\Sigma-\int_{R^n}\begin{bmatrix}x \\ 1\end{bmatrix}\begin{bmatrix}x \\ 1\end{bmatrix}^Td\mathbb{P}(x) \rangle
$$ --> 

<div align="center"><img style="background: white;" src="svg\dSCWgn65FL.svg"></div>


<!-- $$
\begin{array}{ll}
& \sup_{\mathbb{P}\geq 0} L(\mathbb{P},M) \\ = & \langle M,\Sigma \rangle+\sup_{\mathbb{P}\geq0}\int_{R^n} [\mathbb{I}_{\{r\leq-w^Tx\}}(x)-\begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix}]d\mathbb{P}(x) \\
= & \sup_{\mathbb{P}\geq 0}\langle M,\Sigma \rangle \\
& s.t \quad \mathbb{I}_{\{r\leq-w^Tx\}}(x)\leq \begin{bmatrix}x \\ 1\end{bmatrix}^TM\begin{bmatrix}x \\ 1\end{bmatrix} \quad \forall x
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\YzViGICWKo.svg"></div>

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
s.t \quad \begin{bmatrix}x \\ 1\end{bmatrix}^T(M+\begin{bmatrix}0 & -\frac{1}{2} \tau w \\ -\frac{1}{2} \tau w ^T & 1-\tau r\end{bmatrix})\begin{bmatrix}x \\ 1\end{bmatrix} \geq 0 \quad \forall x \in R^n
$$ --> 

<div align="center"><img style="background: white;" src="svg\v4Rp2UNEd3.svg"></div>

所以，DRCC的对偶问题为：

<!-- $$
\begin{array}{ll}
& \inf_{M\in S^{n+1}} \sup_{\mathbb{P}\geq 0} \quad L(\mathbb{P},M) \\
= & \inf \quad \langle M,\Sigma \rangle \\
& s.t \quad M+\begin{bmatrix}0 & \frac{1}{2} \tau w \\ \frac{1}{2} \tau w ^T & \tau r-1\end{bmatrix} \in S_+^{n+1} \\
& \quad \quad M\in S_+^{n+1} \\
& \quad \quad \tau \geq 0 \\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\eVuJteLQfs.svg"></div>

这个问题是一个半正定锥规划问题。另外可以证明，对于DRCC问题，强对偶性成立。至此，DR-VaR问题已经容易求解。

## 引入统计量的估计

前面的模糊集依赖于均值和方差的真实值，但实际上，很多时候我们只能对这些统计量进行估计。如果我们有随机变量的 $N$ 个独立同分布的样本 $\xi_i$ ，就可以引入均值和方差的估计量：

$$\hat{\mu}=\frac{1}{N} \sum_{i=1}^{N} \xi_i$$

$$\hat{\Gamma}=\frac{1}{N} \sum_{i=1}^{N} (\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^T$$

为了在构建模糊集时考虑估计误差，我们可以考虑均值的估计量在一个以均值的真实值为中心的椭圆中：

$$(\mu-\hat{\mu})^T\hat{\Gamma}^{-1}(\mu-\hat{\mu})\leq \gamma_1$$

其中， $\mu=\mathbb{E}_{\mathbb{P}}[\xi]$ 。类似地，我们可以考虑利用方差的估计量给方差设置一个上界：

$$\mathbb{E}_{\mathbb{P}} [(\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^T] \preceq \gamma_2 \hat{\Gamma}$$

不过，此时 $\mathbb{E}_{\mathbb{P}} [(\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^T]$ 并不是方差的真实值，因为我们使用了 $\hat{\mu}$ 而不是 $\mu$ 。

在支撑集 $C\in R^n$ 上考虑满足以上条件的分布 $\mathbb{P}$，我们可以构建模糊集：

<!-- $$
\mathcal{P}=\mathcal{P}(C,\hat{\mu},\hat{\Gamma},\gamma_1,\gamma_2)=
\left\{\begin{array}{ll}
\mathbb{P}\geq0 : & 
\begin{array}{l}
\mathbb{P}(C)=1,  \\
(\mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu})^T\hat{\Gamma}^{-1}(\mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu})\leq \gamma_1,  \\
\mathbb{E}_{\mathbb{P}} [(\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^T] \preceq \gamma_2 \hat{\Gamma}.
\end{array}
\end{array}
\right\}
$$ --> 

<div align="center"><img style="background: white;" src="svg\8jcLjAmjfo.svg"></div>

## 分布鲁棒线性规划

考虑以下分布鲁棒线性规划问题：

$$\inf_{x\in \mathcal{U}} \sup_{\mathbb{P}\in \mathcal{P}} \quad \mathbb{E}_{\mathbb{P}}[h(x,\xi)]$$

其中 $h: R^m \times R^n \rightarrow R$ 是一个带有不确定性的成本函数， $x$ 为决策变量， $\xi$ 为扰动向量。

首先，我们从内层的最大化问题出发：

<!-- $$
\begin{array}{ll}
& \sup_{\mathbb{P}\in \mathcal{P}} \quad \mathbb{E}_{\mathbb{P}}[h(x,\xi)] \\
= & \sup \quad \int_{C} h(x,\xi) d\mathbb{P}(\xi) \\
s.t & \int_{C}d\mathbb{P}(\xi)=1, \\
& \int_{C}(\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^Td\mathbb{P}(\xi) \preceq \gamma_2 \hat{\Gamma},\\
& \int_{C} \begin{bmatrix} \hat{\Gamma} & \xi-\hat{\mu} \\ (\xi-\hat{\mu})^T & \gamma_1\end{bmatrix} d\mathbb{P}(\xi) \succeq 0,\\
& \mathbb{P}\geq 0. \\
= & \sup \quad \int_{C} h(x,\xi) d\mathbb{P}(\xi) \\
s.t & \int_{C}d\mathbb{P}(\xi)=1, \\
& \int_{C}(\xi\xi^T-\xi\hat{\mu}^T-\hat{\mu}\xi^T)d\mathbb{P}(\xi) \preceq \gamma_2 \hat{\Gamma}-\hat{\mu}\hat{\mu}^T,\\
& \begin{bmatrix} \hat{\Gamma} & \int_C\xi d\mathbb{P}(\xi) -\hat{\mu} \\ \int_C\xi^T d\mathbb{P}(\xi) -\hat{\mu}^T & \gamma_1\end{bmatrix} \succeq 0,\\
& \mathbb{P}\geq 0. \\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\nFQ8CK01qX.svg"></div>

转化的过程中用到了舒尔补（Schur complement）：

<!-- $$
\begin{array}{ll}
& (\mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu})^T\hat{\Gamma}^{-1}(\mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu})\leq \gamma_1 \\
\Leftrightarrow & \begin{bmatrix} \hat{\Gamma} & \mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu} \\ (\mathbb{E}_{\mathbb{P}}[\xi]-\hat{\mu})^T & \gamma_1\end{bmatrix} \succeq 0 \\ 
\Leftrightarrow & \mathbb{E}_{\mathbb{P}}\begin{bmatrix} \hat{\Gamma} & \xi-\hat{\mu} \\ (\xi-\hat{\mu})^T & \gamma_1\end{bmatrix} \succeq 0\\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\OvjKxwqaZ3.svg"></div>

同样地，写对偶。对三个约束条件分别引入拉格朗日乘子 $r\in R$ 、 $Q \succeq 0$ 和 $M \succeq 0$ ，其中：

<!-- $$
M=\begin{bmatrix} P & p \\ p^T & s\end{bmatrix}
$$ --> 

<div align="center"><img style="background: white;" src="svg\NsYRGiu90E.svg"></div>

于是，可以写出内层问题的拉格朗日函数：

<!-- $$
\begin{array}{ll}
& L(\mathbb{P},r,Q,P,p,s) \\
= & \int_{C} h(x,\xi) d\mathbb{P}(\xi)+r(1-\int_{C}d\mathbb{P}(\xi)) \\ - & \langle Q,\int_{C}(\xi\xi^T-\xi\hat{\mu}^T-\hat{\mu}\xi^T)d\mathbb{P}(\xi) - \gamma_2 \hat{\Gamma}+\hat{\mu}\hat{\mu}^T \rangle \\ + & \langle P,\hat{\Gamma}\rangle + 2p^T(\int_C\xi d\mathbb{P}(\xi) -\hat{\mu}) + s\gamma_1
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\tXpzXyLtdF.svg"></div>

内层问题的对偶问题为：

<!-- $$
\begin{array}{ll}
& \inf_{\begin{array}{l}r \in R\\ Q \succeq 0\\M\succeq 0 \end{array}} \sup_{\mathbb{P}\geq 0} \quad L(\mathbb{P},r,Q,P,p,s) \\ 
= &\inf \sup \quad \int_C [h(x,\xi)-r-\xi^T Q\xi+2\xi^T Q\hat{\mu}+2p^T\xi] d\mathbb{P}(\xi) \\
& +r+\langle Q,\gamma_2 \hat{\Gamma}-\hat{\mu}\hat{\mu}^T \rangle+\langle P,\hat{\Gamma} \rangle-2p^T\hat{\mu}+s\gamma_1 \\ = & \inf \quad r+\langle Q,\gamma_2 \hat{\Gamma}-\hat{\mu}\hat{\mu}^T \rangle+\langle P,\hat{\Gamma} \rangle-2p^T\hat{\mu}+s\gamma_1 \\ s.t & h(x,\xi)-r-\xi^T Q\xi+2\xi^T Q\hat{\mu}+2p^T\xi \leq 0, \quad \forall \xi \in C \\ & r \in R\\ & Q \succeq 0\\ & M\succeq 0
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\nnv5zilYnt.svg"></div>

此时，问题还可进一步简化。由于：

<!-- $$
M=\begin{bmatrix} P & p \\ p^T & s\end{bmatrix} \succeq 0
$$ --> 

<div align="center"><img style="background: white;" src="svg\VsR5Xmo84E.svg"></div>

所以，若问题存在最优解，则 $P=\frac{1}{s}p{p}^T$ 或 $s=0$ 。在这两种情况下，都有 $s=\frac{1}{\sqrt{\gamma_1}} {\sqrt{p^T \hat{\Gamma} p}}=\frac{1}{\sqrt{\gamma_1}} \Vert \hat{\Gamma}^{\frac{1}{2}}p \Vert_2$ 。再令 $q=p+Q\hat{\mu}$ ，则内层问题的对偶问题简化为：

<!-- $$
\begin{array}{ll}
\inf & r+\gamma_2\langle \hat{\Gamma},Q \rangle+\hat{\mu}^T Q\hat{\mu} -2\hat{\mu}^Tq+2\sqrt{\gamma_1} \Vert \Gamma^{\frac{1}{2}} (q-Q\hat{\mu}) \Vert_2 \\
s.t & h(x,\xi)-r-\xi^T Q\xi+2\xi^T q \leq 0, \quad \forall \xi \in C \\ & r \in R\\ & Q \succeq 0\\ & q \in R^n
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\7WCg7BNZzU.svg"></div>

wow，这又是一个二阶锥规划问题！而且可以证明，强对偶性成立。所以，分布鲁棒线性规划问题被写为：

<!-- $$
\begin{array}{ll}
& \inf_{x\in \mathcal{U}} \sup_{\mathbb{P}\in \mathcal{P}} \quad \mathbb{E}_{\mathbb{P}}[h(x,\xi)] \\
= & \inf \quad r+t \\
s.t & h(x,\xi)-r-\xi^T Q\xi+2\xi^T q \leq 0, \quad \forall \xi \in C \\
& t \geq \gamma_2\langle \hat{\Gamma},Q \rangle+\langle \hat{\mu}\hat{\mu}^T,Q\rangle -2\hat{\mu}^Tq+2\sqrt{\gamma_1} \Vert \Gamma^{\frac{1}{2}} (q-Q\hat{\mu}) \Vert_2, \\ 
& r\in R,\quad Q \succeq 0,\quad q \in R^n,\quad x\in \mathcal{U}, \quad t\in R
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\d0T3RmL30A.svg"></div>

此问题现在已经变为一个经典鲁棒优化问题，可以用经典鲁棒优化的算法求解。进一步地，在一些假设下，此问题可以在多项式时间内找到最优解。

## 模糊集大小的选择

现在还剩下一个问题，模糊集的大小如何选择？模糊集越大，其包含真实分布的可能性就越高，但模型也越保守。因此，我们需要在模糊集的大小上做一些权衡。具体到上面的分布鲁棒线性规划问题，就是如何选择合适的 $\gamma_1$ 和 $\gamma_2$ 。

回顾经验均值 $\hat{\mu}$ 和经验方差 $\hat{\Gamma}$ 的计算方法：

$$\hat{\mu}=\frac{1}{N} \sum_{i=1}^{N} \xi_i$$

$$\hat{\Gamma}=\frac{1}{N} \sum_{i=1}^{N} (\xi_i-\hat{\mu})(\xi_i-\hat{\mu})^T$$

先选择 $\gamma_1$ 。设 $\mu$ 和 $\Gamma$ 分别为真实分布 $\mathbb{P^*}$ 下 $\xi$ 的均值和方差。假设真实分布的支撑集有界，即：

$$\exists R\geq 0: \quad P_\mathbb{P^*}((\xi-\mu)^T\Gamma^{-1}(\xi-\mu)\leq R^2)=1$$

考虑以下概率：

<!-- $$
\begin{array}{ll}
& P_\mathbb{P^*}((\hat{\mu}-\mu)^T\Gamma^{-1}(\hat{\mu}-\mu)\leq t) \\
= & P_\mathbb{P^*}(\Vert \Gamma^{-\frac{1}{2}}(\hat{\mu}-\mu) \Vert_2^2 \leq t) \\
= & P_\mathbb{P^*}(\Vert \Gamma^{-\frac{1}{2}}(\frac{1}{N} \sum_{i=1}^{N} \xi_i-\mu) \Vert_2^2 \leq t) \\
= & P_\mathbb{P^*}(\Vert \frac{1}{N} \sum_{i=1}^{N} \Gamma^{-\frac{1}{2}} (\xi_i-\mu) \Vert_2^2 \leq t) \\
= & P_\mathbb{P^*}(\Vert \frac{1}{N} \sum_{i=1}^{N} \zeta_i \Vert_2^2 \leq t) \\
\end{array}
$$ --> 

<div align="center"><img style="background: white;" src="svg\bTZzN3Z8pp.svg"></div>

其中， $\zeta=\Gamma^{-\frac{1}{2}} (\xi-\mu)$ 的均值为 $0$ ，方差为 $I$ 。根据 Mcdiarmid 不等式，选择 $t=\frac{R^2}{N}(2+\sqrt{2\ln\frac{1}{\delta}})^2$ 时，有：

$$P_\mathbb{P^*}(\Vert \frac{1}{N} \sum_{i=1}^{N} \zeta_i \Vert_2^2 \leq \frac{R^2}{N}(2+\sqrt{2\ln\frac{1}{\delta}})^2) \geq 1-\delta$$

还原为：

$$P_\mathbb{P^*}((\hat{\mu}-\mu)^T\Gamma^{-1}(\hat{\mu}-\mu) \leq \frac{R^2}{N}(2+\sqrt{2\ln\frac{1}{\delta}})^2) \geq 1-\delta$$

最后需要处理的一件事情是，上式中出现的是真实方差 $\Gamma$ ，但我们在定义 $\mathcal{P}$ 时用的是经验方差 $\hat{\Gamma}$，所以需要对经验方差 $\hat{\Gamma}$ 和真实方差 $\Gamma$ 之间的距离进行限制。

使用类似的思路，可以选择 $\gamma_2$ 。利用一些集中不等式进行推导可以证明，当样本数量足够多时，在给定 $\delta$ 的情况下，可使用以下式子选择 $\gamma_1$ 和 $\gamma_2$ ：

$$\gamma_1=\frac{\beta}{1-\alpha-\beta}$$

$$\gamma_2=\frac{1+\beta}{1-\alpha-\beta}$$

其中：

$$\alpha=\frac{R^2}{\sqrt{N}}(\sqrt{1-\frac{n}{R^4}}+\sqrt{2\ln\frac{4}{\delta}})$$

$$\beta=\frac{R^2}{\sqrt{N}}(2+\sqrt{2\ln\frac{2}{\delta}})^2$$

$n$ 为 $\xi$ 的维度。

## 参考文献

- Delage, E., & Ye, Y. (2010). Distributionally Robust Optimization Under Moment Uncertainty with Application to Data-Driven Problems. Operations Research, 58(3), 595–612.

- Ghaoui, L. E., Oks, M., & Oustry, F. (2003). Worst-Case Value-At-Risk and Robust Portfolio Optimization: A Conic Programming Approach. Operations Research, 51(4), 543–556.

- Shapiro, A. (2001). On Duality Theory of Conic Linear Problems. In M. Á. Goberna & M. A. López (Eds.), Semi-Infinite Programming (Vol. 57, pp. 135–165). Springer US.








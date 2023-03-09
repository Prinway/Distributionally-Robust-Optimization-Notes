# Conic Linear Programming 锥线性规划

## 二阶锥

二阶锥（second-order cone，又称ice-cream/Lorentz cone）的形式为：
 $$
Q^{n+1}=\{(t,x)\in \mathbb{R} \times \mathbb{R}^n:t\geq \Vert x\Vert_2\}
$$ 
即：满足 $t\geq \Vert x\Vert_2$ 的所有 $(t,x)$ 。

$Q$ 的上标表示了它的维度。因 $t$ 为1维， $x$ 为 $n$ 维向量，故 $Q$ 为 $n+1$ 维。从几何的角度看，在 $x$ 为2维的情况下，二阶锥表示了一个圆锥上方的所有空间，所以也被称为“冰淇淋锥”。

## 非负象限锥

非负象限锥很简单，即 $R_+^n$ 。在2维的情况下，指的就是第一象限。

## 半正定锥

设 $S^n$ 为 $n\times n$ 的对称矩阵，则半正定锥（semidefinite cone）的形式为：
 $$
S_+^n=\{Y\in S^n:u^TYu\geq 0\quad \forall u\in \mathbb{R}^n \}
$$ 

即：包含所有的 $n$ 阶对称半正定矩阵。

## 锥线性规划与线性规划的关系

为什么说锥线性规划是线性规划的拓展呢？回顾线性规划标准型：
 $$
\begin{aligned}
\min &\quad c^Tx \\
s.t &\quad Ax=b \\
 &\quad x\geq 0
\end{aligned}
$$ 
从锥规划的角度看， $x\geq0$ 实际上就是 $x\in R_+^n$ 。当我们把此处的非负象限锥 $R_+^n$ 改为二阶锥或者半正定锥，自然就从线性规划推广出了锥线性规划。
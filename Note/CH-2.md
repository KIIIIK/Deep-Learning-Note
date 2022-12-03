# Linear Algebra

## Norms

$Frobenius \ norm$: $||A||_F = \sqrt{\sum_{i,j}A_{i,j}^2}$

## Eigendecomposition

$A = V diag(\lambda)V^{-1}$，如果$A$是实对称矩阵的话，则可以分解为$A = Q \Lambda Q^T$，其中Q是正交矩阵，$A$是方阵。

## Singular Value Decomposition

$A=UDV^T$，$A$是$m \times n$的矩阵，$U$是$m \times m$的矩阵，$D$是$m \times n$的矩阵，$V$是$n \times n$的矩阵。$U,V$都是正交矩阵，$D$是对角阵，但不一定得是方阵。
$D$的对角元素被称为奇异值，$U$的各列向量称为左奇异向量，$V$的各列向量被称为右奇异向量。左奇异向量是$AA^T$的特征向量，右奇异向量是$A^TA$的特征向量。$A$的非0奇异值是$A^TA$特征值的平方根，也是$AA^T$特征值的平方根。
所有的矩阵都可以进行奇异值分解。

## The Moore-Penrose Pseudoinverse

$A$伪逆的定义是$A^+ = \underset{\alpha \rightarrow 0}{lim}(A^T A + \alpha I)^{-1}A^T$，通常不这样计算，而是采用奇异值分解来计算，$A^+ = V D^+ U^T$，其中$U,D,V$是$A$的奇异值分解，$D^+$是对角阵$D$非零元素的倒数，然后再转置。
当$A$的列数多于行数时，解$Ax=y$时可采用伪逆，即$x=A^+ y$，这时的$x$具有最小的欧式范数。
当$A$的行数多于列数时，这时$Ax=y$可能是无解的，如果用伪逆计算的话，这时欧式范数$||Ax-y||_2$最小。

## The Trace Operator

$Tr(A) = \underset{i} \sum A_{i,i}$。
迹的操作可以使一些计算变得简便，如$||A||_F = \sqrt {Tr(AA^T)}$
公式：$Tr(ABC) = Tr(CAB) = Tr(BCA)$
如果$A \in \mathbb{R}^{m \times m}$，$B \in \mathbb{R}^{n \times m}$，有$Tr(AB) = Tr(BA)$，即使$AB \in \mathbb{R}^{m \times m}$，$BA \in \mathbb{R}^{n \times n}$。

## Principal Components Analysis

PCA是有损压缩，会丢失一部分信息。
假设我们现在有$\lbrace x^{(1)}, ..., x^{(m)} \rbrace \in \mathbb{R}^n$，也就是有$m$个样本，每个样本都是n维的列向量。
对于每一个样本$x^{(i)} \in \mathbb{R}^n$如果有相应的$c^{(i)} \in \mathbb{l}$，$l$远远小于$n$，就可以用更少的空间去存储原始数据。
我们想要找到一个编码函数，使得$f(\pmb{x})=c$，和一个解码函数，使得$x \approx g(c)=g(f(x))$。
PCA取决于我们解码函数的选择。为了让解码变得简单，我们选择用矩阵操作，$g(c)=Dc$，$D \in \mathbb{R}^{n \times l}$，是解码矩阵。
为了让编码更容易，PCA限制$D$的各个列向量相互正交，但这并不意味着$D$是正交矩阵，因为正交矩阵必须得是方阵。此时$D$仍然不是唯一的，为了让$D$是唯一的，再限制$D$的列向量都已单位化。
我们要寻找最优的$c^*=\underset{c} {\arg\min} ||x-g(c)||_2 = \underset{c} {\arg\min} ||x-g(c)||_2^2$。
又可以简化为
$$
\begin{aligned}
||x-g(c)||_2 &=(x-g(c))^T(x-g(c)) \\
&=x^Tx - x^Tg(c) - g(c)^Tx + g(c)^Tg(c) \\
&=x^Tx - 2x^Tg(c) + g(c)^Tg(c)
\end{aligned}
$$
$(x^Tg(c)和g(c)^Tx都为常数且相等)$
首项不含$c$，所以
$$
\begin{aligned}
c^* &= \underset{c}{\arg\min}-2x^Tg(c) + g(c)^Tg(c) \\
&= \underset{c}{\arg\min}-2x^TDc + c^TD^TDc \\
&= \underset{c}{\arg\min}-2x^TDc + c^TI_lc \\
&= \underset{c}{\arg\min}-2x^TDc + c^Tc
\end{aligned}
$$
对$c$进行求导
$$
\begin{aligned}
\nabla_c(-2x^TDc + c^Tc) = 0 \\
-2D^Tx + 2c = 0 \\
c = D^Tx
\end{aligned}
$$
所以可得编码函数$f(x)=D^TX$。
定义PCA重构操作为:
$$
r(x) = g(f(x)) = DD^Tx
$$
接下来就要选择编码矩阵$D$，考虑所有的样本有:
$$
D^*= \underset{D}{\arg\min} \sqrt{\underset{i,j}{\sum}(x_j^{(i)}-r(x^{(i)})_j)^2} \\
subject \ to(D^TD = I_l)
$$
简化起见，考虑$l=1$的情况，这时候的$D$就仅仅是一个$n \times 1$的列向量，用$d$来表示，可得
$$
d^* = \underset{d}{\arg\min} \underset{i}{\sum}||x^{(i)}-dd^Tx^{(i)}||_2^2 \quad subject \ to ||d||_2 = 1
$$
因为$d^Tx^{(i)}$是一个常数，所以上式可变为
$$
\begin{aligned}
d^*&= \underset{d}{\arg\min} \underset{i}{\sum}||x^{(i)}-d^Tx^{(i)}d||_2^2  \\
&=\underset{d}{\arg\min} \underset{i}{\sum}||x^{(i)}-x^{(i)T}dd||_2^2
\end{aligned} \quad subject \ to ||d||_2 = 1
$$
上面我们都是只考虑了一个样本点，接下来考虑所有的样本点，可得
$$
\begin{aligned}
d^* &= \underset{d}{\arg\min} ||X-Xdd^T||_F^2 \quad subject \ to \ d^Td = 1 \\
&= \underset{d}{\arg\min} Tr((X-XddT)^T(X-XddT)) \\
&= \underset{d}{\arg\min} Tr(X^TX-X^TXdd^T-dd^TX^TX+dd^TX^TXdd^T) \\
&= \underset{d}{\arg\min} Tr(X^TX)-Tr(X^TXdd^T)-Tr(dd^TX^TX)+Tr(dd^TX^TXdd^T) \\
&= \underset{d}{\arg\min} -Tr(X^TXdd^T)-Tr(dd^TX^TX)+Tr(dd^TX^TXdd^T) \\
&= \underset{d}{\arg\min} -2Tr(X^TXdd^T)+Tr(dd^TX^TXdd^T) \\
&= \underset{d}{\arg\min} -2Tr(X^TXdd^T)+Tr(X^TXdd^Tdd^T) \\
&= \underset{d}{\arg\min} -2Tr(X^TXdd^T)+Tr(X^TXdd^T) \quad d^Td=1 \\
&= \underset{d}{\arg\min} -Tr(X^TXdd^T) \\
&= \underset{d}{\arg\max} Tr(X^TXdd^T) \\
&= \underset{d}{\arg\max} Tr(d^TX^TXd) \quad subject \ to \ d^Td = 1
\end{aligned}
$$
所以，最优的$d$就是$X^TX$最大特征值对应的特征向量。

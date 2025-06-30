# 附录：数学基础

本附录包含模式识别中常用的数学知识和计算方法，为理解和实现各种算法提供数学基础。

## 1. 行列式的计算方法

### 1.1 定义与基本性质

**定义**：$n$ 阶方阵 $\mathbf{A} = [a_{ij}]_{n \times n}$ 的行列式定义为：
$$\det(\mathbf{A}) = |\mathbf{A}| = \sum_{p} (-1)^{\tau(p)} a_{1p_1} a_{2p_2} \cdots a_{np_n}$$

其中 $p = (p_1, p_2, \ldots, p_n)$ 是 $(1, 2, \ldots, n)$ 的一个排列，$\tau(p)$ 是排列 $p$ 的逆序数。

**基本性质**：
1. $\det(\mathbf{A}^T) = \det(\mathbf{A})$
2. $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
3. $\det(k\mathbf{A}) = k^n\det(\mathbf{A})$（$\mathbf{A}$ 为 $n$ 阶矩阵）
4. 交换两行（列），行列式变号
5. 一行（列）乘以常数加到另一行（列），行列式不变

### 1.2 具体计算方法

#### 方法一：代数余子式展开（适用于小矩阵）

对于 $n$ 阶矩阵，按第 $i$ 行展开：
$$\det(\mathbf{A}) = \sum_{j=1}^{n} a_{ij} A_{ij} = \sum_{j=1}^{n} a_{ij} (-1)^{i+j} M_{ij}$$

其中 $M_{ij}$ 是元素 $a_{ij}$ 的余子式，$A_{ij}$ 是代数余子式。

**示例**：计算 $3 \times 3$ 矩阵行列式
$$\mathbf{A} = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{pmatrix}$$

按第一行展开：
$$|\mathbf{A}| = 1 \cdot \begin{vmatrix} 5 & 6 \\ 8 & 9 \end{vmatrix} - 2 \cdot \begin{vmatrix} 4 & 6 \\ 7 & 9 \end{vmatrix} + 3 \cdot \begin{vmatrix} 4 & 5 \\ 7 & 8 \end{vmatrix}$$
$$= 1 \cdot (45-48) - 2 \cdot (36-42) + 3 \cdot (32-35) = -3 + 12 - 9 = 0$$

#### 方法二：高斯消元法（适用于大矩阵）

通过初等行变换将矩阵化为上三角矩阵，然后计算对角线元素的乘积。

**算法步骤**：
1. 对矩阵进行初等行变换，化为上三角矩阵 $\mathbf{U}$
2. 记录行交换次数 $s$
3. $\det(\mathbf{A}) = (-1)^s \prod_{i=1}^{n} u_{ii}$

#### 方法三：LU分解法

将矩阵分解为 $\mathbf{A} = \mathbf{LU}$，其中 $\mathbf{L}$ 是下三角矩阵，$\mathbf{U}$ 是上三角矩阵。
$$\det(\mathbf{A}) = \det(\mathbf{L})\det(\mathbf{U}) = \prod_{i=1}^{n} l_{ii} \prod_{i=1}^{n} u_{ii}$$

### 1.3 特殊矩阵的行列式

- **对角矩阵**：$\det(\text{diag}(d_1, d_2, \ldots, d_n)) = \prod_{i=1}^{n} d_i$
- **三角矩阵**：对角线元素的乘积
- **范德蒙德矩阵**：$\det(V) = \prod_{1 \leq i < j \leq n} (x_j - x_i)$

## 2. 矩阵逆的求法

### 2.1 逆矩阵的定义与条件

**定义**：对于 $n$ 阶方阵 $\mathbf{A}$，如果存在 $n$ 阶方阵 $\mathbf{B}$ 使得 $\mathbf{AB} = \mathbf{BA} = \mathbf{I}$，则称 $\mathbf{B}$ 为 $\mathbf{A}$ 的逆矩阵，记为 $\mathbf{A}^{-1}$。

**存在条件**：矩阵 $\mathbf{A}$ 可逆当且仅当 $\det(\mathbf{A}) \neq 0$。

**基本性质**：
1. $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
2. $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
3. $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$
4. $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$

### 2.2 求逆方法

#### 方法一：伴随矩阵法（适用于小矩阵）

$$\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \text{adj}(\mathbf{A})$$

其中伴随矩阵 $\text{adj}(\mathbf{A}) = [A_{ji}]_{n \times n}$，$A_{ij}$ 是代数余子式。

**示例**：求 $2 \times 2$ 矩阵的逆
$$\mathbf{A} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \quad \mathbf{A}^{-1} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

#### 方法二：高斯-约旦消元法（推荐）

**算法步骤**：
1. 构造增广矩阵 $[\mathbf{A} | \mathbf{I}]$
2. 通过初等行变换将左侧化为单位矩阵
3. 右侧即为 $\mathbf{A}^{-1}$

**示例**：
$$\begin{pmatrix} 2 & 1 & | & 1 & 0 \\ 3 & 2 & | & 0 & 1 \end{pmatrix} \rightarrow \begin{pmatrix} 1 & 0 & | & 2 & -1 \\ 0 & 1 & | & -3 & 2 \end{pmatrix}$$

因此 $\mathbf{A}^{-1} = \begin{pmatrix} 2 & -1 \\ -3 & 2 \end{pmatrix}$

#### 方法三：LU分解法

如果 $\mathbf{A} = \mathbf{LU}$，则求解 $\mathbf{A}^{-1}$ 相当于求解 $n$ 个线性方程组：
$$\mathbf{Ax}_i = \mathbf{e}_i, \quad i = 1, 2, \ldots, n$$

其中 $\mathbf{e}_i$ 是第 $i$ 个标准基向量。

### 2.3 数值稳定性考虑

对于数值计算，推荐使用：
1. **QR分解**：$\mathbf{A} = \mathbf{QR}$，则 $\mathbf{A}^{-1} = \mathbf{R}^{-1}\mathbf{Q}^T$
2. **SVD分解**：$\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$，则 $\mathbf{A}^{-1} = \mathbf{V}\boldsymbol{\Sigma}^{-1}\mathbf{U}^T$

## 3. 矩阵广义逆（伪逆）

### 3.1 定义与背景

当矩阵 $\mathbf{A}$ 不是方阵或不可逆时，需要使用广义逆矩阵。最常用的是**Moore-Penrose伪逆**。

**定义**：矩阵 $\mathbf{A}$ 的Moore-Penrose伪逆 $\mathbf{A}^+$ 满足以下四个条件：
1. $\mathbf{AA}^+\mathbf{A} = \mathbf{A}$
2. $\mathbf{A}^+\mathbf{AA}^+ = \mathbf{A}^+$
3. $(\mathbf{AA}^+)^T = \mathbf{AA}^+$
4. $(\mathbf{A}^+\mathbf{A})^T = \mathbf{A}^+\mathbf{A}$

### 3.2 计算方法

#### 方法一：基于SVD分解（推荐）

设 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ 是SVD分解，其中：
- $\mathbf{U} \in \mathbb{R}^{m \times m}$，$\mathbf{V} \in \mathbb{R}^{n \times n}$ 为正交矩阵
- $\boldsymbol{\Sigma} = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r, 0, \ldots, 0)$

则伪逆为：
$$\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^T$$

其中 $\boldsymbol{\Sigma}^+ = \text{diag}(\sigma_1^{-1}, \sigma_2^{-1}, \ldots, \sigma_r^{-1}, 0, \ldots, 0)$

#### 方法二：基于正规方程

**情况1**：$\mathbf{A}$ 列满秩（$m \geq n$，$\text{rank}(\mathbf{A}) = n$）
$$\mathbf{A}^+ = (\mathbf{A}^T\mathbf{A})^{-1}\mathbf{A}^T$$

**情况2**：$\mathbf{A}$ 行满秩（$m \leq n$，$\text{rank}(\mathbf{A}) = m$）
$$\mathbf{A}^+ = \mathbf{A}^T(\mathbf{AA}^T)^{-1}$$

### 3.3 应用

伪逆主要用于求解**超定或欠定线性方程组**：
$$\mathbf{Ax} = \mathbf{b}$$

**最小二乘解**：$\mathbf{x} = \mathbf{A}^+\mathbf{b}$

在模式识别中，伪逆常用于：
- 线性回归
- 主成分分析(PCA)
- 线性判别分析(LDA)

## 4. 特征值与特征向量

### 4.1 定义与基本概念

**定义**：对于 $n$ 阶方阵 $\mathbf{A}$，如果存在非零向量 $\mathbf{v}$ 和标量 $\lambda$ 使得：
$$\mathbf{Av} = \lambda\mathbf{v}$$

则称 $\lambda$ 为 $\mathbf{A}$ 的**特征值**，$\mathbf{v}$ 为对应的**特征向量**。

**特征多项式**：$p(\lambda) = \det(\mathbf{A} - \lambda\mathbf{I}) = 0$

**基本性质**：
1. $\text{tr}(\mathbf{A}) = \sum_{i=1}^{n} \lambda_i$（迹等于特征值之和）
2. $\det(\mathbf{A}) = \prod_{i=1}^{n} \lambda_i$（行列式等于特征值之积）
3. $\mathbf{A}$ 和 $\mathbf{A}^T$ 有相同的特征值
4. 相似矩阵有相同的特征值

### 4.2 计算方法

#### 方法一：特征多项式法（小矩阵）

**步骤**：
1. 计算特征多项式 $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$
2. 求解多项式方程得到特征值
3. 对每个特征值 $\lambda_i$，求解 $(\mathbf{A} - \lambda_i\mathbf{I})\mathbf{v} = \mathbf{0}$ 得到特征向量

**示例**：
$$\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}$$

特征多项式：
$$\det(\mathbf{A} - \lambda\mathbf{I}) = \det\begin{pmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{pmatrix} = (3-\lambda)(2-\lambda) = 0$$

特征值：$\lambda_1 = 3, \lambda_2 = 2$

对于 $\lambda_1 = 3$：
$$(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$$
特征向量：$\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

对于 $\lambda_2 = 2$：
$$(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = \mathbf{0}$$
特征向量：$\mathbf{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

#### 方法二：幂法（主特征值）

用于求解**最大特征值**及其对应的特征向量。

**算法**：
1. 选择初始向量 $\mathbf{v}^{(0)}$
2. 迭代：$\mathbf{v}^{(k+1)} = \frac{\mathbf{A}\mathbf{v}^{(k)}}{\|\mathbf{A}\mathbf{v}^{(k)}\|}$
3. 特征值估计：$\lambda \approx \mathbf{v}^{(k)T}\mathbf{A}\mathbf{v}^{(k)}$

#### 方法三：QR算法（所有特征值）

这是数值计算中最常用的方法：

**算法步骤**：
1. 初始化：$\mathbf{A}_0 = \mathbf{A}$
2. 对于 $k = 0, 1, 2, \ldots$：
   - QR分解：$\mathbf{A}_k = \mathbf{Q}_k\mathbf{R}_k$
   - 更新：$\mathbf{A}_{k+1} = \mathbf{R}_k\mathbf{Q}_k$
3. $\mathbf{A}_k$ 收敛到上三角矩阵，对角线元素即为特征值

### 4.3 特殊矩阵的特征值

#### 对称矩阵
- 所有特征值都是实数
- 特征向量相互正交
- 可对角化：$\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$

#### 正定矩阵
- 所有特征值都大于0
- 常用判定方法：
  1. 所有主子式大于0
  2. 所有特征值大于0
  3. 存在可逆矩阵 $\mathbf{P}$ 使得 $\mathbf{A} = \mathbf{P}^T\mathbf{P}$

## 5. 矩阵求导

### 5.1 基本概念与记号

**标记约定**：
- $\mathbf{x} \in \mathbb{R}^n$ 为列向量
- $\mathbf{A} \in \mathbb{R}^{m \times n}$ 为矩阵
- $f: \mathbb{R}^n \to \mathbb{R}$ 为标量函数

**梯度定义**：
$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \frac{\partial f}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### 5.2 常用求导公式

#### 基本公式
1. $\frac{\partial \mathbf{a}^T\mathbf{x}}{\partial \mathbf{x}} = \mathbf{a}$
2. $\frac{\partial \mathbf{x}^T\mathbf{A}}{\partial \mathbf{x}} = \mathbf{A}^T$
3. $\frac{\partial \mathbf{x}^T\mathbf{x}}{\partial \mathbf{x}} = 2\mathbf{x}$
4. $\frac{\partial \mathbf{x}^T\mathbf{A}\mathbf{x}}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$

**特别地**，当 $\mathbf{A}$ 对称时：$\frac{\partial \mathbf{x}^T\mathbf{A}\mathbf{x}}{\partial \mathbf{x}} = 2\mathbf{A}\mathbf{x}$

#### 二阶导数（Hessian矩阵）
$$\mathbf{H} = \nabla^2 f(\mathbf{x}) = \frac{\partial^2 f}{\partial \mathbf{x} \partial \mathbf{x}^T} = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{pmatrix}$$

**常用二阶导数**：
1. $\frac{\partial^2 \mathbf{x}^T\mathbf{A}\mathbf{x}}{\partial \mathbf{x} \partial \mathbf{x}^T} = \mathbf{A} + \mathbf{A}^T$
2. 当 $\mathbf{A}$ 对称时：$\frac{\partial^2 \mathbf{x}^T\mathbf{A}\mathbf{x}}{\partial \mathbf{x} \partial \mathbf{x}^T} = 2\mathbf{A}$

### 5.3 矩阵对矩阵的导数

#### 迹的导数
1. $\frac{\partial \text{tr}(\mathbf{A})}{\partial \mathbf{A}} = \mathbf{I}$
2. $\frac{\partial \text{tr}(\mathbf{AB})}{\partial \mathbf{A}} = \mathbf{B}^T$
3. $\frac{\partial \text{tr}(\mathbf{A}^T\mathbf{B})}{\partial \mathbf{A}} = \mathbf{B}$
4. $\frac{\partial \text{tr}(\mathbf{A}\mathbf{B}\mathbf{A}^T)}{\partial \mathbf{A}} = \mathbf{A}(\mathbf{B} + \mathbf{B}^T)$

#### 行列式的导数
1. $\frac{\partial \ln|\mathbf{A}|}{\partial \mathbf{A}} = (\mathbf{A}^{-1})^T$
2. $\frac{\partial |\mathbf{A}|}{\partial \mathbf{A}} = |\mathbf{A}|(\mathbf{A}^{-1})^T$

### 5.4 在机器学习中的应用

#### 最小二乘法
目标函数：$J(\mathbf{w}) = \frac{1}{2}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$

梯度：
$$\frac{\partial J}{\partial \mathbf{w}} = -\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) = \mathbf{X}^T\mathbf{X}\mathbf{w} - \mathbf{X}^T\mathbf{y}$$

最优解：$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

#### 逻辑回归
目标函数：$J(\mathbf{w}) = -\sum_{i=1}^{n} [y_i \ln \sigma(\mathbf{w}^T\mathbf{x}_i) + (1-y_i) \ln(1-\sigma(\mathbf{w}^T\mathbf{x}_i))]$

梯度：$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{X}^T(\boldsymbol{\sigma} - \mathbf{y})$

其中 $\boldsymbol{\sigma} = [\sigma(\mathbf{w}^T\mathbf{x}_1), \ldots, \sigma(\mathbf{w}^T\mathbf{x}_n)]^T$

## 6. 其他重要数学概念

### 6.1 向量空间与线性变换

#### 向量空间的基本概念
- **线性无关**：向量组 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ 线性无关当且仅当方程 $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$ 只有零解
- **基与维数**：向量空间 $V$ 的基是 $V$ 的一个线性无关的生成集，基中向量的个数称为 $V$ 的维数
- **正交基**：基中向量两两正交
- **标准正交基**：正交基且每个向量的模长为1

#### 线性变换
线性变换 $T: \mathbb{R}^n \to \mathbb{R}^m$ 可以用矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 表示：
$$T(\mathbf{x}) = \mathbf{A}\mathbf{x}$$

**重要性质**：
- **零空间（核）**：$\text{null}(\mathbf{A}) = \{\mathbf{x} : \mathbf{A}\mathbf{x} = \mathbf{0}\}$
- **列空间（像）**：$\text{col}(\mathbf{A}) = \{\mathbf{A}\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$
- **秩-零化定理**：$\text{rank}(\mathbf{A}) + \text{nullity}(\mathbf{A}) = n$

### 6.2 二次型与正定性

#### 二次型
$n$ 元二次型是形如下式的函数：
$$f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x} = \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij}x_ix_j$$

其中 $\mathbf{A}$ 是对称矩阵（不失一般性）。

#### 正定性判定
矩阵 $\mathbf{A}$ 正定的等价条件：
1. 所有特征值大于0
2. 所有顺序主子式大于0
3. 存在可逆矩阵 $\mathbf{P}$ 使得 $\mathbf{A} = \mathbf{P}^T\mathbf{P}$
4. 对所有非零向量 $\mathbf{x}$，都有 $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$

**Sylvester判据**：
- **正定**：所有顺序主子式 $D_k > 0$
- **负定**：$(-1)^k D_k > 0$，$k = 1, 2, \ldots, n$
- **半正定**：所有主子式 $\geq 0$

### 6.3 矩阵分解

#### QR分解
任意矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ (列满秩) 可分解为：
$$\mathbf{A} = \mathbf{Q}\mathbf{R}$$
其中 $\mathbf{Q} \in \mathbb{R}^{m \times n}$ 列正交，$\mathbf{R} \in \mathbb{R}^{n \times n}$ 上三角且对角元素为正。

**Gram-Schmidt正交化过程**：
$$\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|}, \quad \mathbf{q}_k = \frac{\mathbf{a}_k - \sum_{j=1}^{k-1}(\mathbf{a}_k^T\mathbf{q}_j)\mathbf{q}_j}{\|\mathbf{a}_k - \sum_{j=1}^{k-1}(\mathbf{a}_k^T\mathbf{q}_j)\mathbf{q}_j\|}$$

#### Cholesky分解
对于正定矩阵 $\mathbf{A}$，存在唯一的下三角矩阵 $\mathbf{L}$（对角元素为正）使得：
$$\mathbf{A} = \mathbf{L}\mathbf{L}^T$$

**计算公式**：
$$l_{ii} = \sqrt{a_{ii} - \sum_{k=1}^{i-1} l_{ik}^2}$$
$$l_{ji} = \frac{1}{l_{ii}}\left(a_{ji} - \sum_{k=1}^{i-1} l_{jk}l_{ik}\right), \quad j > i$$

### 6.4 范数与条件数

#### 向量范数
- **1-范数**：$\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$
- **2-范数（欧几里得范数）**：$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$
- **$\infty$-范数**：$\|\mathbf{x}\|_\infty = \max_{1 \leq i \leq n} |x_i|$
- **$p$-范数**：$\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$

#### 矩阵范数
- **Frobenius范数**：$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^T\mathbf{A})}$
- **谱范数（2-范数）**：$\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A})$（最大奇异值）
- **1-范数**：$\|\mathbf{A}\|_1 = \max_{1 \leq j \leq n} \sum_{i=1}^{m} |a_{ij}|$（最大列和）
- **$\infty$-范数**：$\|\mathbf{A}\|_\infty = \max_{1 \leq i \leq m} \sum_{j=1}^{n} |a_{ij}|$（最大行和）

#### 条件数
矩阵 $\mathbf{A}$ 的条件数定义为：
$$\text{cond}(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|$$

对于2-范数：$\text{cond}_2(\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}$

**意义**：条件数衡量线性方程组 $\mathbf{Ax} = \mathbf{b}$ 的数值稳定性。条件数越大，方程组越病态。

---

**总结**：本附录涵盖了模式识别中最常用的数学工具和计算方法。这些数学基础对于理解PCA、LDA、SVM等算法的原理和实现至关重要。在实际应用中，建议使用成熟的数值计算库（如LAPACK、BLAS）来进行矩阵运算，以确保数值稳定性和计算效率。
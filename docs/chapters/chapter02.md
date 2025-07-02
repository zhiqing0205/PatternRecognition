# 2. 贝叶斯决策论

贝叶斯决策论为模式识别提供了理论基础，通过概率方法来进行最优决策。

## 贝叶斯决策理论基础

### 贝叶斯定理

对于模式识别问题，我们有：

$$P(\omega_i|\mathbf{x}) = \frac{p(\mathbf{x}|\omega_i)P(\omega_i)}{p(\mathbf{x})}$$

其中：
- $P(\omega_i|\mathbf{x})$：后验概率
- $p(\mathbf{x}|\omega_i)$：类条件概率密度函数（似然函数）
- $P(\omega_i)$：先验概率
- $p(\mathbf{x})$：证据因子

### 贝叶斯决策规则

**最小错误率决策**：选择具有最大后验概率的类别
$$\hat{\omega} = \arg\max_i P(\omega_i|\mathbf{x})$$

**最小风险决策**：当不同错误的代价不同时
$$\hat{\omega} = \arg\min_i R(\alpha_i|\mathbf{x})$$

其中风险函数为：$R(\alpha_i|\mathbf{x}) = \sum_{j=1}^{c} \lambda_{ij} P(\omega_j|\mathbf{x})$

## 多元正态分布的判别函数推导

多元正态分布是模式识别中最重要的概率分布之一。对于 $d$ 维特征向量 $\mathbf{x}$，多元正态分布的概率密度函数为：

$$p(\mathbf{x}|\omega_i) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_i|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i)\right)$$

为了简化计算，我们通常使用判别函数 $g_i(\mathbf{x})$，其中：

$$g_i(\mathbf{x}) = \ln p(\mathbf{x}|\omega_i) + \ln P(\omega_i)$$

### 情况一：协方差矩阵为对角阵（独立同分布）

当各特征相互独立且方差可能不同时，协方差矩阵为对角阵：

$$\boldsymbol{\Sigma}_i = \begin{pmatrix}
\sigma_{i1}^2 & 0 & \cdots & 0 \\
0 & \sigma_{i2}^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & \sigma_{id}^2
\end{pmatrix}$$

#### 推导过程

**第一步**：计算行列式和逆矩阵
$$|\boldsymbol{\Sigma}_i| = \prod_{j=1}^{d} \sigma_{ij}^2$$

$$\boldsymbol{\Sigma}_i^{-1} = \begin{pmatrix}
1/\sigma_{i1}^2 & 0 & \cdots & 0 \\
0 & 1/\sigma_{i2}^2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1/\sigma_{id}^2
\end{pmatrix}$$

**第二步**：计算二次型
$$(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) = \sum_{j=1}^{d} \frac{(x_j - \mu_{ij})^2}{\sigma_{ij}^2}$$

**第三步**：得到判别函数
$$g_i(\mathbf{x}) = -\frac{1}{2}\sum_{j=1}^{d} \frac{(x_j - \mu_{ij})^2}{\sigma_{ij}^2} - \frac{1}{2}\sum_{j=1}^{d} \ln(2\pi\sigma_{ij}^2) + \ln P(\omega_i)$$

**特点**：
- 决策边界为超椭球面
- 各特征维度独立处理
- 计算复杂度较低

### 情况二：所有类的协方差矩阵相同

当所有类别具有相同的协方差矩阵时，即 $\boldsymbol{\Sigma}_i = \boldsymbol{\Sigma}, \forall i$，这是线性判别分析(LDA)的基础假设。

#### 推导过程

**第一步**：展开判别函数
$$g_i(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) - \frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\boldsymbol{\Sigma}| + \ln P(\omega_i)$$

**第二步**：展开二次型
$$(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) = \mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - 2\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\mathbf{x} + \boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i$$

**第三步**：消除公共项
由于 $\mathbf{x}^T\boldsymbol{\Sigma}^{-1}\mathbf{x}$ 和 $\ln|\boldsymbol{\Sigma}|$ 对所有类别都相同，在比较不同类别时可以忽略。

**第四步**：得到线性判别函数
$$g_i(\mathbf{x}) = \boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\mathbf{x} - \frac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$$

这可以写成线性形式：
$$g_i(\mathbf{x}) = \mathbf{w}_i^T\mathbf{x} + w_{i0}$$

其中：
- $\mathbf{w}_i = \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i$（权重向量）
- $w_{i0} = -\frac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_i + \ln P(\omega_i)$（偏置项）

#### 决策边界

两类 $\omega_i$ 和 $\omega_j$ 之间的决策边界由 $g_i(\mathbf{x}) = g_j(\mathbf{x})$ 确定：

$$(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)^T\boldsymbol{\Sigma}^{-1}\mathbf{x} = \frac{1}{2}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_j)^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}_i + \boldsymbol{\mu}_j) - \ln\frac{P(\omega_i)}{P(\omega_j)}$$

**特点**：
- 决策边界为超平面（线性边界）
- 这是**线性判别分析(LDA)**的理论基础
- 计算效率高，广泛应用于实际问题

### 情况三：协方差矩阵任意（一般情况）

这是最一般的情况，每个类别都有自己独特的协方差矩阵 $\boldsymbol{\Sigma}_i$，对应**二次判别分析(QDA)**。

#### 推导过程

**完整的判别函数**：
$$g_i(\mathbf{x}) = -\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) - \frac{1}{2}\ln|\boldsymbol{\Sigma}_i| - \frac{d}{2}\ln(2\pi) + \ln P(\omega_i)$$

**第一步**：展开二次型
$$(\mathbf{x} - \boldsymbol{\mu}_i)^T\boldsymbol{\Sigma}_i^{-1}(\mathbf{x} - \boldsymbol{\mu}_i) = \mathbf{x}^T\boldsymbol{\Sigma}_i^{-1}\mathbf{x} - 2\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\mathbf{x} + \boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i$$

**第二步**：整理得到二次判别函数
$$g_i(\mathbf{x}) = -\frac{1}{2}\mathbf{x}^T\boldsymbol{\Sigma}_i^{-1}\mathbf{x} + \boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\mathbf{x} + w_{i0}$$

其中：
$$w_{i0} = -\frac{1}{2}\boldsymbol{\mu}_i^T\boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i - \frac{1}{2}\ln|\boldsymbol{\Sigma}_i| + \ln P(\omega_i)$$

**第三步**：二次型的标准形式
判别函数可以写成：
$$g_i(\mathbf{x}) = \mathbf{x}^T\mathbf{W}_i\mathbf{x} + \mathbf{w}_i^T\mathbf{x} + w_{i0}$$

其中：
- $\mathbf{W}_i = -\frac{1}{2}\boldsymbol{\Sigma}_i^{-1}$（二次项系数矩阵）
- $\mathbf{w}_i = \boldsymbol{\Sigma}_i^{-1}\boldsymbol{\mu}_i$（线性项系数向量）
- $w_{i0}$ 如上所定义（常数项）

#### 决策边界

两类之间的决策边界由 $g_i(\mathbf{x}) = g_j(\mathbf{x})$ 确定，这是一个**二次方程**：

$$\mathbf{x}^T(\mathbf{W}_i - \mathbf{W}_j)\mathbf{x} + (\mathbf{w}_i - \mathbf{w}_j)^T\mathbf{x} + (w_{i0} - w_{j0}) = 0$$

**特点**：
- 决策边界为二次曲面（椭圆、抛物线、双曲线等）
- 这是二次判别分析(QDA)的理论基础
- 能够处理更复杂的分类问题，但计算复杂度较高
- 需要更多的训练样本来准确估计协方差矩阵

### 三种情况的比较

| 情况 | 协方差矩阵假设 | 判别函数形式 | 决策边界 | 参数数量 | 适用场景 |
|------|----------------|--------------|----------|----------|----------|
| **情况一** | 对角阵（独立） | 加权欧氏距离 | 超椭球面 | $O(cd)$ | 特征独立，方差不同 |
| **情况二** | 所有类相同 | 线性函数 | 超平面 | $O(d^2 + cd)$ | 经典LDA，效率高 |
| **情况三** | 任意矩阵 | 二次函数 | 二次曲面 | $O(cd^2)$ | 复杂分布，QDA |

其中 $c$ 是类别数，$d$ 是特征维数。

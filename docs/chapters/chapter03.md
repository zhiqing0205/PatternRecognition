# 3. 最大似然估计和贝叶斯参数估计

## 问题：什么是维数灾难？如何缓解维数灾难？

### 维数灾难：

当维度增加时，由于样本点不增加，所以在高维空间中，数据样本的密度变得更加稀疏了，更难以分类

### 如何解决：

1. 数据层面：增加先验，如过采样，数据增广，专家标记等
2. 模型层面：使得函数更加光滑，如加上正则化项
3. 特征层面：降维，如PCA，LDA

## 问题：过拟合是什么，为什么会出现，如何缓解？
### 过拟合是什么？
指分类器模型过于复杂，对训练样本学习得"太好"，以至于把噪声和样本的偶然特性也学了进去。这导致它能很好地拟合训练数据，但在面对新的、未见过的数据时，分类性能会很差。文中图3-4的10阶多项式曲线就是一个例子。

### 为什么会出现？

主要原因是**训练样本数量不足**以支撑复杂模型的参数估计。当模型参数很多（如高维数据下的协方差矩阵），而训练样本很少时，就容易发生过拟合。

### 如何缓解？

1. **降维**：减少特征数量或组合特征。
2. **简化模型**：例如，假设不同类别具有相同的协方差矩阵，从而减少需要估计的参数。
3. **使用缩并(Shrinkage)技术**：将估计出的协方差矩阵与一个更简单的矩阵（如对角阵或单位阵）进行加权平均。
4. **采用启发式方法**：例如，强制假设特征间统计独立，将协方差矩阵的非对角元素置零。

## 问题：主成分分析（PCA）和Fisher判别分析（LDA）是什么，二者有什么区别？

### 主成分分析 (Principal Component Analysis, PCA)

是一种旨在寻找能以最小均方误差**最有效代表**原始数据的投影方向的降维方法。它通过寻找数据整体散布（方差）最大的主轴（即本征向量），将高维数据投影到低维空间，从而达到数据压缩和表示的目的。

### Fisher判别分析 (Fisher Discriminant Analysis)

也是一种降维方法，但其目标是寻找能够**最有效区分**不同类别数据的投影方向。它通过最大化"类间散布"与"类内散布"的比值，找到一个或多个使投影后各类别尽可能分开、而类别内部尽可能紧凑的投影方向，服务于分类任务。

### 二者的区别

1. **目标不同**：PCA的目标是**数据表示**，寻找方差最大的方向；Fisher判别分析的目标是**数据分类**，寻找类别可分性最好的方向。
2. **对类别信息的利用不同**：PCA在计算时不考虑样本的类别标签，是一种无监督的方法。而Fisher判别分析则明确利用样本的类别信息来计算类内散布和类间散布，是一种有监督的方法。

一个形象的例子是，PCA可能会为了保留数据共有的主要特征（如字母"O"和"Q"的圆形部分），而丢弃掉恰好能区分它们细微差异（如"Q"的尾巴）的方向。Fisher判别分析则会着重寻找能凸显这个"尾巴"差异的方向。

## 问题：主成分分析（PCA）或线性判别分析（LDA）的特征表示更优越吗？

**没有一种方法是绝对"更优越"的，它们的优越性取决于具体任务目标。**

1. **对于数据表示和压缩，主成分分析（PCA）更优越。** PCA的目标是寻找能以最小均方误差最有效代表原始数据的方向。它关注的是数据整体的方差，旨在保留数据中最重要的结构信息，非常适合用于数据压缩。
2. **对于分类任务，Fisher判别分析（LDA）更优越。** 该方法的目标是寻找能最大程度区分不同类别的投影方向。文中明确指出，PCA所寻找的表示主轴"并没有理由表明主成分对区分不同的类别有什么大作用"，甚至可能丢弃对分类至关重要的特征（如区分字母"O"和"Q"的"尾巴"）。而Fisher判别分析正是为有效分类而设计的。

因此，选择哪种方法取决于任务目标是数据表示还是分类。

## 问题：在降维和分类时，PCA和LDA有什么区别？当分布服从高斯分布时，PCA和LDA有什么区别？当分布服从什么分布时，PCA和LDA等价？

### 1. 在降维和分类时，PCA和LDA有什么区别？

二者在降维和分类中的核心区别在于其**目的**和**对类别信息的利用**。

- **目的不同**：
    - **PCA** 的目的是**数据表示**。它寻找能最大程度保留数据原始方差（即信息量）的投影方向，旨在用更少的维度来紧凑地表示数据，而不关心这些方向是否有利于分类。
    - **LDA** 的目的是**数据分类**。它寻找能使不同类别在投影后尽可能分开的投影方向，即最大化类间距离同时最小化类内方差。其降维结果是为后续的分类任务服务的。
- **对类别信息的利用不同**：
    - **PCA** 是**无监督**的。它在计算散布矩阵 $\mathbf{S}$ 时，将所有样本不论类别都混合在一起，不使用任何类别标签信息。
    - **LDA** 是**有监督**的。它明确利用样本的类别标签来计算类内散布矩阵 $\mathbf{S}_W$ 和类间散布矩阵 $\mathbf{S}_B$，以找到最优的分类投影。

### 2. 当分布服从高斯分布时，PCA和LDA有什么区别？

当数据服从高斯分布时，它们的基本区别依然存在，但LDA会展现出更强的理论优势。

- **PCA** 会找到高斯分布数据构成的"椭球云团"的主轴方向。这些主轴是数据方差最大的方向，但不一定是对分类最有用的方向。
- **LDA** 在这种情况下表现更优。如果数据服从**协方差矩阵相同**的多元高斯分布，那么Fisher线性判别（LDA）找到的最优投影方向 $\mathbf{w}$ 与贝叶斯最优分类器的判决边界方向是**一致的**。这意味着LDA找到的降维方向在理论上是**最优的分类方向**。

### 3. 当分布服从什么分布时，PCA和LDA等价？

PCA和LDA找到的投影方向等价，这种情况非常特殊且罕见，需要满足以下严格的条件：

1. **各类别具有相同的、球形的协方差矩阵**：即 $\mathbf{\Sigma}_i = \sigma^2\mathbf{I}$ 对所有类别 $i$ 都成立。这意味着每个类别内部的数据分布都是一个没有特定方向性的"圆球"，并且所有类别的"圆球"大小都一样。这使得类内散布矩阵 $\mathbf{S}_W$ 成为一个对角矩阵 $(k\mathbf{I})$，它在所有方向上的散布都相同，因此在最大化 $J(\mathbf{w})$ 的比率时，分母 $\mathbf{w}^T \mathbf{S}_W \mathbf{w}$ 不再对方向有偏好。
2. **数据的主要方差完全由类间差异主导**：当满足条件1时，总散布矩阵 $\mathbf{S}_T = \mathbf{S}_W + \mathbf{S}_B = k\mathbf{I} + \mathbf{S}_B$。PCA的目标是最大化 $\mathbf{w}^T \mathbf{S}_T \mathbf{w}$，这等价于最大化 $\mathbf{w}^T \mathbf{S}_B \mathbf{w}$。而LDA的目标也是最大化 $\mathbf{w}^T \mathbf{S}_B \mathbf{w}$。此时，两个方法的目标函数变得一致。

**直观理解**：想象在二维平面上有两个（或多个）距离很远、各自呈圆形且大小相同的点云。此时，能够最好地分开这两个点云的方向（LDA的目标），恰好也是整个数据集方差最大的方向（PCA的目标）。但在绝大多数实际情况中，类内分布是椭球形的且各不相同，PCA和LDA会给出截然不同的结果。

综上，只有在各类分布本身是球形同构且类间分离是数据方差的绝对主导因素时，PCA和LDA才会等价。

## PCA（主成分分析）数学推导

### 1. 问题定义

PCA的目标是找到一个投影方向 $\mathbf{w}$，使得数据在该方向上投影后的方差最大。

设有 $n$ 个 $d$ 维数据点：$\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$

### 2. 数据中心化

首先对数据进行中心化处理：
$$\tilde{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$$

其中样本均值为：
$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

### 3. 投影方差最大化

数据在单位向量 $\mathbf{w}$（$\|\mathbf{w}\|=1$）方向上的投影为：
$$y_i = \mathbf{w}^T\tilde{\mathbf{x}}_i$$

投影后数据的方差为：
$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \bar{y})^2$$

由于数据已中心化，$\bar{y} = 0$，因此：
$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}y_i^2 = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{w}^T\tilde{\mathbf{x}}_i)^2$$

### 4. 目标函数

$$\text{Var}(y) = \frac{1}{n}\sum_{i=1}^{n}\mathbf{w}^T\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T\mathbf{w} = \mathbf{w}^T\left(\frac{1}{n}\sum_{i=1}^{n}\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T\right)\mathbf{w}$$

定义协方差矩阵：
$$\mathbf{S} = \frac{1}{n}\sum_{i=1}^{n}\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T$$

因此目标函数为：
$$\max_{\mathbf{w}} \mathbf{w}^T\mathbf{S}\mathbf{w} \quad \text{subject to } \|\mathbf{w}\|^2 = 1$$

### 5. 拉格朗日乘子法求解

构建拉格朗日函数：
$$L(\mathbf{w}, \lambda) = \mathbf{w}^T\mathbf{S}\mathbf{w} - \lambda(\mathbf{w}^T\mathbf{w} - 1)$$

对 $\mathbf{w}$ 求偏导并令其为零：
$$\frac{\partial L}{\partial \mathbf{w}} = 2\mathbf{S}\mathbf{w} - 2\lambda\mathbf{w} = 0$$

得到特征值方程：
$$\mathbf{S}\mathbf{w} = \lambda\mathbf{w}$$

### 6. 解的性质

- $\mathbf{w}$ 是协方差矩阵 $\mathbf{S}$ 的特征向量
- $\lambda$ 是对应的特征值，等于投影方差：$\mathbf{w}^T\mathbf{S}\mathbf{w} = \lambda$
- 为最大化方差，选择最大特征值对应的特征向量
- 前 $k$ 个主成分对应前 $k$ 个最大特征值的特征向量

## LDA（线性判别分析）数学推导

### 1. 问题定义

LDA的目标是找到一个投影方向 $\mathbf{w}$，使得投影后类间距离最大、类内距离最小。

设有 $C$ 个类别，第 $i$ 类有 $n_i$ 个样本，总样本数 $n = \sum_{i=1}^{C} n_i$

### 2. 类内散布矩阵

第 $i$ 类的均值：
$$\boldsymbol{\mu}_i = \frac{1}{n_i}\sum_{\mathbf{x} \in \omega_i}\mathbf{x}$$

第 $i$ 类的散布矩阵：
$$\mathbf{S}_i = \sum_{\mathbf{x} \in \omega_i}(\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T$$

类内散布矩阵：
$$\mathbf{S}_W = \sum_{i=1}^{C}\mathbf{S}_i = \sum_{i=1}^{C}\sum_{\mathbf{x} \in \omega_i}(\mathbf{x} - \boldsymbol{\mu}_i)(\mathbf{x} - \boldsymbol{\mu}_i)^T$$

### 3. 类间散布矩阵

总体均值：
$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i = \frac{1}{n}\sum_{i=1}^{C}n_i\boldsymbol{\mu}_i$$

类间散布矩阵：
$$\mathbf{S}_B = \sum_{i=1}^{C}n_i(\boldsymbol{\mu}_i - \boldsymbol{\mu})(\boldsymbol{\mu}_i - \boldsymbol{\mu})^T$$

### 4. Fisher判别准则

在投影方向 $\mathbf{w}$ 上：
- 投影后第 $i$ 类均值：$\tilde{\mu}_i = \mathbf{w}^T\boldsymbol{\mu}_i$
- 投影后类内散布：$\tilde{s}_W^2 = \mathbf{w}^T\mathbf{S}_W\mathbf{w}$
- 投影后类间散布：$\tilde{s}_B^2 = \mathbf{w}^T\mathbf{S}_B\mathbf{w}$

Fisher判别准则（以二分类为例）：
$$J(\mathbf{w}) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

### 5. 广义特征值问题

目标是最大化：
$$J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

对 $\mathbf{w}$ 求导并令其为零：
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2\mathbf{S}_B\mathbf{w}(\mathbf{w}^T\mathbf{S}_W\mathbf{w}) - 2\mathbf{S}_W\mathbf{w}(\mathbf{w}^T\mathbf{S}_B\mathbf{w})}{(\mathbf{w}^T\mathbf{S}_W\mathbf{w})^2} = 0$$

简化得到广义特征值方程：
$$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$$

等价于：
$$\mathbf{S}_W^{-1}\mathbf{S}_B\mathbf{w} = \lambda\mathbf{w}$$

### 6. 解的性质

- 最优投影方向 $\mathbf{w}$ 是 $\mathbf{S}_W^{-1}\mathbf{S}_B$ 的特征向量
- 对应的特征值 $\lambda$ 等于Fisher判别准则的值
- 对于 $C$ 类问题，最多有 $C-1$ 个有意义的判别方向
- 选择最大的几个特征值对应的特征向量作为投影方向

## 多变量高斯分布的最大似然估计推导

多变量高斯分布在模式识别中应用广泛，其参数的最大似然估计是基础且重要的内容。我们分两种情况进行推导。

### 情况一：均值已知，协方差矩阵未知

设 $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_n$ 是来自 $d$ 维多变量高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的独立同分布样本，其中均值 $\boldsymbol{\mu}$ 已知，需要估计协方差矩阵 $\boldsymbol{\Sigma}$。

#### 1. 似然函数

多变量高斯分布的概率密度函数为：
$$p(\mathbf{x}|\boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

似然函数为：
$$L(\boldsymbol{\Sigma}) = \prod_{i=1}^{n} p(\mathbf{x}_i|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

#### 2. 对数似然函数

$$\ln L(\boldsymbol{\Sigma}) = \sum_{i=1}^{n} \ln p(\mathbf{x}_i|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$= -\frac{nd}{2}\ln(2\pi) - \frac{n}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu})$$

#### 3. 求导与最大化

利用矩阵求导公式：
- $\frac{\partial \ln|\boldsymbol{\Sigma}|}{\partial \boldsymbol{\Sigma}} = (\boldsymbol{\Sigma}^{-1})^T = \boldsymbol{\Sigma}^{-1}$（因为 $\boldsymbol{\Sigma}$ 对称）
- $\frac{\partial \text{tr}(\mathbf{A}\boldsymbol{\Sigma}^{-1})}{\partial \boldsymbol{\Sigma}} = -\boldsymbol{\Sigma}^{-1}\mathbf{A}\boldsymbol{\Sigma}^{-1}$

注意到：
$$\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu}) = \text{tr}\left(\boldsymbol{\Sigma}^{-1}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T\right)$$

设 $\mathbf{S} = \sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$，则：

$$\frac{\partial \ln L(\boldsymbol{\Sigma})}{\partial \boldsymbol{\Sigma}} = -\frac{n}{2}\boldsymbol{\Sigma}^{-1} + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{S}\boldsymbol{\Sigma}^{-1} = 0$$

#### 4. 最大似然估计解

从上式得到：
$$n\mathbf{I} = \mathbf{S}\boldsymbol{\Sigma}^{-1}$$

因此：
$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})(\mathbf{x}_i - \boldsymbol{\mu})^T$$

### 情况二：均值和协方差矩阵均未知

当均值 $\boldsymbol{\mu}$ 和协方差矩阵 $\boldsymbol{\Sigma}$ 都未知时，需要同时估计这两个参数。

#### 1. 对数似然函数

$$\ln L(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{nd}{2}\ln(2\pi) - \frac{n}{2}\ln|\boldsymbol{\Sigma}| - \frac{1}{2}\sum_{i=1}^{n}(\mathbf{x}_i - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu})$$

#### 2. 对均值求导

$$\frac{\partial \ln L}{\partial \boldsymbol{\mu}} = \sum_{i=1}^{n}\boldsymbol{\Sigma}^{-1}(\mathbf{x}_i - \boldsymbol{\mu}) = 0$$

得到：
$$\hat{\boldsymbol{\mu}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

#### 3. 对协方差矩阵求导

将估计的均值代入，设 $\mathbf{S} = \sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$，类似于情况一的推导：

$$\frac{\partial \ln L}{\partial \boldsymbol{\Sigma}} = -\frac{n}{2}\boldsymbol{\Sigma}^{-1} + \frac{1}{2}\boldsymbol{\Sigma}^{-1}\mathbf{S}\boldsymbol{\Sigma}^{-1} = 0$$

#### 4. 最大似然估计解

$$\hat{\boldsymbol{\mu}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i$$

$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

## 问题：最大似然估计得到的结果是有偏的还是无偏的，二者差距多少，哪一个是更好的结果？

### 1. 均值估计的无偏性

**结论**：均值的最大似然估计是**无偏的**。

**证明**：
$$E[\hat{\boldsymbol{\mu}}_{ML}] = E\left[\frac{1}{n}\sum_{i=1}^{n}\mathbf{x}_i\right] = \frac{1}{n}\sum_{i=1}^{n}E[\mathbf{x}_i] = \frac{1}{n}\sum_{i=1}^{n}\boldsymbol{\mu} = \boldsymbol{\mu}$$

因此 $\hat{\boldsymbol{\mu}}_{ML}$ 是 $\boldsymbol{\mu}$ 的无偏估计。

### 2. 协方差矩阵估计的有偏性

**结论**：协方差矩阵的最大似然估计是**有偏的**。

**分析**：考虑情况二中的协方差矩阵估计：
$$\hat{\boldsymbol{\Sigma}}_{ML} = \frac{1}{n}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

#### 有偏性证明

对于标量情况，我们知道：
$$E\left[\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right] = \frac{n-1}{n}\sigma^2$$

推广到多变量情况：
$$E[\hat{\boldsymbol{\Sigma}}_{ML}] = \frac{n-1}{n}\boldsymbol{\Sigma}$$

**证明思路**：
- 当使用样本均值 $\hat{\boldsymbol{\mu}}_{ML}$ 而非真实均值 $\boldsymbol{\mu}$ 时，样本偏差的平方和会系统性地偏小
- 这是因为样本均值是使平方和最小的点，导致低估了真实的方差

#### 偏差量化

- **偏差**：$\text{Bias}[\hat{\boldsymbol{\Sigma}}_{ML}] = E[\hat{\boldsymbol{\Sigma}}_{ML}] - \boldsymbol{\Sigma} = -\frac{1}{n}\boldsymbol{\Sigma}$
- **相对偏差**：$\frac{1}{n}$，随样本量增加而减小

### 3. 无偏估计

为得到协方差矩阵的无偏估计，使用**贝塞尔修正**：

$$\hat{\boldsymbol{\Sigma}}_{无偏} = \frac{1}{n-1}\sum_{i=1}^{n}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_{ML})^T$$

**验证无偏性**：
$$E[\hat{\boldsymbol{\Sigma}}_{无偏}] = \frac{1}{n-1} \cdot \frac{n-1}{n} \cdot n \cdot \boldsymbol{\Sigma} = \boldsymbol{\Sigma}$$

### 4. 比较与选择

**有偏估计 vs 无偏估计**：

| 特性 | 最大似然估计（有偏） | 无偏估计 |
|------|---------------------|----------|
| **偏差** | $-\frac{1}{n}\boldsymbol{\Sigma}$ | $0$ |
| **方差** | 较小 | 较大 |
| **均方误差** | 小样本时可能更小 | 大样本时更优 |
| **实际应用** | 机器学习中常用 | 统计推断中常用 |

**哪个更好？**

- **大样本情况**（$n$ 很大）：两者差别很小，$\frac{1}{n} \approx 0$，都是渐近无偏的
- **小样本情况**：
  - 如果关注**无偏性**（如统计推断），选择无偏估计
  - 如果关注**预测精度**（如机器学习），最大似然估计的较小方差可能更有价值
- **实践建议**：
  - 统计分析：使用无偏估计（除以 $n-1$）
  - 机器学习：使用最大似然估计（除以 $n$），因为偏差在大数据下可忽略，且计算简单

### 5. 渐近性质

当 $n \to \infty$ 时：
- 两种估计都收敛到真实值：$\hat{\boldsymbol{\Sigma}}_{ML} \to \boldsymbol{\Sigma}$，$\hat{\boldsymbol{\Sigma}}_{无偏} \to \boldsymbol{\Sigma}$
- 最大似然估计具有渐近正态性和渐近效率性
- 在大样本下，最大似然估计是"最优"的
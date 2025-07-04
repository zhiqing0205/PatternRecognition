# 4. 非参数技术

## 问题：比较参数技术与非参数技术的优缺点

### 参数技术：

- **优点**：当概率分布的参数形式已知或能被准确假设时，它需要的数据量远少于非参数方法，计算和存储效率更高。
- **缺点**：其核心问题是必须预先假设概率密度的函数形式。这种假设在实际应用中往往不成立，例如，经典的参数形式多为单峰分布，而现实数据常呈现多峰分布，导致模型与实际情况不符。

### 非参数技术：

- **优点**：具有很强的通用性，无需对概率分布做任何形式上的假设，能处理任意复杂的分布。只要训练样本足够多，理论上总能收敛到真实模型。
- **缺点**：为了获得精确结果，通常需要"惊人"数量的训练样本，导致时间和存储开销巨大。更严重的是，它受"维数灾难"的制约，即所需样本量随特征维度的增加呈指数级增长。

## 问题：数据驱动的算法是什么含义，请举例分析一下

### 数据驱动算法的含义

**数据驱动算法**是指决策过程直接基于训练数据，不依赖分布假设或参数化模型的算法。核心特征：局部性决策、记忆型学习、让数据"自己说话"。

### Parzen窗方法

**基本原理**：在查询点周围放置窗口，统计窗口内样本数量估计概率密度。

**数据驱动特性**：
- 每个训练样本都对密度估计有贡献，无预设分布形式
- 查询点密度主要由邻近样本决定
- 数据密集区域自动产生高密度估计

### K近邻方法

**基本原理**：找到与查询点最近的K个训练样本，用邻居类别标签决定分类。

**数据驱动特性**：
- 不对数据分布做任何假设，决策边界完全由训练数据决定
- 每个预测只依赖最近K个样本，实现局部决策
- 能自然处理非线性边界，复杂度由数据本身决定
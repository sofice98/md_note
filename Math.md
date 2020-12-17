# 应用随机过程



## 计数过程

$\{N(t),t\geqslant0\}$，$N(t)$表示到时刻 t 为止发生的事件的总数

需满足：

1. $N(t)\geqslant0$
2. $N(t)$取整数值
3. 若$s<t$，则$N(s)\leqslant N(t)$
4. 对于$s<t$，$N(t)-N(s)$表示在区间 (s,t] 中发生的事件个数

**独立增量：**发生在不相交的时间区间中的事件个数彼此独立

> **独立增量过程一定是马尔科夫过程**

**平稳增量（时间齐次）：**在任意时间区间中发生的事件个数只依赖于时间区间的长度





## 泊松过程

计数过程$\{N(t),t\geqslant0\}$称为具有速率$\lambda(\lambda>0)$的泊松过程，如果：

1. $N(0)=0$

2. $\{N(t),t\geqslant0\}$过程有平稳增量和独立增量

3. $P\{N(t+\Delta t)-N(t)=1\}=\lambda \Delta t+o(\Delta t)$

4. $P\{N(t+\Delta t)-N(t)\geqslant 2\}=o(\Delta t)$

   







# 最优化方法

最优化问题，即求一个多元函数在某个给定集合上的**极值**：
$$
min\,f(x)\\
s.t.\,x\in K
$$
其中 s.t.=subject to(受限于)，K 为**可行域**，f(x) 为**目标函数**，x为**决策变量**

- 线性规划和非线性规划：可行域是有限维空间中的一个子集
- 组合优化或网络规划：可行域中的元素有限
- 动态规划：可行域是一个依赖时间的决策序列
- 最优控制：可行域是无穷维空间中的一个连续子集




$$
\large{
min\,f(x)\\
s.t.\,h_i(x)=0,\quad i=1,...,l\\
\quad\quad\,\, g_i(x)\geqslant 0,\quad i=1,...,m
}
$$

其中$f(x),h_i(x),g_i(x)$都是定义在 $\R^n$上的连续可微的多元实值函数
$$
\large{
记\quad E=\{i:h_i(x)=0\},\quad I=\{i:g_i(x)\geqslant 0\}\\
\begin{cases}
无约束优化\quad E\bigcup I=\emptyset\\
约束优化
\begin{cases}
等式约束优化\quad E\not=\emptyset且 I=\emptyset\\
不等式约束优化\quad E=\emptyset且 I\not=\emptyset\\
线性规划\quad 目标函数和约束函数都是线性函数\\
二次规划\quad 目标函数是二次函数，约束函数是线性函数
\end{cases}
\end{cases}
}
$$





















































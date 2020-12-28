# 数学

$K$为标签数，$d$为特征数，$K_i$为$X_i$的类别数，$N$为样本数，$D$为数据集，$D_c$为数据子集

+ **梯度下降法**

  每次将估计值向梯度的负方向进行修改，可加上步长（学习率），当每次只是用一个误分类的样本进行更新时，称为**随机梯度下降法**，更新过后以前的误分类点可能会正确分类

+ **牛顿法**

  使用海森矩阵和泰勒展开公式，近似的用点的函数值表示实际函数值，由此推导出递推公式，当使用另一个矩阵近似海森矩阵以简化计算量时，称为**拟牛顿法**

+ **拉格朗日乘子法**

  考虑函数f(x)在约束g(x)下求极值的问题，有如下结论：

  - 对于约束曲面上的任一点$x$，该点的梯度$\nabla g(x)$正交与约束曲面
  - 在最优点$x^*$，目标函数在该点的梯度$\nabla f(x^*)$正交与约束曲面

  因此有$\nabla f(x^*)+\lambda\nabla g(x^*)=0,\lambda\not=0$，此式即为拉格朗日函数对x求偏导得到

  最优化问题，广义拉格朗日函数，KKT条件如下：
  $$
  min_xf(x)\\
  s.t.\quad h_i(x)=0，g_j(x)\leqslant0\\
  L(x,\lambda,\mu)=f(x)+\sum_{i=1}^m\lambda_ih_i(x)+\sum_{j=1}^n\mu_jg_j(x)\\
  KKT:\begin{cases} 
  \nabla_x L(x,\lambda,\mu)=0\\
  h_i(x)=0\\
  g_j(x))\leqslant0\\
  \mu_j \geqslant 0\\
  \mu_jg_j(x)=0\\
  \end{cases}
  $$

+ **拉格朗日对偶性**

$$
定义\quad \theta_P(x)=max_{\lambda,\mu;\,\mu_i\geqslant0}\,L(x,\lambda,\mu)\\
  则\quad \theta_P(x)=\begin{cases} 
  f(x),\quad x满足原始问题约束\\
  +\infty,\quad 其他
  \end{cases}\\
  原始最优化问题等价于极小极大问题\quad p^*=min_x\theta_P(x)=min_x\,\,max_{\lambda,\mu;\,\mu_i\geqslant0}\,L(x,\lambda,\mu)\\
  定义下确界\quad \theta_D(\lambda,\mu)=min_x\,L(x,\lambda,\mu)=inf_x\,L(x,\lambda,\mu)\\
  则有对偶问题（极大极小问题）\quad d^*=max_{\lambda,\mu;\,\mu_i\geqslant0}\theta_D(\lambda,\mu)=max_{\lambda,\mu;\,\mu_i\geqslant0}\,\,min_xL(x,\lambda,\mu)\\可证明\quad d^* \leqslant p^*
$$

​       将原最优化问题，转化为求广义拉格朗日函数的极小极大问题，再转化为极大极小问题的对偶问题，两个问题在一定条件下等价，需要满足KKT条件

+ **凸优化问题**

  具有形式：
  $$
  min_{w}f(w)\\
  s.t. g_i(w)\leqslant0 i=1,2,...,k\\
  h_i(w)=0, i=1,2,...,k
  $$
  其中，f(w)和g(x)是连续可微凸函数，h(w)是仿射函数

  当f(w)是二次函数且g(w)是仿射函数时，上述问题成为**凸二次规划问题**

+ **核方法**

  优化问题中，当正则化项$\Omega(||h||_H)$单调递增，损失函数非负时，优化问题的最优解总可以表示成核函数的线性组合：$h^*(x)=\sum_{i=1}^m\alpha_iK(x,x_i)$

+ **信息论**

  默认log底数为2

  - **信息量**：$I(x_0)=-logP(x_0)$，表示获取信息的多少，事件发生概率越大，获取到的信息量越少，P为频率（0log0=0）
  - **熵(Entropy)**：$H(X)=-\sum_iP(x_i)logP(x_i)$，是信息量的期望，熵越大随机变量不确定性越大，当随机变量各取值概率一样时熵达到最大，单位比特
  - **条件熵**：$H(Y|X)=\sum_{i=1}^np_iH(Y|X=x_i)，X有n种取值$ ，表示在已知随机变量X的条件下随机变量Y的不确定性
  - **信息增益(InfoGain)**：$g(D,A)=H(D)-H(D|A)$，表示得知特征A的信息而使数据集D的分类的不确定性减少的程度，信息增益大的特征具有更强的分类能力
  - **信息增益比**：$g_R(D,A)=\cfrac{g(D,A)}{H_A(D)}$
  - **相对熵（KL散度）**：$D_{KL}(P||Q)=\sum_iP(x_i)ln\cfrac{P(x_i)}{Q(x_i)}=-H(P(x))+H(P,Q)$，表示如果用P来描述目标问题，而不是用Q来描述目标问题，得到的信息增量。在机器学习中，P往往用来表示样本的真实分布，Q用来表示模型所预测的分布，相对熵的值越小，表示P分布和Q分布越接近。可使用$D_{KL}(y||\hat{y})$评估label和predicts之间的差距
  - **交叉熵**：$H(P,Q)=-\sum_iP(x_i)lnQ(x_i)$ ，KL散度中前一部分P的熵不变，可使用交叉熵代替均方误差等作为loss函数，表示两个概率分布之间的距离
  - **基尼指数**：$Gini(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2，K为类别数;二分类Gini(p)=2p_1p_2$；数据集被条件（$特征Ai==a$）划分为两个子集，则  $Gini(D,Ai)=p_{D_1}Gini(D_1)+p_{D_2}Gini(D_2)$  （**2分类只有1种分类方法，n分类有n种分类方法**）

+ **监督式降维**

  在二维平面上，要求把点投影到一条直线上，直线的方向向量为$w$，则点X投影点到原点的距离为$w^TX$，假设只有两个类别0、1，降维后，希望同类别投影点近（协方差小），不同类别投影点远（中心距离大），即：
  $$
  记第i类:集合X_i\quad均值向量\mu_i\quad协方差矩阵\Sigma_i\quad中心投影距离w^T\mu_i\quad投影点协方差w^T\Sigma_iw\\
  最大化目标：J=\cfrac{||w^T\mu_0-w^T\mu_1||_2^2}{w^T\Sigma_0w+w^T\Sigma_1w}=
  \cfrac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T(\Sigma_0+\Sigma_1)w}\\
  定义\quad类间散度矩阵：S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)\quad 类内散度矩阵：S_w=\Sigma_0+\Sigma_1\\
  则：J=\cfrac{w^TS_bw}{w^TS_ww}\\
  解得：w^*=s_W^{-1}(\mu_0-\mu_1)
  $$

+ **EM算法**

  用于含有隐变量的概率模型参数的极大似然估计

  不同的初值可能得到不同的参数估计值；不能保证找到全局最优值

  可用于生成模型的无监督学习
  $$
  记\quad 观测随机变量（不完全数据）Y\quad 隐随机变量Z\quad完全数据Y+Z\quad模型参数\theta（初值\theta^{(0)}）\\
  迭代求\quad L(\theta)=logP(Y|\theta) \quad的最大似然估计\quad\\
  E步：第i+1次迭代计算Q函数\quad Q(\theta,\theta^{(i)})=E_z[logP(Y,Z|\theta)|Y,\theta^{(i)}]=\sum_Z logP(Y,Z|\theta)P(Z|Y,\theta^{(i)})\\
  M步：\theta^{(i+1)}=arg\,max_\theta Q(\theta,\theta^{(i)})
  $$
  推广有GEM算法

+ **高斯混合模型**

  用任意概率分布代替高斯分布密度可得到一般混合模型

  高斯混合模型可以用EM算法估 计参数 $\theta=(\alpha_k,\theta_k)$
  $$
  具有分布概率:P(y|\theta)=\sum_{k=1}^K\alpha_k\phi(y|\theta_k)\\
  \alpha_k\geqslant0,\sum_{k=1}^K\alpha_k=1;\quad \phi(y|\theta_k)是高斯分布密度，\theta_k=（\mu_k,\sigma_k^2）
  $$

+ **聚类距离**

  - 闵可夫斯基距离

    $d_{ij}=(\sum_{k=1}^m|x_{ki}-x_{kj}|^p)^{\frac1p},p\geqslant 1$，$p=$1为曼哈顿距离，$p=2$为欧氏距离，$p=\infin$为切比雪夫距离（取各个坐标数值差的绝对值的最大值）

  - 马哈拉诺比斯距离

    $d_{ij}=[(x_i-x_j)^TS^{-1}(x_i-x_j)]^{\frac12}$，S 为 X 的协方差矩阵

  - 相关系数

  - 夹角余弦

- **前向分步算法**

  为得到加法模型：$f(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)$，需要求解损失函数极小化问题：$min_{\beta_m,\gamma_m}\sum_{i=1}^NL(y_i,\sum_{m=1}^M\beta_mb(x;\gamma_m))$，前向分步算法将同时求解$m=1 toM$所有参数问题简化为逐次求解各个$\beta_m,\gamma_m$

  1. 初始化$f_0(x)=0$
  2. 对于$m=1,2,...,M$，$(\beta_m,\gamma_m)=argmin_{\beta,\gamma}\sum_{i=1}^NL(y_i,f_{m-1}(x_i)+\beta b(x;\gamma)),\quad f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m))$
  3. 得到加法模型：$f(x)=f_M(x)=\sum_{m=1}^M\beta_mb(x;\gamma_m)$



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

## 范数

向量$x$的范数满足：
$$
\large{
正定性：||x||\geqslant 0,||x||=0\Leftrightarrow x=0\\
齐次性：||\lambda x||=|\lambda|\,||x||,\lambda\in \R\\
三角不等式：||x+y||\leqslant ||x||+||y||
}
$$
矩阵$A\in \R^{n\times n}$的范数还需满足：
$$
\large{
乘法：||AB||\leqslant ||A||B||\\
向量范数与矩阵范数相容：||Ax||\leqslant||A||_\mu ||x||\\
向量范数的算子（矩阵）范数：||A||_\mu=max_{x\not=0} \cfrac{||Ax||}{||x||}=max_{||x||=1}||Ax||
}
$$


常用范数：

- p-范数：$\large ||x||_p=(\sum_{i=1}^n |x_i|^p)^{\frac{1}{p}}$
- 1-范数：$\large ||x||_1=\sum_{i=1}^n |x_i|$，对应算子范数—行和范数：$\large ||A||_{\infty}=max_{1\leqslant i\leqslant n}\sum_{j=1}^n |a_{ij}|$
- 2-范数：$\large ||x||_2=(\sum_{i=1}^n |x_i|^2)^{\frac{1}{2}}$，对应算子范数—列和范数：$\large ||A||_1=max_{1\leqslant j\leqslant n}\sum_{i=1}^n |a_{ij}|$
- $\infty$-范数：$\large ||x||_{\infty}=max_{1\leqslant i\leqslant n}|x_i|$，对应算子范数—谱范数：$\large ||A||_2=max\{\lambda|\lambda\in\lambda(A^TA)\}$

- F-范数：$\large ||A||_F=(\sum_{i=1}^n\sum_{j=1}^n a_{ij}^2)^{1/2}=\sqrt{tr(A^TA)}$

## 凸集与凸函数

**凸集**

$\forall x,y\in D,\forall \lambda\in[0,1],\lambda x+(1-\lambda)y\in D$，即集合中任意两点的线段仍属于该集合

**性质**

- $\alpha D=\{y|y=\alpha x,x\in D\}$是凸集
- 交集$D_1 \cap D_2$是凸集
- 和集$D_1+D_2=\{z|z=x+y,x\in D_1,y\in D_2\}$是凸集



**凸函数**

设函数$f:D\subset \R^n \rightarrow\R$，其中 $D$ 为凸集

- 凸函数：$\forall x,y\in D,\forall \lambda\in[0,1],f(\lambda x+(1-\lambda)y)\leqslant \lambda f(x)+(1-\lambda)f(y)$
- 严格凸函数：$\forall x,y\in D,x\not=y,\forall \lambda\in[0,1],f(\lambda x+(1-\lambda)y)< \lambda f(x)+(1-\lambda)f(y)$
- 一致凸函数：$\exist\gamma>0, \forall x,y\in D,\forall \lambda\in[0,1],f(\lambda x+(1-\lambda)y)+\cfrac{1}{2}\lambda(1-\lambda)\gamma ||x-y||^2\leqslant \lambda f(x)+(1-\lambda)f(y)$





















































































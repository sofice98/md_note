[TOC]

# Machine Learning

**方法=模型+策略+算法**

模型的假设空间内包含可能的函数和参数，要通过策略计算损失函数和风险函数，写出最优的算法。损失函数的期望为风险函数，目的是期望最小，但不能直接计算，由经验风险估计。经验风险最小化（ERM）认为经验风险最小的模型最优，但易过拟合，因此利用结构风险最小化（SRM），加上正则化或惩罚项，即可**使经验风险与模型复杂度同时小**

**模式识别**：用计算的方法根据样本的特征将样本划分到一定的类别中去

借助数学模型理解数据

+ 有监督学习（supervised learning）：对数据的若干特征与若干标签（类型）之间的关联性进行建模的过程
  - 分类 （classifification）
  - 回归（regression）

+ 无监督学习（unsupervised learning）：对不带任何标签的数据特征进行建模，通常被看成是一种“让数据自己介绍自己”的过程
  - 聚类（clustering）
  - 降维（dimensionality reduction）

+ 半监督学习（semi-supervised learning）：在数据标签不完整时使用

**机器学习步骤：**

1. 确定场景类型：数据是什么，需要得到什么，是什么问题

2. 定义损失函数（loss function）：搭建模型的目标是使模型预测的值和实际值接近，因此需要定义损失函数来评估模型效果

3. 提取特征：数据清洗->直接使用还是提取特征->数值型特征还是类别型特征

4. 确定模型形式并估计参数

5. 评估模型效果

   

**特征矩阵**：通常被简记为变量 X。它是维度 为 [n_samples, n_features] 的二维矩阵

**样本**（即每一行）通常是指数据集中的每个对象

**特征**（即每一列）通常是指每个样本都具有的某种量化观测值

**目标数组**：通常简记为 y，一般是一维数组，其长度就是样本总数 n_samples



**Scikit-Learn 评估器 API** 的常用步骤如下所示（后面介绍的示例都是按照这些步骤进行的）。 

1. 通过从 Scikit-Learn 中导入适当的评估器类，选择模型类。 

2. 用合适的数值对模型类进行实例化，配置模型超参数（hyperparameter）。 

3. 整理数据，通过前面介绍的方法获取特征矩阵和目标数组。 

4. 调用模型实例的 fit() 方法对数据进行拟合。 

5. 对新数据应用模型： 

+ 在有监督学习模型中，通常使用 predict() 方法预测新数据的标签； 
+ 在无监督学习模型中，通常使用 transform() 或 predict() 方法转换或推断数据的性质。



**模型持久化**

![image-20200604224059670](\MachineLearning\模型持久化.png)

python-python：内置库pickle

```python
import pickle
model = linear_model.LinearRegression()
model.fit(data[["x"]], data[["y"]])
# 使用pickle存储模型
pickle.dump(model, open(modelPath, "wb"))
# 使用pickle读取已有模型
model = pickle.load(open(modelPath, "rb"))
```

python-java：预测模型标记语言PMML

```python
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
# 利用sklearn2pmml将模型存储为PMML
model = PMMLPipeline([("regressor", linear_model.LinearRegression())])
model.fit(data[["x"]], data["y"])
sklearn2pmml(model, "linear.pmml", with_repr=True)
```



# 模型验证

在选择模型和超参数之后，通过对训练数据进行学习，对比模型对已知数据的预测值与实际值的差异

**模型陷阱**：

+ 使用模型对未知数据做预测：侧重准确度，易受过度拟合干扰——交叉检验
+ 借助模型分析数据的联动效应：侧重可靠性，易受模型幻觉干扰——惩罚项，假设检验

**留出集**

```python
 from sklearn.model_selection import train_test_split
 # 每个数据集分一半数据
 X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5) 
 # 用模型拟合训练数据
 model.fit(X1, y1) 
 # 在测试集中评估模型准确率
 y2_model = model.predict(X2) 
 accuracy_score(y2, y2_model)
```

**交叉检验**

```python
from sklearn.model_selection import cross_val_score 
cross_val_score(model, X, y, cv=5)
```

**最优模型**

+ 欠拟合：模型灵活性低，偏差高，模型在验证集的表现与在训练集的表现类似

+ 过拟合：模型灵活性高，方差高，模型在验证集的表现远远不如在训练集的表现

<img src="\MachineLearning\验证曲线示意图.png" style="zoom: 80%;" />

**评估模型结果**

![image-20200611141531731](\MachineLearning\数据预测分类.png)

+ **查准率**：$Precision=\cfrac{TP}{TP+FP}$ 表示预测为正的样例中有多少是真正的正样例

+ **查全率**：$Recall=\cfrac{TP}{TP+FN}$ 表示样本中的正例有多少被预测正确

+ **精确度**：$Accuracy=\cfrac{TP+TN}{TT+FN+FP+TN}$ 表示分类正确的样本数占样本总数的比例，非平衡数据集会发生准确度悖论从而导致**失真**

+ **平衡查准率与查全率**：$F_\beta=(1+\beta^2)\cfrac{P·R}{\beta^2·P+R}$ 当$\beta$靠近0时，$F_\beta$偏向查准率P，当$\beta$靠近正无穷时，$F_\beta$偏向查全率R，$\beta=1$时为$F1-score$

+ **ROC空间**（Receiver Operating Characteristic）：真阳性率 $TPR=\cfrac{TP}{TP+FN}$，伪阳性率 $FPR=\cfrac{FP}{FP+TN}$，以FPR伪横轴，以TPR为纵轴画图，得到的是ROC空间，其中：

  - 越靠近左上角预测准确率越高
  - 对角线为无意义的随机猜测
  - 对角线下方是把结果搞反了，做相反预测即可
  - 设置不同阈值参数可得到一个点，连起来就是ROC曲线。曲线下方阴影面积为AUC，代表模型预测正确的概率，不依赖于阈值，取决于模型本身
  - 当测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变

  ![image-20200611150737627](\MachineLearning\ROC.png)
  
  ```python
  from sklearn.metrics import roc_curve, auc
  
  logitModel = LogisticRegression()
  logitModel.fit(trainData[features], trainData[label])
  logitProb = logitModel.predict_proba(testData[features])[:, 1]
  fpr, tpr, _ = roc_curve(testData[label], logitProb)
  _auc = auc(fpr, tpr)
  fig = plt.figure(figsize=(6, 6), dpi=80)
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(fpr, tpr, s, label="%s:%s; %s=%0.2f" % ("模型", i,"曲线下面积（AUC）", _auc))
  ```
  



**泛化能力**

该方法学习到的模型对未知数据的预测能力

泛化误差（generalization error）等于模型对未知数据预测的误差期望，使用**泛化误差上界**来判断学习算法的优劣

# 数学

+ **梯度下降法**

  每次将估计值向梯度的负方向进行修改，可加上步长（学习率），当每次只是用一个误分类的样本进行更新时，称为**随机梯度下降法**，更新过后以前的误分类点可能会正确分类

+ **牛顿法**

  使用海森矩阵和泰勒展开公式，近似的用点的函数值表示实际函数值，由此推导出递推公式，当使用另一个矩阵近似海森矩阵以简化计算量时，称为**拟牛顿法**

+ **拉格朗日对偶性**

  将原最优化问题，求广义拉格朗日函数的极小极大值，转化为求极大极小值问题的（对偶问题），两个问题在一定条件下等价，需要满足KKT条件







# 有监督学习

## 感知机

**原始形式**

给定训练集$X,y\in\{-1,1\}$，感知机$sign(w·x+b)$学习的损失函数为$L(w,b)=-\sum_{x_i\in m}y_i(w·x_i+b)$

感知机算法是误分类驱动的，采用随机梯度下降法，每次选取一个误分类点对w,b进行更新：$w:=w+\eta y_ix_i，b:=b+\eta y_i$，$\eta\in(0,1]$称为学习率（步长），误分条件为：$y_i(w·x_i+b)\leqslant0$

**对偶形式**

令$\alpha_i=n_i\eta$，则误分条件为：$y_i(\sum_{j=1}^Na_jy_jx_j·x_i+b)\leqslant0$，更新：$\alpha_i:=\alpha_i+\eta，b:=b+\eta y_i$，结果求得$w=\sum_{i=1}^N\alpha_iy_ix_i$。在求误分条件时所用内积可以先求出来存在矩阵中（**Gram矩阵**）：$G=[x_i·x_j]_{N\times N}$

可以证明感知机的原始和对偶形式都是收敛的

```python
from sklearn.linear_model import Perceptron
# 参数：正则化项penalty=None/l1/l2 正则化系数alpha 学习率eta0 最大迭代次数max_iter=5 终止阈值tol=None
perceptron = Perceptron()
perceptron.fit(X,y)
# 模型参数w，b和迭代次数
w = perceptron.coef_ 
b = perceptron.intercept_
it = perceptron.n_iter_
# 预测准确率
perceptron.score(X,y)
```

**当特征维度大时用对偶形式，当样本多时用原始形式**

## 线性回归

简洁，高效，容易理解

特征（features）为自变量，标签（labels）为因变量



**数学推导**

给定一组数据其中包括特征矩阵 ![[公式]](https://www.zhihu.com/equation?tex=X) , 目标变量向量 ![[公式]](https://www.zhihu.com/equation?tex=y) :

![[公式]](https://www.zhihu.com/equation?tex=y+%3D+%5Cleft%5B+%5Cbegin%7Bmatrix%7D+y_1+%5C%5C+y_2+%5C%5C+%3A+%5C%5C+y_m+%5Cend%7Bmatrix%7D+%5Cright%5D) 

![[公式]](https://www.zhihu.com/equation?tex=X+%3D+%5Cleft%5B+%5Cbegin%7Bmatrix%7D+1+%26+x_%7B11%7D+%26+x_%7B12%7D+%26+%E2%80%A6+%26+x_%7B1n%7D+%5C%5C+1+%26+x_%7B21%7D+%26+x_%7B22%7D+%26+%E2%80%A6+%26+x_%7B2n%7D+%5C%5C+%3A+%26+%3A+%26+%3A+%26+%E2%80%A6+%26+%3A+%26+%5C%5C+1+%26+x_%7Bn1%7D+%26+x_%7Bn2%7D+%26+%E2%80%A6+%26+x_%7Bnn%7D+%5C%5C+%5Cend%7Bmatrix%7D+%5Cright%5D) 

其中 ![[公式]](https://www.zhihu.com/equation?tex=X) 第一列为截距项，我们做线性回归是为了得到一个最优回归系数向量 ![[公式]](https://www.zhihu.com/equation?tex=w) 使得当我们给定一个 ![[公式]](https://www.zhihu.com/equation?tex=x) 能够通过 ![[公式]](https://www.zhihu.com/equation?tex=y%3Dxw) 预测 ![[公式]](https://www.zhihu.com/equation?tex=y) 的值。其中 ![[公式]](https://www.zhihu.com/equation?tex=w+%3D+%5Cleft%5B+%5Cbegin%7Bmatrix%7D+w_0%5C%5C+w_1+%5C%5C+w_2+%5C%5C+%3A+%5C%5C+w_n+%5Cend%7Bmatrix%7D+%5Cright%5D)

这里采用平方误差求最优![[公式]](https://www.zhihu.com/equation?tex=w) ：![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bm%7D+%28y_i+-+x_%7Bi%7D%5E%7BT%7Dw%29%5E2) 

对于上述式子 ![[公式]](https://www.zhihu.com/equation?tex=f%28w%29) 可以通过梯度下降等方法得到最优解。但是使用矩阵表示将会使求解和程序更为简单：![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+%3D+%28y+-+Xw%29%5E%7BT%7D%28y+-+Xw%29) 

将 ![[公式]](https://www.zhihu.com/equation?tex=f%28w%29+) 对 ![[公式]](https://www.zhihu.com/equation?tex=w) 求导可得：![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+f%28w%29%7D%7B%5Cpartial+w%7D+%3D+-2X%5E%7BT%7D%28y+-+Xw%29) 

使其等于0，便可得到：![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7Bw%7D+%3D+%28X%5E%7BT%7DX%29%5E%7B-1%7DX%5E%7BT%7Dy) 

​	



**损失函数**

+ Least Absolute Deviations(LAS)： $L=\sum_i|y_i-\hat{y}_i|$ 
  
  对异常值更加稳定
  
+ Odinary Least Squares(OLS)： $L=\sum_i(y_i-\hat{y}_i)^2$
  数学基础更加扎实，与统计学最大似然估计法的结果一致

**模型评估**

+ 均方差(MSE）:    $MSE=\cfrac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2=\cfrac{1}{n}L$，平均误差=$\sqrt{MSE}$
+ 决定系数（coefficient of determination）：$SS_{tot}=\sum_i(y_i-\bar{y})^2$； $SS_{res}=\sum_i(y_i-\hat{y}_i)^2$； $R^2=1-\cfrac{SS_{res}}{SS_{tot}}$，决定系数表示有多少的因变量变化能由模型解释

```python
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

def linearModel(data):
    """
    线性回归模型机器学习
    
    参数
    ----
    data : DataFrame，建模数据
    """
    features = ["x"]
    labels = ["y"]
    # 划分训练集和测试集
    trainData = data[:15]
    testData = data[15:]
    
    # 产生并训练模型
    # 创建一个线性回归模型
    model = linear_model.LinearRegression()
    # 训练模型，估计模型参数
    model.fit(trainData[features], trainData[labels])
    #  斜率：model.coef_[0]，截距：model.intercept_
    
    # 评价模型效果
    # 均方差(The mean squared error)，均方差越小越好
    error = np.mean(
        (model.predict(testData[features]) - testData[labels]) ** 2)
    # 决定系数(Coefficient of determination)，决定系数越接近1越好
    score = model.score(testData[features], testData[labels])
    
def linearModel(data):
    """
    线性回归统计性质分析

    参数
    ----
    data : DataFrame，建模数据
    """
    features = ["x"]
    labels = ["y"]
    Y = data[labels]
    # 加入常量变量
    X = sm.add_constant(data[features])
    # 构建模型
    model = sm.OLS(Y, X)
    re = model.fit()
    
    # 分析模型效果
    # 整体统计分析结果
    print(re.summary())
    # 用f_test检测x对应的系数a是否显著，P-value小于0.05则拒绝
    print("检验假设x的系数等于0：")
    print(re.f_test("x=0"))
    
    # const并不显著，去掉这个常量变量
    model = sm.OLS(Y, data[features])
    resNew = model.fit()
    # 输出新模型的分析结果
    print(resNew.summary())
```

> 基函数回归
>
> 通过基函数对原始数据进行变换，从而将变量间的线性回归模型转换为非线性回归模型

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline 
# 7次多项式回归模型
poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

rng = np.random.RandomState(1) 
x = 10 * rng.rand(50) 
y = np.sin(x) + 0.1 * rng.randn(50) 
poly_model.fit(x[:, np.newaxis], y) 
yfit = poly_model.predict(xfit[:, np.newaxis]) 
plt.scatter(x, y) 
plt.plot(xfit, yfit);
```



**惩罚项**

+ Lasso回归：使用L1范数  $L=\sum_i(y_i-ax_i-bx_i-c)^2+\alpha(|a|+|b|+|c|)$

+ Ridge回归（岭回归）：使用L2范数  $L=\sum_i(y_i-ax_i-bx_i-c)^2+\alpha(a^2+b^2+c^2)$

*超参数*$\alpha>0$时，惩罚项会随a,b,c绝对值的增大而增大，即参数越远离0，惩罚就越大，因此在寻找L的最小值时，这一项会迫使参数估计值向0靠近

超参数存在时，采用网格搜寻（设置几组针对超参数的控制变量组，来找最小均方差的超参数）和验证集（将数据分为训练集，验证集，测试集）

```python
# scikit
reg = linear_model.Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg = linear_model.Lasso(alpha=0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
# statsmodels
model = sm.OLS(Y, X)
res = model.fit_regularized(alpha=alpha)
```

下图为$\alpha$改变时三个参数的变化规律

![image-20200604222642765](\MachineLearning\惩罚项.png)

+ 前向逐步回归：贪心，开始所有权重设为1，每一步对某个权重增加或减少一个值，迭代若干次即可收敛得到超参数

```python
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    """
    前向逐步回归

    参数
    ----
    xArr : []，输入数据
    yArr : []，预测数据
    eps  : 步长
    numIt: 迭代次数
    """
    xMat = mat(xArr)
    yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #can also regularize ys but will get smaller coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) #testing code remove
    #ws = zeros((n,1)) #初始所有权重为0
    ws = ones((n,1)) #初始所有权重为1
    #wsMax = ws.copy()
    Mat = zeros((numIt, n))
    for i in range(numIt):
        if i%1000 ==0 : #每迭代1000次，打印归回系数
            print("第%d次迭代后回归系数为："%i,end ="")
            print(ws.T)
        lowestError = inf #初始误差设为正无穷大
        for j in range(n):
            for sign in [-1,1]: # 贪心算法，左右试探
                wsTest = ws.copy() # 初始化
                wsTest[j] += eps*sign #eps 为步长
                yTest = xMat*wsTest #预测值
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        Mat[i,:]=ws.T
    return Mat
```

相比假设检验更加自动化，工程化，理论基础较差，可解释性差



**假设检验**

理论基础更牢，更容易理解数据之间关系，需要较多人工干预，随意性较大



**局部加权线性回归（Locally Weighted Linear Regression, LWLR）**

一种非参数算法，没有训练步骤，而是直接使用训练集进行预测。在做预测时，更多地参考距离预测样本近的已知样本，而更少地参考距离预测样本远的已知样本

原均方差和修改后均方差如下所示：

$\mathcal{L} = \dfrac{1}{2} \left[ \left(y^{(1)} -  \boldsymbol{\theta}^T \mathbf{x}^{(1)} \right)^2 + \left(y^{(2)} -  \boldsymbol{\theta}^T \mathbf{x}^{(2)} \right)^2 + \cdots +  \left(y^{(m)} - \boldsymbol{\theta}^T \mathbf{x}^{(m)} \right)^2 \right]$

$\begin{align*} \mathcal{L} &= \dfrac{1}{2} \left[ w^{(1)}  \left(y^{(1)} - \boldsymbol{\theta}^T \mathbf{x}^{(1)} \right)^2 +  w^{(2)} \left(y^{(2)} - \boldsymbol{\theta}^T \mathbf{x}^{(2)} \right)^2 + \cdots + w^{(m)} \left(y^{(m)} - \boldsymbol{\theta}^T  \mathbf{x}^{(m)} \right)^2 \right] \\ &= \frac{1}{2} \sum_{i=1}^m  w^{(i)} \left(y^{(i)} - \boldsymbol{\theta}^T \mathbf{x}^{(i)}\right)^2 \end{align*}$

修改后的加上了权重w：

$w^{(i)} = \exp \left( - \dfrac{\left(\mathbf{x}^{(i)} - \mathbf{x}\right)^2}{2k^2} \right)$

缺点是每个点做预测时都要使用整个数据集，计算量大，可改进











## 逻辑回归

认为观察得到的结果是经过一层变换后的结果，由正效用$y_i^*$和负效用$y_i^-$确定（称为隐含变量），

probit回归：$P(y_i=1)=1-F_{\epsilon}(-\Chi_i\gamma)$  由此计算顾客购买的比例，但正态分布的分布函数无法表示，需要近似

可以用sigmoid函数（逻辑分布的分布函数）近似正态分布，它表示某一方竞争胜出的概率

近似后的模型为：*逻辑回归模型*  $P(y_i=1)=\cfrac{1}{1+e^{-\Chi_i\beta}}$  ， $ln\cfrac{P(y_i=1)}{1-P(y_i=1)}=\Chi_i\beta$  ，也就是假设**事件发生比**（odds，$\cfrac{P}{1-P}$）的对数为线性模型，事实是将非线性模型转化为线性模型



<img src="\MachineLearning\逻辑函数近似.png" alt="image-20200611103957090" style="zoom:80%;" />

------

信息论中：

**信息量**  $I(x_0)=-lnP(x_0)$ 表示获取信息的多少，事件发生概率越大，获取到的信息量越少

**熵**  $H(X)=\sum_iP(x_i)I(x_i)$ 表示信息量的期望

**相对熵（KL散度）**  $D_{KL}(P||Q)=\sum_iP(x_i)ln\cfrac{P(x_i)}{Q(x_i)}=-H(P(x))+H(P,Q)$ 表示如果用P来描述目标问题，而不是用Q来描述目标问题，得到的信息增量。在机器学习中，P往往用来表示样本的真实分布，Q用来表示模型所预测的分布，相对熵的值越小，表示P分布和Q分布越接近。可使用$D_{KL}(y||\hat{y})$评估label和predicts之间的差距

**交叉熵** $H(P,Q)=-\sum_iP(x_i)lnQ(x_i)$ ，KL散度中前一部分P的熵不变，一般使用交叉熵作为loss函数

-------

**LOSS函数**：$LL=-\sum_i[y_iln\hat{y_i}+(1-y_i)ln(1-\hat{y_i})]$ 

```python
from sklearn.linear_model import LogisticRegression
data = pd.read_csv(path)
cols = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "label"]
data = data[cols]
data[["age", "hours_per_week", "education_num", "label_code"]].hist(
     rwidth=0.9, grid=False, figsize=(8, 8), alpha=0.6, color="grey")
plt.show()

# 显示基本统计信息
print(data.describe(include="all"))
# 计算education_num, label交叉报表
cross1 = pd.crosstab(pd.qcut(data["education_num"],  [0, .25, .5, .75, 1]), data["label"])
print(cross1)
# 将交叉报表图形化
props = lambda key: {"color": "0.45"} if ' >50K' in key else {"color": "#C6E2FF"}
mosaic(cross1[[" >50K", " <=50K"]].stack(), properties=props)
# 计算hours_per_week, label交叉报表
cross2 = pd.crosstab(pd.cut(data["hours_per_week"], 5), data["label"])
# 将交叉报表归一化，利于分析数据
cross2_norm = cross2.div(cross2.sum(1).astype(float), axis=0)
print(cross2_norm)
# 图形化归一化后的交叉报表
cross2_norm.plot(kind="bar", color=["#C6E2FF", "0.45"], rot=0)
plt.show()

# 将数据分为训练集和测试集
trainSet, testSet = train_test_split(data, test_size=0.2, random_state=2310)
# 搭建逻辑回归模型，并训练模型
formula = "label_code ~ age + education_num + capital_gain + capital_loss + hours_per_week"
model = sm.Logit.from_formula(formula, data=data)
re = model.fit()

# 整体统计分析结果
print(re.summary())
# 计算各个变量对事件发生比的影响
# conf里面的三列，分别对应着估计值的下界、上界和估计值本身
conf = re.conf_int()
conf['OR'] = re.params
conf.columns = ['2.5%', '97.5%', 'OR']
print(np.exp(conf))
# 计算各个变量的边际效应
print(re.get_margeff(at="overall").summary())

# 计算事件发生的概率
testSet["prob"] = re.predict(testSet)
# 根据预测的概率，得出最终的预测
testSet["pred"] = testSet.apply(lambda x: 1 if x["prob"] > alpha else 0, axis=1)
```

**多元分类**

+ 多元逻辑回归：使用多个隐含变量模型
+ One-vs.-all：将多元问题分为多个二元子问题

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial', solver='sag',max_iter=1000, random_state=42)
# model = LogisticRegression(multi_class='ovr', solver='sag',max_iter=1000, random_state=42)
model.fit(data[features], data[labels])
```

**非均衡数据集**

数据标签偏向若干个，可通过修改损失函数里不同类别的权重来解决

```python
# 通过调整各个类别的比重，解决非均衡数据集的问题
positiveWeight = len(Y[Y>0]) / float(len(Y))
classWeight = {1: 1. / positiveWeight, 0: 1. / (1 - positiveWeight)}
# 为了消除惩罚项的干扰，将惩罚系数设为很大
model = LogisticRegression(class_weight=classWeight, C=1e4)
model.fit(X, Y.ravel())
```



## 支持向量机

**数学原理**

如图，空间中的直线可以用一个线性方程来表示：$\beta·(T-\beta)=0$ ，因此，可设$f(X)=\beta·X$，则$f(X)=0$表示一条直线，且$|f(X)|$与点$X$到直线$f(X)=0$的距离成正比，$f(X)$的符号表示点$X$到直线的垂线方向

<img src="\MachineLearning\svm向量基础.png" alt="image-20200615204603657" style="zoom: 50%;" />

应用到svm中：

<img src="\MachineLearning\svm原理1.png" alt="image-20200615210914591" style="zoom: 67%;" />

设分割线为$\beta·X+b=0$，可得到上下两条虚线和类别分类条件，转化为数学语言并化简：

<img src="\MachineLearning\svm原理2.png" alt="image-20200615211105746" style="zoom: 67%;" />

<img src="\MachineLearning\svm原理3.png" alt="image-20200615211802211" style="zoom:67%;" />

对于**线性不可分问题**，需要加入损失函数：$y_i(w·X_i+c)\geqslant1-\xi_i,\quad\xi\geqslant0$，其中$\xi_i$与点 i 离相应虚线的距离成正比，表示数据 i 这一点违反自身分类原则的程度，所有点的$\xi_i$和越小越好，因此加入**超参数C**，合并损失函数：

<img src="\MachineLearning\svm损失项.png" alt="image-20200615213257591" style="zoom:67%;" />

将不等式变形并改为等式，赋值给$\xi_i$一个值令其满足两个不等式限制条件，再带入损失函数中，得到：

<img src="\MachineLearning\svm损失函数.png" alt="image-20200615215527836" style="zoom:67%;" />

损失函数的前一部分为**L2惩罚项**，后一部分为**预测损失LL（hinge loss）**

其中超参数C是模型预测损失的权重，C越大表示模型越严格，margin越小，考虑的点越少，称为**hard margin**，C越小考虑的点越多，称为**soft margin**

<img src="\MachineLearning\svm超参数.png" alt="image-20200615221059133" style="zoom:67%;" />

将上边的有限制条件的原始SVM问题转化为更一般的、适于非线性的对偶问题：

<img src="\MachineLearning\svm对偶.png" alt="image-20200622180901407" style="zoom:67%;" />

事实上是寻找与被测数据相似的训练数据，并将相应的因变量加权平均得到最后的预测值。只有在虚线上或虚线内的点权重才不为0，其他点权重都为0：

<img src="\MachineLearning\svm支持向量.png" alt="image-20200622181820083" style="zoom:50%;" />

**核函数**

$K(x_i,x_j)=\phi(x_i)·\phi(x_j)$ ，其中$\phi(x)$为空间变换

利用核函数，可以极大减少模型运算量，并且完成未知的空间变换，实际应用中，可以用网格搜寻的办法找到最合适的核函数

<img src="\MachineLearning\常用核函数.png" alt="image-20200622182951633" style="zoom:67%;" />

**Scale variant**

不带惩罚项的线性回归和逻辑回归对特征的线性变换是稳定的，但SVM对线性变化不稳定，变量的权重改变不可被修复。可以用**归一化**来去掉干扰因素



```python 
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, laplacian_kernel
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel

model = SVC(kernel=linear_kernel, coef0=1)
model.fit(data[["x1", "x2"]], data["y"])
```



## 决策树

使用树形决策流程，将数据进行分类

寻找最优决策树是一个NP完全问题，只能退而求其次使用贪心算法



**评判标准**

要求规则能基本把节点上不同类别的数据分离开，需要定义一个量来衡量数据类别单一程度

**类别在节点上的占比**：$p_{mi}=\cfrac{1}{N_m}\sum_j1_{\{y_j=i\}}$

**不纯度**：$H_m$，越接近0，表示数据类别越单一，有以下几个常用指标

<img src="\MachineLearning\不纯度指标.png" alt="image-20200622194324629" style="zoom:50%;" />

带有split的为相应指标的不纯度计算

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
# 单独使用决策树
dtModel = DecisionTreeClassifier(max_depth=2)
dtModel.fit(trainData[features], trainData[label])
dtProb = dtModel.predict_proba(testData[features])[:, 1]
```

决策树优点在于能够考虑多个变量，而且变量的线性变换是稳定的，缺点是模型最后一步算法比较薄弱。为了得到更好的预测效果，可以使用**模型联结主义**，将决策树作为整体模型的一部分和其他模型嵌套使用

**剪枝**

决策树模型属于非参模型，容易发生过拟合问题，解决方法为剪枝

+ 前剪枝：作用于决策树生成过程中，如设置阈值限制高度
+ 后剪枝：作用与决策树生成之后，将一些不太必要的子树剪掉，剪掉的节点的纯度下降不明显

后剪枝中应用较广泛的为**消除误差剪枝法（REP）**，将数据分为训练集、剪枝集、测试集，将不符合剪枝集分类的子树剪掉，且符合从下往上按层下边的全部剪完再剪上边的，称为bottom-up restriction

## 集成方法

为了是模型间的组合更加自动化，只使用一种模型最好，将机器学习中比较简单的模型（弱学习）组合成一个预测效果好的复杂模型，即为集成方法（ensemble method）

**树的集成**

+ 平均方法（averaging methods）

  **随机森林（random forests）**

  各个树相互独立时，可以降低犯错概率

  对于分类问题，结果等于各个树中出现次数最多的类别

  对于回归问题，结果等于各个树结果的平均值

  随机来源及scikit-learn函数如下：

  <img src="\MachineLearning\随机森林.png" alt="image-20200623110624949" style="zoom: 67%;" />

  **随机森林高维映射（random forest embedding）**

  可以将随机森林当作非监督式学习使用，随机抽取特征组合成合成数据，与原始数据一起进行决策树训练。当分类结果误差较小时，说明各变量间的相关关系比较强烈

  <img src="\MachineLearning\rfe.png" alt="image-20200623111544046" style="zoom:50%;" />

  使用随机森林将低维数据映射到高维后，可以与其他模型联结：

  <img src="\MachineLearning\rfe2.png" alt="image-20200623111941192" style="zoom:67%;" />

  ```python
  from sklearn.ensemble import RandomTreesEmbedding
  from sklearn.naive_bayes import BernoulliNB
  from sklearn.pipeline import Pipeline
  
  pipe = Pipeline([("embedding", RandomTreesEmbedding(random_state=1024)),
          ("model", BernoulliNB())])
  pipe.fit(data[["x1", "x2"]], data["y"])
  prob = pipe.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:, 0]
  # 将模型的预测结果可视化
  # 生成100*100的预测点集
  x1 = np.linspace(min(data["x1"]) - 0.2, max(data["x1"]) + 0.2, 100)
  x2 = np.linspace(min(data["x2"]) - 0.2, max(data["x2"]) + 0.2, 100)
  X1, X2 = np.meshgrid(x1, x2)
  # 预测点的类别
  prob = pipe.predict_proba(np.c_[X1.ravel(), X2.ravel()])[:, 0]
  prob = prob.reshape(X1.shape)
  # 画轮廓线
  ax.contourf(X1, X2, prob, levels=[0.5, 1], colors=["gray"], alpha=0.4)
  plt.show()
  ```

  

+ 提升方法（boosting methods）

  **梯度提升决策树（gradient-boosted trees，GBTs）**

  使用梯度提升法，得到更好的预测结果

  <img src="\MachineLearning\梯度提升.png" alt="image-20200623115608709" style="zoom: 50%;" />

  <img src="\MachineLearning\梯度下降提升.png" alt="image-20200623115819454" style="zoom: 60%;" />

  GBTs的算法步骤如下：

  <img src="\MachineLearning\gbts细节.png" alt="image-20200623120120936" style="zoom:60%;" />

  GBTs损失函数里没有惩罚项，容易过拟合，可使用XGBoost算法或者与其他模型联结来解决

## 生成式模型

关心数据$\{X,y\}$是如何生成的，X代表事物的表象，y代表事物的内在

<img src="\MachineLearning\贝叶斯定理.png" style="zoom:60%;" />



利用贝叶斯框架，简化版预测公式：$\hat{y}=argmax_yP(X|y)P(y)$



### 朴素贝叶斯

**naive Bayes assumption**：假设各特征相互独立：$P(x_1,x_2,...,x_n|y)=\prod_{i=1}^nP(x_i|y)$ ，越独立效果越好

包含：伯努利模型，多项式模型，高斯模型

> NLP特征提取字典法：将出现的文字组成一个字典X，出现的记为1，X为非常稀疏的向量

+ **伯努利模型**

  $P(x_i=1|y)=p_{i,y};P(x_i=0|y)=1-p_{i,y}$ 

  多元伯努利模型的变量y的分布概率$\hat{\theta_l}$等于各类别在训练数据中的占比，每个字的条件概率$\hat{p_{j,l}}$等于这个字在这个类别里出现的比例

  训练集没有出现过的字没办法预测，这时可加入**平滑项**，将$\hat{p_{j,l}}$的计算公式分母上加$2\alpha$，分子上加$\alpha$（平滑系数） 

+ **多项式模型**

  更改特征提取方法，X的长度与文本字数相同，第i个元素表示第i个位置上出现的文字的字典序号

  变量y的分布概率$\hat{\theta_l}$等于各类别在训练数据中的占比，每个字的条件概率$\hat{p_{j,l}}$等于这个字的出现次数占这个类别的总字数的比例

+ **TD-IDF**

  文字对文本的影响主要有：

  - TF：某个文字在文本中出现的比例越高，与主题越相关

  - IDF：如果某个文字在几乎所有文本中都出现，说明它是常用字

  $TF_{i,k}=x_{i,k}/\sum_kx_{i,k}$，$IDF_k=ln(m/\sum_i1_{\{x_{i,k}>0\}})$，$TFIDF_{i,k}=TF_{i,k}IDF_{k}$

  常对文本向量进行TF-IDF 变换后在使用多项式模型

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

def trainBernoulliNB(data):
    """
    使用伯努利模型对数据建模
    """
    # 生成量化文本向量
    vect = CountVectorizer(token_pattern=r"(?u)\b\w+\b", binary=True)
    X = vect.fit_transform(data["content"])
    # 生成量化标签字典
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    model = BernoulliNB()
    model.fit(X, Y)
    return vect, le, model

def trainMultinomialNB(data):
    """
    使用多项式模型对数据进行建模
    """
    pipe = Pipeline([("vect", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ("model", MultinomialNB())])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    pipe.fit(data["content"], Y)
    return le, pipe

def trainMultinomialNBWithTFIDF(data):
    """
    使用TFIDF+多项式模型对数据建模
    """
    pipe = Pipeline([("vect", CountVectorizer(token_pattern=r"(?u)\b\w+\b")),
        ("tfidf", TfidfTransformer(norm=None, sublinear_tf=True)),
        ("model", MultinomialNB())])
    le = LabelEncoder()
    Y = le.fit_transform(data["label"])
    pipe.fit(data["content"], Y)
    return le, pipe

def trainModel(trainData, testData, testDocs, docs):
    """
    对分词后的文本数据分别使用多项式和伯努利模型进行分类
    """
    # 伯努利模型
    vect, le, model = trainBernoulliNB(trainData)
    pred = le.classes_[model.predict(vect.transform(testDocs))]
    # 传入准确值和预测值，生成分析报告
    print(classification_report(
        le.transform(testData["label"]),
        model.predict(vect.transform(testData["content"])),
        target_names=le.classes_))
    # 多项式模型
    le, pipe = trainMultinomialNB(trainData)
    pred = le.classes_[pipe.predict(testDocs)]
    print(classification_report(
        le.transform(testData["label"]),
        pipe.predict(testData["content"]),
        target_names=le.classes_))
    # TFIDF+多项式模型
    le, pipe = trainMultinomialNBWithTFIDF(trainData)
    pred = le.classes_[pipe.predict(testDocs)]
    print(classification_report(
        le.transform(testData["label"]),
        pipe.predict(testData["content"]),
        target_names=le.classes_))
```



### 判别分析

discriminant analysis，与朴素贝叶斯相比，允许变量间存在关系

+ **线性判别分析（LinearDiscriminantAnalysis，LDA）**

  只能处理连续型变量，$X|y=0\sim N(\mu_0,\Sigma)$，$X|y=1\sim N(\mu_1,\Sigma)$，$P(y=0)=\theta_0$，$P(y=1)=\theta_1$

  模型要求：

  1. 变量服从正态分布，因此要连续；
  2. 对于不同类别，自变量协方差一样，只是期望不一样，只关心各类别中心位置；
  3. 协方差$\Sigma$是对角矩阵时，变量相互独立

  参数估计：

  1. $\theta_l$：等于各类别在训练数据中的占比
  2. $\mu_l$：等于训练数据里各类别的平均值
  3. $\Sigma$：等于各类别内部协方差的加权平均，权重为类别内数据的个数
  
  LDA的预测公式与逻辑回归的一样，在满足模型要求时，往往生成式模型效果更好
  
  LDA可以用作降维，$\mu_l$与降维理论中$\mu_l$一样，$\Sigma$与降维理论中$\cfrac{1}{m}s_W$一样
  
  ```python
  from sklearn import datasets
  from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
  # 生成数字集
  digits = datasets.load_digits()
  X = digits.data
  y = digits.target
  #输入降维后的维度进行降维
  model = LinearDiscriminantAnalysis(n_components=3)
  model.fit(X, y)
  newX = model.transform(X)
  ```
  
+ **二次判别分析（QuadraticDiscriminantAnalysis，LDA）**

  不要求自变量分布的协方差矩阵一样，但无降维功能，当协方差矩阵为对角矩阵时，二次判别变成高斯模型

  调用GaussianNB或QuadraticDiscriminantAnalysis建模

### 隐马尔科夫模型

当数据之间不再独立，数据间的顺序会对数据本身造成影响，此时称为序列数据，需要用隐马尔科夫模型（Hidden Markov Model，HMM）来解决

**马尔科夫链**

描述的是一个随机过程，随机变量是相互关联的：$P(y_i|y_{i-1},y_{i-2},...,y_0)=P(y_i|y_{i-1})$，并且$P(y_0,...,y_{i-1},y_{i+1},...,y_n|y_i)=P(y_0,...,y_{i-1}|y_i)P(y_{i+1},...,y_n|y_i)$，即在已知当前状态的条件下，未来和过去相互独立

当$y_i$取值离散且链平稳时，马尔科夫链可由**转移矩阵**和**初始分布**表示，转移矩阵Q为n*n方阵，$Q_{p,q}=P(y_i=q|y_{i-1}=p)$，并且也包括先验概率（常用朴素贝叶斯），数据的**联合概率**为$P(X,y)=P(y_0)P(y_1|y_0)...P(y_i|y_{i-1})\prod_jP(X_j|y_j)$

**应用**

+ 监督式学习（中文分词）

  multinomial HMM，包含Viterbi算法求解模型预测结果

  将文字分为若干种状态，再去根据y估计模型参数

+ 非监督式学习（股票市场）

  要用到最大期望算法（EM）来估计模型参数和预测量：先随机生成模型参数，通过E step求得预测值y，再通过M step求得估算的模型参数，重复交叉使用即可得到所有参数

  假设股票的日收益率和成交量服从正态分布，此时隐马尔科夫模型称为Gaussian HMM

  ```python
  from matplotlib.finance import candlestick_ochl
  from hmmlearn.hmm import GaussianHMM
  
  cols = ["r_5", "r_20", "a_5", "a_20"]
  # 参数为内在状态个数，先验分布的协方差矩阵类型，迭代次数
  model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000,
                      random_state=2010)
  model.fit(data[cols])
  hiddenStatus = model.predict(data[cols])
  ```

  





# 无监督学习



## k-means



## 降维

理论：

在二维平面上，要求把点投影到一条直线上，直线的方向向量为$\vec{k}$，|k|=1，则点X1投影点到原点的距离为$kX_1^T$，假设只有两个类别0、1，

**类别中心**：$u_l=\cfrac{1}{N_l}\sum_{i=1}^m1_{\{yi=l\}}kX_1^T$，$N_l$为类别数据个数

**内部方差**：$v_l=\sum_{i=1}^m1_{\{yi=l\}}(kX_1^T-u_l)^2$

降维后，希望各类别中心越远越好，各类别方差越小越好，即

**最大化类别距离**：$max_k(u_0-u_1)^2/(v_0+v_1)$

记**原始数据的类别中心**：$\mu_l=\cfrac{1}{N_l}\sum_{i=1}^m1_{\{yi=l\}}X_i$，则$u_l=k\mu_l^T$

**内部方差矩阵**：$s_l=\sum_{i=1}^m1_{\{yi=l\}}(X_i-\mu_l)^T(X_i-\mu_l)$，则$v_l=ks_lk^T$

记：$s_W=(s_0+s_1)$，$s_B=(\mu_0-\mu_1)^T(\mu_0-\mu_1)$

则最大化类别距离：$max_kks_Bk^T/ks_Wk^T$，得$k^*=s_W^{-1}(\mu_0-\mu_1)$



# 特征工程

找到与问题有关的任何信息，把它们转换成特征矩阵的数值

### 分类特征

非数值数据类型分类数据——独热编码

```python
# data = [ 
# {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'}, 
# {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'}, 
# {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'}, 
# {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'} 
# ]
from sklearn.feature_extraction import DictVectorizer
# sparse=True为稀疏矩阵
vec = DictVectorizer(sparse=False, dtype=int) 
vec.fit_transform(data)
# array([[ 0, 1, 0, 850000, 4], 
#        [ 1, 0, 0, 700000, 3], 
#        [ 0, 0, 1, 650000, 3], 
#        [ 1, 0, 0, 600000, 2]], dtype=int64)
# 查看特征名称
vec.get_feature_names()
# ['neighborhood=Fremont', 
# 'neighborhood=Queen Anne', 
# 'neighborhood=Wallingford', 
# 'price', 
# 'rooms']
```






























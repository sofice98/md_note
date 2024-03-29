

# Machine Learning

**机器学习步骤：**

1. 确定场景类型：数据是什么，需要得到什么，是什么问题

2. 定义损失函数（loss function）：搭建模型的目标是使模型预测的值和实际值接近，因此需要定义损失函数来评估模型效果

3. 提取特征：数据清洗->直接使用还是提取特征->数值型特征还是类别型特征

4. 确定模型形式并估计参数

5. 评估模型效果

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

![image-20200604224059670](..\Resources\模型持久化.png)

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
 from sklearn.metrics import accuracy_score
 # 得到训练集测试集
 # 想要每次都一样：保存，指定random_state，使用数据ID
 X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5) 
 # 用模型拟合训练数据
 model.fit(X1, y1) 
 # 在测试集中评估模型准确率
 y2_model = model.predict(X2) 
 accuracy_score(y2, y2_model)
```

**交叉检验**

```python
from sklearn.model_selection import cross_val_score,KFold 
# cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores.mean())
# KFold
kfold = KFold(n_splits=NFOLDS, shuffle=True, random_state=1998)
kf = kfold.split(train_df)
for train_idx, validate_idx in kf:
    # 切割训练集&验证集
    X_train, y_train = train_df[feature_names].iloc[train_idx, :], Y_train.iloc[train_idx, :]
    X_valid, y_valid = train_df[feature_names].iloc[validate_idx, :], Y_train.iloc[validate_idx]
```

**自助法（Bootstrap）**

每次随机从数据集D中挑选一个样本拷贝到子集$D^{\\'}$中，约有36.8%的样本未出现在$D^{\\'}$中；将$D^{\\'}$作为训练集，$D\setminus D^{\\'}$作为测试集

- 在数据集较小，难以划分训练/测试集时有效

- 能从初始数据集中产生多个不同训练集

- 改变了初始数据集分布，引入估计偏差

**偏差与方差**

对算法的期望泛化错误率进行拆解：（$E[\cdot]$为期望，$\bar{f}(x)=E[f(x)]$，$y_D$为数据集中的标签）

方差：$var(x)=E[(f(x)-\bar{f}(x))^2]$，度量同样大小训练集变动导致的学习性能变化，即**数据扰动**

偏差：$bias^2(x)=(y-\bar{f}(x))^2$，度量算法期望预测与真实结果偏离程度，即**准确率**

噪声：$\epsilon^2=E[(y-y_D)^2]$，度量当前任务上任何算法能达到的期望泛化误差下界，即**问题难度**

**最优模型**

+ 欠拟合：模型灵活性低，偏差高，模型在验证集的表现与在训练集的表现类似——增加输入特征项，增加参数，减少正则化项

+ 过拟合：模型灵活性高，方差高，模型在验证集的表现远远不如在训练集的表现——数据清洗，增大训练集，增多正则化项

<img src="..\Resources\验证曲线示意图.png" style="zoom: 80%;" />

**评估模型结果**

![image-20200611141531731](..\Resources\数据预测分类.png)

+ **查准率**：$Precision=\cfrac{TP}{TP+FP}$ 表示预测为正的样例中有多少是真正的正样例

+ **查全率**：$Recall=\cfrac{TP}{TP+FN}$ 表示样本中的正例有多少被预测正确

+ **精确度**：$Accuracy=\cfrac{TP+TN}{TT+FN+FP+TN}$ 表示分类正确的样本数占样本总数的比例，非平衡数据集会发生准确度悖论从而导致**失真**

+ **平衡查准率与查全率**：$F_\beta=(1+\beta^2)\cfrac{P·R}{\beta^2·P+R}$ 当$\beta$靠近0时，$F_\beta$偏向查准率P，当$\beta$靠近正无穷时，$F_\beta$偏向查全率R，$\beta=1$时为$F1-score$

+ **ROC空间**（Receiver Operating Characteristic）：真阳性率 $TPR=\cfrac{TP}{TP+FN}$，伪阳性率 $FPR=\cfrac{FP}{FP+TN}$，以FPR伪横轴，以TPR为纵轴画图，得到的是ROC空间，其中：

  - 越靠近左上角预测准确率越高
  - 对角线为无意义的随机猜测
  - 对角线下方是把结果搞反了，做相反预测即可
  - 设置不同阈值参数可得到一个点，连起来就是ROC曲线。曲线下方阴影面积为AUC，代表模型预测正确的概率，不依赖于阈值，取决于模型本身
  - 当测试集中的正负样本的分布变换（**类别不平衡**）的时候，ROC曲线能够保持不变

  ![image-20200611150737627](..\Resources\ROC.png)
  
  ```python
  from sklearn.metrics import roc_curve, auc
  
  logitModel = LogisticRegression()
  logitModel.fit(trainData[features], trainData[label])
  logitProb = logitModel.predict_proba(testData[features])[:, 1]
  # 得到False positive rate和True positive rate
  fpr, tpr, _ = roc_curve(testData[label], logitProb)
  # 得到AUC
  _auc = auc(fpr, tpr)
  # 为在Matplotlib中显示中文，设置特殊字体
  plt.rcParams["font.sans-serif"]=["SimHei"]
  fig = plt.figure(figsize=(6, 6), dpi=80)
  ax = fig.add_subplot(1, 1, 1)
  ax.plot(fpr, tpr, "k", label="%s; %s = %0.2f" % ("ROC曲线", "曲线下面积（AUC）", auc))
  ax.fill_between(fpr, tpr, color="grey", alpha=0.6)
  legend = plt.legend(shadow=True)
  plt.show()
  ```
  



**泛化能力**

该方法学习到的模型对未知数据的预测能力

泛化误差（generalization error）等于模型对未知数据预测的误差期望，使用**泛化误差上界**来判断学习算法的优劣



**超参数调优**

```python
# 网格搜索
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model,{
    'max_depth':range(2,6),
    'n_estimators':[20,50,70],
    'learning_rate':list(floatrange(0.1,1.1,0.1)),
})
```





  # 搜索方法

穷举搜索

贪心搜索：每一步选取一个概率最大的，进行下一步扩展，相当于k=1的束搜索

束搜索(beam search)：选取一个k，每一步选取所有搜索空间中前k个概率最大的，进行下一步扩展；搜索空间比贪心大，能更好的找到全局最优解



  

  

  

  













# 计算示例

## 分类

| ID   | A1 年龄 | A2 有工作 | A3 有自己的房子 | A4 信贷情况 | y 类别 |
| ---- | ------- | --------- | --------------- | ----------- | ------ |
| 1    | 青年    | 否        | 否              | 一般        | 否     |
| 2    | 青年    | 否        | 否              | 好          | 否     |
| 3    | 青年    | 是        | 否              | 好          | 是     |
| 4    | 青年    | 是        | 是              | 一般        | 是     |
| 5    | 青年    | 否        | 否              | 一般        | 否     |
| 6    | 青年    | 否        | 否              | 一般        | 否     |
| 7    | 中年    | 否        | 否              | 好          | 否     |
| 8    | 中年    | 是        | 是              | 好          | 是     |
| 9    | 中年    | 否        | 是              | 非常好      | 是     |
| 10   | 中年    | 否        | 是              | 非常好      | 是     |
| 11   | 老年    | 否        | 是              | 非常好      | 是     |
| 12   | 老年    | 否        | 是              | 好          | 是     |
| 13   | 老年    | 是        | 否              | 好          | 是     |
| 14   | 老年    | 是        | 否              | 非常好      | 是     |
| 15   | 老年    | 否        | 否              | 一般        | 否     |

年龄：6青年(1)，4中年(2)，5老年(3)

工作：10否(0)，5是(1)

有自己的房子：9否(0)，6是(1)

信贷情况：5一般(1)，6好(2)，4非常好(3)

类别：6否(-1)，9是(1)

- **信息增益（ID3）**

  数据集熵：$H(D)=-\cfrac{6}{15}log_2\cfrac{6}{15} -\cfrac{9}{15}log_2\cfrac{9}{15}=0.971$

  各特征信息增益：

  A1：$g(D,A1)=H(D)-H(D|A1)=H(D)-\sum_{i=1}^3p_iH(D|A1=a_i)\\  =H(D)-[\cfrac{6}{15}H(D_{A11})+\cfrac{4}{15}H(D_{A12})+\cfrac{5}{15}H(D_{A13})]\\  =0.971-[ \cfrac{6}{15}(-\cfrac{4}{6}log_2\cfrac{4}{6}-\cfrac{2}{6}log_2\cfrac{2}{6})  +\cfrac{4}{15}(-\cfrac{1}{4}log_2\cfrac{1}{4}-\cfrac{3}{4}log_2\cfrac{3}{4})+\cfrac{5}{15}(-\cfrac{1}{5}log_2\cfrac{1}{5}-\cfrac{4}{5}log_2\cfrac{4}{5})]\\ =0.971-0.824=0.147$

  A2：$g(D,A2)=H(D)-[\cfrac{10}{15}H(D_{A21})+\cfrac{5}{15}H(D_{A22})]\\  =0.971-[ \cfrac{10}{15}(-\cfrac{6}{10}log_2\cfrac{6}{10}-\cfrac{4}{10}log_2\cfrac{4}{10})  +\cfrac{5}{10}(-\cfrac{0}{5}log_2\cfrac{0}{5}-\cfrac{5}{5}log_2\cfrac{5}{5})]\\  =0.324$

  A3：$g(D,A3)=H(D)-[\cfrac{9}{15}H(D_{A31})+\cfrac{6}{15}H(D_{A32})]\\  =0.971-[ \cfrac{9}{15}(-\cfrac{6}{9}log_2\cfrac{6}{9}-\cfrac{3}{9}log_2\cfrac{3}{9})  +\cfrac{6}{15}(-\cfrac{0}{6}log_2\cfrac{0}{6}-\cfrac{6}{6}log_2\cfrac{6}{6})]\\  =0.420$

  A4：$g(D,A4)=H(D)-[\cfrac{5}{15}H(D_{A41})+\cfrac{6}{15}H(D_{A42})+\cfrac{4}{15}H(D_{A43})]\\  =0.971-[ \cfrac{5}{15}(-\cfrac{4}{5}log_2\cfrac{4}{5}-\cfrac{1}{5}log_2\cfrac{1}{5})  +\cfrac{6}{15}(-\cfrac{2}{6}log_2\cfrac{2}{6}-\cfrac{4}{6}log_2\cfrac{4}{6})]  +\cfrac{4}{15}(-\cfrac{0}{4}log_2\cfrac{0}{4}-\cfrac{4}{4}log_2\cfrac{4}{4})\\  =0.363$

  其中A3的信息增益最大，因此选择A3作为根节点划分标准进行划分；

  子树再在划分后的数据子集A31、A32上计算其他特征信息增益，再划分；

  直到子节点全属同类或信息增益小于阈值，则置为叶子节点

- **Gini系数（CART分类）**

  A1：$Gini(D,A1=1)=\cfrac{6}{15}(2\times\cfrac{4}{6}\times\cfrac{2}{6})+\cfrac{9}{15}(2\times\cfrac{2}{9}\times\cfrac{7}{9})=0.39\\  Gini(D,A1=2)=\cfrac{4}{15}(2\times\cfrac{1}{4}\times\cfrac{3}{4})+\cfrac{11}{15}(2\times\cfrac{5}{11}\times\cfrac{6}{11})=0.46\\  Gini(D,A1=3)=\cfrac{5}{15}(2\times\cfrac{1}{5}\times\cfrac{4}{5})+\cfrac{10}{15}(2\times\cfrac{5}{10}\times\cfrac{5}{10})=0.11$

  ​		老年为A1的最好划分

  A2：$Gini(D,A2=0)=0.32$

  A3：$Gini(D,A3=0)=0.27$

  A4：$Gini(D,A4=1)=0.36\\ Gini(D,A4=2)=0.47\\ Gini(D,A4=3)=0.32$

  ​		非常好为A4的最好划分

  综上，$Gini(D,A1=3)$为根节点最好划分

- **AdaBoost算法**

  使用决策树桩为基分类器

  1. 初始权值分布：$D_1(...,w_{1i},...),\quad w_{1i}=\cfrac1{15},\quad i=1,2,...,15$

  2. 对 $m=1$，取 $Gini$ 系数最小的特征 $Gini(D,A1=3)$ 作为决策树桩划分

     计算分类误差率为：$e_1=\sum_{i=1}^Nw_{1i}I(G_1(x_i)\not=y_i)=6\times\cfrac1{15}=0.4$，基分类器：$G_1(x)=\begin{cases}1,\quad\,\,\,\, A1=3\\ -1,\quad A1\not =3 \end{cases}$

     $G_1$系数：$\alpha_1=\cfrac12ln\cfrac{1-e_1}{e_1}=0.2027$

     更新权值分布，分类正确的：$w_{2i}=w_{1i}\times e^{-\alpha_1}$，分类错误的：$w_{2i}=w_{1i}\times e^{\alpha_1}$，归一化：$w_{2i}=\cfrac{w_{2i}}{\sum_{i=1}^N w_{2i}}$

     | ID       | 1      | 2      | 3      | 4      | 5      | 6      | 7      | 8      | 9      | 10     | 11     | 12     | 13     | 14     | 15     |
     | -------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
     | $w_{2i}$ | 0.0556 | 0.0556 | 0.0833 | 0.0833 | 0.0556 | 0.0556 | 0.0556 | 0.0833 | 0.0833 | 0.0833 | 0.0556 | 0.0556 | 0.0556 | 0.0556 | 0.0833 |

     $sign[f_1(x)]=sign[0.2027G_1(x)]$有6个误分类点

  3. 对 $m=2$，计算 $Gini$ 系数时要考虑权重：

     A1：$Gini(D,A1=1)=0.3889\times(2\times\cfrac{0.2224}{0.3889}\times\cfrac{0.1666}{0.3889})+0.6111\times(2\times\cfrac{0.1389}{0.6111}\times\cfrac{0.4722}{0.6111})=0.4052,\quad 其中\sum_{i=1}^6w_{2i}=0.3889,\sum_{i=7}^15w_{2i}=0.6111$

     计算好所有 $Gini$ 系数，取最小的作为划分特征，再计算系数，更新权重

     更新分类器：$sign[f_2(x)]=sign[\alpha_1G_1(x)+\alpha_2G_2(x)]$

  4. 重复直到误分类率小于阈值即可得到最终分类器


- **朴素贝叶斯**

  使用带有拉普拉斯平滑的贝叶斯公式（忽略分母$P(x)$）

  $p(y=-1)=\cfrac{7}{17},\quad p(y=-1)=\cfrac{10}{17}$

  | $p(A(i)=i|y=i)$ | $y = - 1$      | $y = 1$         |
  | --------------- | -------------- | --------------- |
  | A1=1            | $\cfrac{5}{9}$ | $\cfrac{3}{12}$ |
  | A1=2            | $\cfrac{2}{9}$ | $\cfrac{4}{12}$ |
  | A1=3            | $\cfrac{2}{9}$ | $\cfrac{5}{12}$ |
  | A2=0            | $\cfrac{7}{8}$ | $\cfrac{5}{11}$ |
  | A2=1            | $\cfrac{1}{8}$ | $\cfrac{6}{11}$ |
  | A3=0            | $\cfrac{7}{8}$ | $\cfrac{4}{11}$ |
  | A3=1            | $\cfrac{1}{8}$ | $\cfrac{7}{11}$ |
  | A4=1            | $\cfrac{4}{9}$ | $\cfrac{3}{12}$ |
  | A4=2            | $\cfrac{4}{9}$ | $\cfrac{4}{12}$ |
  | A4=3            | $\cfrac{1}{9}$ | $\cfrac{5}{12}$ |

  给定样本：$x(A1=1,A2=0,A3=0,A4=3)$

  $p(y=-1)p(A1=1|y=-1)p(A2=0|y=-1)p(A3=0|y=-1)p(A4=3|y=-1)=0.01946$

  $p(y=1)p(A1=1|y=1)p(A2=0|y=1)p(A3=0|y=1)p(A4=3|y=1)=0.01013$

  因此取$y=-1$

## 回归

| $x_i$ | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $y_i$ | 5.56 | 5.70 | 5.91 | 6.40 | 6.80 | 7.05 | 8.90 | 8.70 | 9.00 | 9.05 |

- **CART回归**

  1. 选择最优的特征 j 和分切点 s：

     | 分切点(s) | 1.5   | 2.5   | 3.5  | 4.5  | 5.5  | 6.5      | 7.5  | 8.5   | 9.5   |
     | --------- | ----- | ----- | ---- | ---- | ---- | -------- | ---- | ----- | ----- |
     | $c_1$     | 5.56  | 5.63  | 5.72 | 5.89 | 6.07 | **6.24** | 6.62 | 6.88  | 7.11  |
     | $c_2$     | 7.5   | 7.73  | 7.99 | 8.25 | 8.54 | **8.91** | 8.92 | 9.03  | 9.05  |
     | $loss$    | 15.72 | 12.07 | 8.36 | 5.78 | 3.91 | **1.93** | 8.01 | 11.73 | 15.74 |

     例：$s=6.5$ 时，$c_1=\cfrac16\sum_{i=1}^6y_i=6.24,\quad c_2=\cfrac14\sum_{i=7}^{10}y_i=6.24,\quad loss=\sum_{i=1}^6(y_i-c_1)^2+\sum_{i=7}^{10}(y_i-c_2)^2=1.93$

     当分切点取 $s=6.5$，损失最小 $loss(s=6.5)=1.93$，此时划分出两个分支，分别是： $R_1=\{1,2,3,4,5,6\},\quad c_1=6.42,\quad R_2=\{7,8,9,10\},\quad c_2=8.91$

  2. 对子集再划分，直到满足条件为止

+ **提升树， GBDT**

  1. 初始化 $f_0(x)=0$，残差 $r_{1i}=y_i-f_0(x_i)$，选择最优的特征 j 和分切点 s：

     | 分切点(s) | 1.5   | 2.5   | 3.5  | 4.5  | 5.5  | 6.5      | 7.5  | 8.5   | 9.5   |
     | --------- | ----- | ----- | ---- | ---- | ---- | -------- | ---- | ----- | ----- |
     | $c_1$     | 5.56  | 5.63  | 5.72 | 5.89 | 6.07 | **6.24** | 6.62 | 6.88  | 7.11  |
     | $c_2$     | 7.5   | 7.73  | 7.99 | 8.25 | 8.54 | **8.91** | 8.92 | 9.03  | 9.05  |
     | $loss$    | 15.72 | 12.07 | 8.36 | 5.78 | 3.91 | **1.93** | 8.01 | 11.73 | 15.74 |

     $T_1=\begin{cases}6.24,\quad x<6.5\\ 8.91,\quad x\geqslant 6.5 \end{cases},\quad f_1(x)=T_1(x)$

  2. 求 $T_2$，拟合下表残差：

     | $x_i$    | 1     | 2     | 3     | 4    | 5    | 6    | 7     | 8     | 9    | 10   |
     | -------- | ----- | ----- | ----- | ---- | ---- | ---- | ----- | ----- | ---- | ---- |
     | $r_{2i}$ | -0.68 | -0.54 | -0.33 | 0.16 | 0.56 | 0.81 | -0.01 | -0.21 | 0.09 | 0.14 |

     $T_2=\begin{cases}-0.52,\quad x<3.5\\ 0.22,\quad\,\,\,\,\, x\geqslant 3.5 \end{cases},\quad f_2(x)=f_1(x)+T_2(x)=\begin{cases}5.72,\quad x<3.5\\ 6.46,\quad 3.5\leqslant x< 6.5\\ 9.13,\quad x\geqslant 6.5 \end{cases}$

  3. 继续求残差，拟合得到 $f_3,...,f_n$，直到损失误差达到要求


























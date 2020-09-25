@Sofice

数据科学

常用的可以查找数据集的网站以及一些在计算机视觉常用的图像数据集：

1. **Kaggle 数据集**:每个数据集都是一个小型社区，用户可以在其中讨论数据、查找公共代码或在内核中创建自己的项目。包含各式各样的真实数据集。
2. **Amazon 数据集**：该数据源包含多个不同领域的数据集，如：公共交通、生态资源、卫星图像等。网页中也有一个搜索框来帮助用户寻找想要的数据集，还有所有数据集的描述和使用示例，这些数据集信息丰富且易于使用！
3. **UCI机器学习资源库**：来自加州大学信息与计算机科学学院的大型资源库，包含100多个数据集。用户可以找到单变量和多变量时间序列数据集，分类、回归或推荐系统的数据集。
4. **谷歌数据集搜索引擎**：这是一个可以按名称搜索数据集的工具箱。
5. **微软数据集**：2018年7月，微软与外部研究社区共同宣布推出“Microsoft Research Open Data”。它在云中包含一个数据存储库，用于促进全球研究社区之间的协作。它提供了一系列用于已发表研究的、经过处理的数据集。
6. **Awesome Public Datasets Collection**：Github 上的一个按“主题”组织的数据集，比如生物学、经济学、教育学等。大多数数据集都是免费的，但是在使用任何数据集之前，用户需要检查一下许可要求。
7. **计算机视觉数据集**：Visual Data包含一些可以用来构建计算机视觉(CV)模型的大型数据集。用户可以通过特定的CV主题查找特定的数据集，如语义分割、图像标题、图像生成，甚至可以通过解决方案(自动驾驶汽车数据集)查找特定的数据集。

常用的部分图像数据集：

1. **Mnist**: 手写数字数据集，包含 60000 张训练集和 10000 张测试集。（但该数据集通常只是作为简单 demo 使用，如果要验证算法模型的性能，最好在更大数据集上进行测试，实验结果才有足够的可信度）
2. **Cifar**：分为 Cifar10 和 Cifar100。前者包含 60000 张图片，总共10个类别，每类 6000 张图片。后者是 100 个类别，每个类别 600 张图片。类别包括猫狗鸟等动物、飞机汽车船等交通工具。
3. **ImageNet**：应该是目前最大的开源图像数据集，包含 1500 万张图片，2.2 万个类别。
4. **LFW**：人脸数据集，包含13000+张图片和1680个不同的人。
5. **CelebA**：人脸数据集，包含大约 20w 张图片，总共 10177个不同的人，以及每张图片都有 5 个位置标注点，40 个属性信息

# Numpy

用于Python数值计算基础包

引用：`import numpy as np`

## ndarray多维数组对象

### 生成

| 函数名                | 描述                              |
| --------------------- | --------------------------------- |
| array                 | 将列表，元组，数组等转化为ndarray |
| arange                | 内建函数                          |
| ones                  | 全1,给定形状和数据类型            |
| ones_like             | 全1，给定数组生成一个形状一样的   |
| zeros，zeros_like     | 全0                               |
| empty，empty_like     | 空数组                            |
| full，full_like       | 指定数值                          |
| eye, identity         | 主对角线矩阵                      |
| reshape               | 改变数组维度                      |
| linspace(0, 1, 5)     | 均匀从0-1的5个数                  |
| random.random((3, 3)) | 随机0-1数3*3个                    |

### 属性

shape：数组每一维度数量

dtype：数据类型（每一个元素类型都相同）

ndim：维度

### 算术

带标量计算的算数操作，会把计算参数传递给数组的每一个元素。

不同尺寸的数组间操作会用到**广播特性**

### 索引

- 切片：

  得到一份视图而并非拷贝（拷贝要用`arr[5:8].copy()`）

  `arr[:, i:i+1]`得到第i列

  `arr[i]`得到第i行

  对切片赋值会对切出的所有元素赋值

- 布尔索引：

  ```python
  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
  data = np.random.randn(7, 4)
  # 用布尔索引，产生布尔数组array([ True, False, False, True, False, False, False])
  data[names == 'Bob']
  # 非
  data[~(names == 'Bob')]
  # 或
  data[(names == 'Bob') | (names == 'Will')]
  ```

- 神器索引：**将数据复制到新数组**

  ```python
  arr = np.arange(32).reshape((8, 4))
  # 按顺序选取4，3，0，6行
  arr[[4, 3, 0, 6]]
  # 每一行再选取第二维度
  arr[[1, 5, 7, 2], [0, 3, 1, 2]]
  # 改变选取的新数组的每一行中元素的顺序
  arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
  ```



## 通用函数

快速的逐元素数组函数

```python
# 平方
np.sqrt(arr)
# 返回小数，整数部分
remainder, whole_part = np.modf(arr)
```

### 一元函数

| 函数名                   | 描述                                                   |
| ------------------------ | ------------------------------------------------------ |
| abs, fabs                | 整数、浮点数绝对值                                     |
| sqrt, square             | 平方根，平方                                           |
| exp                      | exp x                                                  |
| log, log10, log2         | 以e,10,2为底的对数                                     |
| sign                     | 符号函数                                               |
| ceil, floor              | 向上、向下取整                                         |
| rint                     | 保留到整数，并保持dtype                                |
| modf                     | 返回小数、整数部分                                     |
| isnan, isinf             | 是否是NaN，是否是无限                                  |
| sin, cos, arcsin, arccos | 三角函数，反三角函数                                   |
| logical_not              | 按位取反                                               |
| tile                     | 重复增加数组                                           |
| argsort                  | 排序，返回从小到大的下标                               |
| linespace                | 产生线性序列                                           |
| squeeze                  | 从数组的形状中删除单维度条目，即把shape中为1的维度去掉 |

### 二元函数

| 函数名                                                    | 描述                                 |
| --------------------------------------------------------- | ------------------------------------ |
| add, subtract, multiply, divide(floor_divide)             | 加，减，乘，除（省略余数）           |
| power                                                     | 乘方                                 |
| maximum，fmax, minimum, fmin                              | 对应元素最大最小值，fmax,fmin忽略NaN |
| mod                                                       | 求模                                 |
| copysign                                                  | 复制符号值                           |
| greater, greater_equal, less,less_equal, equal, not_equal | \>,>=,<,<=,=,!=                      |
| logical_and, logical_or, logical_xor                      | &,\|,^                               |
| concatenate([x, y])                                       | 合并                                 |
| x1, x2, x3 = np.split(x, [3, 5])                          | 分裂                                 |



## 面向数组操作

### 条件逻辑

```python
# x if condition else y
result = np.where(cond, x, y)
# True就取第一个数组，False就取第二个数组
np.where([True, False], [1, 2], [3, 4])
# 将正值设为2，负值设为-2
np.where(arr > 0, 2, -2)
```

### 数学统计

```python
# 两种方法计算
arr.mean()
np.mean(arr)
# 纵向计算
arr.mean(axis=0)
# 横向计算
arr.mean(axis=1)
```

| 方法            | 描述               |
| --------------- | ------------------ |
| sum             | 和                 |
| mean            | 平均值             |
| std, var        | 标准差，方差       |
| min, max        | 最大值，最小值     |
| argmin, argmax  | 最大值最小值位置   |
| cumsum, cumprod | 累计和，累计积     |
| sort,argsort    | 排序，原始顺序下表 |

###  布尔值数组

```python
bools = np.array([False, False, True, False])
# 是否有True
bools.any()
# 是否全是True
bools.all()
# 按位
np.sum((inches > 0.5) & (inches < 1))
# 掩码
 x[x < 5]
```

### 集合操作

| 方法              | 描述                               |
| ----------------- | ---------------------------------- |
| unique(x)         | 唯一值，并排序                     |
| intersect1d(x, y) | 交集，并排序                       |
| union1d(x, y)     | 并集，并排序                       |
| in1d(x, y)        | x中元素是否包含在y，返回布尔值数组 |
| setdiff1d(x, y)   | 差集，在x中但不在y中               |
| setxor1d(x, y)    | 异或集，在并集但不属于交集的元素   |

## 存储

```python
# 存储
np.save('arrays', arr)
# 载入
np.load('arrays.npy')
```

## 线性代数

转置：`arr.T` 

numpy.linalg中的方法

| 方法  | 描述               |
| ----- | ------------------ |
| diag  | 返回对角元素       |
| dot   | 矩阵乘法           |
| trace | 对角元素和         |
| det   | 行列式             |
| eig   | 特征值，特征向量   |
| inv   | 逆矩阵             |
| solve | 求解Ax = b         |
| lstsq | Ax = b的最小二乘解 |



## 随机数

numpy.random

| 方法              | 描述                   |
| ----------------- | ---------------------- |
| seed，RandomState | 随机种子，只使用一次   |
| permutation       | 返回一个序列的随机排列 |
| shuffle           | 随机排列一个序列       |
| rand              | 0-1均匀分布（维度）    |
| randint           | 给定范围的均匀分布     |
| randn             | 均值0方差1的正态分布   |
| binomial          | 二项分布               |
| normal            | 正态（高斯）分布       |
| beta              | beta分布               |
| chisquare         | 卡方分布               |
| gamma             | 伽马分布               |
| uniform           | [0,1)均匀分布          |

## 结构化数组

```python
# 创建
data = np.zeros(4, dtype={'names':('name', 'age', 'weight'), 
                          'formats':('U10', 'i4', 'f8')})
# 导入
data['name'] = ['Alice', 'Bob', 'Cathy', 'Doug']
# 获取一个实例
data[0]
```



# Pandas

`import pandas as pd`

`from pandas import Series, DataFrame`

## Series

+ 特殊的字典，具有数据对齐特性，可切片

+ 属性：

  values：值

  index：索引

  name, index.name：名字


```python
# 生成序列（字典）
obj = pd.Series([4, 7, -5, 3])
obj2 = pd.Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj3 = pd.Series(sdata, index=states) #sdata是字典，index缺省时按键的字典序排序
# 索引访问
obj2[['c', 'a', 'd']]
# 运算
obj2 * 2
np.exp(obj2)
# 检测索引是否存在
'b' in obj2
# 值是否有效
obj4.isnull()
```



## DataFrame

指定行列的二维索引
+ 属性：

  index,columns：行，列索引标签

  values：返回二维ndarray

  index.name, columns.name：名字

```python
# 利用包含等长度Numpy数组列表或字典，可指定列或索引顺序
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}
data = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                      index=['one', 'two', 'three', 'four','five', 'six'])
# 访问，返回Series,视图
frame['state'];frame.values[1] # 按列
frame.loc['three'] # 按行
# 赋值
frame2['debt'] = 16.5
frame2['debt'] = np.arange(6.)
val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
# 删除列
del frame2['eastern']
# 转置
frame3.T
```



## Index

索引对象

+ 无法修改

+ 可重复 

| 方法         | 描述                                         |
| ------------ | -------------------------------------------- |
| append       | 将额外的索引对象粘贴到原索引，生成一个新索引 |
| difference   | 差集                                         |
| intersection | 并集                                         |
| union,&      | 交集                                         |
| isin         | 表示每一个值是否在传值容器中的布尔数组       |
| delete       | 按索引位删除                                 |
| drop         | 按索引值删除                                 |
| insert       | 按位插入                                     |
| is_monotonic | 是否递增                                     |
| is_unique    | 是否唯一                                     |
| unique       | 返回唯一值序列                               |

  

## 基本功能

### 索引器

避免整数索引显式隐式的混乱

+ loc：显式
+ iloc：隐式

### 重建索引

```python
obj = obj.reindex(['a', 'b', 'c', 'd', 'e'])
frame.loc[['a', 'b', 'c', 'd'], states]
```

reindex参数

| 参数       | 描述                                                         |
| ---------- | ------------------------------------------------------------ |
| index      | 新建作为索引的序列                                           |
| method     | 插值方式；ffill向前填充，bfill向后填充                       |
| fill_value | 缺失数据时的替代值                                           |
| limit      | 填充时，所需填充的最大尺寸间隙（以元素数量）                 |
| tolerance  | 填充时，所需填充的不精确匹配下的最大尺寸间隙（以绝对数字距离） |
| level      | 匹配MultiIndex级别的简单索引                                 |
| copy       | True，索引相同时总是复制数据                                 |

### 轴向删除

```python
# series
obj.drop(['d', 'c'])
# dataframe,列删除
data.drop('two', axis=1)
#inplace=True删除原对象中值，真删除
obj.drop('c', inplace=True)
```

### 切片

```python
# series
obj = pd.Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
# a 0.0
# b 1.0
# c 2.0
# d 3.0
obj['b'] # 即obj[1]
obj[2:4] # 即obj['b':'d']包括尾部
obj[obj < 2]
# dataframe
data = pd.DataFrame(np.arange(16).reshape((4, 4)),
                    index=['Ohio', 'Colorado', 'Utah', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
# 先选择行，再选择列
# loc轴标签
data.loc['Colorado', ['two', 'three']]
data.loc[:'Utah', 'two']
# iloc整数标签
data.iloc[2]
data.iloc[[1, 2], [3, 0, 1]]
```

### 缺失值

isnull() ：创建一个布尔类型的掩码标签缺失值。 

notnull() ：与 isnull() 操作相反。 

dropna() ：返回一个剔除缺失值的数据。 可以axis选择行列

fillna() ：返回一个填充了缺失值的数据副本。

### 合并

```python
pd.concat([ser1, ser2])
pd.concat([df3, df4], axis='col')
```

| 参数                  | 说明             |
| --------------------- | ---------------- |
| verify_integrity=True | 捕捉错误         |
| ignore_index=True     | 创建新的整数索引 |
| keys=['x', 'y']       | 多级索引         |



# Mataplotlib

<img src="./DataScience/anatomy.png" style="zoom: 80%;" />



# 数据提取

```python
data = pandas.read_csv('data.csv')
# 数据若干行
print(data.head(3))
# 数据基本统计
print(data.describe(include="all"))
# 展示数据有几类
data["Age"].unique()
```

**处理空缺值**

```python
# 使用中间值填充
data["Age"] = data["Age"].fillna(data["Age"].median())
```

**数据编码**

```python
# 二值
data['Sex'] = data['Sex'].apply(lambda r : 1 if r == "male" else 0)
# 多值
data['Embarked'] = data['Embarked'].apply(lambda r : 1 if r == "C" else r)
data['Embarked'] = data['Embarked'].apply(lambda r : 2 if r == "S" else r)
data['Embarked'] = data['Embarked'].apply(lambda r : 3 if r == "Q" else r)
```







# Tensorflow

**计算图**：是包含节点和边的网络 

**运算操作对象**（Operation Object, OP）:张量（tensor）对象（常量、变量和占位符）

为了构建计算图，需要定义所有要执行的常量、变量和运算操作

TensorFlow 支持以下三种类型的张量：

1. **常量**：常量是其值不能改变的张量

   ```python
   t_1 = tf.constant([4,3,2])
   t_2 = tf.constant(1，shape=[11])
   ```

2. **变量**：当一个量在会话中的值需要更新时，使用变量来表示

   ```python
   t_a = tf.Variable(t_1)# 用常量初始化
   t_b = tf.Variable(t_a, name='b')# 用变量初始化
   initial_op = tf.global_variables_initializer()# 必须显式初始化所有的声明变量
   saver = tf.train.Saver()# 保存变量
   ```

3. **占位符**：用于将值输入 TensorFlow 图中。它们可以和  feed_dict  一起使用来输入数据。在训练神经网络时，它们通常用于提供新的训练样本。在会话中运行计算图时，可以为占位符赋值。这样在构建一个计算图时不需要真正地输入数据。需要注意的是，占位符不包含任何数据，因此不需要初始化它们

   ```python
   tf.placeholder(dtype,shape=None,name=None)
   ```
   



## 常用公共函数

```python
# 对应元素加减乘除，平方，次方，开方，自增减
tf.add(a, b)
tf.subtract(a, b)
tf.multiply(a, b)
tf.divide(b, a)
tf.square(a)
tf.pow(a, 3)
tf.sqrt(a)
x.assign_add(delta_x)
x.assign_sub(delta_x)
# 运算
tf.reduce_min(x)
tf.reduce_max(x)
tf.reduce_mean(x)
tf.reduce_sum(x)
# 判断
c = tf.where(tf.greater(a, b), a, b)  # 若a>b，返回a对应位置的元素，否则返回b对应位置的元素
tf.equal(pred, y_test)
tf.greater(a, b)2
tf.less(a, b)
# 枚举循环
seq = ['one', 'two', 'three']
for i, element in enumerate(seq):
    print(i, element)
# 最值索引
tf.argmin(test, axis=0)
tf.argmax(test, axis=0)
# 矩阵乘
tf.matmul(a, b)
# 强制转换
tf.cast(x1, tf.int32)
# 特定数组
tf.zeros([2,3],tf.int32)
tf.ones_like(t_1)
tf.fill([2,3],10)#全为指定值
tf.linspace(start,stop,num)# 包括stop，num为个数
tf.range(start=0,limit,delta=1)# 不包括limit，delta为增量
# 随机数
tf.set_random_seed(54)
tf.random.normal([M,N], mean=0.0, stddev=1.0, seed)# 正态分布
tf.random.truncated_normal([M,N], mean=0.0, stddev=1.0, seed)# 截断式正态分布
tf.random.uniform([M,N], minval=0.0, maxval=1.0)# 均匀分布
tf.random_crop(t_1,[2,5],seed)# 随机切片
tf.random_shuffle(t_1)# 沿着第一维随机排列张量
# numpy与tensor转化
b=tf.convert_to_tensor(a,dtype=tf.int32)
b.numpy()
# 拼接
tf.concat([a,b],axis=0)
tf.stack([a,b], axis=0)# 增加新维度
```



## 数值计算

```python
# 特征标签绑定，batch为配合enumerate，一次循环一个batch
dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(32)
# 梯度下降
x = tf.Variable(tf.constant(3.0))
with tf.GradientTape() as tape:
    loss = tf.pow(x, 2)
grad = tape.gradient(y, x)
x.assign_sub(grad)
# 多个参数梯度更新
variables = [w1, b1]
grads = tape.gradient(loss, variables)
w1.assign_sub(lr * grads[0])
b1.assign_sub(lr * grads[1])

# 将标签转化为独热编码，再由argmax得到原编号
labels = tf.constant([1, 0, 2])  
output = tf.one_hot(labels, depth=classes)#classes分类个数
labels = tf.argmax(output, axis=1)
# 去掉y中纬度1
tf.squeeze(y) 
# 使y_dim符合概率分布
tf.nn.softmax(y_dim) 
# 激活函数
tf.nn.sigmoid(x)
tf.math.tanh(x)
tf.nn.relu(x)
tf.nn.leaky_relu(x)
# 均方误差
loss_mse = tf.reduce_mean(tf.square(y_train - y))
# 交叉熵
tf.losses.categorical_crossentropy(y_, y)# y_标准，y预测
tf.nn.softmax_cross_entropy_with_logits(y_, y)# 集成softmax和交叉熵
# 正则化项
tf.nn.l2_loss(w1)
# 生成网格待预测特征
# xx在-3到3之间以步长为0.1，yy在-3到3之间以步长0.1,生成间隔数值点
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
# 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)
```



## keras

**Embedding**

一种单词编码方法，以低维向量实现了编码，这种编码通过神经网络训练优化，能表达出单词的相关性

```python
tf.keras.layers.Embedding(词汇表大小，编码维度)
# 编码维度就是用几个数字表达一个单词
# 对1-100进行编码， [4] 编码为 [0.25, 0.1, 0.11]
tf.keras.layers.Embedding(100, 3)

# 使x_train符合Embedding输入要求：[送入样本数， 循环核时间展开步数] ，
# 此处整个数据集送入所以送入，送入样本数为len(x_train)；输入4个字母出结果，循环核时间展开步数为4。
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    Embedding(26, 2),
    SimpleRNN(10),
    Dense(26, activation='softmax')
])
```



### 多层前馈神经网络

六步法

1. import
2. train,test
3. model = tf.keras.models.Sequential
4. model.compile
5. model.fit
6. model.summary

```python
# 1
import tensorflow as tf
from sklearn import datasets
import numpy as np

# 2
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

# 3
model = tf.keras.models.Sequential([网络结构])
# 网络结构
# 拉直层
tf.keras.layers.Flatten()
# 全连接层
# activation=relu,softmax,sigmoid,tanh
# kernel_regularizer=tf.keras.regularizers.l1(),tf.keras.regularizers.l2()
tf.keras.layers.Dense(神经元个数, activation='激活函数', kernel_regularizer=正则化)
# 卷积层
tf.keras.layers.Conv2D ()
# LSTM层
tf.keras.layers.LSTM()  

# 4
# optimizer=’sgd' or tf.keras.optimizers.SGD(lr=0.1)
# loss='mse' or tf.keras.MeanSquaredError()
# metrics='accuracy'都是数值 or 'categorical_accuracy'都是独热码 or ''sparse_categorical_accuracy'y_是数值y是独热码
model.compile(optimizer=优化器, loss=损失函数, metrics=[数值和独热码])

# 5
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=测试集比例, validation_freq=多少次epoch测试一次)
# 6
model.summary()
```

**自定义model**，代替Sequential

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
class IrisModel(Model):
    # 定义网络结构块
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
    # 调用网络结构块，实现前向传播
    def call(self, x):
        y = self.d1(x)
        return y

model = IrisModel()
```

**数据增强**

```python
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个维度,从(60000, 28, 28)reshape为(60000, 28, 28, 1)

image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(x_train)
model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test),
          validation_freq=1)
```

**断点续训**

```python
# 读取
checkpoint_save_path = ".\checkpoint\fashion.ckpt"	#先定义出存放模型的路径和文件名，“.ckpt”文件在生成时会同步生成索引表
if os.path.exists(checkpoint_save_path + '.index'):		#判断是否有索引表，就可以知道是否报存过模型
    model.load_weights(checkpoint_save_path)
# 保存
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,		# 文件存储路径
                                                 save_weights_only=True,			# 是否只保留模型参数
                                                 save_best_only=True)				# 是否只保留模型最优参数

history = model.fit(x_train, y_train, batch_size=32, epochs=5, 						# 加入callbacks选项，记录到history中
					validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# acc/loss
acc = history.history['sparse_categorical_accuracy'] # 训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy'] # 测试集准确率
loss = history.history['loss'] # 训练集loss
val_loss = history.history['val_loss'] # 测试集loss

```

**参数提取**

```python
# 设置print输出格式
np.set_printoptions(threshold=np.inf) # np.inf表示无限大
# 提取参数
print(model.trainable_variables)
file = open('.\weights.txt', 'w')
for v in model.trainable_variables:
	file.write(str(v.name) + '\n')
	file.write(str(v.shape) + '\n')
	file.write(str(v.numpy()) + '\n')
file.close()
```

**预测结果**

```python
# model = tf.keras.models.Sequential()
# model.load_weights(model_save_path)
# 预处理数据
# 预测
result = model.predict(x_predict)
pred = tf.argmax(result, axis=1)
```

**数据集**

```python
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
cifar10 = tf.keras.datasets.cifar10
```



### 卷积神经网络

```python
# 卷积主结构
model = tf.keras.models.Sequential([
	Conv2D(filters=6, kernel_size=(5, 5), padding='same'),	#卷积层
	BatchNormalization(),									#BN层	
	Activation('relu'),										#激活层
	MaxPool2D(pool_size=(2, 2), strides=2, padding='same'),	#池化层
	Dropout(0.2),											#dropout层
])
# 卷积层
tf.keras.layers.Conv2D (
	filters = 卷积核个数,
	kernel_size = 卷积核尺寸, 			#正方形写核长整数，或（核高h，核宽w）
	strides = 滑动步长,					#横纵向相同写步长整数，或(纵向步长h，横向步长w)，默认1
	padding = “same” or “valid”, 		#使用全零填充是“same”，不使用是“valid”（默认）
	activation = “ relu ” or “ sigmoid ” or “ tanh ” or “ softmax”等 , 		#如有BN此处不写
	input_shape = (高, 宽 , 通道数)		#输入特征图维度，可省略
)
# 批标准化
tf.keras.layers.BatchNormalization()
# 池化
tf.keras.layers.MaxPool2D(
	pool_size=池化核尺寸，	#正方形写核长整数，或（核高h，核宽w）
	strides=池化步长，		#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
	padding=‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
)
tf.keras.layers.AveragePooling2D(
	pool_size=池化核尺寸，	#正方形写核长整数，或（核高h，核宽w）
	strides=池化步长，		#步长整数， 或(纵向步长h，横向步长w)，默认为pool_size
	padding=‘valid’or‘same’ #使用全零填充是“same”，不使用是“valid”（默认）
)
# 舍弃
tf.keras.layers.Dropout(舍弃的概率)
```



### 循环神经网络

```python
tf.keras.layers.SimpleRNN(记忆体个数，activation=‘激活函数’ ，return_sequences=是否每个时刻输出ht到下一层)
# 参数
	activation=‘激活函数’ （不写，默认使用tanh）
	return_sequences=True 各时间步输出ht
	return_sequences=False 仅最后时间步输出ht（默认）
# 例：
SimpleRNN(3, return_sequences=True)

# RNN要求输入数据（x_train）的维度是三维的[送入样本数，循环核时间展开步数，每个时间步输入特征个数]
# 此处整个数据集送入，送入样本数为len(x_train)；输入1个字母出结果，循环核时间展开步数为1; 表示为独热码有5个输入特征，每个时间步输入特征个数为5
x_train = np.reshape(x_train, (len(x_train), 1, 5))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    SimpleRNN(3),
    Dense(5, activation='softmax')
])
```




































@Sofice

数据科学

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

| 方法            | 描述                                         |
| --------------- | -------------------------------------------- |
| append          | 将额外的索引对象粘贴到原索引，生成一个新索引 |
| difference      | 差集                                         |
| intersection,\| | 并集                                         |
| union,&         | 交集                                         |
| isin            | 表示每一个值是否在传值容器中的布尔数组       |
| delete          | 按索引位删除                                 |
| drop            | 按索引值删除                                 |
| insert          | 按位插入                                     |
| is_monotonic    | 是否递增                                     |
| is_unique       | 是否唯一                                     |
| unique          | 返回唯一值序列                               |

  

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
|                       |                  |
|                       |                  |



# Mataplotlib

<img src="./DataScience/anatomy.png" style="zoom: 80%;" />





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



优化器






































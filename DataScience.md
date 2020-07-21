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

| 函数名                   | 描述                     |
| ------------------------ | ------------------------ |
| abs, fabs                | 整数、浮点数绝对值       |
| sqrt, square             | 平方根，平方             |
| exp                      | exp x                    |
| log, log10, log2         | 以e,10,2为底的对数       |
| sign                     | 符号函数                 |
| ceil, floor              | 向上、向下取整           |
| rint                     | 保留到整数，并保持dtype  |
| modf                     | 返回小数、整数部分       |
| isnan, isinf             | 是否是NaN，是否是无限    |
| sin, cos, arcsin, arccos | 三角函数，反三角函数     |
| logical_not              | 按位取反                 |
| tile                     | 重复增加数组             |
| argsort                  | 排序，返回从小到大的下标 |
| linespace                | 产生线性序列             |

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












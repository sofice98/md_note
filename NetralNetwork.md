@sofice

# 神经网络

![神经网络原始模型](\NetralNetwork\神经网络原始模型.png)
![image-20200602195133779](\NetralNetwork\S阈值函数公式.png)
<img src="\NetralNetwork\S阈值函数.png" alt="image-20200602194935171" style="zoom: 80%;" />

**若干输入，乘上权重，通过S阈值函数，得到输出**

权重初始值可以为随机值，正态分布，均值为0，方差为
$$
\cfrac{1}{\sqrt{inodes}}
$$
`self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))`

## 传播

### 正向传播

![image-20200602195250519](\NetralNetwork\正向传播.png)
![image-20200602195439442](\NetralNetwork\正向传播矩阵.png)
权重矩阵·输入 = 输出

### 反向传播

反向传播误差进行自学习，自动调整权重

梯度下降法求得误差对各传播路径的权重
<img src="\NetralNetwork\反向传播误差计算公式.png" alt="image-20200602195931523" style="zoom:200%;" />
<img src="\NetralNetwork\反向传播误差变量.png" alt="image-20200602200243371" style="zoom:200%;" />
其中，△Wjk为权重改变量，α为学习率，Ek为输出误差，Ok为下一层输出，Oj上一层输出













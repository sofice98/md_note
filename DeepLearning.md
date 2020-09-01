@sofice

# 神经网络

![神经网络原始模型](\NetralNetwork\神经网络原始模型.png)

**若干输入，乘上权重，通过S阈值函数，得到输出**

权重初始值可以为随机值，正态分布，均值为0，方差为
$$
\cfrac{1}{\sqrt{inodes}}
$$
`self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))`

## 多层前馈神经网络

每层神经元与下一层全互连，神经元不存在同层连接，也不存在跨层连接

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



**指数衰减学习率**

$指数衰减学习率=初始学习率*学习率衰减率^{当前轮数/多少轮衰减一次}$

```python
epoch = 40
LR_BASE = 0.2  # 最初学习率
LR_DECAY = 0.99  # 学习率衰减率
LR_STEP = 1  # 喂入多少轮BATCH_SIZE后，更新一次学习率

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5，循环100次迭代。
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:  # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)  # .assign_sub 对变量做自减 即：w -= lr*grads 即 w = w - lr*grads
    print("After %s epoch,w is %f,loss is %f,lr is %f" % (epoch, w.numpy(), loss, lr))

```



激活函数

+ sigmoid函数


  ![image-20200602195133779](\NetralNetwork\S阈值函数公式.png)
  <img src="\NetralNetwork\S阈值函数.png" alt="image-20200602194935171" style="zoom: 80%;" />

+ tanh函数

+ relu函数

+ leaky_relu函数





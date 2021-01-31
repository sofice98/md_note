e.g.

DA-RNN

A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction，Yao Qin et al.，ISCC，2017，arXiv:1704.02971

- contributions：
- defect：
- model：
- experiment：



Tensorflow:Large-scale machine learning on heterogeneous systems

Experience with selecting exemplars from clean data.

# DNN

**语音识别：**

- G. Hinton, L. Deng, D. Y u, G. Dahl, A. Mohamed, N. Jaitly, A. Senior, V . V anhoucke, P . Nguyen,
  T. Sainath, and B. Kingsbury. Deep neural networks for acoustic modeling in speech recognition. IEEE
  Signal Processing Magazine, 2012.

- G. E. Dahl, D. Y u, L. Deng, and A. Acero. Context-dependent pre-trained deep neural networks for large
  vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing - Special
  Issue on Deep Learning for Speech and Language Processing, 2012.

**图像识别：**

- D. Ciresan, U. Meier, and J. Schmidhuber. Multi-column deep neural networks for image classification.
  In CVPR, 2012.
- A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional neural
  networks. In NIPS, 2012.
- Q.V . Le, M.A. Ranzato, R. Monga, M. Devin, K. Chen, G.S. Corrado, J. Dean, and A.Y . Ng. Building
  high-level features using large scale unsupervised learning. In ICML, 2012.
- Y . LeCun, L. Bottou, Y . Bengio, and P . Haffner. Gradient-based learning applied to document recognition.
  Proceedings of the IEEE, 1998.



# RNN及其变体

1. RNN：Learning representations by back-propagating errors

   Backpropagation through time:what it does and how to do it

   Distributed representations, simple recurrent networks, and grammatical structure

2. 梯度消失：Learning long-term dependencies with gradient descent is diffificult

3. 提出LSTM：Long short-term memory. Neural Computation，S. Hochreiter and J. Schmidhuber.，1997.

4. LSTM解决长时间序列问题：LSTM can solve hard long time lag problems，S. Hochreiter and J. Schmidhuber.，1997.

5. 提出GRU：Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation，Cho et al.，2014

6. 评估GRU：Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling

**应用：**

1. 语音识别：Speech recognition with deep recurrent neural networks

2. 图像描述：Deep visual-semantic alignments for generating image descriptions

3. 机器翻译：Recurrent continuous translation models

   On the properties of neural machine translation: Encoder-decoder approaches.

   



# CNN及其变体

## WaveNet

WaveNet: A Generative Model for Raw Audio



## GLU

Language Modeling with Gated Convolutional Networks



## TCN

An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling，Shaojie Bai，2018，arXiv:1803.01271

- contributions：将卷积神经网络应用于处理时间序列。多层卷积神经网络在输入序列上创建多层表示张量，其中邻近的源单元在较低的层上相互作用，而较远的源单元在较高的层上相互作用。 与递归网络模型的链结构相比，层次结构提供了较短的获取长期依赖的路径，例如：原来RNN中捕获o(n)的长度，需要o(n)步，在TCN中只需要o(logn)步
- defect：
- model：
- experiment：





##　应用

**3D人物骨骼提取+显式遮挡**

3D Human Pose Estimationusing Spatio-Temporal Networks with Explicit Occlusion Training

（同时考虑时间空间，tcn，显式遮挡）

**视频动作分割**

MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation

（能用于长时间视频动作分割，比其他用时间卷积的模型的分辨率（每秒几帧）要高，多级卷积，平滑损失）

**视频动作分割检测**

Temporal Convolutional Networks for Action Segmentation and Detection

（一个动作由许多帧过程中的特征变化来定义；1.ED-TCN：池化，上采样，encoder-decoder，有效地捕获远程时间模式；2.Dilated-TCN：膨胀卷积，跳链接）

**时空并行+注意力机制**

Parallel Spatio-Temporal Attention-Based TCN for Multivariate Time Series Prediction

**一维因果卷积**

Seq-U-Net: A One-Dimensional Causal U-Net for Effificient Sequence Modelling






# 序列建模

## 经典Seq2Seq+Encoder-Decoder

Sequence to Sequence Learningwith Neural Network，Sutskever et al.，2014，arXiv:1409.3215

- contributions：提出Encoder-Decoder模型，将输入编码为一个中间表示向量，用来提取源序列特征

- defect：源序列的表示向量固定，会丢失一些有用的信息，长序列上性能快速下降；源序列中每个单元权重一样，不符合直觉，表现也不好

- model：

- experiment：

 ##  其它

1. 将整个输入句子映射到向量：N. Kalchbrenner and P . Blunsom. Recurrent continuous translation models. In EMNLP, 2013.
2. A. Graves. Generating sequences with recurrent neural networks. In Arxiv preprint，arXiv:1308.0850，2013.
3. A. Graves, S. Fern´ andez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks. In ICML, 2006.



# Attention

## 提出Attention机制

Neural Machine Translation by Jointly Learning to Align and Translate，Bahdanau et al.，2015，ICLR，arXiv:1409.0473

- contributions：改进了经典Encoder-Decoder模型，将固定的一个中间表示向量，变成每个目标独有的中间表示向量，它们计算时使用的源单元的权重不同，因此具有注意力机制。在长序列的对齐上表现更好；鲁棒性更强；属于端到端的机器翻译方法，与基于短语（phrase-based，指不加任何神经网络组件的机器翻译方法）的统计机器翻译的翻译性能相媲美

- defect：没有解决集外词；只用了BiRNN+Attention，没有尝试其他结构与Attention结合，没有挖掘到Attention的本质；只使用了单一数据集

- model：

  RNNsearch：BiRNN+Attention

  ![image-20201222195613205](Paper\BiRNN+Attention.png)
  
- experiment：ACL WMT ’14英法基准系统，与RNNencdec，Moses(phrase-based SOTA)比较

  使用BLUE得分，结果显示最差时刻优于RNNencdec，最好时刻优于Moses

## CNN+Attention

Convolutional Sequence to Sequence Learning，Facebook AI Research，2017，ICML，arXiv:1705.03122

- contributions：使用CNN，通过卷积的叠加可以精确地控制上下文的长度，因为卷积之间的叠加可以通过公式直接计算出感受野是多少；大幅度增加并行；门控线性单元简化了梯度传播；具备残差连接，能具有足够深的层数；对于输入的一个单词而言，输入CNN网络，所经过的卷积核和非线性计算数量都是固定的，有助于训练

- defect：可以应用于TCN以改进感受野计算式

- model：

  1. **Position Embeddings**

     卷积处理与RNN不一样，不含有位置信息，需要加入位置向量，给予模型正在处理哪一位置的信息

     k为卷积核长，d为词向量维度，$\large w_i\in R^{d\times 1}$，$\large e_i\in R^{d\times n}$

     词向量：$w=(w_1,...,w_n)$

     位置向量：$p=(p_1,...,p_n)$

     最终表示向量：Encoder输入表示向量 $e=(w_1+p_1,...,w_n+p_n)$（图中上方），Decoder输入表示向量 $g$ 由上一时刻Decode输出词的embedding组成（图中左下方）

  2. **Convolutional Block Structure**

     Encoder 共 $u$ 层卷积层，最后一层输出为$z^u$，Decoder 共 $l$ 层，最后一层输出为$h^l$

     把一次“卷积计算+非线性计算”看作一个单元Convolutional Block，这个单元在一个卷积层内是共享的

     卷积核$\large W\in R^{2d\times kd}$，权重$\large b_w\in R^{2d}$，输出$Y=[A,B]\in R^{2d}$

     经过门控线性单元GLU，得到$v([A,B])=A\otimes \sigma(B) \in R^d$，$\sigma(B)$控制着网络中的信息流，即哪些能够传递到下一个神经元中

     残差连接：$\large h_i^l=v(W^l[h_{i-k/2}^{l-1},...,h_{i+k/2}^{l-1}]+b_w^l)+h_i^{l-1}$

     单层卷积e.g.：把窗口$k*d$大小的元素（下图蓝色部分）卷积$2d$次，输出维度为$2d\times n$下图卷积后上边为A，下边为B），经过GLU得到纵向维度减半的张量，再加上残差即可得到与源序列维度相同的张量

     ![image-20201223101531297](Paper\ConvS2S单层卷积过程.png)

     多层堆叠e.g.：堆叠a层后感受野为 $1+a*(k-1)$

     ![image-20201223103709548](Paper\ConvS2S多层堆叠.png)

  3. **Multi-step Attention** 

     Decoder每一层都单独使用了Attention机制：
     $$
     \large{
     d_i^l=W_d^lh_i^l+b_d^l+g_i\\
     a_{ij}^l=\cfrac{exp(d_i^l\cdot z_j^u)}{\sum_{t=1}^m exp(d_i^l\cdot z_t^u)}\\
     c_i^l=\sum_{j=1}^m a_{ij}^l(z_j^u+e_j)\\
     W_d^l与b_d^l为Decoder第l层的Attention参数
     }
     $$
     得到$c^l$后，更新$h^l=h_i^l+c_i^l$，再输入到Decoder的第$l+1$层中，最终得到Decoder最后一层$h^L$，经过softmax层计算得到下一个目标词的概率

     

  ![image-20201222200518889](Paper\ConvS2S.png)

  上左encoder部分：通过层叠的卷积抽取输入源语言（英语）sequence的特征，图中直进行了一层卷积。卷积之后经过GLU激活做为encoder输出
  下左decoder部分：采用层叠卷积抽取输出目标语言（德语）sequence的特征，经过GLU激活做为decoder输出
  中左attention部分：把decoder和encoder的输出做点乘，做为输入源语言（英语）sequence中每个词权重
  中右Residualconnection：把attention计算的权重与输入序列相乘，加入到decoder的输出中输出输出序列

- experiment：three major WMT translation tasks,



## 实现了完全基于注意力机制的Transformer架构

Attention is All You Need，Google AI Research，2017，arXiv:1706.03762

https://jalammar.github.io/illustrated-transformer/

- contributions：Attention层的好处是能够一步到位捕捉到全局的联系，因为它直接把序列两两比较

- defect：没有RNN或者CNN结构，即使在embedding向量中引入了位置相关的编码，也是一种序列顺序的弱整合，对于位置敏感的任务如增强学习，这是一个问题，评测指标BLEU也并不特别强调语序；Position-wise Feed-Forward Networks，事实上它就是窗口大小为1的一维卷积；Attention做多次然后拼接，跟CNN中的多个卷积核的思想致；

- model：

  1. **Encoder and Decoder Stacks**

     Encoder和Decoder的输入都先做Positional Encoding，再进行N层堆叠；

     Encoder每一层包括multi-head self-attention子层，position-wise fully connected feed-forward network子层，输出维度$d_{model}=512$；

     Decoder每一层包括；加上mask的multi-head self-attention子层（保证只用到过去的信息），接收Encoder输出的multi-head self-attention子层（Key和Value来自于Encoder输出），position-wise fully connected feed-forward network子层；

     每一个子层后都有残差连接和正则化

  2. **Attention**

     - **Scaled Dot-Product Attention**
       $$
       \large{
       Attention(Q,K,V)=softmax(\cfrac{Q\cdot K^T}{\sqrt{d_k}})\cdot V\\
       Q\in\R^{n\times d_k}\quad K\in\R^{m\times d_k}\quad V\in\R^{m\times d_v}\quad
       }
       $$
       使用矩阵点积运算并归一化的Attention计算公式，相当于将$n\times d_k$的序列$Q$编码成了一个新的$n\times d_v$的序列

       使用点积而不使用加法是因为GPU处理矩阵乘法有优化；除以$\sqrt{d_k}$是为了使方差为1，内积不会太大，否则softmax后非0即1

     - **Multi-Head Attention**
       $$
       \large{
       head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)\\
       MultiHead(Q,K,V)=Concat(head_1,...,head_h)\\
       W_i^Q\in\R^{d_k\times d_{model}}\quad 
       W_i^K\in\R^{d_k\times d_{model}}\quad 
       W_i^V\in\R^{d_v\times d_{model}}\quad 
       W_i^O\in\R^{d_{model}\times hd_v}\quad\\
       h=8\quad d_k=d_v=d_{model}/h=64
       }
       $$
       计算多次Attention，每次使用的参数不共享，计算出来h个head的Attention后拼接；

       由于每个head的维度有缩减，计算复杂度与全维度单head的Attention相当

       可以允许模型在不同的表示子空间里学习到相关的信息（比如短语结构信息），多个head学习到的Attention侧重点可能略有不同，这样给了模型更大的容量

     ![image-20201226161211698](Paper\Transformer Attention.png)

  3. **Position-wise Feed-Forward Networks**
     $$
     FFN(x)=Relu(xW_1+b_1)W_2+b_2
     $$
     即窗口大小为1的一维卷积，含一层隐藏层的全连接网络，内层神经元个数：$d_{ff}=2048$

  4. **Positional Encoding**

     以上模型像一个精妙的词袋模型，还是缺少序列信息，为了引入位置信息，将词的位置编号
     $$
     \large{
     PE_{(pos,2i)}=sin(pos/10000^{2i/d_{model}})\\
     PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{model}})
     }
     $$
     将id为 $pos$ 的位置映射为一个$d_{model}$维的位置向量，这个向量的第 $i$ 个元素的数值就是$PE_{(pos,i)}$

     直接训练出来的位置向量和上述公式计算出来的位置向量，效果是接近的

     

![image-20201226145413015](Paper\Transformer.png)

- experiment：WMT 2014 English-German，WMT 2014 English-French



## 其它

1. A Simple Neural Attentive Meta-Learner，Mishra et al.，2017，ICLR，arxiv:1707.03141
2. 人类注意力机制：A dual-stage two-phase model of selective attention



# 时间序列预测

**时间序列的多变量多步预测**

非线性自回归外生(NARX)模型

长时间依赖问题；

外生变量使用时可以用到未来值，预测变量不可以；

增量学习；

**多变量**

多个外生序列之间的关系；

每个外生序列对最终预测值有多少影响，可能各个外生序列的各个时间步对预测值的影响权重都不一样

**多步**

多步预测准确性问题，累计误差怎样减小；

关键在于找到时间序列的内在周期性或内在趋势，可能是大趋势加小趋势的叠加；

**应用**

股票，气温，降雨量，车流量，CPU使用率



## DA-RNN

A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction，Yao Qin et al.，2017，arXiv:1704.02971

- contributions：用于机器翻译的序列模型无法用于多变量时间预测，因为其无法显式选择相关变量做预测，因此提出基于双阶段注意的递归神经网络；相比ARIMA，能建模非线性关系并区分多个外生序列；相比各类predefined非线性模型时间预测方法，能捕捉真正隐藏的非线性关系；计算出了各个外生序列的各个时间步对预测值的影响权重，考虑全面；

- defect：为什么不用GRU，lstm能不能加上对前面预测值的误差修正（In another way, it still does not make good use of the correlation between the driver series and the target series. For example, when we predict the network flow as the target value, we also conclude the relationship between the 
  target value and other driving attributes (such as the click rate, jump rate, page stay time or other attributes) which also contain some information for prediction. In the process of real-time data prediction, the driving series at time T cannot be provided for predicting the target value at the same time T, which is the necessity for the DARNN model.）无法多步预测，误差大

- model：

  任务：给定n个外生序列 $X=(x^1,x^2,...,x^n)^\top=(x_1,x_2,...,x_T)\in \R^{n\times T}$，$x^k=(x_1^k,...,x_T^k)^\top$代表一条长度为$T$的外生序列，$x_t=(x_t^1,...,x_t^n)^\top$代表$t$时刻的n个外生变量，做单步非线性预测：$\hat{y}_T=F(y_1,...,y_{T-1};x_1,...,x_T)$

  实现：第一阶段选择基本刺激特征：在Encoder中引入了输入注意机制，可以自适应地选择相关的外生序列；第二阶段使用分类信息解码刺激：在Decoder中使用时间注意机制在所有时间步中自动选择相关Encoder隐藏状态

  1. **Encoder with input attention**

     Encoder本质是RNN，用来做输入序列到隐变量的映射：$\large h_t=f(h_{t-1},x_t)\in\R^{m\times 1}$；

     先将n个外生序列和上一个LSTM单元的隐状态和细胞状态作为输入，输入到 Input attention Layer 中：$\large e_t^k=v_e^\top tanh(W_e[h_{t-1};s_{t-1}]+U_ex^k)$，其中$\large v_e^\top\in\R^{1\times T},W_e\in\R^{T\times 2m},U_e\in\R^{T\times T}$，即第k个序列在时刻t的权重由上一时刻隐变量和整个第k个序列决定

     再做softmax，得到 t 时刻各变量的注意力权重：$\large \alpha_t^k=softmax(e_t^k)$

     更新 t 时刻输入值：$\large \tilde{x}_t=(\alpha_t^1x_t^1,...,\alpha_t^nx_t^n)^\top$

     用LSTM更新隐变量：$\large h_t=f(h_{t-1},\tilde{x}_t)$

  2. **Decoder with temporal attention**

     求得Encoder隐状态h后，计算Decoder的输入

     Temporal attention Layer计算Encoder隐状态的权重：$\large l_t^i=v_d^\top tanh(W_d[d_{t-1};s'_{t-1}]+U_dh_i),1\leq i\leq T$，其中$\large v_d^\top\in\R^{1\times m},W_e\in\R^{m\times 2p},U_e\in\R^{m\times m}$，$\beta_t^i=softmax(l_t^i)$

     再计算中间表示向量：$\large c_t=\sum_{i=1}^T \beta_t^ih_i$

     将$c_t$与$y_{t-1}$合并作为Decoder输入：$\large \tilde{y}_t=\tilde{w}^\top[y_t;c_t]+\tilde{b}$，其中$\large [y_t;c_t]\in\R^{m+1},\tilde{w}^\top\in\R^{m+1},\tilde{b}\in\R$

     将Decoder最后一个LSTM单元的隐状态作为预测值

  <img src="Paper\DA-RNN.png" alt="image-20201229153523631"  />

- experiment：SML 2010室温预测，NASDAQ 100股票预测

  超参数：窗口T，Encoder隐状态维度m，Decoder隐状态维度p，T=10，m=p=64,128

  模型：NARX RNN, Encoder-Decoder, Attention RNN, Input-Attn-RNN and DA-RNN

  评价指标：RMSE，MAE，MAPE

  有表，有可视化图，结果都明显达到SOTA

  为了研究DA-RNN内输入注意机制的有效性，以噪声外生序列作为输入进行了测试，将81个原始外生序列与81个噪声序列同时作为输入，图中显示能对噪声外生序列分配更小的权重，从而达到抑制效果；表明DA-RNN**对噪声输入具有鲁棒性**

  通过控制变量法比较DA-RNN与Input-Attn-RNN，得出DA-RNN比Input-Attn-RNN**对参数的鲁棒性**更强



## GeoMAN

GeoMAN: Multi-level Attention Networks for Geo-sensory Time Series Prediction



## DSTP-RNN

DSTP-RNN: A dual-stage two-phase attention-based recurrent neural network for long-term and multivariate time series prediction，2019，



## MA-RNN

A Multivariate Time Series Prediction Schema based on Multi-attention in recurrent neural network，Xiang Yin et al.，ISCC，2020



## MA-GAN

- contributions：DA-RNN，DSTP等都是单步预测，应用于多步时误差大；自动调整外生序列权重；通过增加时间-卷积-注意层中时间特征的权重来增强捕获时间依赖性的能力；GAN可以提高MA-RNN模型生成时间序列数据的能力，能有效地预测未来的长期序列；动态的权重裁剪算法，使判别器阶段更加稳定和准确
- defect：
- model：the encoder network, the generator(decoder) network, and the discriminator network
- experiment：



## LSTNet

Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks



## TPA

Temporal Pattern Attention for Multivariate Time Series Forecasting，Shun-Yao，2018，arXiv:1809.04206

- contributions：MST，TPA，对不同时间序列关注不同的时间步长，适用于非周期和非线性任务，可解释
- defect：
- model：
- experiment：



## 时间序列预测其他方法

1. ARMA：Hypothesis Testing in Time Series Analysis
2. ARIMA：Arima models and the box-jenkins methodology.（不能建模非线性关系，也同等对待所有外生序列；只考虑目标序列季节性变化，忽略了驱动序列）
3. NARX RNN：The use of NARX neural networks to predict chaotic time series.
4. NARX长期依赖问题：Learning long-term dependencies in NARX recurrent neural networks
5. NARX RNN：Narmax time series model prediction: feedforward and recurrent fuzzy neural network approaches
6. NARX：Substructure vibration NARX neural network approach for statistical damage inference
7. 核方法：Narx based nonlinear system identification using orthogonal least squares basis hunting
8. 集成方法：Ensemble learning for time series prediction
9. 高斯过程：Integrated pre-processing for bayesian nonlinear system identification with gaussian processes
10. 分层注意网络：Hierarchical attention networks for document classifification. In *NAACL*, 2016（用于分类，不能选择相关外生序列进行预测）



## 应用：

1. 天气预测

   Fine-grained photovoltaic output prediction using a bayesian ensemble

2. 金融市场预测

   Dynamic covariance models for multivariate financial time series

   Stock Market Prediction Based on Generative Adversarial Network，Zhang et al.，2019

3. 复杂动态系统分析

   A regularized linear dynamical system framework for multivariate time series analysis











































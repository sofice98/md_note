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

1. **提出LSTM：**S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.

2. **LSTM解决长时间序列问题：**S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. 1997.

3. **提出GRU：**Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

4. **评估GRU：**Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling



# CNN及其变体

1. **提出WaveNet：**WaveNet: A Generative Model for Raw Audio

2. 
3. Language Modeling with Gated Convolutional Networks

# TCN

1. **提出TCN：**An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling，2018，arXiv:1803.01271

   - contributions：将卷积神经网络应用于处理时间序列。多层卷积神经网络在输入序列上创建多层表示张量，其中邻近的源单元在较低的层上相互作用，而较远的源单元在较高的层上相互作用。 与递归网络模型的链结构相比，层次结构提供了较短的获取长期依赖的路径，例如：原来RNN中捕获o(n)的长度，需要o(n)步，在TCN中只需要o(logn)步

   - defect：

   - dataset：

**应用：**

- **3D人物骨骼提取+显式遮挡：**3D Human Pose Estimationusing Spatio-Temporal Networks with Explicit Occlusion Training

  （同时考虑时间空间，tcn，显式遮挡）

- **视频动作分割：**MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation

  （能用于长时间视频动作分割，比其他用时间卷积的模型的分辨率（每秒几帧）要高，多级卷积，平滑损失）

- **视频动作分割检测：**Temporal Convolutional Networks for Action Segmentation and Detection

  （一个动作由许多帧过程中的特征变化来定义；1.ED-TCN：池化，上采样，encoder-decoder，有效地捕获远程时间模式；2.Dilated-TCN：膨胀卷积，跳链接）

- **时空并行+注意力机制：**Parallel Spatio-Temporal Attention-Based TCN for Multivariate Time Series Prediction

- **一维因果卷积：**Seq-U-Net: A One-Dimensional Causal U-Net for Effificient Sequence Modelling

- **多变量时间序列预测：**Temporal Pattern Attention for Multivariate Time Series Forecasting

  （MST，TPA，对不同时间序列关注不同的时间步长，适用于非周期和非线性任务，可解释）







# 序列建模

1. **经典Seq2Seq+Encoder-Decoder：**Sequence to Sequence Learningwith Neural Network，Sutskever et al.，2014，arXiv:1409.3215

   - contributions：提出Encoder-Decoder模型，将输入编码为一个中间表示向量，用来提取源序列特征

   - defect：源序列的表示向量固定，会丢失一些有用的信息，并且源序列中每个单元权重一样，不符合直觉，在长序列上表现不好

   - model：
   
   - dataset：
   
   

- **将整个输入句子映射到向量：**N. Kalchbrenner and P . Blunsom. Recurrent continuous translation models. In EMNLP, 2013.
- A. Graves. Generating sequences with recurrent neural networks. In Arxiv preprint arXiv:1308.0850,
  2013.
- A. Graves, S. Fern´ andez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: labelling
  unsegmented sequence data with recurrent neural networks. In ICML, 2006.
  
  



# Attention

## 提出Attention机制：Neural Machine Translation by Jointly Learning to Align and Translate

Bahdanau et al.，2015，ICLR，arXiv:1409.0473

- contributions：改进了经典Encoder-Decoder模型，将固定的一个中间表示向量，变成每个目标独有的中间表示向量，它们计算时使用的源单元的权重不同，因此具有注意力机制。在长序列的对齐上表现更好；鲁棒性更强；属于端到端的机器翻译方法，与基于短语（phrase-based，指不加任何神经网络组件的机器翻译方法）的统计机器翻译的翻译性能相媲美

  - defect：结构混乱background->...->related work；没有解决集外词；只用了BiRNN+Attention，没有尝试其他结构与Attention结合，没有挖掘到Attention的本质；只使用了单一数据集

- model：

  RNNsearch：BiRNN+Attention

  ![image-20201222195613205](Paper\BiRNN+Attention.png)
  $$
  Attention(Query,Source)=\sum_{i=1}^{L_x}Similarity(Query,Key_i)*Value_i\\
  1.Similarity:Sim_{点积}=Query\cdot Key_i\quad Sim_{cosine}=\cfrac{Query\cdot Key_i}{||Query||\cdot ||Key_i||}\quad Sim_{MLP}=MLP(Query,Key_i)\\
  2.\alpha_i=softmax(Sim_i)=\cfrac{exp(Sim_i)}{\sum_{k=1}^{T_x}exp(Sim_i)}\\
  3.Attention(Query,Source)=\sum_{i=1}^{L_x}\alpha_i*Value_i
  $$
  
- experiment：ACL WMT ’14英法基准系统，与RNNencdec，Moses(phrase-based SOTA)比较

  使用BLUE得分，结果显示最差时刻优于RNNencdec，最好时刻优于Moses

## **CNN+Attention：**Convolutional Sequence to Sequence Learning

Facebook AI Research，2017，ICML，arXiv:1705.03122

- contributions：使用CNN，通过卷积的叠加可以精确地控制上下文的长度，因为卷积之间的叠加可以通过公式直接计算出感受野是多少；大幅度增加并行；门控线性单元简化了梯度传播；具备残差连接，能具有足够深的层数；对于输入的一个单词而言，输入CNN网络，所经过的卷积核和非线性计算数量都是固定的，有助于训练

- defect：

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

## 实现了完全基于注意力机制的Transformer架构：Attention is All You Need

Google AI Research，2017，arXiv:1706.03762

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

- dataset：WMT 2014 English-German，WMT 2014 English-French



A Simple Neural Attentive Meta-Learner，Mishra et al.，2017，ICLR，arxiv:1707.03141















































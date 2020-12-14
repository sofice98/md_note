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



# LSTM

**LSTM原理：**S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural Computation, 1997.

**LSTM解决长时间序列问题：**S. Hochreiter and J. Schmidhuber. LSTM can solve hard long time lag problems. 1997.



# GRU



# Wavenet

**WaveNet原理：**WaveNet: A Generative Model for Raw Audio



# TCN

**TCN机制：**An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

（将卷积神经网络应用于处理时间序列）

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







# Sequence to Sequence

**经典Seq2Seq+encoder-decoder：**Sequence to Sequence Learningwith Neural Network

- **将整个输入句子映射到向量：**N. Kalchbrenner and P . Blunsom. Recurrent continuous translation models. In EMNLP, 2013.
- A. Graves. Generating sequences with recurrent neural networks. In Arxiv preprint arXiv:1308.0850,
  2013.
- A. Graves, S. Fern´ andez, F. Gomez, and J. Schmidhuber. Connectionist temporal classification: labelling
  unsegmented sequence data with recurrent neural networks. In ICML, 2006.
  
  



# Encoder-Decoder



**加Attention机制：**Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation




















































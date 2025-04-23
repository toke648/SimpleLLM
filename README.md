# Transerformer架构（自然语言处理）
尝试学习和从零构建一个大语言模型

![download](https://github.com/user-attachments/assets/467274c7-c2fc-41ae-a62e-eacdb942bbcc)

就目前我的认知
Transformer架构主要分为编码器、解码器、词表、训练集、训练算法(T5)
### 编码器(Encoder)
Encoder主要负责将输入转换为计算机能够理解的内容（也就是词表中的向量词元）

### 解码器(Decoder)
将词元的向量内容还原回人类可以理解的内容

### 词表(Tokenizer)
模型所使用的词汇是基于词表中有的词元所生成的，词表可以由大量的文本内容训练，并且训练模式需要符合标准BPE格式

### 训练集(Training set)
大量的现实网络中人们的对话、沟通数据，需要确保数据是干净的

### 训练算法(T5)
通过梯度下降等方法降低模型的损失（令模型回复的内容越来越接近训练集的内容）

剩下的我还不太清楚，再研究研究


### 介绍及运行

这是一个基于Transerformer架构的一个小型文本生成模型
你需要先准备好词元训练集并命名为data1.txt、data2.txt

- 安装环境：
    ```sh
      pip install -r requirements.txt
    ```

- 训练词表：打开 [train_tokenizer.ipynbt](train_tokenizer.ipynb) 文件并按顺序运行

- 训练模型：打开 [train_BPE_tokenizer.ipynb](train_BPE_tokenizer.ipynb) 文件并运行






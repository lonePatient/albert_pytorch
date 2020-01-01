### albert_pytorch

This repository contains a PyTorch implementation of the albert model from the paper 

[A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

by Zhenzhong Lan. Mingda Chen....

已经转好的pytorch版本权重，可以直接下载(谷歌盘):

#### 中文权重下载

**google**

- [albert_tiny_zh](https://drive.google.com/open?id=1qAykqB2OXIYXSVMQSt_EDEPCorTAkIvu)
- [albert_small_zh](https://drive.google.com/open?id=1t-DJKAqALgwlO8J_PZ3ZtOy0NwLipGxA)
- [albert_base_zh.zip](https://drive.google.com/open?id=1m_tnylngzEA94xVvBCc3I3DRQNbK32z3)
- [albert_large_zh.zip](https://drive.google.com/open?id=19UZPHfKJZY9BGS4mghuKcFyAIF3nJdlX)
- [albert_xlarge_zh.zip](https://drive.google.com/open?id=1DdZ3-AXaom10nrx8C99UhFYsx1BYYEJx)
- [albert_xxlarge_zh.zip](https://drive.google.com/open?id=1F-Mu9yWvj1XX5WN6gtyxbVLWr610ttgC)

**bright**

- [albert_tiny_bright](https://drive.google.com/open?id=1VBsUJ7R5eWF1VcUBQY6BEn1a9miEvlBr)
- [albert_base_bright.zip](https://drive.google.com/open?id=1HeijHGubWR-ElFnfxUf8IrRx7Ghm1S_Q)
- [albert_large_bright.zip](https://drive.google.com/open?id=1TAuv7OiFN8qbkT6S_VbfVbhkhg2GUF3q)
- [albert_xlarge_bright.zip](https://drive.google.com/open?id=1kMhogQRX0uGWIGdNhm7-3hsmHlrzY_gp)

**说明**: 以上权重只适合该版本。**pytorch**版本为1.1.0 

原始提供下载地址：

- [brightmart](https://github.com/brightmart/albert_zh)
- [google albert](https://github.com/google-research/ALBERT)

#### 使用方式

不同的模型权重加载不同的对应模型原始文件，如下(使用的时候加载对应模型文件即可):
```python
#google version
from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification

# bright version
from model.modeling_albert_bright import AlbertConfig, AlbertForSequenceClassification
```
**注意**，如果需要运行该`run_classifier.py`脚本，需要将`config.json`和`vocab.txt`文件同时放入对应模型的目录中，比如:

```text
├── prev_trained_model
|  └── albert_base
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── vocab.txt
```

#### 预训练

**备注**：仅供参考

n-gram: 原始论文中按照以下分布随机生成n-gram，默认max_n为3

   <p align="center"><img width="200" src="https://lonepatient-1257945978.cos.ap-chengdu.myqcloud.com/n-gram.png" /></p>
1. 将文本数据转化为一行一句格式，并且不同document之间使用`\n`分割

2. 运行以下命令：
```python
python prepare_lm_data_ngram.py \
    --data_dir=dataset/ \
    --vocab_path=vocab.txt \
    --data_name=albert \
    --max_ngram=3 \
    --do_data
```
产生n-gram masking数据集，具体可根据对应数据进行修改代码

3. 运行以下命令：
```python
python run_pretraining.py \
    --data_dir=dataset/ \
    --vocab_path=configs/vocab.txt \
    --data_name=albert \
    --config_path=configs/albert_config_base.json \
    --output_dir=outputs/ \
    --data_name=albert \
    --share_type=all
```
进行模型训练，具体可根据对应数据进行修改代码

#### 测试结果

问题匹配语任务：LCQMC(Sentence Pair Matching)

| 模型 | 开发集(Dev) | 测试集(Test) |
| :------- | :---------: | :---------: |
| albert_base(tf) | 86.4 | 86.3 |
| albert_base(pytorch) | 87.4 | 86.4 |
| albert_tiny | 85.１ | 85.3 |

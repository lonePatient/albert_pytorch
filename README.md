## albert_zh_pytorch

This repository contains a PyTorch implementation of the albert model from the paper 

[A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

by Zhenzhong Lan. Mingda Chen....

## 中文模型模型Download Pre-trained Models of Chinese

模型权重主要来自：[github](https://github.com/brightmart/albert_zh)

1. [albert_large_zh](https://storage.googleapis.com/albert_zh/albert_large_zh.zip) 参数量，层数24，大小为64M

2. [albert_base_zh(小模型体验版)](https://storage.googleapis.com/albert_zh/albert_base_zh.zip), 参数量12M, 层数12，大小为40M

## 预训练模型

１．将文本数据转化为一行一句格式，并且不同document之间使用｀\n｀分割

２．运行｀python prepare_lm_data_mask.py --do_data｀脚本和｀python prepare_lm_data_ngram.py --do_data｀分别生成随机mask和ngram mask格式数据集

３．运行｀python run_pretraining.py｀进行模型预训练

## 下游任务Fine-tuning

１．下载预训练的albert模型

２．运行｀python convert_albert_tf_checkpoint_to_pytorch.py｀将TF模型权重转化为pytorch模型权重

３．下载对应的数据集，比如[LCQMC](https://drive.google.com/open?id=1HXYMqsXjmA5uIfu_SFqP7r_vZZG-m_H0)数据集，包含训练、验证和测试集，训练集包含24万口语化描述的中文句子对，标签为1或0。1为句子语义相似，0为语义不相似。

４．运行`python run_classifier.py`进行Fine-tuning训练

## 结果






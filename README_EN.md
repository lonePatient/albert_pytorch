# albert_en_pytorch

This repository contains a PyTorch implementation of the albert model from the paper 

[A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

by Zhenzhong Lan. Mingda Chen....

arxiv: https://arxiv.org/pdf/1909.11942.pdf

## Dependencies

- pytorch=1.10
- cuda=9.0
- cudnn=7.5
- scikit-learn
- sentencepiece

## Download Pre-trained Models of English

Version 2 of ALBERT models is relased. TF-Hub modules are available:

- https://tfhub.dev/google/albert_base/2
- https://tfhub.dev/google/albert_large/2
- https://tfhub.dev/google/albert_xlarge/2
- https://tfhub.dev/google/albert_xxlarge/2



or download pytorch model (google):

- [albert_base_v2.zip](https://drive.google.com/open?id=1JRGQPaUb2DIwIfugOdopxdoezx040Qrz)
- [albert_large_v2.zip](https://drive.google.com/file/d/1O6FUCQldNwoz5fP9sa6fDmlwauu8uzo8/view?usp=sharing)
- [albert_xlarge_v2.zip](https://drive.google.com/open?id=1fFu98VfoLILpQeS4IeLQ9OZEB18HR8vK)

baidu:

- [albert_base_v2.zip](https://pan.baidu.com/s/1as97ay14pvCaa_8r2WzMSQ)
- [albert_large_v2.zip](https://pan.baidu.com/s/1pdN4h7b8wgL131Zgpo_ohg)
- [albert_xlarge_v2.zip](https://pan.baidu.com/s/1-XyOjI1GjjAxv_9IO14Xwg)

## Fine-tuning

１．Download the Bert pretrained model from [TF-HUb](https://tfhub.dev/google/albert_base)

2. Rename

- albert-config.json to config.json

3. Place `config.json` and `30k-clean.model` into the `prev_trained_model/albert_base_v2` directory.

example:

```text
├── prev_trained_model
|  └── albert_base_v2
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── 30k-clean.model
```

4．run cmd:
```python
python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/albert_base_tf_v2 \
    --bert_config_file=./prev_trained_model/albert_base_v2/config.json \
    --pytorch_dump_path=./prev_trained_model/albert_base_v2/pytorch_model.bin
```

４．run `sh run_classifier_lcqmc.sh`to fine tuning albert model

## Result

Performance of ALBERT on GLUE benchmark results using a single-model setup on **dev**:

|  | Cola| Sst-2| Mnli| Sts-b|
| :------- | :---------: | :---------: |:---------: | :---------: |
| metric | matthews_corrcoef |accuracy |accuracy | pearson |

| model | Cola| Sst-2| Mnli| Sts-b|
| :------- | :---------: | :---------: |:---------: | :---------: |
| albert_base_v2 | 0.5756 | 0.926 | 0.8418 | 0.9091 |
| albert_large_v2 | 0.5851 |0.9507 |  |0.9151 |
| albert_xlarge_v2 | 0.6023 | |  |0.9221 |



[**English Version**](./README.md) | [**中文版说明**](./README_zh.md)

## albert_pytorch

This repository contains a PyTorch implementation of the albert model from the paper 

[A Lite Bert For Self-Supervised Learning Language Representations](https://arxiv.org/pdf/1909.11942.pdf)

by Zhenzhong Lan. Mingda Chen....

## Dependencies

- pytorch=1.10
- cuda=9.0
- cudnn=7.5
- scikit-learn
- sentencepiece

## Download Pre-trained Models of English

Official download links: [google albert](https://github.com/google-research/ALBERT)

Adapt to this version，download pytorch model (google drive):

**v1**

- [albert_base_v1.zip](https://drive.google.com/open?id=1dVsVd6j8rCTpqF4UwnqWuUpmkhxRkEie)
- [albert_large_v1.zip](https://drive.google.com/open?id=18dDXuIHXYWibCLlKX5_rZkFxa3VSc5j1)
- [albert_xlarge_v1.zip](https://drive.google.com/open?id=1jidZkLLFeDuQJsXVtenTvV_LU-AYprJn)
- [albert_xxlarge_v1.zip](https://drive.google.com/open?id=1PV8giuCEAR2Lxaffp0cuCjXh1tVg7Vj_)

**v2**

- [albert_base_v2.zip](https://drive.google.com/open?id=1byZQmWDgyhrLpj8oXtxBG6AA52c8IHE-)
- [albert_large_v2.zip](https://drive.google.com/open?id=1KpevOXWzR4OTviFNENm_pbKfYAcokl2V)
- [albert_xlarge_v2.zip](https://drive.google.com/open?id=1W6PxOWnQMxavfiFJsxGic06UVXbq70kq)
- [albert_xxlarge_v2.zip](https://drive.google.com/open?id=1o0EhxPqjd7yRLIwlbH_UAuSAV1dtIXBM)

## Fine-tuning

１. Place `config.json` and `30k-clean.model` into the `prev_trained_model/albert_base_v2` directory.
example:
```text
├── prev_trained_model
|  └── albert_base_v2
|  |  └── pytorch_model.bin
|  |  └── config.json
|  |  └── 30k-clean.model
```
2．convert albert tf checkpoint to pytorch
```python
python convert_albert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=./prev_trained_model/albert_base_tf_v2 \
    --bert_config_file=./prev_trained_model/albert_base_v2/config.json \
    --pytorch_dump_path=./prev_trained_model/albert_base_v2/pytorch_model.bin
```
The [General Language Understanding Evaluation (GLUE) benchmark](https://gluebenchmark.com/) is a collection of nine sentence- or sentence-pair language understanding tasks for evaluating and analyzing natural language understanding systems.

Before running anyone of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory $DATA_DIR.

3．run `sh scripts/run_classifier_sst2.sh`to fine tuning albert model

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



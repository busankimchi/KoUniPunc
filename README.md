# KoUniPunc

Korean version of [UniPunc](https://ieeexplore.ieee.org/document/9747131).

Original code implementation is from https://github.com/Yaoming95/UniPunc.

# Finetune (English)

```
fairseq-train \
    --arch bert_punc_wav \
    --pretrain-model bert-base-multilingual-cased \
    --w2v2-model-path pretained_model/wav2vec_small.pt \
    --header-model transformer_add_headers_for_bert_base \
    --head-layer-number 2 \
    --criterion label_smoothed_cross_entropy_with_f1_metrics \
    --optimizer adam \
    --lr 0.00001 \
    --lr-scheduler inverse_sqrt \
    --dropout 0.1 \
    --save-dir checkpoints/unipunc \
```

# Inference (English)

```

```

---

# Updated version

# Data aggregation

```
python3 -m src.dataset.data_aggregation
```

# Finetune

```
python3 -m src.train.main --do_train --do_eval --write_pred --report_as_file
```

# Test Data aggregation

```
python3 -m src.dataset.test_data_aggregation
```

# Inference

```
python3 -m src.inference.main
```

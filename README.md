# KoUniPunc

[Unofficial] Korean version of [UniPunc](https://ieeexplore.ieee.org/document/9747131).

Original code implementation is from https://github.com/Yaoming95/UniPunc.

# Installation

## Conda env setting
Creates conda environemnt for this model.
```
$ conda env create --file env.yaml
```

# Process

## Data aggregation/preprocessing
Makes two files, `train.jsonl` and `dev.jsonl`, from raw sparsed datas.
```
$ python3 -m src.dataset.data_aggregation
```


## Finetune
Fine tune the model using aggregated data.

**Flags**
- `--ignore_wav` : Ignore wave signal. This utlizes only the text features

- `--log_prefix` : Log prefix

- `--amp` : Enable fp16 precisions

- `--parallel` : Enable parallel computing for multi-GPUs

```
$ python3 -m src.train.main --do_train --do_eval --write_pred --report_as_file --amp --parallel --use_virtual --log_prefix 221129_debug_metric_6 
```

## Test Data aggregation
Aggergate test data to fit into the trained model.
```
$ python3 -m src.dataset.test_data_aggregation
```

## Inference
Inference using the trained model and the aggregated test data.
```
$ python3 -m src.inference.main
```

## Demo
This project includes a demo using streamlit.
```
$ streamlit run src/demo/main.py
```

# Results
- TBD
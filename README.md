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
$ python3 -m dataset.data_aggregation
```


## Finetune
Fine tune the model using aggregated data.

**Flags**
- `--ignore_wav` : Ignore wave signal. This utlizes only the text features

- `--log_prefix` : Log prefix

- `--amp` : Enable fp16 precisions

- `--parallel` : Enable parallel computing for multi-GPUs

```
$ python3 -m train.main --do_train --do_eval --write_pred --report_as_file --amp --parallel --use_virtual --log_prefix 221205_debug
```


## Inference
Inference using the trained model and the aggregated test data.
```
$ python3 -m inference.main
```

## End-to-end Inference
End-to-end inference using the trained model and a single audio file.
```
$ python3 -m inference.e2e
```

## Demo
This project includes a demo using streamlit.
```
$ streamlit run main.py
```


# Results
- TBD
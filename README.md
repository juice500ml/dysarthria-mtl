# Automatic Severity Assessment of Dysarthric speech by using Self-supervised Model with Multi-task Learning
- Official implementation of the paper: https://arxiv.org/abs/2210.15387
- Accepted to ICASSP 2023
- ICASSP Slides: [slides.pdf](./slides.pdf)

## How to run
```
# STL Baseline
python3 train.py \
    --num_epochs 100 --num_classes 5 --ctc_weight 0.0 \
    --csv_path dataset.csv --prefix BASELINE

# MTL with e=0, alpha=0.1
python3 train.py \
    --num_epochs 100 --num_classes 5 --ctc_weight 0.1 --enable_cls_epochs 0 \
    --csv_path dataset.csv --prefix MTL_E0

# MTL with e=10, alpha=0.1
python3 train.py \
    --num_epochs 100 --num_classes 5 --ctc_weight 0.1 --enable_cls_epochs 10 \
    --csv_path dataset.csv --prefix MTL_E0

# Sadly, we couldn't release the dataset due to privacy concerns.
# Hence, you need to use your own dataset.
```

## Dataset CSV structure
```
>> cat dataset.csv | head -n 2
name,path,category,text,split
FILE_INDEX,/path/to/wav,0,Some text for ASR,train
```
- split: train/valid/test
- category: 0/1/2/3/4 (severity)

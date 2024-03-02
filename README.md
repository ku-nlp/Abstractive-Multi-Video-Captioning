The implementation of the paper "Abstractive Multi-Video Captioning:
Benchmark Dataset Construction and Extensive Evaluation."

AbstrActs dataset is available at https://github.com/ku-nlp/AbstrActs

# Preliminary
VATEX video features (CLIP4Clip):  
https://lotus.kuee.kyoto-u.ac.jp/~r-takahashi/dataset/VATEX_CLIP4Clip.zip

VATEX caption features:  
https://lotus.kuee.kyoto-u.ac.jp/~r-takahashi/dataset/VATEX_caption_features.zip

Download the features above and put them the right directory as below.

```
data
├─AbstrActs
├─VATEX
├─VATEX_caption_features
│  ├─gold
│  └─pred
└─VATEX_CLIP4Clip
    ├─public_test
    └─trainval
```

# Requirements
- python==3.8.11
- numpy==1.19.5
- torch==2.2.1+cu121
- torchtext
- fasttext
- transformers
- sentencepiece
- tqdm
- hydra
- wandb

```bash
conda create -n abstracts python=3.8.11
conda activate abstracts
pip install numpy==1.19.5 fasttext transformers sentencepiece tqdm hydra-core wandb
pip install torch==2.2.1 torchtext --index-url https://download.pytorch.org/whl/cu121
```

# Training & Prediction
Model checkpoints and predictions are saved in `models` directory.

## End-to-End Model (Two Videos, Soft Alignment)
```bash
# Training
$ python src/train.py --config-name=end2end-2videos

# Prediction
$ python src/predict.py --config-name=end2end-2videos run.model_ckpt_name={checkpoint name}
```

## Cascade Model (Two Videos, Soft Alignment)
```bash
# Training (Abstraction Module)
$ python src/train.py --config-name=cascade-abstraction

# Prediction
$ python src/predict.py --config-name=cascade-abstraction run.model_ckpt_name={checkpoint name}
```

## Cascade Gold Model (Two Videos, Soft Alignment)
```bash
# Training (Abstraction Module)
$ python src/train.py --config-name=cascade-abstraction-gold

# Prediction
$ python src/predict.py --config-name=cascade-abstraction-gold run.model_ckpt_name={checkpoint name}
```

## T5 Model (Two Videos)
```bash
# Training
$ python src/train_t5.py

# Prediction
$ python src/predict_t5.py run.model_ckpt_name={checkpoint name}
```

# Citation
If you find this dataset helpful, please cite our publication.

```
Rikito Takahashi, Hirokazu Kiyomaru, Chenhui Chu, Sadao Kurohashi.
Abstractive Multi-Video Captioning: Benchmark Dataset Construction and Extensive Evaluation.
In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), (2024.5).
```

# Acknowledgements
This work was supported by JSPS KAKENHI Grant Number JP23H03454 and Fujitsu.

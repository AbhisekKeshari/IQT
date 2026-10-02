# IQT + Natural Scene Statistics: Baseline Experiments for Full-Reference IQA

Baseline and exploratory experiments for the NTIRE 2022 Perceptual Image Quality Assessment Challenge, run on the [PIPAL](https://github.com/HaomingCai/PIPAL-dataset) dataset.

This repo is the starting point for our multi-scale method. The final approach and paper are in **[IQA-multiscaling](https://github.com/AbhisekKeshari/IQA-multiscaling)** ([arXiv:2204.09779](https://arxiv.org/abs/2204.09779)).

## What's here

- **IQT baseline:** A PyTorch implementation of the Image Quality Transformer ([Cheon et al., CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Cheon_Perceptual_Image_Quality_Assessment_With_Transformers_CVPRW_2021_paper.pdf)), the winning method of NTIRE 2021. Multi-level InceptionResNetV2 features of the reference and distorted images go to a transformer encoder–decoder that regresses a quality score.
- **Natural scene statistics (NSS) features:** `data/data_PIPAL.py` computes BRISQUE-style features for each image. These come from the MSCN coefficients (mean-subtracted, contrast-normalized), fitted with GGD and AGGD distributions across neighbouring-pixel products at two scales. The experiment tested whether hand-crafted statistical features help the learned transformer features on GAN-distorted images. `data/MSCN.py` is a standalone MSCN computation used for analysis.
- **Checkpoint:** `Weights/PIPAL/epoch40.pth` is a trained transformer checkpoint from these runs.

## Structure

```
train.py / test.py / trainer.py   # training, inference, train/eval loops
backbone.py                       # InceptionResNetV2 feature extractor
model/model_main.py               # transformer encoder–decoder regressor
data/                             # PIPAL / LIVE loaders, NSS + MSCN features
IQA_list/                         # image lists with MOS labels
option/config.py, utils/util.py   # config helper, augmentations, data split
```

## Usage

The setup is the same as in [IQA-multiscaling](https://github.com/AbhisekKeshari/IQA-multiscaling#setup). Set `db_path` and `txt_file_name` in the `config` dict of `train.py`, then run:

```bash
python train.py
python test.py
```

> Research code from 2022, kept for reference. Paths in the config dicts point to the original GPU server and need to be changed.

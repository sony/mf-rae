## MeanFlow Transformers with Representation Autoencoders (RAE)<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://www.arxiv.org/abs/2511.13019) 


This repository is based on: 
- [**Diffusion Transformers with Representation Autoencoders**](https://arxiv.org/abs/2510.11690).
- and our previous work, CMT: [**Consistency Mid-Training**](https://github.com/sony/cmt).

## Environment

### Dependency Setup
1. Create environment and install via `uv`:
   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv
   
   # Install PyTorch 2.2.0 with CUDA 12.1
   uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install other dependencies
   uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
   uv pip install "numpy<2" transformers einops omegaconf
   ```

## Data Preparation

1. Download ImageNet-1k **raw data without preprocessing**.
2. Point Stage 1 and Stage 2 scripts to the training split via `--data-path`.

## Pre-Training: Flow Matching

The RAE authors release flow-matching pre-traine models: RAE decoders, DiT<sup>DH</sup> diffusion transformers and stats for latent normalization. To download all models at once:


```bash

cd RAE
pip install huggingface_hub
hf download nyu-visionx/RAE-collections \
  --local-dir models 
```


To download specific models, run:
```bash
hf download nyu-visionx/RAE-collections \
  <remote_model_path> \
  --local-dir models 
```

## Consistent Mid-Training

```bash
bash CMT_256.sh
```

```bash
bash CMT_512.sh
```

## MFT and MFD Post-Training

Make sure to input the CMT checkpoint path obtained from the previous stage.

For instance, on ImageNet 512, they are

```bash
bash MFT_512.sh
```

```bash
bash MFD_512.sh
```

## Distributed sampling for evaluation

Make sure to input the MeanFlow-RAE checkpoint path after training to the config file.

We provide our trained MF-RAE on Google Drive: https://drive.google.com/drive/folders/1EYVyIDKRZeHn6NO7uF5aJ1ycR3lvfnJu?usp=drive_link

```bash
bash Sample_256.sh
```

```bash
bash Sample_512.sh
```

## Evaluation

### ADM Suite FID setup

Use the ADM evaluation suite to score generated samples:

1. Clone the repo:

   ```bash
   git clone https://github.com/openai/guided-diffusion.git
   cd guided-diffusion/evaluation
   ```

2. Create an environment and install dependencies:

   ```bash
   conda create -n adm-fid python=3.10
   conda activate adm-fid
   pip install 'tensorflow[and-cuda]'==2.19 scipy requests tqdm
   ```

3. Download ImageNet statistics (256×256 shown here):

   ```bash
   wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
   ```

4. Evaluate:

   ```bash
   python evaluator.py VIRTUAL_imagenet256_labeled.npz /path/to/samples.npz
   ```


## Acknowledgement

This code is built upon the following repositories:

* [SiT](https://github.com/willisma/sit) - for diffusion implementation and training codebase.
* [DDT](https://github.com/MCG-NJU/DDT) - for some of the DiT<sup>DH</sup> implementation.
* [LightningDiT](https://github.com/hustvl/LightningDiT/) - for the PyTorch Lightning based DiT implementation.
* [MAE](https://github.com/facebookresearch/mae) - for the ViT decoder architecture.
* [RAE](https://github.com/bytetriper/RAE) - for the RAE model and checkpoints.

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    src/train_mfd.py \
    --config configs/mf/training/MF_XL.yaml \
    --data-path /path/to/ImageNet/train \
    --results-dir results_MFD_XL_256 \
    --precision fp32 \
    --ckpt_cmt /path/to/cmt.pt
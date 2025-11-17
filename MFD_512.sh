torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train_mfd_512.py \
    --config configs/mf/training/MF_XL_512.yaml \
    --data-path /path/to/ImageNet/train \
    --results-dir results_MFD_512 \
    --precision fp32  \
    --ckpt_cmt /path/to/cmt.pt

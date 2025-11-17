torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    src/train_cmt.py \
    --config configs/cmt/CMT_256.yaml \
    --data-path /path/to/ImageNet/train \
    --results-dir results_CMT_256 \
    --precision fp32 
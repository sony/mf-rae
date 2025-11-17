torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    src/train_cmt_512.py \
    --config configs/cmt/CMT_512.yaml \
    --data-path /path/to/ImageNet/train \
    --results-dir results_CMT_512 \
    --precision fp32 
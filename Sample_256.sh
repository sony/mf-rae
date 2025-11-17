torchrun --standalone --nnodes=1 --nproc_per_node=2   src/sample_ddp_mf.py   \
    --config configs/mf/sampling/DiTDHXL-DINOv2-B.yaml   \
    --sample-dir samples   \
    --precision fp32   \
    --label-sampling equal   \
    --per-proc-batch-size 8   \
    --mf-step 1

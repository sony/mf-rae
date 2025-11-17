torchrun --standalone --nnodes=1 --nproc_per_node=2   src/sample_ddp_mf_512.py   \
    --config configs/mf/sampling/DiTDHXL-DINOv2-B-512.yaml   \
    --sample-dir samples   \
    --precision fp32   \
    --label-sampling equal   \
    --per-proc-batch-size 8   \
    --only-npz 0 \
    --mf-step 1


torchrun --standalone --nnodes=1 --nproc_per_node=1   src/sample_ddp_mf_512.py   \
    --config configs/mf/sampling/DiTDHXL-DINOv2-B-512.yaml   \
    --sample-dir samples   \
    --precision fp32   \
    --label-sampling equal   \
    --per-proc-batch-size 8   \
    --only-npz 1 \
    --mf-step 1
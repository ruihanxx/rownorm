torchrun --standalone --nproc_per_node=4 train.py \
  --model_name small --dtype float32\
  --batch_size 64 --block_size 512 --grad_accum_steps 16 \
  --opt adamw \
  --lr 8e-4 --min_lr 8e-5 \
  --weight_decay 0.1 \
  --warmup_iters 1000 --lr_decay_iters 40000 --max_iters 40000 \
  --use_fan_scaling 0 --p_exp 1 \
  --log_every 1000 --eval_iters 50 --eval_interval 100
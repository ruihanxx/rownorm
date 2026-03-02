torchrun --standalone --nproc_per_node=4 train.py \
  --model_name small --dtype float32 \
  --batch_size 64 --block_size 512 --grad_accum_steps 16 \
  --opt moonlight \
  --momentum 0.95 --nesterov_mom 0 \
  --lr 2e-3 --min_lr 2e-4 \
  --weight_decay 0.1 --use_fan_scaling 1 \
  --warmup_iters 1000 --lr_decay_iters 40000 --max_iters 40000 \
  --p_exp 2 \
  --log_every 1000 --eval_iters 50 --eval_interval 100
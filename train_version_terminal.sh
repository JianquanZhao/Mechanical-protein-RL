export CUDA_VISIBLE_DEVICES=3
RUN_DIR=/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/full_esm_terminal_fully-change-max-steps-24
mkdir ${RUN_DIR}

setsid env PYTHONUNBUFFERED=1 /home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python training.py \
  --pdb-dir /mnt/nas/jianquanzhao/data/mprl/pdbs/cath \
  --train-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/train_index.txt \
  --val-index /mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/indices/val_index.txt \
  --output-dir "$RUN_DIR" \
  --mode single --device auto \
  --epochs 32  --train-batch-size 8 --max-steps 24 \
  --observation-encoder esm2 --embedding-dim 1280 --esm2-device auto \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --no-minimize --continue-on-update-error \
  --replay-warmup-size 512 --micro-batch-size 4 --gradient-accumulation-steps 4 \
  --replay-capacity 50000 --target-sync-interval 500 \
  --checkpoint-every 240 --plot-every-episodes 240 --rolling-window 100 \
  --validate-every 240 --validation-episodes 16 --enable-tensorboard \
  --terminal-reward-artifact params/hbond_random_forest.joblib \
  --terminal-predicted-pdb-dir /mnt/nas/jianquanzhao/data/mprl/outputs/evaluate-mechanical-predictor/pdb-colab/CATH-test \
  --log-level INFO --log-every-steps 25 --no-resume-logs --no-save-candidates \
  > "$RUN_DIR/train_stdout.log" 2>&1 < /dev/null &

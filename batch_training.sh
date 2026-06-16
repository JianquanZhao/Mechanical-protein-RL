python training.py --mode single --device cuda:2 \
  --episodes 2 --max-steps 10000 \
  --output-dir /tmp/mprl-training-gpu2-smoke \
  --replay-warmup-size 1 \
  --micro-batch-size 2 \
  --gradient-accumulation-steps 2 \
  --checkpoint-every 2 --no-resume-logs

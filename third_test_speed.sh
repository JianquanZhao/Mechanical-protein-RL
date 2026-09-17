cd /mnt/nas/jianquanzhao/gits/Mechanical-protein-RL
PYTHON="/home/jianquanzhao/anaconda24/envs/mprl-vgpt/bin/python"
MODEL_PATH="/mnt/nas/jianquanzhao/data/mprl/outputs/train/add_terminal_reward_train/async_a24_esm8_f8_g1_prioritized_n3_stepx0.025_terminalx8.0_20260821_093310/checkpoints/agent_00302880.pt"
INPUT_DIR=${1:-'/mnt/nas/jianquanzhao/third_test/nianfu/input/clean/'}
OUTPUT_DIR=${2:-'/mnt/nas/jianquanzhao/third_test/nianfu/output/rl'}

$PYTHON run_rl.py \
  --model-path $MODEL_PATH \
  --input-type structure \
  --input-dir $INPUT_DIR \
  --output-dir $OUTPUT_DIR \
  --updates  24 \
  --device cuda:3 \
  --seed  1024 \
  --overwrite \
  --fail-fast \
  --esm-model-dir /mnt/data1/home/jianquanzhao/data/models/esm \
  --log-level INFO

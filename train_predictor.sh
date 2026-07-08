python model/reward_module/mechanical-properties-predictor/train.py \
  --csv-path /mnt/data1/home/jianquanzhao/data/cath/filtered_All_Mechanical_Vectors_cath_all_fasta_results.csv \
  --split-method similarity \
  --similarity-threshold 0.6  \
  --output-dir outputs/mechanical_property_predictor/$1 \
  --embedding-dim 1280 \
  --embedding-cache-dir outputs/mechanical_property_predictor/random_esm2_1280/embedding_cache \
  --patience 1000 \
  --esm2-device cuda:0 \
  --device cuda:0 \
  --epochs 1000 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --dropout 0.2 \
  --weight-decay 1e-3 \
  --target-transform log1p \
  --loss huber \
  --huber-beta 1.0 \
  --strength-loss-weight 3.0 \
  --toughness-loss-weight 1.0 \
  --enable-tensorboard



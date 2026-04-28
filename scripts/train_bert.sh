# Other choices: [sst2, stsb, mrpc, qqp, mnli, qnli, rte, wnli]

python main.py \
  --model_name_or_path='bert-base-uncased' \
  --decoder_cls_last_token=0 \
  --task=cola \
  --max_length=128 \
  --adaptive_length=0 \
  --epochs=3 \
  --batch_size=32 \
  --eval_batch_size=64 \
  --lr=2e-4 \
  --weight_decay=0.0 \
  --seed=42 \
  --verbose=1 \
  --pooling_method=glot \
  --gnn_type=gat \
  --scorer_hidden=128 \
  --gat_hidden_dim=256 \
  --num_layers=2 \
  --jk_mode=cat \
  --graph_adj=threshold \
  --tau=0.8 \
  --proj_dim=256 \
  --precompute_hidden_states=1 \
  --override_precompute=0 \
  --finetune_backbone=0 
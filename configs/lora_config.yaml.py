base_model_name_or_path: distilbert/distilgpt2
output_dir: ./lora-pubmed-distilgpt2

data:
  split: train[:2%]
  max_length: 256

lora:
  r: 4
  alpha: 16
  dropout: 0.05
  target_modules: ["c_attn"]

training:
  batch_size: 4
  num_epochs: 3
  learning_rate: 5e-4
  logging_steps: 10
  save_steps: 100
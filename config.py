CONFIG = {
    "model_name": "distilbert/distilgpt2",
    "output_dir": "./lora-pubmed-distilgpt2",
    "max_length": 256,
    "prompt_length": 128,
    "temperature": 0.8,
    "top_k": 50,
    "num_return_sequences": 1,
    "train_subset": "train[:2%]",
    "num_train_epochs": 3,
    "train_batch_size": 4,
    "learning_rate": 5e-4,
    "logging_steps": 10,
    "save_steps": 100
}

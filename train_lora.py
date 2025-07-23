# train_lora.py
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from config import CONFIG

# === 1. Load Tokenizer & Model ===
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"])

# === 2. Apply LoRA ===
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# === 3. Load and Format Dataset ===
dataset = load_dataset("ccdv/pubmed-summarization", split=CONFIG["train_subset"])
dataset = dataset.map(lambda x: {"text": x["article"]})

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=CONFIG["max_length"])

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === 4. Training Arguments ===
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["train_batch_size"],
    num_train_epochs=CONFIG["num_train_epochs"],
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    learning_rate=CONFIG["learning_rate"],
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# === 5. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# === 6. Train and Save ===
trainer.train()
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

print("✅ Training complete. Model saved to", CONFIG["output_dir"])
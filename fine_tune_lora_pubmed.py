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

# === 1. Load Tokenizer & Model ===
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ Fix: Add pad_token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# === 2. Apply LoRA ===
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["c_attn"],  # Key attention module for GPT2/DistilGPT2
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

#
#This tells the PEFT library:

#Parameter	What it Means
#r=4	Use rank-4 low-rank adapters (A and B matrices)
#lora_alpha=16	Scale the LoRA update by alpha/r = 4.0
#target_modules=["c_attn"]	Insert adapters into the c_attn layer of GPT2
#lora_dropout=0.05	Dropout to regularize the adapters
#bias="none"	Do not adapt the bias terms
#task_type=CAUSAL_LM	This is a language generation model
#model = get_peft_model(model, lora_config)

# === 3. Load Dataset ===
dataset = load_dataset("ccdv/pubmed-summarization", split="train[:2%]")  # small subset to test

# === 4. Format Dataset ===
def format(example):
    return {"text": example["article"]}

dataset = dataset.map(format)

# === 5. Tokenize ===
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=256)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# === 6. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./lora-pubmed-distilgpt2",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    learning_rate=5e-4,
    fp16=torch.cuda.is_available(),  # Enable fp16 if on GPU
    report_to="none",
)

# === 7. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# === 8. Train the Model ===
trainer.train()

# === 9. Save the Model ===
model.save_pretrained("./lora-pubmed-distilgpt2")
tokenizer.save_pretrained("./lora-pubmed-distilgpt2")

print("✅ Training complete. Model saved to ./lora-pubmed-distilgpt2")
#python3 fine_tune_lora_pubmed.py

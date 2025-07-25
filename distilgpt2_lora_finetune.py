import os
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# Load YAML config
def load_config(path="configs/lora_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def tokenize_function(example, tokenizer, max_length):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

def main():
    cfg = load_config()
    base_model_path = cfg["base_model_name_or_path"]

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # Apply LoRA configuration
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    # Load dataset (PubMed summarization by default)
    dataset = load_dataset("ccdv/pubmed-summarization", split=cfg["data"]["split"])
    dataset = dataset.map(lambda x: {"text": x["article"]})

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, cfg["data"]["max_length"]),
        batched=True
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["num_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("\n✅ Training complete. LoRA model saved to", cfg["output_dir"])

if __name__ == "__main__":
    main()
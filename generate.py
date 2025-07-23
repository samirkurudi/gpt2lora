# generate.py
# generate.py
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import CONFIG

# Load Tokenizer
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Models
print("📦 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"])

print("📦 Loading fine-tuned model...")
fine_tuned_model = AutoModelForCausalLM.from_pretrained(CONFIG["output_dir"])

# Generate text from a model
def generate(model, prompt):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                           max_length=CONFIG["prompt_length"])

        if inputs.input_ids.shape[1] == 0:
            return "❌ Invalid prompt: produced empty input_ids."

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=CONFIG["max_length"],
                num_return_sequences=CONFIG["num_return_sequences"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        if output_ids is None or output_ids.shape[0] == 0:
            return "❌ No output generated."

        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    except Exception as e:
        return f"❌ Generation error: {str(e)}"

# Command line interface
if __name__ == "__main__":
    while True:
        prompt = input("\n📝 Enter a prompt (or type 'exit'): ")
        if prompt.lower() == 'exit':
            break

        output_json = {
            "prompt": prompt,
            "fine_tuned_output": generate(fine_tuned_model, prompt),
            "base_output": generate(base_model, prompt)
        }

        print(json.dumps(output_json, indent=2))  # 🎯 Clean JSON-formatted output

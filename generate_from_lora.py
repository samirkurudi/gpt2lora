from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("./lora-pubmed-distilgpt2")

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "./lora-pubmed-distilgpt2")

# Optional: fix pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prompt the model
prompt = "Recent developments in oncology have shown that"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

# Test prompt
prompt = "The future of medicine is"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_new_tokens=50)

# Print output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
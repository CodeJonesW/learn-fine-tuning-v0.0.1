from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datetime import datetime
import os

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("trained-gpt2").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("trained-gpt2")
model.eval()

# Prompt user
prompt = input("Enter a prompt: ")

# Tokenize with attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

# Generate
with torch.no_grad():
    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=300,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Output folder
output_dir = "generated_text"
os.makedirs(output_dir, exist_ok=True)

# Timestamped filename
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"{output_dir}/generated_{timestamp}.txt"

# Save to file
with open(filename, "w", encoding="utf-8") as f:
    f.write(generated_text)

print(f"\n✅ Text generated and saved to {filename}")

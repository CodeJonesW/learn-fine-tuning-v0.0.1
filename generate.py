from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from datetime import datetime
import os

# Load instruction-tuned model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").cuda()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model.eval()

# Prompt user
prompt = input("Enter a prompt: ")

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print('generated text:', generated_text)

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

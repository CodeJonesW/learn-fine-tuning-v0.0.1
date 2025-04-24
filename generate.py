from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from datetime import datetime
import os

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

# Prompt user
prompt = "What is the name of the narrator in *To Kill a Mockingbird*?"

# formatted = f"""### Instruction:
# {prompt}

# ### Input:

# ### Response:
# """
# print('2', formatted)
formatted = prompt
inputs = tokenizer(formatted, return_tensors="pt", padding=False, truncation=False).to("cuda")


# Generate
with torch.no_grad():
    output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode
print('output', output)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print('generated text', generated_text)

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

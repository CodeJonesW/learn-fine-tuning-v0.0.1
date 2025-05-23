from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os

def load_instruction_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Split on '---' blocks
    samples = [sample.strip() for sample in raw_text.split('---') if sample.strip()]
    formatted = []

    for s in samples:
        # Use raw string split and cleanup
        instruction = input_text = response = ""
        for section in s.split("###"):
            if section.strip().startswith("Instruction:"):
                instruction = section.strip().replace("Instruction:", "").strip()
            elif section.strip().startswith("Input:"):
                input_text = section.strip().replace("Input:", "").strip()
            elif section.strip().startswith("Response:"):
                response = section.strip().replace("Response:", "").strip()

        # Only include examples with a valid response
        if response:
            full_prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{response}"
            formatted.append({"text": full_prompt})

    return Dataset.from_list(formatted)


# Load dataset
dataset = load_instruction_dataset("training-data/tkam_finetune_dataset.txt")
print('loading dataset', dataset)

# Load GPT-2 and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model = model.cuda()

# Tokenize data
def tokenize(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

print("🔍 Sample tokenized text:")
print(tokenized[0])
print(tokenizer.decode(tokenized[0]["input_ids"]))

# Training config
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_dir="./logs",
    save_total_limit=1,
    fp16=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# Train!
trainer.train()

# Save model
model.save_pretrained("trained-gpt2-instruct")
tokenizer.save_pretrained("trained-gpt2-instruct")

print("✅ Fine-tuning complete.")

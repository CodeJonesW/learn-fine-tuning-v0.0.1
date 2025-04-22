from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import Dataset
import torch
import os
import re

# ---------- STEP 1: Read and clean markdown ----------

def clean_markdown(text):
    # Remove frontmatter (--- blocks)
    text = re.sub(r"^---[\s\S]*?---", "", text)
    # Remove image markdown: ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Strip markdown links but keep the text: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    print(text)
    return text.strip()

def load_markdown_files_as_dataset(path):
    examples = []
    for fname in os.listdir(path):
        if fname.endswith(".md"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned_text = clean_markdown(raw_text)
                examples.append({"text": cleaned_text})
    return Dataset.from_list(examples)

dataset = load_markdown_files_as_dataset("./blog-data")

# ---------- STEP 2: Load model & tokenizer ----------

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
model = model.cuda()

# ---------- STEP 3: Tokenize ----------

def tokenize_fn(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ---------- STEP 4: Training ----------

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    logging_dir="./logs",
    save_total_limit=1,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

trainer.train()

# ---------- STEP 5: Save model ----------

model.save_pretrained("trained-gpt2")
tokenizer.save_pretrained("trained-gpt2")

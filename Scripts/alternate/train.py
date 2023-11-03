import torch.cuda
from datasets import load_dataset
from transformers import LlamaTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
dataset = load_dataset("Open-Orca/OpenOrca")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("llama")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = r"C:\Users\suran\Desktop\School\1_UNIVERSITY\BENNETT\5thSem\SBUH\llama-2-13b-chat.Q4_K_M (1).gguf"


dataset = load_dataset("Open-Orca/OpenOrca", "1M-GPT4-Augmented.parquet")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


base_model = AutoModelForCausalLM.from_pretrained("Yukang/Llama-2-7b-longlora-100k-ft")
refined_model = "llama-2-7b-selbidesumma"

llama_tokenizer = AutoTokenizer.from_pretrained("Yukang/Llama-2-7b-longlora-100k-ft", trust_remote_code = True)


'''  # Quantization Config
  quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
  )
'''

''' LoRA COnfiG '''
config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query","value"]
    r=8,
    bias="lora_only",
    task_type="CAUSAL_LM"
    modules_to_save=["decode_head"],
)

lora_model = get_peft_model(base_model, config)


train_params = TrainingArguments(
    output_dir="/content/gdrive/My\ Drive/Colab\ Notebooks/models",
    num_train_epochs=3,
    gradient_accumulation_steps=1,
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)


fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=X_train,
    tokenizer=llama_tokenizer,
    args=train_params
    eval_dataset=X_test
    compute_metrics=compute_metrics
)

fine_tuning.train()

fine_tuning.model.save_pretrained(refined_model)
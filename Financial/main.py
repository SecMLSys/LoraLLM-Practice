import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from peft import LoraConfig
from trl import SFTTrainer

## load model
model_name = 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

## load data
dataset = load_dataset('financial_phrasebank.py',
                       name='sentences_75agree')['train']

dataset = dataset.train_test_split(test_size=0.2)
train_data, eval_data = dataset['train'], dataset['test']

## configure Lora fine-tuner
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

## configure hyperparameters
training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch"
    )

trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=1024,
    )

trainer.train()

trainer.save_model('models/financial-sentiment')
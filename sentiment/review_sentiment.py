# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 17:07:31 2025

@author: User
"""

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

# (디버깅) CUDA 에러 동기화
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import (
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate

# 1) 원본 CSV → Hugging Face Dataset 로드
data_files = {"train": "train_reviews.csv", "test": "test_reviews.csv"}
raw_ds = load_dataset("csv", data_files=data_files)

# 'Review'와 'labels'만 남기기
raw_ds = raw_ds.map(lambda ex: {"Review": ex["Review"], "labels": ex["labels"]})

# 2) 토크나이저 & 모델 준비
"""
# Hugging Face에서 사전학습된 BERT 모델 다운로드 (한국어 감성 분석)
model_name = "klue/bert-base"
bert_sentiment_model
# 모델 및 토크나이저 다운로드
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 모델 저장 (로컬 캐싱)
model.save_pretrained("./bert_sentiment_model")
tokenizer.save_pretrained("./bert_sentiment_model")
"""

"""
# 로컬에 저장된 모델 로드
model_path = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
"""
model_name = "monologg/kobert"
tokenizer  = BertTokenizerFast.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ↓↓↓ 여기에 드롭아웃 적용 ↓↓↓
dropout_prob = 0.3   # 원하는 드롭아웃 확률
# (1) config에도 반영
model.config.hidden_dropout_prob = dropout_prob
# (2) 실제 드롭아웃 레이어 교체
model.dropout = nn.Dropout(dropout_prob)

# 3) 토크나이징
def tokenize_fn(examples):
    tokens = tokenizer(
        examples["Review"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = examples["labels"]
    return tokens

tokenized_ds = raw_ds.map(tokenize_fn, batched=True)

# 4) (디버깅) 토큰 ID 범위 확인
vocab_size = tokenizer.vocab_size
max_id = max(max(seq) for seq in tokenized_ds["train"]["input_ids"])
assert max_id < vocab_size, f"Token ID {max_id} ≥ vocab_size {vocab_size}"

# 5) DataCollator & 지표 로드
data_collator = DataCollatorWithPadding(tokenizer)
accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    pred_labels = preds.argmax(axis=-1)
    return {
        "eval_accuracy": accuracy.compute(predictions=pred_labels, references=labels)["accuracy"],
        "eval_f1":       f1.compute(predictions=pred_labels, references=labels, average="weighted")["f1"]
    }

# 6) 콘솔 출력용 콜백
class ConsoleCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\n===== Evaluation Results (epoch {state.epoch:.2f}) =====")
        print(f"  eval_loss:     {metrics['eval_loss']:.4f}")
        print(f"  eval_accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"  eval_f1:       {metrics['eval_f1']:.4f}\n")

# 7) TrainingArguments 설정 (batch=64, lr=1e-5, epochs=30, early stopping)
training_args = TrainingArguments(
    output_dir="./kobert_run",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=30,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
)

# 8) Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset= tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        ConsoleCallback(),
        EarlyStoppingCallback(early_stopping_patience=5)
    ]
)

# 9) 학습 시작
print("▶ Training begins")
trainer.train()

# 10) 로그 저장 & 그래프 그리기
logs = trainer.state.log_history
metrics = [
    {
        "epoch": e.get("epoch"),
        "eval_loss": e.get("eval_loss"),
        "eval_accuracy": e.get("eval_accuracy"),
        "eval_f1": e.get("eval_f1")
    }
    for e in logs if "eval_loss" in e
]
logs_df = pd.DataFrame(metrics)
logs_df.to_csv("training_logs.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10,6))
plt.plot(logs_df["epoch"], logs_df["eval_loss"],     marker='o', label="Validation Loss")
plt.plot(logs_df["epoch"], logs_df["eval_accuracy"], marker='o', label="Validation Accuracy")
plt.plot(logs_df["epoch"], logs_df["eval_f1"],       marker='o', label="Validation F1")
plt.title("Validation Metrics per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score / Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_metrics_plot.png", dpi=300)
plt.show()

# 11) Fine-tuned 모델 & 토크나이저 저장
save_dir = "./kobert_finetuned_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✔ Fine-tuned model & tokenizer saved in '{save_dir}'")

# raw data 0,1,2 라벨링 -> 2제외

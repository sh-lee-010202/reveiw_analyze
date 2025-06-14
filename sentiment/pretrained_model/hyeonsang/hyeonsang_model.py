# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:22:29 2025

@author: User
"""

import time
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import load_dataset
import evaluate

# 1) 모델 이름 설정
model_name = "HyeonSang/kobert-sentiment"

# 2) 원본 CSV → Hugging Face Dataset 로드
data_files = {"train": "train_reviews.csv", "test": "test_reviews.csv"}
raw_ds = load_dataset("csv", data_files=data_files)

# 3) 'Review'와 'labels'만 남기기
raw_ds = raw_ds.map(lambda ex: {"Review": ex["Review"], "labels": ex["labels"]})

# 4) 토크나이저 & 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 5) 토크나이징 함수
def tokenize_fn(examples):
    tokens = tokenizer(
        examples["Review"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = examples["labels"]
    return tokens

# 6) batched 매핑으로 토크나이징
tokenized_ds = raw_ds.map(tokenize_fn, batched=True)

# 7) DataCollator 및 지표 로드
data_collator = DataCollatorWithPadding(tokenizer)
accuracy = evaluate.load("accuracy")
f1       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1":       f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# 8) Early-stopping 및 콘솔 출력 콜백
class ConsoleCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"\n⎯⎯⎯ Evaluation (epoch {state.epoch:.2f}) ⎯⎯⎯")
        print(f"  loss:     {metrics['eval_loss']:.4f}")
        print(f"  accuracy: {metrics['eval_accuracy']:.4f}")
        print(f"  f1:       {metrics['eval_f1']:.4f}\n")

# 9) TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./hs_kobert_run",
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

# 10) Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset= tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[ConsoleCallback(), EarlyStoppingCallback(early_stopping_patience=3)]
)

# 11) 학습 실행
print("▶ Fine-tuning HyeonSang/kobert-sentiment 시작")
trainer.train()

# 12) 로그 추출 & 저장
logs = trainer.state.log_history
metrics = [
    {
        "epoch": e["epoch"],
        "eval_loss":     e["eval_loss"],
        "eval_accuracy": e["eval_accuracy"],
        "eval_f1":       e["eval_f1"]
    }
    for e in logs if "eval_loss" in e
]
logs_df = pd.DataFrame(metrics)
logs_df.to_csv("hs_training_logs.csv", index=False, encoding="utf-8-sig")

# 13) 그래프 시각화
plt.figure(figsize=(10,6))
plt.plot(logs_df["epoch"], logs_df["eval_loss"],     marker='o', label="Loss")
plt.plot(logs_df["epoch"], logs_df["eval_accuracy"], marker='o', label="Accuracy")
plt.plot(logs_df["epoch"], logs_df["eval_f1"],       marker='o', label="F1 Score")
plt.title("HyeonSang/kobert-sentiment Fine-tuning Metrics")
plt.xlabel("Epoch")
plt.ylabel("Score / Loss")
plt.legend()
plt.grid(True)
plt.savefig("hs_training_metrics.png", dpi=300)
plt.show()

# 14) 최종 모델 & 토크나이저 저장
save_dir = "./hs_kobert_finetuned"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"✔ 모델과 토크나이저가 '{save_dir}'에 저장되었습니다.")

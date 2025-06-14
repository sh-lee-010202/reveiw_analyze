# -*- coding: utf-8 -*-
"""
Created on Mon May 12 22:58:09 2025

@author: User
"""
import os
import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    hamming_loss
)
import types

# 1) 데이터 로드 및 전처리
df = pd.read_csv("final_review_labels_with_text.csv", encoding="utf-8-sig") \
       .rename(columns={"리뷰본문":"text"})
CATS = ["맛","가격","서비스","분위기"]

# 2) Dataset 생성 및 split
ds = Dataset.from_pandas(df[["text"] + CATS])
split = ds.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = split["train"], split["test"]

# 3) 토크나이저 & config
MODEL_NAME = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True
)
config = AutoConfig.from_pretrained(
    MODEL_NAME,
    problem_type="multi_task_classification"
)

# 4) 멀티태스크 모델 정의
class MultiTaskBert(nn.Module):
    def __init__(self, model_name, hidden_dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        hs = self.bert.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.heads = nn.ModuleDict({cat: nn.Linear(hs, 3) for cat in CATS})
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.bert(input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        logits = {cat: head(pooled) for cat, head in self.heads.items()}

        loss = None
        if labels is not None:
            losses = [ self.loss_fct(logits[cat], labels[:, i]) 
                       for i, cat in enumerate(CATS) ]
            loss = torch.stack(losses).mean()

        # pack logits into shape [B,4,3]
        stacked = torch.stack([logits[cat] for cat in CATS], dim=1)
        return {"loss": loss, "logits": stacked}

model = MultiTaskBert(MODEL_NAME).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# 5) 토크나이징
def preprocess(example):
    toks = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    toks["labels"] = torch.tensor(
        [example[cat] for cat in CATS], dtype=torch.long
    )
    return toks

train_ds = train_ds.map(preprocess, remove_columns=CATS+["text"])
eval_ds  = eval_ds.map(preprocess,  remove_columns=CATS+["text"])

data_collator = DataCollatorWithPadding(tokenizer)

# 6) metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred                  # logits: (B,4,3), labels: (B,4)
    preds = logits.argmax(axis=-1)              # (B,4)
    labels = labels.astype(int)
    exact_match = float((preds == labels).all(axis=1).mean())
    ham_loss    = hamming_loss(labels.flatten(), preds.flatten())
    y_true, y_pred = labels.flatten(), preds.flatten()
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_cat = {
        f"f1_{cat}": f1_score(labels[:,i], preds[:,i], average="weighted", zero_division=0)
        for i, cat in enumerate(CATS)
    }
    return {
        "exact_match": exact_match,
        "hamming_loss": ham_loss,
        "micro_f1": micro,
        "macro_f1": macro,
        **f1_per_cat
    }

# 7) TrainingArguments & Trainer (tokenizer 인자 제거)
output_dir = "./kobert_senti_categ"
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    logging_steps=200,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8) 학습
trainer.train()

# 9) 최종 메트릭 및 epoch별 로그 저장
final_metrics = trainer.evaluate()
history = [
    {k: v for k,v in entry.items() if k in ("epoch","eval_loss","eval_exact_match","eval_macro_f1")}
    for entry in trainer.state.log_history if "eval_loss" in entry
]

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, ensure_ascii=False, indent=2)
with open(os.path.join(output_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

# 10) 모델 저장
trainer.save_model(output_dir)

# 11) 토크나이저 수동 저장
#    save_pretrained() 호출 없이, filename_prefix 오류 회피
orig_save_vocab = tokenizer.save_vocabulary
def _save_vocab_no_prefix(self, save_directory, *args, **kwargs):
    return orig_save_vocab(save_directory)
tokenizer.save_vocabulary = types.MethodType(_save_vocab_no_prefix, tokenizer)

# 어휘 파일만 저장
tokenizer.save_vocabulary(output_dir)
# tokenizer_config.json 작성
with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
    json.dump(
        {"tokenizer_class": tokenizer.__class__.__name__, **tokenizer.init_kwargs},
        f, ensure_ascii=False, indent=2
    )

print(f"✅ 학습 완료 및 모델·토크나이저(수동) 저장: {output_dir}")






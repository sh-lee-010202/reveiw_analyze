# -*- coding: utf-8 -*-
"""
Created on Fri May 16 16:28:59 2025

@author: User
"""

import os
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel
from safetensors.torch import load_file

# 설정
OUTPUT_DIR = "./kobert_senti_categ"
DEVICE     = torch.device("cpu")
CATS       = ["맛", "가격", "서비스", "분위기"]

# 1) checkpoint에 저장된 config & tokenizer 로드
config    = AutoConfig.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True, trust_remote_code=True)
# config.vocab_size를 실제 tokenizer 크기로 맞춰 줌
config.vocab_size = tokenizer.vocab_size

# 2) 학습 때와 동일한 모델 정의
class MultiTaskBert(nn.Module):
    def __init__(self, checkpoint_dir, config):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            checkpoint_dir,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        hs = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.heads = nn.ModuleDict({cat: nn.Linear(hs, 3) for cat in CATS})

    def forward(self, input_ids, attention_mask=None):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.pooler_output)
        # (batch, 4, 3) logits 반환
        return torch.stack([self.heads[cat](pooled) for cat in CATS], dim=1)

# 3) 모델 인스턴스화 및 가중치 로드
model = MultiTaskBert(OUTPUT_DIR, config).to(DEVICE)
state_dict = load_file(os.path.join(OUTPUT_DIR, "model.safetensors"), device="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# (선택) CPU 동적 양자화로 메모리/연산 절감
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# 4) 추론 함수 (label_map 없이, 정수 0/1/2 반환)
def predict(texts, max_length: int = 128):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        # logits.shape == (batch_size, 4, 3)
        preds = logits.argmax(dim=-1).cpu().numpy()  # (batch,4)

    # 카테고리별 정수 레이블 반환
    return [
        {cat: int(pred_row[i]) for i, cat in enumerate(CATS)}
        for pred_row in preds
    ]

# 5) 사용 예시
if __name__ == "__main__":
    samples = [
    # 1) 맛(taste)
    "스테이크가 육즙 가득하고 적당히 구워져 정말 맛있었습니다.",
    # 2) 가격(price)
    "메뉴 가격이 합리적이고 가성비가 뛰어나서 만족스러웠습니다.",
    # 3) 서비스(service)
    "직원분들이 상냥하게 응대해 주시고, 요청 사항도 즉시 처리해 주셨어요.",
    # 4) 분위기(atmosphere)
    "조명이 은은하고 인테리어가 세련되어 분위기가 매우 고급스러웠습니다."
    ]
    for out in predict(samples):
        print(out)


# 0:부정, 1: 긍정, 2: 해당없음







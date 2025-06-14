# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 14:43:43 2025

@author: User
"""

import torch
import pandas as pd
from transformers import BertTokenizer
from datasets import Dataset

# ✅ 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")

# ✅ 데이터 로드
df_train = pd.read_csv("train_reviews.csv")
df_test = pd.read_csv("test_reviews.csv")

# ✅ 'labels' 열 이름 변경 (Trainer가 labels를 요구함)
df_train = df_train.rename(columns={"Rating": "labels"})
df_test = df_test.rename(columns={"Rating": "labels"})

# ✅ 데이터셋 변환
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# ✅ 데이터 토큰화 함수 정의
def tokenize_function(examples):
    return tokenizer(examples["Review"], padding="max_length", truncation=True, max_length=128)

# ✅ map() 적용 후 캐싱된 데이터셋을 저장
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# ✅ 미리 처리한 데이터셋 저장
train_dataset.save_to_disk("train_dataset")
test_dataset.save_to_disk("test_dataset")

print("train & test 데이터셋이 저장되었습니다.")

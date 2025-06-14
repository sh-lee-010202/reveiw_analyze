# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 15:53:26 2025

@author: User
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# 1) TSV 불러오기 (탭 '\t' 구분자)
df = pd.read_csv("kr3.tsv", sep="\t")

# 2) 중립(Rating==2) 샘플 제거
df = df[df["Rating"] != 2]

# 3) Review, Rating 결측 제거
df = df.dropna(subset=["Review", "Rating"])

# 4) 부정/긍정 분리
df_neg = df[df["Rating"] == 0]
df_pos = df[df["Rating"] == 1]

print(f"원본 클래스 분포 -> 부정: {len(df_neg)}, 긍정: {len(df_pos)}")

# 5) 부정 데이터 오버샘플링(증강) → 긍정 데이터 수에 맞추기
df_neg_aug = resample(
    df_neg,
    replace=True,                   # 중복 허용
    n_samples=len(df_pos),          # 긍정 샘플 수만큼
    random_state=42
)

print(f"증강 후 부정 샘플 수: {len(df_neg_aug)} (긍정과 동일)")

# 6) 긍정 + 증강된 부정 합치고 shuffle
df_balanced = pd.concat([df_pos, df_neg_aug]) \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

print(f"최종 균형 클래스 분포 →\n{df_balanced['Rating'].value_counts()}")

# 7) Train/Test Split (80/20), stratify 적용
train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["Rating"],
    random_state=42
)

# 8) 컬럼명 변경 (Trainer가 기대하는 'labels'로)
train_df = train_df.rename(columns={"Rating": "labels"})
test_df  = test_df.rename(columns={"Rating": "labels"})

# 9) CSV로 저장
train_df[["Review", "labels"]].to_csv("train_reviews.csv", index=False, encoding="utf-8-sig")
test_df[["Review", "labels"]].to_csv("test_reviews.csv", index=False, encoding="utf-8-sig")

print(f"훈련 데이터: {len(train_df)}개 (부정:{sum(train_df['labels']==0)}, 긍정:{sum(train_df['labels']==1)})")
print(f"테스트 데이터: {len(test_df)}개 (부정:{sum(test_df['labels']==0)}, 긍정:{sum(test_df['labels']==1)})")

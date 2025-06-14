# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 20:36:52 2025

@author: User
"""

import re
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from konlpy.tag import Mecab

# ✅ 형태소 분석기 초기화 (MeCab 설치 경로 지정)
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# ✅ GPU 사용 여부 확인
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ✅ T5 모델 로드
model = T5ForConditionalGeneration.from_pretrained("j5ng/et5-typos-corrector").to(device)
tokenizer = T5Tokenizer.from_pretrained("j5ng/et5-typos-corrector")

def clean_texts(text_list):
    cleaned_texts = []
    
    for text in text_list:
        # ✅ None 또는 NaN 값 처리
        if text is None or (isinstance(text, float) and pd.isna(text)):
            text = ""  # 빈 문자열로 변환
        elif isinstance(text, (list, dict, tuple, set)):  # ✅ 리스트, 딕셔너리 등 처리 불가한 자료형 예외 처리
            continue  # 해당 항목은 스킵
        else:
            text = str(text)  # 문자열로 변환

    for text in text_list:
        text = re.sub(r"[!?,.]", " ", text)  # ?, !를 삭제
        text = re.sub(r"\s*\n\s*", " ", text)  # '\n'을 공백으로 변환
        text = re.sub(r"[^가-힣a-zA-Z0-9., ]", "", text)  # 한글, 영어, 숫자, 문장부호(.,)만 남김
        text = re.sub(r"\s+", " ", text).strip()  # 공백 정리
        cleaned_texts.append(text)

    return cleaned_texts

def correct_spelling(text_list):
    corrected_texts = []
    
    with torch.no_grad():  # ✅ 추론 시 torch.no_grad() 적용
        for text in text_list:
            # ✅ 입력 문장 인코딩 & GPU로 이동
            input_encoding = tokenizer("맞춤법을 고쳐주세요: " + text, return_tensors="pt").to(device)

            # ✅ T5 모델 출력 생성 (GPU 사용)
            output_encoding = model.generate(
                input_ids=input_encoding.input_ids,
                attention_mask=input_encoding.attention_mask,
                max_length=128,
                num_beams=5,
                early_stopping=True,
            )

            # ✅ 출력 문장 디코딩
            corrected_text = tokenizer.decode(output_encoding[0], skip_special_tokens=True)
            corrected_texts.append(corrected_text)

    return corrected_texts

def split_sentences(text):
    """
    한국어 문장을 자연스럽게 분리하는 정규 표현식 기반 함수 (모든 요소 유지)
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):  # NaN 체크
        return [""]  # 빈 리스트 반환 (오류 방지)

    pattern = r"([.?!])\s+|((?:다|요|죠|니다|데|는데|하고|지만|라서|이며|면서|고))\s+"
    parts = re.split(pattern, text)

    sentences = []
    temp_sentence = ""

    for part in parts:
        if part is None or part.strip() == "":
            continue

        temp_sentence += part.strip()

        if re.match(r"[.?!]$", part) or part in ["다", "요", "죠", "니다", "데", "는데", "하고", "지만", "라서", "이며", "면서", "고"]:
            sentences.append(temp_sentence.strip())
            temp_sentence = ""

    if temp_sentence:
        sentences.append(temp_sentence.strip())

    return sentences

def split_sentences_list(text_list):
    """
    여러 개의 리뷰 텍스트를 문장 단위로 분리하는 함수
    """
    result = []
    
    for text in text_list:
        sentences = split_sentences(text)
        result.append(sentences)
    
    return result

# ✅ CSV 파일 로드
file_path = r"C:\Users\User\OneDrive\바탕 화면\Develop\review_analyze\reviews_whole.csv"
df = pd.read_csv(file_path, encoding="utf-8")  

# ✅ 3번째 열(Column3)의 NaN, None 값이 있는 행 삭제
df_cleaned = df.dropna(subset=[df.columns[2]])
df_cleaned.to_csv("review_raw.csv", index=False, encoding="utf-8-sig")

# ✅ 3번째 열 추출 및 리스트 변환
reviews = df_cleaned.iloc[:, 2].tolist()

# ✅ 텍스트 전처리
clean_result = clean_texts(reviews)

# ✅ 맞춤법 교정 (GPU 사용)
correct_result = correct_spelling(clean_result)

# ✅ 문장 분리
split_results = split_sentences_list(correct_result)

# ✅ 결과 출력 (최대 5개만 출력)
for i, sentences in enumerate(split_results):
    if i >= 5:
        break
    print(f"리뷰 {i+1}: {sentences}")



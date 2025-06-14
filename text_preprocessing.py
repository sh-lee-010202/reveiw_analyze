# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 00:04:37 2025

@author: User
"""

import re
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from konlpy.tag import Mecab

# ✅ GPU 사용 여부 확인
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ✅ T5 모델 로드
model = T5ForConditionalGeneration.from_pretrained(
    "j5ng/et5-typos-corrector", cache_dir="./model_cache"  # ✅ 모델 캐싱 사용
).to(device)
model = model.half()  # ✅ Half precision 적용 (속도 향상)
model = torch.compile(model)  # ✅ PyTorch 2.0 이상에서 JIT 컴파일 적용 (속도 증가)

tokenizer = T5Tokenizer.from_pretrained(
    "j5ng/et5-typos-corrector", cache_dir="./model_cache"  # ✅ 토크나이저 캐싱 사용
)

# ✅ 형태소 분석기 초기화 (MeCab 설치 경로 지정)
mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')

# ✅ GPU 상태 출력
print("CUDA 사용 가능 여부:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("사용 중인 GPU:", torch.cuda.get_device_name(0))
    print("초기 GPU 메모리 할당량:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
    print("초기 GPU 캐시 메모리:", torch.cuda.memory_reserved(0) / 1024**2, "MB")

# 불필요한 특수문자 및 공백 제거
def clean_texts(text_list):
    cleaned_texts = []
    
    for text in text_list:
        # None 또는 NaN 값 처리
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

# T5 모델 사용해서 맞춤법 교정 (기본 설정 사용)
def correct_spelling(text_list, batch_size=64):
    corrected_texts = []
    
    with torch.no_grad():  # ✅ 그래디언트 계산 비활성화 (추론 속도 증가)
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i + batch_size]

            # ✅ 배치 단위로 인코딩 & fp16 변환
            input_encoding = tokenizer(["맞춤법을 고쳐주세요: " + text for text in batch_texts], 
                                       return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
            
            input_ids = input_encoding.input_ids.to(device).long()  # 입력 데이터도 fp16 변환
            attention_mask = input_encoding.attention_mask.to(device).half()

            # ✅ 디버깅: 첫 번째 배치에서 GPU에서 실행되는지 
            if i == 0:
                print("모델이 실행되는 디바이스:", next(model.parameters()).device)
                print("input_ids 위치:", input_ids.device, "| dtype:", input_ids.dtype)
                print("attention_mask 위치:", attention_mask.device, "| dtype:", attention_mask.dtype)

            try:
                torch.cuda.synchronize()  # ✅ GPU 연산이 끝날 때까지 대기

                # ✅ 최적화된 `generate()`
                output_encoding = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=100,  # ✅ 문장 길이 제한 → 속도 증가
                    num_beams=2,  # ✅ 빔 서치 개수 줄임 → 속도 증가
                    use_cache=True,  # ✅ 캐시 사용하여 속도 증가
                    early_stopping=True,
                )

                corrected_batch = [tokenizer.decode(output, skip_special_tokens=True) for output in output_encoding]
            except Exception as e:
                print(f"오류 발생: {e}")
                corrected_batch = batch_texts  # 오류 발생 시 원본 유지

            corrected_texts.extend(corrected_batch)

            # ✅ 진행 상황 출력 (100개 단위)
            if i % 100 == 0:
                print(f"{i}개 처리 완료 | GPU 메모리 사용량: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    return corrected_texts


# 정규 표현식 기반 서술어 기준 문장 분리
def split_sentences(text):
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

# 입력 형태 리스트 형식
def split_sentences_list(text_list):
    """
    여러 개의 리뷰 텍스트를 문장 단위로 분리하는 함수
    """
    result = []
    
    for i, text in enumerate(text_list):
        sentences = split_sentences(text)
        result.append(sentences)
        
        # 진행 상황 출력 (1000개 단위)
        if i % 1000 == 0:
            print(f"문장 분리 진행 상황: {i}개 완료")

    return result


# CSV 파일 로드
file_path = "./review_raw.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# 3번째 열 추출 및 리스트 변환
reviews = df.iloc[:, 2].tolist()

# 텍스트 전처리
print("텍스트 정리 중...")
clean_result = clean_texts(reviews)

# 맞춤법 교정 (GPU 사용 여부 확인)
print("맞춤법 교정 시작...")
correct_result = correct_spelling(clean_result)

# 문장 분리 실행
print("문장 분리 시작...")
split_results = split_sentences_list(correct_result)

# 최종 결과 출력 (최대 5개만 출력)
for i, sentences in enumerate(split_results):
    if i >= 5:
        break
    print(f"리뷰 {i+1}: {sentences}")

# 최종 GPU 메모리 사용량 확인
if torch.cuda.is_available():
    print("최종 GPU 메모리 사용량:", torch.cuda.memory_allocated(0) / 1024**2, "MB")

    
df_result = pd.DataFrame(split_results, columns=["문장"])
df_result.to_csv("sentences.csv", encoding="utf-8")






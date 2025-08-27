# emotion_engine.py (최종 수정 완료 - 수동 로드 버전)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

def load_emotion_classifier():
    """
    수동으로 다운로드한 로컬 폴더에서, 훈련된 감정 분석 모델을 불러와 준비하는 함수.
    """
    # ---!!! 1. 우리가 직접 다운로드한 파일들을 넣어둔 로컬 폴더 경로를 사용합니다 !!!---
    MODEL_PATH = "E:/sentiment_analysis_project/my-local-model"
    
    print(f"로컬 경로 '{MODEL_PATH}'에서 모델을 불러옵니다...")
    try:
        # 2. 해당 경로가 실제로 존재하는지 확인합니다.
        if not os.path.isdir(MODEL_PATH):
            print(f"❌ 경로에 폴더가 존재하지 않습니다! '{MODEL_PATH}' 폴더를 만들고 그 안에 모델 파일들을 모두 다운로드했는지 확인해주세요.")
            return None

        # 3. 이 로컬 경로에서 토크나이저와 모델을 불러옵니다.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("✅ 로컬 모델 파일 로딩 성공!")
        
    except Exception as e:
        print(f"❌ 모델 또는 토크나이저를 불러오는 중 오류가 발생했습니다: {e}")
        return None

    # 4. GPU 사용 설정 및 파이프라인 생성 (이하 코드는 동일)
    device = 0 if torch.cuda.is_available() else -1
    emotion_classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    
    return emotion_classifier

def predict_emotion(classifier, text):
    """
    준비된 파이프라인(classifier)과 텍스트를 받아 감정을 예측하는 함수.
    """
    if not text or not text.strip():
        return "내용 없음"
    if classifier is None:
        return "오류: 감정 분석 엔진이 준비되지 않았습니다."
        
    result = classifier(text)
    return result[0]['label']
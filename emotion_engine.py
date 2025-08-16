# emotion_engine.py
# 감정 분석 엔진 모듈

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def load_emotion_classifier():
    """
    훈련된 감정 분석 모델과 파이셔플라인을 불러와 준비하는 함수.
    이 함수는 서버가 시작될 때 딱 한 번만 호출됩니다.
    """
    # 훈련된 모델이 저장된 최종 경로
    MODEL_PATH = 'E:/sentiment_analysis_project/results/checkpoint-9681' 
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"❌ 해당 경로에 모델이 없습니다: {MODEL_PATH}")
        return None

    # GPU 사용 설정
    device = 0 if torch.cuda.is_available() else -1
    
    # 파이프라인 생성
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
    이 함수는 사용자가 메시지를 보낼 때마다 호출됩니다.
    """
    if not text.strip():
        return "내용 없음" # 빈 텍스트 처리
        
    result = classifier(text)
    # 결과에서 감정 라벨만 추출하여 반환 (예: '기쁨')
    return result[0]['label']
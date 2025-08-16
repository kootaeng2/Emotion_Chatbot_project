import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. 훈련이 완료된 모델과 토크나이저가 저장된 경로 설정
#    경로 앞뒤에 불필요한 따옴표가 들어가지 않도록 하고,
#    경로 구분자는 역슬래시(\) 대신 슬래시(/)를 사용하는 것이 안전합니다.
MODEL_PATH = 'E:/sentiment_analysis_project/results/checkpoint-9681' 

# 2. 저장된 모델과 토크나이저 불러오기
print(f"'{MODEL_PATH}'에서 훈련된 모델을 불러오는 중입니다...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✅ 저장된 모델과 토크나이저를 성공적으로 불러왔습니다.")
except Exception as e:
    print(f"❌ 모델을 불러오는 중 오류가 발생했습니다: {e}")
    print("MODEL_PATH 경로가 정확한지, 해당 폴더 안에 모델 파일(pytorch_model.bin 등)이 있는지 확인해주세요.")
    exit()

# GPU가 사용 가능하면 모델을 GPU로 이동, 아니면 CPU 사용 (0은 첫 번째 GPU)
device = 0 if torch.cuda.is_available() else -1

# 3. 감정 분석 파이프라인 생성
emotion_classifier = pipeline(
    "text-classification", 
    model=model, 
    tokenizer=tokenizer, 
    device=device
)
print("✅ 감정 분석 파이프라인이 준비되었습니다.")

# 4. 예측해 볼 새로운 문장들 (자유롭게 수정하거나 추가해보세요!)
test_sentences = [
    "오늘 점심 메뉴는 정말 최고였어! 기분 좋은 하루야.",
    "과제가 너무 많아서 스트레스 받아. 밤새야 할 것 같아...",
    "이 영화 결말 실화냐? 진짜 어이없고 화난다.",
    "친구가 내 생일을 잊어버려서 너무 서운했어.",
    "길에서 우연히 어릴 적 친구를 만났어! 엄청 놀랐네.",
    "아무런 감정도 느껴지지 않는 평범한 날이다."
]

print("\n--- 새로운 문장 감정 분석 테스트 ---")
for sentence in test_sentences:
    # 파이프라인으로 예측 실행
    result = emotion_classifier(sentence)
    
    # 결과 형식: [{'label': '기쁨', 'score': 0.99...}]
    print(f"\n입력 문장: \"{sentence}\"")
    print(f"▶ AI 예측 감정: {result[0]['label']} (신뢰도: {result[0]['score']:.2f})")
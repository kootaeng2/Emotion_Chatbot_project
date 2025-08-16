# main.py (모델 훈련을 위한 새로운 시작점)

import pandas as pd
import json
import re
from transformers import AutoTokenizer

# --- 1. 데이터 로딩 및 전처리 ---

print("--- [Phase 1] 데이터 로딩 및 전처리 시작 ---")
# 파일 경로 설정
file_path = './data/'

# 훈련/검증 데이터 로딩 (이전과 동일)
with open(file_path + 'training-label.json', 'r', encoding='utf-8') as file:
    training_data_raw = json.load(file)
with open(file_path + 'validation-label.json', 'r', encoding='utf-8') as file:
    validation_data_raw = json.load(file)

# DataFrame 생성 함수 (코드를 깔끔하게 하기 위해 함수로 묶음)
def create_dataframe(data_raw):
    extracted_data = []
    for dialogue in data_raw:
        try:
            emotion_type = dialogue['profile']['emotion']['type']
            dialogue_content = dialogue['talk']['content']
            full_text = " ".join(list(dialogue_content.values()))
            if full_text and emotion_type:
                extracted_data.append({'text': full_text, 'emotion': emotion_type})
        except KeyError:
            continue
    return pd.DataFrame(extracted_data)

df_train = create_dataframe(training_data_raw)
df_val = create_dataframe(validation_data_raw)

# 텍스트 정제
def clean_text(text):
    return re.sub(r'[^가-힣a-zA-Z0-9 ]', '', text)

df_train['cleaned_text'] = df_train['text'].apply(clean_text)
df_val['cleaned_text'] = df_val['text'].apply(clean_text)
print("✅ 데이터 로딩 및 전처리 완료!")


# --- 2. AI 모델링 준비 ---
print("\n--- [Phase 2] AI 모델링 준비 시작 ---")
# 모델 및 토크나이저 불러오기
MODEL_NAME = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 텍스트 토큰화
train_tokenized = tokenizer(list(df_train['cleaned_text']), return_tensors="pt", max_length=128, padding=True, truncation=True)
val_tokenized = tokenizer(list(df_val['cleaned_text']), return_tensors="pt", max_length=128, padding=True, truncation=True)

# 라벨 인코딩
unique_labels = sorted(df_train['emotion'].unique())
label_to_id = {label: id for id, label in enumerate(unique_labels)}
id_to_label = {id: label for label, id in label_to_id.items()}
df_train['label'] = df_train['emotion'].map(label_to_id)
df_val['label'] = df_val['emotion'].map(label_to_id)
print("✅ 토큰화 및 라벨 인코딩 완료!")
print("이제 모델 훈련을 위한 모든 준비가 끝났습니다.")


# -----------------------------------------------------------
# --- [Phase 3] 모델 학습 및 평가 ---
# -----------------------------------------------------------

import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("\n--- [Phase 3] 모델 학습 및 평가 시작 ---")

# 1. PyTorch Dataset 클래스 정의
# 토큰화된 데이터와 라벨을 하나로 묶어 PyTorch가 이해할 수 있는 형태로 만들어주는 클래스입니다.
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # input_ids, attention_mask 등을 모두 가져옵니다.
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Dataset 인스턴스 생성
train_dataset = EmotionDataset(train_tokenized, df_train['label'].tolist())
val_dataset = EmotionDataset(val_tokenized, df_val['label'].tolist())
print("✅ PyTorch 데이터셋 생성이 완료되었습니다.")

# 2. AI 모델 불러오기
# AutoModelForSequenceClassification은 텍스트 분류 문제에 특화된 klue/roberta-base 모델을 불러옵니다.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(unique_labels), # 분류해야 할 감정의 종류 개수
    id2label=id_to_label,           # 숫자 라벨을 감정 이름으로 변환할 때 사용
    label2id=label_to_id            # 감정 이름을 숫자 라벨로 변환할 때 사용
)
# GPU가 사용 가능하면 모델을 GPU 메모리에 올립니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ 모델 로딩 완료! 모델은 {device}에서 실행됩니다.")


# 3. 모델 성능 평가를 위한 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 4. 훈련을 위한 상세 설정(Arguments) 정의
training_args = TrainingArguments(
    output_dir='./results',          # 훈련 결과물(모델 등)이 저장될 디렉토리
    num_train_epochs=3,              # 전체 데이터를 총 3번 반복해서 학습
    per_device_train_batch_size=16,  # 한 번에 GPU에 올릴 훈련 데이터 개수 (메모리 부족 시 8로 줄이기)
    per_device_eval_batch_size=64,   # 한 번에 GPU에 올릴 평가 데이터 개수
    warmup_steps=500,                # 초기 학습률을 점진적으로 올리는 단계 수
    weight_decay=0.01,               # 과적합 방지를 위한 가중치 감소
    logging_dir='./logs',            # 로그가 저장될 디렉토리
    logging_steps=500,               # 500 스텝마다 로그(학습 현황) 출력
    evaluation_strategy="steps",     # 500 스텝마다 평가 실행
    eval_steps=500,                  # 평가 실행 주기
    save_strategy="steps",           # 500 스텝마다 모델 저장
    save_steps=500,
    load_best_model_at_end=True,     # 훈련 종료 후 가장 성능이 좋았던 모델을 자동으로 불러옴
    report_to="none"                 # 외부 서비스(wandb 등)에 로그 전송 안 함
)

# 5. Trainer 정의
# 모델, 훈련 설정, 훈련/검증 데이터셋, 평가 함수를 모두 트레이너에게 전달합니다.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# 6. 모델 훈련 시작!
print("\n🔥 AI 모델 훈련을 시작합니다... (RTX 4060 기준 약 30분~1시간 소요 예상)")
trainer.train()

print("\n🎉 모델 훈련 완료!")

# 7. 최종 모델 성능 평가
print("\n--- 최종 모델 성능 평가 ---")
final_evaluation = trainer.evaluate()
print(final_evaluation)

print("\n모든 과정이 성공적으로 끝났습니다! results 폴더에서 훈련된 모델을 확인하세요.")
# app.py

from flask import Flask, render_template, request, jsonify
# 우리가 이전에 만들었던 AI 모듈들을 불러옵니다.
from emotion_engine import load_emotion_classifier, predict_emotion
from recommender import recommender
import random

# --- 1. Flask 앱 및 AI 모델 준비 ---
app = Flask(__name__)

print("AI 챗봇 서버를 준비 중입니다...")
emotion_classifier = load_emotion_classifier()
recommender = Recommender()
emotion_emoji_map = {
    '기쁨': '😄', '행복': '😊', '사랑': '❤️',
    '불안': '😟', '슬픔': '😢', '상처': '💔', '두려움': '😨',
    '분노': '😠', '혐오': '🤢', '짜증': '😤',
    '놀람': '😮',
    '중립': '😐',
}
print("✅ AI 챗봇 서버가 성공적으로 준비되었습니다.")


# --- 2. 라우팅 설정 (손님 응대 규칙) ---

# 기본 페이지 (http://127.0.0.1:5000)
@app.route("/")
def home():
    """웹사이트의 첫 화면을 보여줍니다."""
    # 'templates' 폴더 안에 있는 'emotion_homepage.html' 파일을 찾아서 보여줍니다.
    return render_template("emotion_homepage.html")


# 추천 API (JavaScript가 호출하는 주소)
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """일기 내용을 받아 감정을 분석하고, 추천을 반환합니다."""
    # JavaScript에서 보낸 'diary' 데이터를 받습니다.
    user_diary = request.json.get("diary")
    if not user_diary:
        return jsonify({"error": "일기 내용이 없습니다."}), 400

    # AI 엔진으로 감정을 분석합니다.
    predicted_emotion = predict_emotion(emotion_classifier, user_diary)
    
    # 추천 로직 실행 ('수용'과 '전환' 중 하나를 랜덤 선택)
    choice = random.choice(["수용", "전환"])
    recommendations = recommender.recommend(predicted_emotion, choice)
    
    if recommendations and "아직 준비된 추천이 없어요" not in recommendations[0]:
        # 추천 목록에서 하나를 랜덤으로 선택하여 한 문장으로 만듭니다.
        recommendation_text = f"'{choice}'을 위한 추천: {random.choice(recommendations)}"
    else:
        recommendation_text = "아직 준비된 추천이 없네요."

    # 최종 결과를 JSON 형태로 웹페이지에 돌려줍니다.
    response_data = {
        "emotion": predicted_emotion,
        "emoji": emotion_emoji_map.get(predicted_emotion, '🤔'),
        "recommendation": recommendation_text
    }
    return jsonify(response_data)


# --- 3. 서버 실행 ---
if __name__ == "__main__":
    app.run(debug=True)
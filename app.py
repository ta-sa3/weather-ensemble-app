import os
import requests
import json
from flask import Flask, render_template_string, request
from google import genai

app = Flask(__name__)

# APIキーの設定 (GitHub Secrets または環境変数から取得)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 監視・判定対象のエリアリスト（主要区および周辺都市の緯度・経度）
LOCATIONS = {
    "名古屋市港区": {"lat": 35.1084, "lon": 136.8853},
    "名古屋市守山区": {"lat": 35.2017, "lon": 136.9856},
    "名古屋市緑区": {"lat": 35.0822, "lon": 136.9744},
    "名古屋市千種区": {"lat": 35.1706, "lon": 136.9458},
    "一宮市": {"lat": 35.3017, "lon": 136.7956},
    "豊田市": {"lat": 35.0833, "lon": 137.1500},
    "四日市市": {"lat": 34.9664, "lon": 136.6261},
    "岐阜市": {"lat": 35.4233, "lon": 136.7606}
}

def get_detailed_weather(lat, lon):
    """Open-Meteo APIから雨量・風向・風速・湿度データを取得"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ["precipitation", "rain"],
        "hourly": ["relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure"],
        "timezone": "Asia/Tokyo",
        "forecast_minutely_15": 12, # 今後3時間分
        "forecast_days": 1
    }
    res = requests.get(url, params=params)
    return res.json()

def analyze_stability_with_gemini(location_name, weather_data):
    """風速・湿度・風向データを含めて気象の安定度と降水リスクをGeminiで判定"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    あなたは高度な気象アナリストです。
    以下のデータは【{location_name}】の今後3時間の気象予測データ(Open-Meteo)です。

    データ概要:
    - 15分単位の降水量予測: {weather_data.get('minutely_15', {})}
    - 時間単位の環境データ (湿度・風速・風向・気圧): {weather_data.get('hourly', {})}

    【分析依頼】
    以下のフォーマットに従って、気象判定を行ってください：

    1. **総合評価**（例: 安定・やや不安定・荒天の警戒が必要 など）
    2. **大気の安定度と風・湿度の分析**:
       - 湿度や風速、風向の変化から、ゲリラ豪雨や急な前線通過のリスク、大気の安定性を簡潔に解説してください。
    3. **雨のリスクと推移**:
       - 今後3時間以内に雨が降るか、何時ごろに強まるか。
    4. **住民への具体的なアドバイス**:
       - 洗濯物、傘の準備、外出時の注意点など。

    回答は親しみやすく読みやすいMarkdown形式（150〜250文字程度）で作成してください。
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

# 簡易UIテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>気象安定度・雨雲解析ダッシュボード</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 30px auto; padding: 20px; background: #f4f7f8; }
        .card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        select, button { padding: 10px; font-size: 16px; border-radius: 6px; border: 1px solid #ccc; }
        button { background: #007bff; color: white; border: none; cursor: pointer; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; background: #eef6ff; padding: 15px; border-radius: 8px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="card">
        <h2>🌦️ 安定度・強風・雨雲リアルタイム解析</h2>
        <form method="POST">
            <label for="location"><b>エリアを選択:</b></label>
            <select name="location" id="location">
                {% for loc in locations %}
                    <option value="{{ loc }}" {% if loc == selected_loc %}selected{% endif %}>{{ loc }}</option>
                {% endfor %}
            </select>
            <button type="submit">AI分析を実行</button>
        </form>

        {% if result %}
        <div class="result">
            <h3>📍 {{ selected_loc }} の気象解析結果</h3>
            <div>{{ result }}</div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    selected_loc = "名古屋市港区"
    result = None

    if request.method == "POST":
        selected_loc = request.form.get("location")
        
    if selected_loc in LOCATIONS:
        coords = LOCATIONS[selected_loc]
        weather_data = get_detailed_weather(coords["lat"], coords["lon"])
        result = analyze_stability_with_gemini(selected_loc, weather_data)

    return render_template_string(HTML_TEMPLATE, locations=LOCATIONS.keys(), selected_loc=selected_loc, result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

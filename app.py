import os
import requests
import json
import streamlit as st
from google import genai

# ページ基本設定
st.set_page_config(page_title="気象安定度・雨雲解析ダッシュボード", layout="centered")

# APIキーの取得（Streamlit Secrets または環境変数）
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# エリアリスト（緯度・経度）
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
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "minutely_15": ["precipitation", "rain"],
        "hourly": ["relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "surface_pressure"],
        "timezone": "Asia/Tokyo",
        "forecast_minutely_15": 12,
        "forecast_days": 1
    }
    res = requests.get(url, params=params)
    return res.json()

def analyze_stability_with_gemini(location_name, weather_data):
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

# UI表示
st.title("🌦️ 安定度・強風・雨雲リアルタイム解析")

selected_loc = st.selectbox("エリアを選択してください", list(LOCATIONS.keys()))

if st.button("AI分析を実行"):
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY が設定されていません。Streamlit の Secrets 設定を確認してください。")
    else:
        with st.spinner("気象データ取得＆AI分析中..."):
            coords = LOCATIONS[selected_loc]
            weather_data = get_detailed_weather(coords["lat"], coords["lon"])
            result = analyze_stability_with_gemini(selected_loc, weather_data)
            
            st.subheader(f"📍 {selected_loc} の気象解析結果")
            st.markdown(result)

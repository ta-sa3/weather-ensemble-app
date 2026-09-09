import os
import time
import requests
import json
import streamlit as st
import pandas as pd
from google import genai

# ページ基本設定
st.set_page_config(page_title="気象安定度・雨雲解析ダッシュボード", layout="centered")

# APIキーの取得（Streamlit Secrets または 環境変数を安全に取得）
GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# エリアリスト（緯度・経度）
LOCATIONS = {
    "名古屋市西区": {"lat": 35.1950, "lon": 136.8878},
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
        "hourly": [
            "temperature_2m",            # 気温 (℃)
            "precipitation_probability", # 降水確率 (%)
            "precipitation",             # 降水量 (mm)
            "relative_humidity_2m",      # 湿度 (%)
            "wind_speed_10m",            # 風速 (m/s)
            "wind_direction_10m",        # 風向 (度)
            "surface_pressure"           # 気圧 (hPa)
        ],
        "timezone": "Asia/Tokyo",
        "forecast_minutely_15": 12,
        "forecast_days": 1
    }
    res = requests.get(url, params=params)
    return res.json()

def analyze_stability_with_gemini(location_name, weather_data, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    あなたは高度な気象アナリストです。
    以下のデータは【{location_name}】の今後3時間の気象予測データ(Open-Meteo)です。

    データ概要:
    - 15分単位の降水量予測: {weather_data.get('minutely_15', {})}
    - 時間単位の環境データ (気温・降水確率・降水量・湿度・風速・風向・気圧): {weather_data.get('hourly', {})}

    【分析依頼】
    以下のフォーマットに従って、気象判定を行ってください：

    1. **総合評価**（例: 安定・やや不安定・荒天の警戒が必要 など）
    2. **気温・降水量・大気安定度の分析**:
       - 気温や降水確率の推移、湿度・風速から見た急な雨や前線通過のリスクを簡潔に解説してください。
    3. **雨のリスクと推移**:
       - 今後3時間以内に雨が降るか、降水量・確率のピークは何時ごろか。
    4. **住民への具体的なアドバイス**:
       - 服装選び、洗濯物、傘の準備、外出時の注意点など。

    回答は親しみやすく読みやすいMarkdown形式（150〜250文字程度）で作成してください。
    """

    # 試行するモデル順（高負荷時のフォールバック用）
    models_to_try = ['gemini-3.6-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        # 各モデルで最大2回リトライ
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                # 503エラー（一時的高負荷）の場合は1.5秒待ってリトライ
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(1.5)
                    continue
                else:
                    raise e
                    
    raise Exception("現在Gemini APIが混雑しています。数十秒おいてから再度お試しください。")

# UI表示
st.title("🌦️ 安定度・気温・雨雲リアルタイム解析")

selected_loc = st.selectbox("エリアを選択してください", list(LOCATIONS.keys()))

if st.button("AI分析・データ取得を実行"):
    if not GEMINI_API_KEY:
        st.error("🔑 APIキーが検出されませんでした。Streamlit Cloudの Secrets で GEMINI_API_KEY を確認してください。")
    else:
        with st.spinner("気象データ取得＆AI分析中..."):
            try:
                coords = LOCATIONS[selected_loc]
                weather_data = get_detailed_weather(coords["lat"], coords["lon"])
                
                # 今後の時間別データを表示用カードとして整形
                hourly = weather_data.get("hourly", {})
                times = hourly.get("time", [])[:6] # 今後6時間分
                temps = hourly.get("temperature_2m", [])[:6]
                probs = hourly.get("precipitation_probability", [])[:6]
                precips = hourly.get("precipitation", [])[:6]

                # 気象データの主要メトリクス表示
                st.subheader(f"📊 {selected_loc} の直近気象データ")
                col1, col2, col3 = st.columns(3)
                if temps:
                    col1.metric("現在の気温", f"{temps[0]} ℃")
                if probs:
                    col2.metric("現在の降水確率", f"{probs[0]} %")
                if precips:
                    col3.metric("直近の予測降水量", f"{precips[0]} mm")

                # 時間別詳細表
                if times:
                    df = pd.DataFrame({
                        "時間": [t.split("T")[1] for t in times],
                        "気温 (℃)": temps,
                        "降水確率 (%)": probs,
                        "降水量 (mm)": precips
                    })
                    st.dataframe(df, use_container_width=True)

                # GeminiのAI解析結果表示
                result = analyze_stability_with_gemini(selected_loc, weather_data, GEMINI_API_KEY)
                st.subheader(f"🤖 AI気象解析コメント")
                st.markdown(result)

            except Exception as e:
                st.error(f"解析中にエラーが発生しました: {e}")

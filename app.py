import os
import time
import requests
import json
import streamlit as st
import pandas as pd
from google import genai

# ページ基本設定
st.set_page_config(page_title="気象安定度・雨雲解析ダッシュボード", layout="centered")

# APIキーの取得
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
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure"
        ],
        "timezone": "Asia/Tokyo",
        "forecast_minutely_15": 12,
        "forecast_days": 2
    }
    res = requests.get(url, params=params)
    return res.json()

def analyze_stability_with_gemini(location_name, weather_data, api_key):
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    あなたは高度な気象アナリストです。
    以下のデータは【{location_name}】の気象予測データ(Open-Meteo)です。

    データ概要:
    - 15分単位の降水量予測: {weather_data.get('minutely_15', {})}
    - 時間単位の環境データ (気温・降水確率・降水量・湿度・風速・風向・気圧): {weather_data.get('hourly', {})}

    【分析依頼】
    以下のフォーマットに従って、気象判定を行ってください：
    1. **総合評価**
    2. **気温・降水量・大気安定度の分析**
    3. **雨のリスクと推移**
    4. **住民への具体的なアドバイス**

    回答は親しみやすく読みやすいMarkdown形式（150〜250文字程度）で作成してください。
    """

    # 3.6-flash を最優先とし、制限時のみフォールバック
    models_to_try = ['gemini-3.6-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            # 429(上限超過) または 503(混雑) の場合は次のモデルへフォールバック
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            elif "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(1.5)
                continue
            else:
                raise e
                    
    return "⚠️ 全モデルの無料利用枠上限に達しました。夕方16時の制限解除後にお試しいただくか、上の表・グラフデータをご参照ください。"

# UI表示
st.title("🌦️ 安定度・気温・雨雲リアルタイム解析")

selected_loc = st.selectbox("エリアを選択してください", list(LOCATIONS.keys()))

# ボタンを横並びに配置
col_btn1, col_btn2 = st.columns(2)
get_data_clicked = col_btn1.button("📊 気象データを取得 (無料)")
analyze_ai_clicked = col_btn2.button("🤖 AI分析を実行 (API消費)")

# データ保持用セッションステートの初期化
if "current_weather" not in st.session_state:
    st.session_state.current_weather = None
if "current_location" not in st.session_state:
    st.session_state.current_location = None

# 1. 気象データ取得ボタンが押された場合
if get_data_clicked:
    with st.spinner("最新の気象データを取得中..."):
        coords = LOCATIONS[selected_loc]
        st.session_state.current_weather = get_detailed_weather(coords["lat"], coords["lon"])
        st.session_state.current_location = selected_loc

# 気象データが保持されている場合に表示
if st.session_state.current_weather and st.session_state.current_location == selected_loc:
    weather_data = st.session_state.current_weather
    hourly = weather_data.get("hourly", {})
    times_all = [t.replace("T", " ") for t in hourly.get("time", [])]
    temps_all = hourly.get("temperature_2m", [])
    probs_all = hourly.get("precipitation_probability", [])
    precips_all = hourly.get("precipitation", [])

    st.subheader(f"📊 {selected_loc} の気象状態")
    c1, c2, c3 = st.columns(3)
    if temps_all:
        c1.metric("現在の気温", f"{temps_all[0]} ℃")
    if probs_all:
        c2.metric("現在の降水確率", f"{probs_all[0]} %")
    if precips_all:
        c3.metric("直近の予測降水量", f"{precips_all[0]} mm")

    st.subheader("📅 予報データ")
    tab1, tab2, tab3 = st.tabs(["直近6時間", "24時間一覧", "24時間グラフ"])

    with tab1:
        df_6h = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_all[:6]],
            "気温 (℃)": temps_all[:6],
            "降水確率 (%)": probs_all[:6],
            "降水量 (mm)": precips_all[:6]
        })
        st.dataframe(df_6h, use_container_width=True)

    with tab2:
        df_24h = pd.DataFrame({
            "日時": times_all[:24],
            "気温 (℃)": temps_all[:24],
            "降水確率 (%)": probs_all[:24],
            "降水量 (mm)": precips_all[:24]
        })
        st.dataframe(df_24h, use_container_width=True, height=300)

    with tab3:
        chart_data = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_all[:24]],
            "気温 (℃)": temps_all[:24],
            "降水量 (mm)": precips_all[:24]
        }).set_index("時間")
        st.line_chart(chart_data)

# 2. AI分析ボタンが押された場合
if analyze_ai_clicked:
    if not st.session_state.current_weather or st.session_state.current_location != selected_loc:
        coords = LOCATIONS[selected_loc]
        st.session_state.current_weather = get_detailed_weather(coords["lat"], coords["lon"])
        st.session_state.current_location = selected_loc

    if not GEMINI_API_KEY:
        st.error("🔑 APIキーが検出されませんでした。Secrets を確認してください。")
    else:
        with st.spinner("Gemini API (3.6-flash) でAI気象解析を生成中..."):
            result = analyze_stability_with_gemini(selected_loc, st.session_state.current_weather, GEMINI_API_KEY)
            st.subheader("🤖 AI気象解析コメント")
            st.markdown(result)

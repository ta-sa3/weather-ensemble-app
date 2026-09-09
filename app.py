import os
import time
import requests
import json
import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from google import genai

# タイムゾーンの設定（日本時間）
JST = pytz.timezone("Asia/Tokyo")

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
    headers = {"Cache-Control": "no-cache"}
    res = requests.get(url, params=params, headers=headers)
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

    models_to_try = ['gemini-3.6-flash', 'gemini-2.5-flash', 'gemini-1.5-flash']
    
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                continue
            elif "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(1.5)
                continue
            else:
                raise e
                    
    return "⚠️ 全モデルの無料利用枠上限に達しました。時間をおいてから再度お試しいただくか、上のデータをご参照ください。"

# UI表示
st.title("🌦️ 安定度・気温・雨雲リアルタイム解析")

# セッション状態の初期化
if "current_weather" not in st.session_state:
    st.session_state.current_weather = None
if "current_location" not in st.session_state:
    st.session_state.current_location = None
if "fetched_at" not in st.session_state:
    st.session_state.fetched_at = None

selected_loc = st.selectbox("エリアを選択してください", list(LOCATIONS.keys()))

# エリアを変更したら古いデータを自動クリア
if st.session_state.current_location != selected_loc:
    st.session_state.current_weather = None
    st.session_state.current_location = selected_loc

# ボタンの並び
col_btn1, col_btn2 = st.columns(2)
get_data_clicked = col_btn1.button("🔄 最新データを取得 (無料)")
analyze_ai_clicked = col_btn2.button("🤖 AI分析を実行 (API消費)")

# 1. 「最新データを取得」ボタンが押された場合
if get_data_clicked:
    with st.spinner("最新の気象データを取得中..."):
        coords = LOCATIONS[selected_loc]
        st.session_state.current_weather = get_detailed_weather(coords["lat"], coords["lon"])
        st.session_state.current_location = selected_loc
        st.session_state.fetched_at = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

# 気象データが存在する場合に表示
if st.session_state.current_weather:
    weather_data = st.session_state.current_weather
    hourly = weather_data.get("hourly", {})
    
    times_raw = hourly.get("time", [])
    temps_all = hourly.get("temperature_2m", [])
    probs_all = hourly.get("precipitation_probability", [])
    precips_all = hourly.get("precipitation", [])

    # 日本時間での「現在の年月日時」を取得
    now_jst = datetime.now(JST)
    now_str = now_jst.strftime("%Y-%m-%dT%H:00")

    start_idx = 0
    if now_str in times_raw:
        start_idx = times_raw.index(now_str)
    else:
        for idx, t_str in enumerate(times_raw):
            t_dt = JST.localize(datetime.strptime(t_str, "%Y-%m-%dT%H:%M"))
            if t_dt >= now_jst:
                start_idx = max(0, idx - 1)
                break

    # 現在以降のデータを抽出
    times_now = [t.replace("T", " ") for t in times_raw[start_idx:]]
    temps_now = temps_all[start_idx:]
    probs_now = probs_all[start_idx:]
    precips_now = precips_all[start_idx:]

    st.subheader(f"📊 {selected_loc} の気象状態")
    if st.session_state.fetched_at:
        st.caption(f"最終取得日時: {st.session_state.fetched_at} (JST)")

    c1, c2, c3 = st.columns(3)
    if temps_now:
        c1.metric("現在の気温", f"{temps_now[0]} ℃")
    if probs_now:
        c2.metric("現在の降水確率", f"{probs_now[0]} %")
    if precips_now:
        c3.metric("直近の予測降水量", f"{precips_now[0]} mm")

    st.subheader("📅 予報データ")
    tab1, tab2, tab3 = st.tabs(["直近6時間", "これからの24時間", "24時間グラフ"])

    with tab1:
        # 今現在からの直近6時間分（インデックス非表示）
        df_6h = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_now[:6]],
            "気温 (℃)": temps_now[:6],
            "降水確率 (%)": probs_now[:6],
            "降水量 (mm)": precips_now[:6]
        })
        st.dataframe(df_6h, use_container_width=True, hide_index=True)

    with tab2:
        # 今現在からの24時間分（インデックス非表示）
        df_24h = pd.DataFrame({
            "日時": times_now[:24],
            "気温 (℃)": temps_now[:24],
            "降水確率 (%)": probs_now[:24],
            "降水量 (mm)": precips_now[:24]
        })
        st.dataframe(df_24h, use_container_width=True, height=300, hide_index=True)

    with tab3:
        # これからの24時間グラフ
        chart_data = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_now[:24]],
            "気温 (℃)": temps_now[:24],
            "降水量 (mm)": precips_now[:24]
        }).set_index("時間")
        st.line_chart(chart_data)

# 2. AI分析ボタンが押された場合
if analyze_ai_clicked:
    if not st.session_state.current_weather:
        coords = LOCATIONS[selected_loc]
        st.session_state.current_weather = get_detailed_weather(coords["lat"], coords["lon"])
        st.session_state.current_location = selected_loc
        st.session_state.fetched_at = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

    if not GEMINI_API_KEY:
        st.error("🔑 APIキーが検出されませんでした。Secrets を確認してください。")
    else:
        with st.spinner("Gemini API でAI気象解析を生成中..."):
            result = analyze_stability_with_gemini(selected_loc, st.session_state.current_weather, GEMINI_API_KEY)
            st.subheader("🤖 AI気象解析コメント")
            st.markdown(result)

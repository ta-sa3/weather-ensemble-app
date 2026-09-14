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
st.set_page_config(page_title="気象安定度・雨雲解析ダッシュボード (JMA高精度版)", layout="centered")

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

def evaluate_cape(cape_val):
    """CAPE値による大気安定度評価ステータスを返す"""
    if cape_val is None:
        return "データなし", "⚪"
    if cape_val < 100:
        return "安定", "🟢"
    elif cape_val < 1000:
        return "やや不安定", "🟡"
    elif cape_val < 2500:
        return "不安定 (雷雨・突風リスク)", "🟠"
    else:
        return "非常に不安定 (激しい雷雨・ゲリラ豪雨警戒)", "🔴"

def get_detailed_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "models": "jma_seamless",  # 気象庁（JMA）高解像度モデルを指定
        "minutely_15": ["precipitation", "rain"],
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cape"  # 大気安定度（CAPE: 対流利用可能エネルギー）
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
    以下のデータは【{location_name}】の気象庁（JMA）ベースの数値予報データです。

    データ概要:
    - 15分単位の降水量予測: {weather_data.get('minutely_15', {})}
    - 時間単位の環境データ (気温・降水確率・降水量・湿度・風速・風向・気圧・CAPE): {weather_data.get('hourly', {})}

    【特に重点的に分析するポイント】
    - CAPE（対流利用可能エネルギー）の値に注目し、大気の積乱雲発達リスクや突風・雷雨の可能性を評価してください。

    【分析依頼フォーマット】
    1. **総合評価**
    2. **大気安定度（CAPE）と積乱雲・雷雨リスク**
    3. **気温・降水量・風の推移**
    4. **住民への具体的なアドバイス（外出・雨具・防災面）**

    回答は親しみやすく読みやすいMarkdown形式（200〜300文字程度）で作成してください。
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
st.title("🌦️ 安定度・気温・雨雲解析 (JMAモデル版)")

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
get_data_clicked = col_btn1.button("🔄 最新データを取得 (JMAモデル)")
analyze_ai_clicked = col_btn2.button("🤖 AI分析を実行 (Gemini)")

# 1. 「最新データを取得」ボタンが押された場合
if get_data_clicked:
    with st.spinner("気象庁(JMA)高精度モデルからデータ取得中..."):
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
    capes_all = hourly.get("cape", [])

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
    capes_now = capes_all[start_idx:] if capes_all else [0] * len(times_now)

    st.subheader(f"📊 {selected_loc} の気象状態 (JMAモデル)")
    if st.session_state.fetched_at:
        st.caption(f"最終取得日時: {st.session_state.fetched_at} (JST)")

    c1, c2, c3, c4 = st.columns(4)
    if temps_now:
        c1.metric("現在の気温", f"{temps_now[0]} ℃")
    if probs_now:
        c2.metric("現在の降水確率", f"{probs_now[0]} %")
    if precips_now:
        c3.metric("直近の予測降水量", f"{precips_now[0]} mm")
    if capes_now:
        cape_val = capes_now[0]
        status_text, icon = evaluate_cape(cape_val)
        c4.metric("大気安定度 (CAPE)", f"{cape_val} J/kg", delta=f"{icon} {status_text}", delta_color="off")

    st.subheader("📅 予報データ")
    tab1, tab2, tab3 = st.tabs(["直近6時間", "これからの24時間", "24時間グラフ"])

    with tab1:
        # 今現在からの直近6時間分
        df_6h = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_now[:6]],
            "気温 (℃)": temps_now[:6],
            "降水確率 (%)": probs_now[:6],
            "降水量 (mm)": precips_now[:6],
            "CAPE (J/kg)": capes_now[:6]
        })
        st.dataframe(df_6h, use_container_width=True, hide_index=True)

    with tab2:
        # 今現在からの24時間分
        df_24h = pd.DataFrame({
            "日時": times_now[:24],
            "気温 (℃)": temps_now[:24],
            "降水確率 (%)": probs_now[:24],
            "降水量 (mm)": precips_now[:24],
            "CAPE (J/kg)": capes_now[:24]
        })
        st.dataframe(df_24h, use_container_width=True, height=300, hide_index=True)

    with tab3:
        # これからの24時間グラフ（気温・降水量・CAPE）
        chart_data = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_now[:24]],
            "気温 (℃)": temps_now[:24],
            "降水量 (mm)": precips_now[:24],
            "CAPE (J/kg)": capes_now[:24]
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
        with st.spinner("Gemini API で大気安定度・雷雨リスクを解析中..."):
            result = analyze_stability_with_gemini(selected_loc, st.session_state.current_weather, GEMINI_API_KEY)
            st.subheader("🤖 AI気象解析コメント")
            st.markdown(result)

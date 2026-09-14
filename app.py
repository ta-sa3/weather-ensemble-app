import os
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
        "minutely_15": ["precipitation", "rain"],
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "cape"
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
    以下のデータは【{location_name}】の数値予報データです。

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

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI分析中にエラーが発生しました: {str(e)}"

# UI表示
st.title("🌦️ 安定度・気温・雨雲解析ダッシュボード")

# セッション状態の初期化
if "current_weather" not in st.session_state:
    st.session_state.current_weather = None
if "ai_analysis" not in st.session_state:
    st.session_state.ai_analysis = None
if "current_location" not in st.session_state:
    st.session_state.current_location = None
if "fetched_at" not in st.session_state:
    st.session_state.fetched_at = None

selected_loc = st.selectbox("エリアを選択してください", list(LOCATIONS.keys()))

# エリアを変更したら前回の結果をクリア
if st.session_state.current_location != selected_loc:
    st.session_state.current_weather = None
    st.session_state.ai_analysis = None
    st.session_state.current_location = selected_loc

# 一括実行ボタン
run_clicked = st.button("🚀 最新気象データ取得 ＆ AI分析を実行", use_container_width=True)

# ボタンが押された場合の処理（一括実行）
if run_clicked:
    if not GEMINI_API_KEY:
        st.error("🔑 APIキーが検出されませんでした。Secrets を確認してください。")
    else:
        with st.spinner("気象データ取得 ＆ Gemini AI解析を実行中..."):
            coords = LOCATIONS[selected_loc]
            # 1. 天気データの取得
            weather_data = get_detailed_weather(coords["lat"], coords["lon"])
            st.session_state.current_weather = weather_data
            st.session_state.fetched_at = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")

            # 2. Gemini AIでの分析実行
            ai_result = analyze_stability_with_gemini(selected_loc, weather_data, GEMINI_API_KEY)
            st.session_state.ai_analysis = ai_result

# 結果が存在する場合の表示処理
if st.session_state.current_weather:
    weather_data = st.session_state.current_weather
    hourly = weather_data.get("hourly", {})
    times_raw = hourly.get("time", [])

    # None対策処理
    temps_all = [t if t is not None else 0.0 for t in hourly.get("temperature_2m", [])]
    probs_all = [p if p is not None else 0 for p in hourly.get("precipitation_probability", [])]
    precips_all = [pr if pr is not None else 0.0 for pr in hourly.get("precipitation", [])]
    capes_all = [c if c is not None else 0 for c in hourly.get("cape", [])]

    # 現在時刻のインデックス特定
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

    times_now = [t.replace("T", " ") for t in times_raw[start_idx:]]
    temps_now = temps_all[start_idx:]
    probs_now = probs_all[start_idx:]
    precips_now = precips_all[start_idx:]
    capes_now = capes_all[start_idx:] if capes_all else [0] * len(times_now)

    # --- 1. AI分析結果 ---
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI気象解析コメント")
        st.info(st.session_state.ai_analysis)

    st.markdown("---")

    # --- 2. リアルタイム数値指標 ---
    st.subheader(f"📊 {selected_loc} の最新状態")
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

    # --- 3. 詳細データ表示（タブ） ---
    st.subheader("📅 予報データ")
    tab1, tab2, tab3 = st.tabs(["直近6時間", "これからの24時間", "24時間グラフ"])

    with tab1:
        df_6h = pd.DataFrame({
            "時間": [t.split(" ")[1] for t in times_now[:6]],
            "気温 (℃)": temps_now[:6],
            "降水確率 (%)": probs_now[:6],
            "降水量 (mm)": precips_now[:6],
            "CAPE (J/kg)": capes_now[:6]
        })
        st.dataframe(df_6h, use_container_width=True, hide_index=True)

    with tab2:
        df_24h = pd.DataFrame({
            "日時": times_now[:24],
            "気温 (℃)": temps_now[:24],
            "降水確率 (%)": probs_now[:24],
            "降水量 (mm)": precips_now[:24],
            "CAPE (J/kg)": capes_now[:24]
        })
        st.dataframe(df_24h, use_container_width=True, height=300, hide_index=True)

    with tab3:
        times_24h = [t.split(" ")[1] for t in times_now[:24]]
        
        st.caption("🌡️ **気温の推移 (℃)**")
        df_temp = pd.DataFrame({"時間": times_24h, "気温 (℃)": temps_now[:24]}).set_index("時間")
        st.line_chart(df_temp)

        st.caption("🌧️ **降水量の推移 (mm)**")
        df_precip = pd.DataFrame({"時間": times_24h, "降水量 (mm)": precips_now[:24]}).set_index("時間")
        st.bar_chart(df_precip)  # 降水量は棒グラフ表示で直感的に分かりやすく

        st.caption("⚡ **大気安定度 CAPE (J/kg)**")
        df_cape = pd.DataFrame({"時間": times_24h, "CAPE (J/kg)": capes_now[:24]}).set_index("時間")
        st.line_chart(df_cape)

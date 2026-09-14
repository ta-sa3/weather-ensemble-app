import os
import requests
import json
import streamlit as st
import pandas as pd
from datetime import datetime
import pytz
from google import genai
import plotly.express as px

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
if "chart_idx" not in st.session_state:
    st.session_state.chart_idx = 0

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

    # --- 1. リアルタイム数値指標 ---
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

    st.markdown("---")

    # --- 2. 詳細データ表示（予報データ） ---
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

        # 1. 気温用データ
        df_temp = pd.DataFrame({"時間": times_24h, "気温 (℃)": temps_now[:24]}).set_index("時間")

        # 2. 降水量用（Plotlyで色分け）
        precip_levels = []
        for val in precips_now[:24]:
            if val <= 10:
                precip_levels.append("10mm以下 (普通〜やや強い雨)")
            elif val <= 20:
                precip_levels.append("10mm〜20mm (注意レベル)")
            else:
                precip_levels.append("20mm超 (土砂降り・大雨警戒)")

        df_precip_plotly = pd.DataFrame({
            "時間": times_24h,
            "降水量 (mm)": precips_now[:24],
            "雨の強さ": precip_levels
        })

        fig_precip = px.bar(
            df_precip_plotly,
            x="時間",
            y="降水量 (mm)",
            color="雨の強さ",
            color_discrete_map={
                "10mm以下 (普通〜やや強い雨)": "#2196F3",  # 青
                "10mm〜20mm (注意レベル)": "#FFC107",      # 黄
                "20mm超 (土砂降り・大雨警戒)": "#F44336"    # 赤
            }
        )
        fig_precip.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # 3. CAPE用（Plotlyで色分け）
        cape_levels = []
        for val in capes_now[:24]:
            if val <= 1000:
                cape_levels.append("1000以下 (安定・やや不安定)")
            elif val <= 2500:
                cape_levels.append("1000〜2500 (不安定・雷雨リスク)")
            else:
                cape_levels.append("2500超 (非常に不安定・豪雨警戒)")

        df_cape_plotly = pd.DataFrame({
            "時間": times_24h,
            "CAPE (J/kg)": capes_now[:24],
            "リスク区分": cape_levels
        })

        fig_cape = px.bar(
            df_cape_plotly,
            x="時間",
            y="CAPE (J/kg)",
            color="リスク区分",
            color_discrete_map={
                "1000以下 (安定・やや不安定)": "#4CAF50",      # 緑
                "1000〜2500 (不安定・雷雨リスク)": "#FF9800",  # 橙
                "2500超 (非常に不安定・豪雨警戒)": "#F44336"    # 赤
            }
        )
        fig_cape.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # スライド表示用リストの設定
        charts_info = [
            {
                "title": "🌡️ 気温の推移 (℃)",
                "data": df_temp,
                "type": "line",
                "color": ["#FF5722"]
            },
            {
                "title": "🌧️ 降水量の推移 (mm)",
                "type": "plotly",
                "fig": fig_precip
            },
            {
                "title": "⚡ 大気安定度 CAPE (J/kg)",
                "type": "plotly",
                "fig": fig_cape
            }
        ]

        # ナビゲーションコントロール（前へ / タイトル / 次へ）
        col_prev, col_title, col_next = st.columns([1, 4, 1])

        if col_prev.button("◀ 前へ", use_container_width=True):
            st.session_state.chart_idx = (st.session_state.chart_idx - 1) % len(charts_info)
            st.rerun()

        if col_next.button("次へ ▶", use_container_width=True):
            st.session_state.chart_idx = (st.session_state.chart_idx + 1) % len(charts_info)
            st.rerun()

        # 現在選択されているグラフ情報を取得
        current_chart = charts_info[st.session_state.chart_idx]

        # タイトルとページインジケーターの表示
        col_title.markdown(
            f"<h4 style='text-align: center; margin: 0;'>{current_chart['title']}</h4>"
            f"<p style='text-align: center; color: gray; margin: 0;'>({st.session_state.chart_idx + 1} / {len(charts_info)})</p>",
            unsafe_allow_html=True
        )

        st.markdown("")

        # 該当するグラフを出力
        if current_chart["type"] == "line":
            st.line_chart(current_chart["data"], color=current_chart["color"])
        elif current_chart["type"] == "plotly":
            st.plotly_chart(current_chart["fig"], use_container_width=True)

        # 降水量（2番目）の凡例ガイド
        if st.session_state.chart_idx == 1:
            st.caption("""
            **【降水量 色分け凡例】**
            - 🟦 **青色 (10mm以下)**: 通常の雨〜やや強い雨
            - 🟨 **黄色 (10mm〜20mm)**: ザーザー雨・注意レベル
            - 🟥 **赤色 (20mm超)**: 土砂降り・大雨警戒レベル
            """)

        # CAPE（3番目）の凡例ガイド
        if st.session_state.chart_idx == 2:
            st.caption("""
            **【CAPE 色分け凡例】**
            - 🟢 **緑色 (1000以下)**: 安定〜やや不安定
            - 🟠 **橙色 (1000〜2500)**: 不安定（雷雨・突風リスク）
            - 🔴 **赤色 (2500超)**: 非常に不安定（激しい雷雨・ゲリラ豪雨警戒）
            """)

    st.markdown("---")

    # --- 3. AI分析結果 ---
    if st.session_state.ai_analysis:
        st.subheader("🤖 AI気象解析コメント")
        st.info(st.session_state.ai_analysis)

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from google import genai
from datetime import datetime, timedelta, timezone

# ==========================================
# 1. ページ基本設定
# ==========================================
st.set_page_config(
    page_title="5大モデルAIアンサンブル気象予報",
    page_icon="🌀",
    layout="wide"
)

CITY_COORDINATES = {
    "東京": (35.6895, 139.6917),
    "大阪": (34.6937, 135.5023),
    "名古屋": (35.1815, 136.9066),
    "札幌": (43.0621, 141.3544),
    "福岡": (33.5904, 130.4017),
    "仙台": (38.2682, 140.8694)
}

# 日本標準時 (JST)
JST = timezone(timedelta(hours=9))

# ==========================================
# 2. データ取得関数 (Open-Meteo API)
# JMA / JMA-MSM / GFS / ECMWF / ICON の5モデルを取得 (ttl=300秒)
# ==========================================
@st.cache_data(ttl=300)
def fetch_hourly_ensemble_data(latitude, longitude):
    # 5大モデルを指定: jma_seamless, jma_msm, gfs_seamless, ecmwf_ifs025, icon_seamless
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
        f"&hourly=precipitation_probability,precipitation"
        f"&timezone=Asia%2FTokyo"
        f"&models=jma_seamless,jma_msm,gfs_seamless,ecmwf_ifs025,icon_seamless"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        raw_times = hourly.get("time", [])
        
        if not raw_times:
            st.error("気象データの時刻情報が空です。")
            return None, None
            
        # 現在時刻 (JST) のフォーマットを作成
        now_jst = datetime.now(JST)
        current_time_str = now_jst.strftime("%Y-%m-%dT%H:00")
        
        # 配列内から現在時刻以降のインデックスを探す
        start_idx = 0
        for idx, t in enumerate(raw_times):
            if t >= current_time_str:
                start_idx = idx
                break
                
        end_idx = start_idx + 12
        
        # 直近12時間分切り出し
        target_times = [t.split("T")[1] for t in raw_times[start_idx:end_idx]]
        
        # 降水確率
        prob_jma = hourly.get("precipitation_probability_jma_seamless", hourly.get("precipitation_probability", []))[start_idx:end_idx]
        prob_msm = hourly.get("precipitation_probability_jma_msm", [])[start_idx:end_idx]
        prob_gfs = hourly.get("precipitation_probability_gfs_seamless", [])[start_idx:end_idx]
        prob_ecm = hourly.get("precipitation_probability_ecmwf_ifs025", [])[start_idx:end_idx]
        prob_ico = hourly.get("precipitation_probability_icon_seamless", [])[start_idx:end_idx]
        
        # 降水量
        precip_jma = hourly.get("precipitation_jma_seamless", hourly.get("precipitation", []))[start_idx:end_idx]
        precip_msm = hourly.get("precipitation_jma_msm", [])[start_idx:end_idx]
        precip_gfs = hourly.get("precipitation_gfs_seamless", [])[start_idx:end_idx]
        precip_ecm = hourly.get("precipitation_ecmwf_ifs025", [])[start_idx:end_idx]
        precip_ico = hourly.get("precipitation_icon_seamless", [])[start_idx:end_idx]

        prob_data = {
            "時刻": target_times,
            "気象庁 (JMA)": prob_jma,
            "気象庁局地 (MSM)": prob_msm,
            "米国 (GFS)": prob_gfs,
            "欧州 (ECMWF)": prob_ecm,
            "ドイツ (ICON)": prob_ico
        }
        
        precip_data = {
            "時刻": target_times,
            "気象庁 (JMA)": precip_jma,
            "気象庁局地 (MSM)": precip_msm,
            "米国 (GFS)": precip_gfs,
            "欧州 (ECMWF)": precip_ecm,
            "ドイツ (ICON)": precip_ico
        }
        
        return pd.DataFrame(prob_data), pd.DataFrame(precip_data)
    except Exception as e:
        st.error(f"気象データの取得に失敗しました: {e}")
        return None, None

# ==========================================
# 3. GeminiによるAI分析関数 (gemini-3.6-flash)
# ==========================================
@st.cache_data(ttl=300)
def analyze_ensemble_with_gemini(api_key, city_name, df_prob, df_precip):
    client = genai.Client(api_key=api_key)
    
    combined_data = {
        "降水確率_時系列(%)": df_prob.to_dict(orient="records"),
        "降水量_時系列(mm)": df_precip.to_dict(orient="records")
    }
    
    prompt = f"""
あなたは高度な数値予報モデルの解析を行う専門気象予報士です。
以下の都市における直近12時間の【気象庁(JMA標準) / 気象庁(JMA局地MSM) / 米国(GFS) / 欧州(ECMWF) / ドイツ(ICON)】の5大数値予報モデルの1時間ごと降水データをもとに、時間軸でのアンサンブル分析を行ってください。

【都市名】: {city_name}
【1時間ごとの5モデル比較データ (JSON)】:
{json.dumps(combined_data, ensure_ascii=False, indent=2)}

【分析要件】
1. **降水量・降水タイミングの一致度**:
   - 5つのモデル間で「降り始める時刻」「雨のピーク時刻」「雨が止む時刻」にズレがあるかを分析してください。
   - 特に日本の局地モデル(JMA MSM)と海外モデル(GFS/ECMWF/ICON)の見解の違いに着目してください。
2. **予測確信度判定**:
   - 確信度を【高・中・低】の3段階で判定し、その理由を述べてください。
3. **時間帯別のリスクとアクション**:
   - ユーザーがいつから傘を持つべきか、どの時間帯に強雨・局地的大雨のリスクがあるかを分かりやすくアドバイスしてください。

【出力フォーマット】
Markdown形式で、見出しを使って見やすく出力してください。
---
### ⏱️ 時間軸アンサンブル解析サマリー
- **降り始め予想時刻**: 
- **予測確信度**: 【高 / 中 / 低】

### 🌧️ 1時間ごとのモデル別見解・比較
- **降り始めのタイミング**: 
- **降水のピーク時間帯と強度**: 
- **各モデル(JMA標準/MSM/GFS/ECMWF/ICON)の特徴とズレ**: 

### ⚠️ お出かけ時のリスクシナリオ・アドバイス
---
"""

    target_model = 'gemini-3.6-flash'
    last_error = None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            last_error = e
            time.sleep(attempt + 1)

    return f"AIの分析中にエラーが発生しました: {last_error}"

# ==========================================
# 4. Streamlit 画面構成（UI）
# ==========================================
st.title("🌀 5大モデルAIアンサンブル気象予報")
st.caption("日本(JMA標準/MSM局地)・米国(GFS)・欧州(ECMWF)・ドイツ(ICON)の5大モデルをグラフで比較し、Geminiが分析します。")

st.sidebar.header("⚙️ 設定")

# Streamlit Secretsからキーを自動取得
if "GEMINI_API_KEY" in st.secrets:
    api_key_input = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("🔑 APIキー自動読み込み完了")
else:
    api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studioで発行したAPIキーを入力してください")

selected_city = st.sidebar.selectbox("対象エリアを選択", list(CITY_COORDINATES.keys()))

if st.sidebar.button("🔄 手動でデータを再更新"):
    st.cache_data.clear()
    st.rerun()

# ------------------------------------------
# 開いた瞬間に現在時刻基準で自動更新する処理
# ------------------------------------------
if not api_key_input:
    st.warning("⚠️ Streamlit SecretsにAPIキーを設定するか、サイドバーに入力してください。")
else:
    lat, lng = CITY_COORDINATES[selected_city]
    
    with st.spinner(f"最新の5大気象モデルデータを取得中 ({selected_city})..."):
        df_prob, df_precip = fetch_hourly_ensemble_data(lat, lng)
        
        if df_prob is not None and df_precip is not None:
            st.subheader(f"📊 {selected_city}の時系列モデル比較データ")
            
            # 5モデル用カラーパレット設定
            color_map = {
                "気象庁 (JMA)": "#1f77b4",     # 青
                "気象庁局地 (MSM)": "#17becf", # 水色
                "米国 (GFS)": "#ff7f0e",       # オレンジ
                "欧州 (ECMWF)": "#2ca02c",     # 緑
                "ドイツ (ICON)": "#d62728"     # 赤
            }
            
            tab1, tab2, tab3 = st.tabs(["🌧️ 降水確率 (%)", "💧 降水量 (mm/h)", "📋 Rawデータ表"])
            
            with tab1:
                df_prob_melted = df_prob.melt(id_vars=["時刻"], var_name="気象モデル", value_name="降水確率(%)")
                fig_prob = px.line(
                    df_prob_melted, x="時刻", y="降水確率(%)", color="気象モデル",
                    markers=True, title="現在時刻から12時間先までの降水確率推移",
                    color_discrete_map=color_map
                )
                fig_prob.update_yaxes(range=[0, 105])
                st.plotly_chart(fig_prob, use_container_width=True)
                
            with tab2:
                df_precip_melted = df_precip.melt(id_vars=["時刻"], var_name="気象モデル", value_name="降水量(mm)")
                fig_precip = px.line(
                    df_precip_melted, x="時刻", y="降水量(mm)", color="気象モデル",
                    markers=True, title="現在時刻から12時間先までの予想降水量推移",
                    color_discrete_map=color_map
                )
                st.plotly_chart(fig_precip, use_container_width=True)
                
            with tab3:
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("降水確率(%)")
                    st.dataframe(df_prob, hide_index=True)
                with col2:
                    st.caption("降水量(mm/h)")
                    st.dataframe(df_precip, hide_index=True)

            st.markdown("---")
            st.subheader("🤖 Geminiによるアンサンブル解析結論")
            
            with st.spinner("Geminiが5大モデルのデータを集計・解析中..."):
                ai_result = analyze_ensemble_with_gemini(api_key_input, selected_city, df_prob, df_precip)
                st.markdown(ai_result)

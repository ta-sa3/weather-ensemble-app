%%writefile app.py
import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from google import genai

# ==========================================
# 1. ページ基本設定
# ==========================================
st.set_page_config(
    page_title="マルチモデルAIアンサンブル気象予報",
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

# ==========================================
# 2. データ取得関数 (Open-Meteo API)
# ==========================================
@st.cache_data(ttl=1800)
def fetch_hourly_ensemble_data(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=precipitation_probability,precipitation&timezone=Asia%2FTokyo&models=jma_seamless,gfs_seamless,ecmwf_ifs025"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = [t.split("T")[1] for t in hourly.get("time", [])[:12]] # 直近12時間分
        
        prob_jma = hourly.get("precipitation_probability_jma_seamless", hourly.get("precipitation_probability", []))[:12]
        prob_gfs = hourly.get("precipitation_probability_gfs_seamless", [])[:12]
        prob_ecm = hourly.get("precipitation_probability_ecmwf_ifs025", [])[:12]
        
        precip_jma = hourly.get("precipitation_jma_seamless", hourly.get("precipitation", []))[:12]
        precip_gfs = hourly.get("precipitation_gfs_seamless", [])[:12]
        precip_ecm = hourly.get("precipitation_ecmwf_ifs025", [])[:12]

        prob_data = {
            "時刻": times,
            "気象庁 (JMA)": prob_jma,
            "米国 (GFS)": prob_gfs,
            "欧州 (ECMWF)": prob_ecm
        }
        
        precip_data = {
            "時刻": times,
            "気象庁 (JMA)": precip_jma,
            "米国 (GFS)": precip_gfs,
            "欧州 (ECMWF)": precip_ecm
        }
        
        return pd.DataFrame(prob_data), pd.DataFrame(precip_data)
    except Exception as e:
        st.error(f"気象データの取得に失敗しました: {e}")
        return None, None

# ==========================================
# 3. GeminiによるAI分析関数 (最新モデル gemini-3.6-flash を指定)
# ==========================================
def analyze_ensemble_with_gemini(api_key, city_name, df_prob, df_precip):
    client = genai.Client(api_key=api_key)
    
    combined_data = {
        "降水確率_時系列(%)": df_prob.to_dict(orient="records"),
        "降水量_時系列(mm)": df_precip.to_dict(orient="records")
    }
    
    prompt = f"""
あなたは高度な数値予報モデルの解析を行う専門気象予報士です。
以下の都市における直近12時間の【気象庁(JMA) / 米国(GFS) / 欧州(ECMWF)】の1時間ごと降水データをもとに、時間軸でのアンサンブル分析を行ってください。

【都市名】: {city_name}
【1時間ごとのモデル比較データ (JSON)】:
{json.dumps(combined_data, ensure_ascii=False, indent=2)}

【分析要件】
1. **降水量・降水タイミングの一致度**:
   - 3モデルの間で「降り始める時刻」「雨のピーク時刻」「雨が止む時刻」にズレがあるかを分析してください。
2. **予測確信度判定**:
   - 確信度を【高・中・低】の3段階で判定し、その理由を述べてください。
3. **時間帯別のリスクとアクション**:
   - ユーザーがいつから傘を持つべきか、どの時間帯に強雨のリスクがあるかを分かりやすくアドバイスしてください。

【出力フォーマット】
Markdown形式で、見出しを使って見やすく出力してください。
---
### ⏱️ 時間軸アンサンブル解析サマリー
- **降り始め予想時刻**: 
- **予測確信度**: 【高 / 中 / 低】

### 🌧️ 1時間ごとのモデル別見解・比較
- **降り始めのタイミング**: 
- **降水のピーク時間帯と強度**: 
- **各モデルの特徴とズレ**: 

### ⚠️ お出かけ時のリスクシナリオ・アドバイス
---
"""

    target_model = 'gemini-3.6-flash'
    last_error = None

    # サーバー混雑発生時は最大3回まで自動リトライ
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
st.title("🌀 マルチモデルAIアンサンブル気象予報")
st.caption("日本(JMA)・米国(GFS)・欧州(ECMWF)の3大気象モデルを折れ線グラフで比較し、Geminiが予測確信度を判定します。")

st.sidebar.header("⚙️ 設定")
api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Google AI Studioで発行したAPIキーを入力してください")
selected_city = st.sidebar.selectbox("対象エリアを選択", list(CITY_COORDINATES.keys()))

if st.sidebar.button("アンサンブル解析を実行", type="primary"):
    if not api_key_input:
        st.warning("⚠️ サイドバーに Gemini API キーを入力してください。")
    else:
        lat, lng = CITY_COORDINATES[selected_city]
        
        with st.spinner("気象モデルデータを取得＆解析中..."):
            df_prob, df_precip = fetch_hourly_ensemble_data(lat, lng)
            
            if df_prob is not None and df_precip is not None:
                st.subheader(f"📊 {selected_city}の時系列モデル比較データ")
                
                tab1, tab2, tab3 = st.tabs(["🌧️ 降水確率 (%)", "💧 降水量 (mm/h)", "📋 Rawデータ表"])
                
                with tab1:
                    df_prob_melted = df_prob.melt(id_vars=["時刻"], var_name="気象モデル", value_name="降水確率(%)")
                    fig_prob = px.line(
                        df_prob_melted, x="時刻", y="降水確率(%)", color="気象モデル",
                        markers=True, title="12時間先までの降水確率推移",
                        color_discrete_map={"気象庁 (JMA)": "#1f77b4", "米国 (GFS)": "#ff7f0e", "欧州 (ECMWF)": "#2ca02c"}
                    )
                    fig_prob.update_yaxes(range=[0, 105])
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                with tab2:
                    df_precip_melted = df_precip.melt(id_vars=["時刻"], var_name="気象モデル", value_name="降水量(mm)")
                    fig_precip = px.line(
                        df_precip_melted, x="時刻", y="降水量(mm)", color="気象モデル",
                        markers=True, title="12時間先までの予想降水量推移",
                        color_discrete_map={"気象庁 (JMA)": "#1f77b4", "米国 (GFS)": "#ff7f0e", "欧州 (ECMWF)": "#2ca02c"}
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
                ai_result = analyze_ensemble_with_gemini(api_key_input, selected_city, df_prob, df_precip)
                st.markdown(ai_result)

else:
    st.info("👈 サイドバーでAPIキーを入力し、「アンサンブル解析を実行」ボタンを押してください。")

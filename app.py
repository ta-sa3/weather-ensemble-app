with tab3:
        # グラフ切り替え用のインデックスをセッション状態に初期化
        if "chart_idx" not in st.session_state:
            st.session_state.chart_idx = 0

        times_24h = [t.split(" ")[1] for t in times_now[:24]]

        # 表示するグラフのデータリストを定義
        charts_info = [
            {
                "title": "🌡️ 気温の推移 (℃)",
                "data": pd.DataFrame({"時間": times_24h, "気温 (℃)": temps_now[:24]}).set_index("時間"),
                "type": "line"
            },
            {
                "title": "🌧️ 降水量の推移 (mm)",
                "data": pd.DataFrame({"時間": times_24h, "降水量 (mm)": precips_now[:24]}).set_index("時間"),
                "type": "bar"
            },
            {
                "title": "⚡ 大気安定度 CAPE (J/kg)",
                "data": pd.DataFrame({"時間": times_24h, "CAPE (J/kg)": capes_now[:24]}).set_index("時間"),
                "type": "line"
            }
        ]

        # 矢印ボタンとタイトル表示領域（3列レイアウト）
        col_prev, col_title, col_next = st.columns([1, 4, 1])

        # 「◀ 前へ」ボタンの挙動
        if col_prev.button("◀ 前へ", use_container_width=True):
            st.session_state.chart_idx = (st.session_state.chart_idx - 1) % len(charts_info)
            st.rerun()

        # 「次へ ▶」ボタンの挙動
        if col_next.button("次へ ▶", use_container_width=True):
            st.session_state.chart_idx = (st.session_state.chart_idx + 1) % len(charts_info)
            st.rerun()

        # 現在のグラフ情報を取得
        current_chart = charts_info[st.session_state.chart_idx]

        # 中央にタイトルと現在のページ番号を表示
        col_title.markdown(
            f"<h4 style='text-align: center; margin: 0;'>{current_chart['title']}</h4>"
            f"<p style='text-align: center; color: gray; margin: 0;'>({st.session_state.chart_idx + 1} / {len(charts_info)})</p>",
            unsafe_allow_html=True
        )

        st.markdown("")  # グラフとの間に少し余白

        # 該当するタイプのグラフを描画
        if current_chart["type"] == "line":
            st.line_chart(current_chart["data"])
        else:
            st.bar_chart(current_chart["data"])

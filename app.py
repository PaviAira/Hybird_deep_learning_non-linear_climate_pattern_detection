import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import plotly.graph_objects as go

st.set_page_config(
    page_title="Hybrid Climate Intelligence Dashboard",
    layout="wide"
)
st.title("🌍 Hybrid Deep Learning Climate Intelligence Dashboard")
st.markdown("""
<style>
.dashboard-card {
    background: rgba(15, 20, 30, 0.45);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);

    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 25px;

    padding: 35px;
    margin-bottom: 20px;

    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.feature-card {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 15px;
    text-align: center;
    font-weight: 600;
    margin: 5px;
    font-size: 19px;
}

.metric-card {
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
}

.metric-label {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}
</style>
""", unsafe_allow_html=True)
latest_temp = 0.869
max_temp = 1.276
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="feature-card">
        📈<br>
        Trend Analysis
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="feature-card">
        🌊<br>
        Seasonal Variations
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="feature-card">
        ⚠️<br>
        Climate Anomaly Detection
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="feature-card">
        🔮<br>
        Future Forecasting
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🌡 Latest Global Temperature</div>
        <div class="metric-value">{latest_temp:.3f} °C</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">🔥 Maximum Recorded</div>
        <div class="metric-value">{max_temp:.3f} °C</div>
    </div>
    """, unsafe_allow_html=True)

# Background styling for the main app
st.markdown("""
<style>
.stApp {
    background-image: 
    linear-gradient(
        rgba(0,0,0,0.35),
        rgba(0,0,0,0.45)
    ),
    url("https://images.stockcake.com/public/1/d/1/1d1250cf-09ca-43c6-8aed-1718674e7aba/earth-from-above-stockcake.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)

# Sidebar and inputs custom UI styling
st.markdown("""
<style>
/* SIDEBAR PANEL */
[data-testid="stSidebar"] {
    background:
    radial-gradient(circle at top left, rgba(0,245,212,0.75) 0%, transparent 28%),
    radial-gradient(circle at bottom right, rgba(0,187,249,0.65) 0%, transparent 30%),
    linear-gradient(to bottom, #020617, #050816);
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0 0 25px rgba(0,245,212,0.08),
        0 8px 32px rgba(0,0,0,0.45);
    color: white;
}

/* REMOVE TRANSPARENCY */
[data-testid="stSidebar"] > div:first-child {
    background: transparent;
}

/* TEXT */
[data-testid="stSidebar"] * {
    color: white;
}

/* DROPDOWN */
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* SLIDER */
.stSlider > div[data-baseweb="slider"] {
    padding-top: 10px;
}

/* METRIC TEXT */
.css-1d391kg {
    color: white;
}
</style>
""", unsafe_allow_html=True)
# Shared layout configuration to guarantee visible text, grids, and settings over white backgrounds
white_plot_layout_defaults = dict(
    template="plotly_white",
    paper_bgcolor="#f8fafc",
    plot_bgcolor="#f8fafc",
    font=dict(color="#1e293b", size=12),
    legend=dict(
        font=dict(color="#334155"),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="#cbd5e1",
        borderwidth=1
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor="#e2e8f0",
        linecolor="#64748b",
        tickfont=dict(color="#334155")
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#e2e8f0",
        linecolor="#64748b",
        tickfont=dict(color="#334155")
    )
)

plotly_display_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': []
}

# ============================================================
# LOAD MODEL + SCALERS
# ============================================================
@st.cache_resource
def load_modeling_assets():
    model = tf.keras.models.load_model(r"D:\BCA & MSC Core Notes\MSC\MSC 2ND YEAR\MSC Major project\Codes\hybrid_climate_model.keras")
    with open(r"D:\BCA & MSC Core Notes\MSC\MSC 2ND YEAR\MSC Major project\Codes\scaler_X.pkl","rb") as f:
        scaler_X = pickle.load(f)
    with open(r"D:\BCA & MSC Core Notes\MSC\MSC 2ND YEAR\MSC Major project\Codes\scaler_y.pkl","rb") as f:
        scaler_y = pickle.load(f)
    return model, scaler_X, scaler_y

try:
    hybrid_model, scaler_X, scaler_y = load_modeling_assets()
except Exception as e:
    st.error(f"❌ Error loading model assets: {e}")

# ============================================================
# LOAD DATASET
# ============================================================
@st.cache_data
def load_dataset():
    df = pd.read_csv(r"D:\BCA & MSC Core Notes\MSC\MSC 2ND YEAR\MSC Major project\Dataset\truncated_data.csv",index_col=0)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

df = load_dataset()

# ============================================================
# FEATURE CONFIGURATION
# ============================================================
target_col = "Global_Temp_Avg"
feature_cols = [
    col for col in df.columns
    if col not in ['Date', target_col]
]
sequence_length = 60

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("🎯 Configuration Panel")
min_year = int(df['Date'].dt.year.min())
max_year = int(df['Date'].dt.year.max())
analysis_type = st.sidebar.selectbox(
    "Select Climatic Analysis Option:",
    [
        "Trend Analysis",
        "Seasonal Variations",
        "Anomaly Detection",
        "Future Forecasting"
    ])

# Conditionally display the year range slider
if analysis_type != "Future Forecasting":
    start_year, end_year = st.sidebar.slider(
        "Select Year Range:",
        min_value=min_year,
        max_value=max_year,
        value=(2000, 2025)
    )
else:
    start_year, end_year = min_year, max_year 

# ============================================================
# SEQUENCE GENERATOR
# ============================================================
def generate_sequences_for_range(dataframe, start_yr, end_yr):
    mask = (
        (dataframe['Date'].dt.year >= start_yr) &
        (dataframe['Date'].dt.year <= end_yr)
    )
    filtered_df = dataframe[mask].copy().reset_index(drop=True)
    if len(filtered_df) <= sequence_length:
        return None, None, None
    X_scaled = scaler_X.transform(filtered_df[feature_cols])
    y_scaled = scaler_y.transform(filtered_df[[target_col]])
    X_seq = []
    y_seq = []
    dates_seq = []
    for i in range(len(filtered_df) - sequence_length):
        X_seq.append(X_scaled[i:i + sequence_length])
        y_seq.append(y_scaled[i + sequence_length])
        dates_seq.append(filtered_df['Date'].iloc[i + sequence_length])
    return np.array(X_seq), np.array(y_seq), dates_seq

X_window, y_window, output_dates = generate_sequences_for_range(df,start_year,end_year)

# ============================================================
# VALIDATION CHECK & DASHBOARD OUTPUTS
# ============================================================
if X_window is None:
    st.warning(f"⚠️ Selected range is too short. Please select at least {sequence_length // 12} years.")
else:
    if analysis_type == "Future Forecasting":
        st.subheader(f"📊 Dashboard Outcome: {analysis_type}")
    else:
        st.subheader(f"📊 Dashboard Outcome: {analysis_type} ({start_year} - {end_year})")
    
    # ========================================================
    # 1. TREND ANALYSIS (TRANSFORMER)
    # ========================================================
    if analysis_type == "Trend Analysis":
      with st.spinner("Extracting long-term climate trend patterns..."):
        trend_df = df[
            (df['Date'].dt.year >= start_year) &
            (df['Date'].dt.year <= end_year)
        ].copy()
        yearly_trend = (trend_df.set_index('Date')['Global_Temp_Avg'].resample('YE').mean())
        # TRANSFORMER-INSPIRED LONG TERM SMOOTHING
        transformer_trend = yearly_trend.rolling(window=5,center=True).mean()
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=yearly_trend.index,
                y=yearly_trend.values,
                mode='lines',
                name='Observed Climate Signal',
                line=dict(color="#0F6506", width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=transformer_trend.index,
                y=transformer_trend.values,
                mode='lines',
                name='Transformer Long-Term Trend',
                line=dict(color='#FF4B4B', width=4)
            )
        )
        fig.add_hline(y=yearly_trend.mean(),line_dash="dot",line_color="gray")
        
        fig.update_layout(
            **white_plot_layout_defaults,
            title=dict(text="Long-Term Global Climate Trend Evolution", font=dict(color="#0f172a", size=18)),
            xaxis_title="Timeline",
            yaxis_title="Global Temperature Anomaly (°C)",
            height=700,
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, theme=None, config=plotly_display_config)

        # INSIGHTS ENGINE - TREND ANALYSIS
        first_yr_val = yearly_trend.values[0]
        last_yr_val = yearly_trend.values[-1]
        overall_diff = last_yr_val - first_yr_val
        trend_direction = "an upward warming vector" if overall_diff > 0 else "a cooling/stabilizing vector"
        
        with st.expander("📚 Interpret the Plot: Understanding Trend line", expanded=True):
            st.markdown("""
                <style>
                .interpret-box {
                    background: rgba(15, 15, 15, 0.45);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 15px;
                    margin: 20px 25px;
                    padding: 25px;
                    color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                </style>
                """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="interpret-box"; style="color: #FFFFFF; line-height: 1.6;">
                    <ul>
                        <li><strong>What this plot shows:</strong> This graph illustrates how global temperature anomalies have changed over time. The green line represents the observed yearly temperature values, while the red line shows the long-term climate trend after smoothing out short-term fluctuations. This helps reveal the overall direction of climate change more clearly.</li>
                        <li><strong>The Red Trend Line:</strong> The red line highlights the underlying long-term warming pattern learned from the historical data. While the green line may rise and fall from year to year due to natural climate variability, the red trend line focuses on the broader climate movement. The steady upward movement of this line indicates a continuous increase in global temperatures over the decades.</li>
                        <li><strong>Scientific Takeaway:</strong> The overall upward trend suggests that the Earth's climate has been warming significantly over time. Temporary decreases or flat periods can occur due to natural factors such as volcanic eruptions, ocean circulation patterns, or La Niña events. However, the long-term increase is primarily associated with rising greenhouse gas concentrations and increased heat retention in the atmosphere. The persistent growth of the trend line indicates that warming is not a short-term fluctuation but a long-term climate pattern.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ========================================================
    # 2. SEASONAL VARIATIONS
    # ========================================================
    elif analysis_type == "Seasonal Variations":
        with st.spinner("Analyzing seasonal climate dynamics..."):
            seasonal_extractor = tf.keras.Model(
                inputs=hybrid_model.input,
                outputs=hybrid_model.get_layer("LSTM_Feature_Extraction").output)
            lstm_embeddings = seasonal_extractor.predict(X_window,verbose=0)
            seasonal_signal = np.mean(lstm_embeddings,axis=1)
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=output_dates,
                    y=seasonal_signal,
                    mode='lines',
                    name="Seasonal Climate Signal",
                    line=dict(color="#5A5AEE", width=3)
                )
            )
            fig.add_hline(y=0,line_dash="dot",line_color="gray")
            
            fig.update_layout(
                **white_plot_layout_defaults,
                title=dict(text="LSTM-Based Seasonal Climate Dynamics", font=dict(color="#0f172a", size=18)),
                xaxis_title="Timeline",
                yaxis_title="Temporal Seasonal Index",
                height=650
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, config=plotly_display_config)

            # INSIGHTS ENGINE - SEASONAL VARIATIONS
            with st.expander("📚 Interpret the Plot: Understanding Seasonal Variations", expanded=True):
                st.markdown("""
                <style>
                .interpret-box {
                    background: rgba(15, 15, 15, 0.45);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 15px;
                    margin: 20px 25px;
                    padding: 25px;
                    color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="interpret-box"; style="color: #FFFFFF; line-height: 1.6;">
                    <ul>
                        <li><strong>What this plot shows:</strong> This graph illustrates the seasonal climate patterns identified by the LSTM branch of the hybrid deep learning model. The model analyzes historical climate sequences and extracts recurring temperature variations that occur over time. These patterns help reveal how climate conditions naturally fluctuate across different periods.</li>
                        <li><strong>The Blue Seasonal Signal:</strong> The blue line represents the seasonal climate signal learned by the LSTM network. Peaks in the graph indicate periods where temperatures are generally higher than the seasonal average, while dips represent periods where temperatures are lower than the seasonal average. The continuous oscillating pattern reflects the natural warming and cooling cycles present within the climate system.</li>
                        <li><strong>Scientific Takeaway:</strong> The repeating rise-and-fall pattern demonstrates the presence of regular seasonal climate behavior. These variations are influenced by factors such as changes in solar radiation, atmospheric circulation, ocean currents, and natural weather cycles. While seasonal fluctuations are expected, noticeable changes in the height, frequency, or intensity of the peaks and troughs may indicate shifts in climate behavior over time. The gradual upward tendency observed in recent decades suggests that seasonal patterns are occurring within an overall warming climate system.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # =======================================================
    # 3. ANOMALY DETECTION
    # ========================================================
    elif analysis_type == "Anomaly Detection":
        with st.spinner("Detecting abnormal climate anomalies..."):
            preds_scaled = hybrid_model.predict(X_window,verbose=0)
            preds_celsius = scaler_y.inverse_transform(preds_scaled).flatten()
            actuals_celsius = scaler_y.inverse_transform(y_window).flatten()
            errors = np.abs(actuals_celsius - preds_celsius)
            z_scores = (errors - np.mean(errors)) / np.std(errors)
            anomalies = z_scores > 2.0
            anomaly_dates = [
                output_dates[i]
                for i in range(len(anomalies))
                if anomalies[i]
            ]
            anomaly_vals = [
                actuals_celsius[i]
                for i in range(len(anomalies))
                if anomalies[i]
            ]
            
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=output_dates,
                    y=actuals_celsius,
                    mode='lines',
                    name="Observed Temperature",
                    line=dict(color="#7703A4", width=2)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=output_dates,
                    y=preds_celsius,
                    mode='lines',
                    name="Expected Climate Baseline",
                    line=dict(color="#F36BDA", dash="dash", width=2)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=anomaly_dates,
                    y=anomaly_vals,
                    mode='markers',
                    name="Climate Anomaly",
                    marker=dict(color="red", size=10, symbol="triangle-up")
                )
            )
            
            fig.update_layout(
                **white_plot_layout_defaults,
                title=dict(text="Hybrid Deep Climate Anomaly Mapping", font=dict(color="#0f172a", size=18)),
                xaxis_title="Timeline",
                yaxis_title="Temperature Anomaly (°C)",
                height=700
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, config=plotly_display_config)

            # INSIGHTS ENGINE - ANOMALY DETECTION
            total_anomalies = len(anomaly_dates)
            with st.expander("📚 Interpret the Plot: Understanding Climate Anomalies", expanded=True):
                st.markdown("""
                <style>
                .interpret-box {
                    background: rgba(15, 15, 15, 0.45);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 15px;
                    margin: 20px 25px;
                    padding: 25px;
                    color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="interpret-box"; style="color: #FFFFFF; line-height: 1.6;">
                    <ul>
                        <li><strong>What this plot shows:</strong> This graph compares the observed global temperature values with the climate baseline estimated by the hybrid model. The solid purple line represents the actual recorded temperatures, while the dashed pink line represents the expected climate behavior. By comparing these two patterns, the model can identify periods where temperatures deviate significantly from normal conditions.</li>
                        <li><strong>What do the Red Markers mean?</strong> The red markers highlight climate anomalies, which occur when the observed temperature differs noticeably from the expected baseline. These points indicate unusual climate events where temperatures are either much higher or lower than what the model considers typical for that period. A larger number of anomaly points suggests greater climate instability and more frequent departures from normal conditions.</li>
                        <li><strong>Physical World Causes:</strong> Climate anomalies are often associated with significant environmental and atmospheric events such as strong El Niño episodes, volcanic eruptions, ocean circulation changes, heatwaves, or other extreme weather conditions. The increasing concentration of anomaly markers in recent decades suggests that unusual temperature events are becoming more frequent. This pattern indicates growing climate variability and highlights the increasing occurrence of extreme climate behavior within an overall warming climate system.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # ========================================================
    # 4. FUTURE FORECASTING
    # ========================================================
    elif analysis_type == "Future Forecasting":
        with st.spinner("Generating climate forecast..."):
            temp_df = df.copy()
            temp_df['Date'] = pd.to_datetime(temp_df['Date'])
            temp_df.set_index('Date', inplace=True)
            historical_yearly = (temp_df['Global_Temp_Avg'].resample('YE').mean().to_frame())
            historical_yearly.columns = ['Anomaly']
            
            # Monthly future dates: 2026 → 2030
            forecast_dates = pd.date_range(start='2026-01-01',end='2030-12-01',freq='MS')
            
            # Latest trend value
            current_trend = temp_df['Trend_Component'].iloc[-1]
            # Average warming drift from last 10 years
            annual_drift = (temp_df['Warming_Velocity_10y'].tail(120).mean()) / 120
            # Monthly seasonal behavior
            seasonal_cycle = (temp_df.groupby(temp_df.index.month)['Seasonal_Component'].mean())
            
            forecast_list = []
            for i, date in enumerate(forecast_dates):
                # Trend progression
                projected_trend = current_trend + (i * annual_drift)
                # Seasonal influence
                month_effect = seasonal_cycle[date.month]
                # Final forecast
                forecast_value = projected_trend + month_effect
                forecast_list.append(forecast_value)
                
            # FORECAST DATAFRAME
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Global_Temp_Avg_Forecast': forecast_list
            })
            forecast_df.set_index('Date', inplace=True)
            forecast_df = forecast_df.round(3)
            forecast_yearly = (forecast_df.resample('YE').mean())
            forecast_yearly.columns = ['Anomaly']
            
            yearly_master = pd.concat([historical_yearly, forecast_yearly])
            yearly_master.index.name = 'Year'
            yearly_master['Source'] = [
                'Historical' if y < 2026 else 'Forecast'
                for y in yearly_master.index.year
            ]
            
            # GRAPH PREPARATION
            historical_plot = yearly_master[yearly_master['Source'] == 'Historical']
            forecast_plot = yearly_master[yearly_master['Source'] == 'Forecast']
            
            fig = go.Figure()
            # Historical Climate Trend
            fig.add_trace(
                go.Scatter(
                    x=historical_plot.index.year,
                    y=historical_plot['Anomaly'],
                    mode='lines',
                    name='Historical Climate Trend',
                    line=dict(color="#A10487", width=3)
                )
            )
            # Forecast Trend
            fig.add_trace(
                go.Scatter(
                    x=forecast_plot.index.year,
                    y=forecast_plot['Anomaly'],
                    mode='lines+markers',
                    name='Future Forecast',
                    line=dict(color='#d9a700', width=4, dash='dash'), 
                    marker=dict(size=9)
                )
            )
            # Transition Connector
            fig.add_trace(
                go.Scatter(
                    x=[historical_plot.index.year[-1], forecast_plot.index.year[0]],
                    y=[historical_plot['Anomaly'].iloc[-1], forecast_plot['Anomaly'].iloc[0]],
                    mode='lines',
                    showlegend=False,
                    line=dict(color='#d9a700', dash='dot', width=2)
                )
            )
            
            fig.update_layout(
                **white_plot_layout_defaults,
                title=dict(text="Future Global Climate Forecast (2026-2030)", font=dict(color="#0f172a", size=18)),
                xaxis_title="Year",
                yaxis_title="Global Temperature Anomaly (°C)",
                height=700,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True, theme=None, config=plotly_display_config)
            
            # INSIGHTS ENGINE - FUTURE FORECASTING
            final_2030_val = forecast_yearly['Anomaly'].iloc[-1]
            with st.expander("📚 Interpret the Plot: Understanding Predictive Climate Baselines", expanded=True):
                st.markdown("""
                <style>
                .interpret-box {
                    background: rgba(15, 15, 15, 0.45);
                    backdrop-filter: blur(12px);
                    -webkit-backdrop-filter: blur(12px);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 15px;
                    margin: 20px 25px;
                    padding: 25px;
                    color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                }
                </style>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="interpret-box"; style="color: #FFFFFF; line-height: 1.6;">
                    <ul>
                        <li><strong>What this plot shows:</strong> This graph uses historical climate data and the patterns learned from the model to estimate future global temperature changes up to the year 2030. It combines long-term warming trends with recurring climate patterns to project how temperatures may change in the coming years.</li>
                        <li><strong>The Golden Dash Line:</strong> The golden dashed line represents the forecasted temperatures from 2026 to 2030. These values are predictions made using past climate behavior and warming trends. According to the forecast, the global temperature anomaly is expected to reach approximately 1.035 °C by the year 2030.</strong>.</li>
                        <li><strong>Scientific Takeaway:</strong> The continuously rising trend suggests that global temperatures are likely to keep increasing throughout the decade. This happens because the Earth's climate system, especially the oceans, stores large amounts of heat and releases it slowly over time. In addition, climate feedback processes such as melting ice and increased greenhouse gas concentrations can further contribute to warming. As a result, temperature increases may continue even if climate conditions begin to improve.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # ====================================================
            # FORECAST TABLE
            # ====================================================
            st.subheader("📋 Forecasted Climate Anomalies")
            display_table = forecast_yearly.copy()
            display_table.index = display_table.index.strftime('%Y')
            st.dataframe(
                display_table.style.format({'Anomaly': '{:.3f}'}),
                use_container_width=True
            )
# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #fffefc; font-size: 0.95rem; line-height: 1.6;">
        <strong>About this Project: An advanced climate intelligence system leveraging hybrid 
        Transformer attention structures and LSTM network dynamics to analyze historical global temperature trends, 
        map environmental anomalies, and generate predictive future climate baselines. </strong>
    </div>
    """, 
    unsafe_allow_html=True
)
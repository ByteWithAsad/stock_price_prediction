# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="StockPredict AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS for Beautiful UI
# ----------------------------
st.markdown("""
<style>
    /* Main container */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Title styles */
    .main-title {
        color: white;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .sub-title {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cards */
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        transition: transform 0.3s ease;
        height: 100%;
        border-top: 5px solid;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Timeline */
    .timeline-item {
        display: flex;
        align-items: center;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    .timeline-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        color: #667eea;
    }
    
    /* Progress bars */
    .confidence-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #00b09b, #96c93d);
        border-radius: 4px;
    }
    
    /* Section headers */
    .section-header {
        color: #333;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Tooltips */
    .tooltip-icon {
        color: #667eea;
        cursor: help;
        margin-left: 0.5rem;
    }
    
    /* Sentiment Analysis Styles */
    .sentiment-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 2rem 0;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    
    .sentiment-header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .sentiment-emoji-large {
        font-size: 2.5rem;
        margin-right: 1rem;
    }
    
    .sentiment-main-title {
        color: #1e40af;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .sentiment-subtitle {
        color: #64748b;
        margin: 0.25rem 0 0 0;
    }
    
    .sentiment-message-box {
        font-size: 1.2rem;
        color: #334155;
        line-height: 1.6;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: white;
        border-radius: 10px;
        border-left: 4px solid #60a5fa;
    }
    
    .observations-box {
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 2px solid #e2e8f0;
    }
    
    .observations-title {
        color: #475569;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .observations-list {
        color: #475569;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .observations-list li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header with Hero Section
# ----------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div class="main-container">
        <h1 class="main-title">📊 StockPredict AI</h1>
        <p class="sub-title">Your Personal AI-Powered Stock Market Assistant</p>
        <div style="text-align: center; margin-top: 1.5rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; 
            display: inline-block; margin: 0 0.5rem;">
                🤖 AI-Powered
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; 
            display: inline-block; margin: 0 0.5rem;">
                📈 Real-Time Analysis
            </span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; 
            display: inline-block; margin: 0 0.5rem;">
                🎯 Easy to Use
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# How It Works Section
# ----------------------------
st.markdown('<h2 class="section-header">🎯 How It Works</h2>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="prediction-card">
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">📥</span>
        </div>
        <h3 style="text-align: center; color: #333;">1. Upload Data</h3>
        <p style="color: #666; text-align: center;">We analyze your historical stock data to understand patterns</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="prediction-card">
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">🤖</span>
        </div>
        <h3 style="text-align: center; color: #333;">2. AI Analysis</h3>
        <p style="color: #666; text-align: center;">Our smart algorithm predicts future trends automatically</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="prediction-card">
        <div style="text-align: center; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">📊</span>
        </div>
        <h3 style="text-align: center; color: #333;">3. Get Results</h3>
        <p style="color: #666; text-align: center;">Receive clear, actionable insights in simple language</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Input Section with Visual Timeline
# ----------------------------
st.markdown('<h2 class="section-header">🔮 Make Your Prediction</h2>', unsafe_allow_html=True)

# Create a timeline visualization for forecast period
col1, col2 = st.columns([2, 2])

with col1:
    st.markdown("""
    <div class="prediction-card" style="border-top-color: #00b09b;">
        <h3 style="color: #333; margin-bottom: 1.5rem;">📅 Forecast Timeline</h3>
    """, unsafe_allow_html=True)
    
    # Interactive timeline
    forecast_days = st.slider(
        "**How many days ahead do you want to see?**",
        min_value=1,
        max_value=30,
        value=7,
        help="Slide to choose prediction period (1-30 days)"
    )
    
    # Visual timeline indicator
    progress = forecast_days / 30
    st.markdown(f"""
    <div style="margin-top: 1rem;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Short-term (1-10 days)</span>
            <span>Medium-term (11-20 days)</span>
            <span>Long-term (21-30 days)</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {progress*100}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="prediction-card" style="border-top-color: #667eea;">
        <h3 style="color: #333; margin-bottom: 1.5rem;">💡 What This Means</h3>
    """, unsafe_allow_html=True)
    
    # Timeline items as separate markdown calls
    if forecast_days <= 7:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-icon">🎯</div>
            <div>
                <strong>High Precision Mode (1-7 days)</strong>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Best accuracy for immediate trading decisions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif forecast_days <= 14:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-icon">📊</div>
            <div>
                <strong>Strategic Planning (8-14 days)</strong>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Ideal for weekly investment strategies</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif forecast_days <= 21:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-icon">📈</div>
            <div>
                <strong>Trend Analysis (15-21 days)</strong>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Good for identifying medium-term trends</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="timeline-item">
            <div class="timeline-icon">🔮</div>
            <div>
                <strong>Long-term Outlook (22-30 days)</strong>
                <p style="margin: 0; color: #666; font-size: 0.9rem;">Broad trend insights with wider confidence intervals</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Always show the full timeline explanation
    st.markdown("""
    <div style="margin-top: 1.5rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
        <p style="margin: 0; color: #666; font-size: 0.9rem;"><strong>💡 Tip:</strong> Shorter periods give more accurate predictions, while longer periods show broader trends.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Load Data Function
# ----------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('stocks.csv', parse_dates=['Date'], index_col='Date')
        df = df.asfreq('B')
        df['Close'] = pd.to_numeric(df['Close'].str.replace(',', ''), errors='coerce')
        df['Close'].interpolate(inplace=True)
        return df
    except:
        # Create sample data for demo if file not found
        dates = pd.date_range(start='2023-01-01', periods=200, freq='B')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(200) * 2)
        df = pd.DataFrame({'Close': prices}, index=dates)
        return df

# ----------------------------
# Train Model Function
# ----------------------------
@st.cache_data
def train_model(_df):
    try:
        model = SARIMAX(_df['Close'],
                       order=(1,1,1),
                       seasonal_order=(1,1,1,5)).fit(disp=False)
        return model
    except:
        return None

# ----------------------------
# Generate Forecast
# ----------------------------
df = load_data()
model = train_model(df)

if model and df is not None:
    # Generate forecast
    forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), 
                                  periods=forecast_days, 
                                  freq='B')
    
    forecast = model.get_forecast(steps=forecast_days)
    forecast_series = pd.Series(forecast.predicted_mean, index=forecast_index)
    
    conf_int = forecast.conf_int()
    lower_series = pd.Series(conf_int['lower Close'].values, index=forecast_index)
    upper_series = pd.Series(conf_int['upper Close'].values, index=forecast_index)
    
    # Calculate key metrics
    current_price = df['Close'].iloc[-1]
    final_prediction = forecast_series.iloc[-1]
    avg_prediction = forecast_series.mean()
    price_change_pct = ((final_prediction - current_price) / current_price) * 100
    avg_change_pct = ((avg_prediction - current_price) / current_price) * 100
    
    # Calculate confidence level
    avg_uncertainty = (upper_series - lower_series).mean()
    confidence_level = max(0, 100 - (avg_uncertainty / current_price * 500))
    
    # ----------------------------
    # Key Predictions Dashboard
    # ----------------------------
    st.markdown('<h2 class="section-header">📊 Your Stock Predictions</h2>', unsafe_allow_html=True)
    
    # Create metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card" style="border-top-color: #00b09b;">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div style="color: #666; font-size: 0.9rem;">Today's closing price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "🟢" if price_change_pct > 0 else "🔴"
        st.markdown(f"""
        <div class="prediction-card" style="border-top-color: #667eea;">
            <div class="metric-label">Predicted Price {color}</div>
            <div class="metric-value">${final_prediction:.2f}</div>
            <div style="color: {'#10b981' if price_change_pct > 0 else '#ef4444'}; font-weight: 600;">
                {price_change_pct:+.1f}% change
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prediction-card" style="border-top-color: #f59e0b;">
            <div class="metric-label">Average Forecast</div>
            <div class="metric-value">${avg_prediction:.2f}</div>
            <div style="color: #666; font-size: 0.9rem;">
                Average over next {forecast_days} days
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="prediction-card" style="border-top-color: #8b5cf6;">
            <div class="metric-label">AI Confidence</div>
            <div class="metric-value">{confidence_level:.0f}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_level}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ----------------------------
    # Interactive Visualizations
    # ----------------------------
    st.markdown('<h2 class="section-header">📈 Visual Forecast Analysis</h2>', unsafe_allow_html=True)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["🎯 Main Forecast", "📊 Confidence Range", "📅 Daily Details"])
    
    with tab1:
        # Main interactive chart
        fig = go.Figure()
        
        # Historical data with trend line
        hist_data = df['Close'].iloc[-100:]
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.values,
            mode='lines',
            name='Historical Trend',
            line=dict(color='#667eea', width=3),
            hovertemplate='<b>Date:</b> %{x|%b %d}<br><b>Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        # Forecast with gradient effect
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_series.values,
            mode='lines+markers',
            name='AI Prediction',
            line=dict(color='#10b981', width=4, dash='solid'),
            marker=dict(size=8, color='#10b981'),
            hovertemplate='<b>Prediction:</b> $%{y:.2f}<br><b>Date:</b> %{x|%b %d}<extra></extra>'
        ))
        
        # Add current price marker
        fig.add_trace(go.Scatter(
            x=[df.index[-1]],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(size=15, color='#ef4444', symbol='star'),
            hovertemplate='<b>Current Price:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f'📈 Stock Price Forecast - Next {forecast_days} Days',
                font=dict(size=24, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(248, 249, 250, 0.9)',
            paper_bgcolor='white',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Confidence interval visualization
        fig2 = go.Figure()
        
        # Create gradient confidence interval
        x_rev = forecast_index[::-1]
        
        fig2.add_trace(go.Scatter(
            x=list(forecast_index) + list(x_rev),
            y=list(upper_series.values) + list(lower_series.values[::-1]),
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Range',
            hovertemplate='Confidence Range<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_series.values,
            mode='lines',
            name='Most Likely Path',
            line=dict(color='#10b981', width=3),
            hovertemplate='<b>Predicted:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=forecast_index,
            y=lower_series.values,
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.5)', dash='dash'),
            name='Lower Bound',
            hovertemplate='<b>Minimum Expected:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig2.add_trace(go.Scatter(
            x=forecast_index,
            y=upper_series.values,
            mode='lines',
            line=dict(color='rgba(16, 185, 129, 0.5)', dash='dash'),
            name='Upper Bound',
            hovertemplate='<b>Maximum Expected:</b> $%{y:.2f}<extra></extra>'
        ))
        
        fig2.update_layout(
            title=dict(
                text='📊 Prediction Confidence Range',
                font=dict(size=24, color='#333'),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Confidence explanation
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin-top: 1rem;">
            <h4 style="color: #333; margin-bottom: 0.5rem;">📝 Understanding Confidence Ranges</h4>
            <p style="color: #666; margin: 0;">
                The <strong style="color: #10b981;">green area</strong> shows where the stock price is most likely to be.
                <br>
                Wider ranges mean more uncertainty, while narrower ranges indicate higher confidence.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        # Daily predictions table with emojis
        st.markdown("### 📅 Daily Forecast Breakdown")
        
        # Create enhanced dataframe
        forecast_df = pd.DataFrame({
            '📅 Date': forecast_index.strftime('%a, %b %d'),
            '📊 Day': [f'Day {i+1}' for i in range(forecast_days)],
            '💰 Predicted': ['${:,.2f}'.format(x) for x in forecast_series.values],
            '📈 Change': ['{:+.2f}%'.format(((x - current_price) / current_price * 100)) 
                         for x in forecast_series.values],
            '🎯 Confidence': [f'{(1 - (u - l) / x) * 100:.0f}%' 
                           for x, l, u in zip(forecast_series.values, lower_series.values, upper_series.values)]
        })
        
        # Display with alternating row colors
        st.dataframe(
            forecast_df,
            column_config={
                "📅 Date": "Date",
                "📊 Day": "Day",
                "💰 Predicted": "Predicted Price",
                "📈 Change": "Change from Today",
                "🎯 Confidence": "AI Confidence"
            },
            hide_index=True,
            use_container_width=True
        )
    
    # ----------------------------
    # MARKET SENTIMENT ANALYSIS SECTION (FIXED - NO RAW HTML)
    # ----------------------------
    st.markdown('<h2 class="section-header">📊 Market Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    # Calculate sentiment metrics
    avg_daily_movement = abs(price_change_pct / forecast_days)
    
    # Create a container for the sentiment card
    sentiment_container = st.container()
    
    with sentiment_container:
        # Apply the sentiment container style
        st.markdown('<div class="sentiment-container">', unsafe_allow_html=True)
        
        # Header with emoji and title
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.markdown('<div class="sentiment-emoji-large">📊</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<h2 class="sentiment-main-title">👍 Positive Outlook</h2>', unsafe_allow_html=True)
            st.markdown(f'<p class="sentiment-subtitle">Based on {forecast_days}-day forecast</p>', unsafe_allow_html=True)
        
        # Message box
        st.markdown('<div class="sentiment-message-box">Moderate growth expected. Good time for strategic investments.</div>', unsafe_allow_html=True)
        
        # Observations section
        st.markdown('<div class="observations-box">', unsafe_allow_html=True)
        st.markdown('<p class="observations-title"><strong>Key Observations:</strong></p>', unsafe_allow_html=True)
        
        # Create the observations list
        observations_html = f"""
        <ul class="observations-list">
            <li>Predicted {price_change_pct:+.1f}% change over {forecast_days} days</li>
            <li>Average daily movement: {avg_daily_movement:.2f}%</li>
            <li>Confidence level: {confidence_level:.0f}%</li>
        </ul>
        """
        st.markdown(observations_html, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close observations-box
        st.markdown('</div>', unsafe_allow_html=True)  # Close sentiment-container
    
    # ----------------------------
    # AI Insights Section
    # ----------------------------
    st.markdown('<h2 class="section-header">🤖 AI Insights</h2>', unsafe_allow_html=True)
    
    # Generate insights based on prediction
    if price_change_pct > 5:
        sentiment = "📈 Bullish"
        insight = "Strong upward trend predicted. Consider holding or increasing position."
        emoji = "🚀"
        color = "#10b981"
    elif price_change_pct > 0:
        sentiment = "👍 Positive"
        insight = "Moderate growth expected. Good time for strategic investments."
        emoji = "📊"
        color = "#3b82f6"
    elif price_change_pct > -5:
        sentiment = "🤔 Neutral"
        insight = "Stable performance expected. Monitor market conditions closely."
        emoji = "⚖️"
        color = "#f59e0b"
    else:
        sentiment = "📉 Bearish"
        insight = "Downward trend predicted. Consider reviewing your strategy."
        emoji = "🛡️"
        color = "#ef4444"
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-card">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2rem; margin-right: 1rem;">{emoji}</span>
                <div>
                    <h3 style="margin: 0; color: {color};">{sentiment} Outlook</h3>
                    <p style="margin: 0; color: #666;">Based on {forecast_days}-day forecast</p>
                </div>
            </div>
            <p style="color: #333; font-size: 1.1rem; line-height: 1.6;">{insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Mini gauge for sentiment
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=price_change_pct,
            title={'text': "Market Sentiment"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [-10, 10]},
                'bar': {'color': color},
                'steps': [
                    {'range': [-10, -5], 'color': "#fef2f2"},
                    {'range': [-5, 0], 'color': "#fffbeb"},
                    {'range': [0, 5], 'color': "#f0f9ff"},
                    {'range': [5, 10], 'color': "#f0fdf4"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': price_change_pct
                }
            }
        ))
        
        fig3.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig3, use_container_width=True)
    
    # ----------------------------
    # Action Items
    # ----------------------------
    st.markdown('<h2 class="section-header">🎯 Recommended Actions</h2>', unsafe_allow_html=True)
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        st.markdown("""
        <div class="prediction-card">
            <div style="text-align: center;">
                <span style="font-size: 2rem;">📊</span>
                <h4 style="margin: 1rem 0;">Monitor Daily</h4>
                <p style="color: #666; font-size: 0.9rem;">Check predictions daily as they update with new data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col2:
        st.markdown("""
        <div class="prediction-card">
            <div style="text-align: center;">
                <span style="font-size: 2rem;">🔔</span>
                <h4 style="margin: 1rem 0;">Set Alerts</h4>
                <p style="color: #666; font-size: 0.9rem;">Watch for when prices approach confidence boundaries</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with action_col3:
        st.markdown("""
        <div class="prediction-card">
            <div style="text-align: center;">
                <span style="font-size: 2rem;">📱</span>
                <h4 style="margin: 1rem 0;">Stay Updated</h4>
                <p style="color: #666; font-size: 0.9rem;">Re-run analysis weekly for fresh predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ----------------------------
    # Refresh Section
    # ----------------------------
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔄 Generate New Prediction", type="primary", use_container_width=True):
            st.rerun()
    
    # ----------------------------
    # Footer
    # ----------------------------
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem 0;">
            <p style="margin-bottom: 0.5rem;">
                <strong>StockPredict AI</strong> • Powered by SARIMA Time Series Analysis
            </p>
            <p style="font-size: 0.9rem; margin: 0;">
                🤖 AI-driven predictions • 📊 Real-time analysis • 🎯 Actionable insights
                <br>
                <span style="color: #ef4444; font-size: 0.8rem;">
                    ⚠️ Educational tool only. Not financial advice. Past performance ≠ future results.
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Error state
    st.error("""
    ⚠️ **Unable to load data**
    
    Please ensure you have a `stocks.csv` file with:
    - `Date` column in YYYY-MM-DD format
    - `Close` column with numeric prices
    
    Example:
    ```
    Date,Close
    2024-01-01,150.25
    2024-01-02,152.30
    ```
    """)
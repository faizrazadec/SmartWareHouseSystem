import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import sys
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.forecasting import run_inference, PRODUCTS_TO_PLOT, DATA_PATH

try:
    st.set_page_config(page_title="Forecast Dashboard", layout="wide")
except Exception:
    pass

st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    # .main-frame {
    #     background-color: #1e1e1e;
    #     border-radius: 8px;
    #     padding: 12px;
    #     margin: 5px;
    #     box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    # }
    .metric-box {
        background-color: #2c2f36;
        border-radius: 6px;
        padding: 5px;
        margin-bottom: 3px;
        color: white;
        text-align: center;
        font-size: 0.8em;
    }
    .kpi-metric {
        background-color: #2c2f36;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        color: white;
        text-align: center;
    }
    h1 {
        font-size: 1.5em !important;
        margin: 0 0 5px 0 !important;
    }
    h2 {
        color: #61dafb;
        font-size: 1.2em !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load historical data and prepare it for evaluation"""
    try:
        historical_df = pd.read_pickle(DATA_PATH)
        
        forecast_df = run_inference(historical_df)
        
        # forecast_df.to_csv("forecast_next_30_days.csv")
        
        test_df = historical_df[-30:].copy()
        
        historical_train = historical_df[:-30].copy()
        historical_forecast = run_inference(historical_train, forecast_days=30)
        
        historical_forecast.index = test_df.index
        
        return historical_df, forecast_df, test_df, historical_forecast
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def calculate_metrics(actual, predicted):
    """Calculate forecast accuracy metrics between actual and predicted values"""
    metrics = {}
    
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual_clean = actual[mask]
    predicted_clean = predicted[mask]
    
    if len(actual_clean) == 0:
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'MAPE': np.nan,
            'R2': np.nan
        }
    
    metrics['MAE'] = mean_absolute_error(actual_clean, predicted_clean)
    metrics['RMSE'] = sqrt(mean_squared_error(actual_clean, predicted_clean))
    
    nonzero_mask = actual_clean != 0
    if sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((actual_clean[nonzero_mask] - predicted_clean[nonzero_mask]) / actual_clean[nonzero_mask])) * 100
        metrics['MAPE'] = mape
    else:
        metrics['MAPE'] = np.nan
        
    metrics['R2'] = r2_score(actual_clean, predicted_clean)
    
    return metrics

def main():
    st.markdown('<div class="main-frame">', unsafe_allow_html=True)
    
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title("Forecast Dashboard")
    
    with st.spinner("Loading forecast data..."):
        historical_df, forecast_df, test_df, historical_forecast = load_and_prepare_data()
    
    if historical_df is None:
        st.error("Failed to load data. Please check the data paths and models.")
        return
    
    all_products = historical_df.columns.tolist()
    default_product = [p for p in PRODUCTS_TO_PLOT if p in all_products][0:1]
    
    with header_col2:
        st.markdown("##### 🔍 Product Selection")
        selected_products = st.multiselect(
            "Choose products",
            options=all_products,
            default=default_product,
            key="product_selector"
        )
    
    if not selected_products:
        st.warning("Please select at least one product to display metrics.")
    
    all_metrics = {}
    for product in selected_products:
        if product in test_df.columns and product in historical_forecast.columns:
            product_metrics = calculate_metrics(test_df[product].values, historical_forecast[product].values)
            all_metrics[product] = product_metrics
    
    for product in selected_products:
        st.markdown(f"### Analysis for: {product}")
        
        graphs_col, metrics_col = st.columns(2)
        
        has_metrics = product in all_metrics
        
        with graphs_col:
            chart_data = []
            
            if product in test_df.columns and product in historical_forecast.columns:
                for date, value in zip(test_df.index, test_df[product]):
                    chart_data.append({
                        'Date': date,
                        'Value': value,
                        'Type': 'Actual'
                    })
                
                for date, value in zip(historical_forecast.index, historical_forecast[product]):
                    chart_data.append({
                        'Date': date,
                        'Value': value,
                        'Type': 'Forecast'
                    })
            
            chart_df = pd.DataFrame(chart_data)
            
            if not chart_df.empty:
                st.markdown("#### Historical Forecast vs. Actual")
                product_chart = alt.Chart(chart_df).mark_line().encode(
                    x='Date:T',
                    y='Value:Q',
                    color='Type:N',
                    strokeDash=alt.condition(
                        alt.datum.Type == 'Actual',
                        alt.value([0]),
                        alt.value([5, 5])
                    ),
                    tooltip=['Date:T', 'Value:Q', 'Type:N']
                ).properties(
                    height=200 
                )
                
                st.altair_chart(product_chart, use_container_width=True)
            
            st.markdown("#### Future Forecast (Next 30 Days)")
            
            future_data = []
            
            last_15_days = historical_df[-15:].copy()
            
            if product in last_15_days.columns and product in forecast_df.columns:
                for date, value in zip(last_15_days.index, last_15_days[product]):
                    future_data.append({
                        'Date': date,
                        'Value': value,
                        'Type': 'Historical'
                    })
                
                for date, value in zip(forecast_df.index, forecast_df[product]):
                    future_data.append({
                        'Date': date,
                        'Value': value,
                        'Type': 'Forecast'
                    })
            
            future_df = pd.DataFrame(future_data)
            
            if not future_df.empty:
                future_chart = alt.Chart(future_df).mark_line().encode(
                    x='Date:T',
                    y='Value:Q',
                    color='Type:N',
                    strokeDash=alt.condition(
                        alt.datum.Type == 'Historical',
                        alt.value([0]),
                        alt.value([5, 5])
                    ),
                    tooltip=['Date:T', 'Value:Q', 'Type:N']
                ).properties(
                    height=200 
                )
                
                st.altair_chart(future_chart, use_container_width=True)
        
        with metrics_col:
            if has_metrics:
                st.markdown("#### Performance Metrics")
                metrics = all_metrics[product]
                
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:#2c2f36; border-radius:8px; padding:8px; margin:3px 0; text-align:center; height:100%;">
                        <div style="font-size:0.8em; color:#61dafb;">Mean Absolute Error</div>
                        <div style="font-size:1.6em; font-weight:bold; margin:3px 0;">{metrics['MAE']:.2f}</div>
                        <div style="font-size:0.7em;">Lower is better</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="background-color:#2c2f36; border-radius:8px; padding:8px; margin:3px 0; text-align:center; height:100%;">
                        <div style="font-size:0.8em; color:#61dafb;">Mean Absolute % Error</div>
                        <div style="font-size:1.6em; font-weight:bold; margin:3px 0;">{metrics['MAPE']:.2f}%</div>
                        <div style="font-size:0.7em;">Lower is better</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    r2_color = "#4CAF50" if metrics['R2'] > 0.7 else "#FFA500" if metrics['R2'] > 0.5 else "#FF5252"
                    st.markdown(f"""
                    <div style="background-color:#2c2f36; border-radius:8px; padding:8px; margin:3px 0; text-align:center; height:100%;">
                        <div style="font-size:0.8em; color:#61dafb;">R² Score</div>
                        <div style="font-size:1.6em; font-weight:bold; margin:3px 0; color:{r2_color};">{metrics['R2']:.3f}</div>
                        <div style="font-size:0.7em;">Higher is better</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    if product in forecast_df.columns:
                        forecast_vals = forecast_df[product].values
                        if len(forecast_vals) > 0:
                            first_half = forecast_vals[:len(forecast_vals)//2]
                            second_half = forecast_vals[len(forecast_vals)//2:]
                            
                            first_half_avg = np.mean(first_half)
                            second_half_avg = np.mean(second_half)
                            
                            trend_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0
                            
                            trend_color = "#4CAF50" if trend_pct > 0 else "#FF5252"
                            trend_arrow = "↑" if trend_pct > 0 else "↓"
                            
                            st.markdown(f"""
                            <div style="background-color:#2c2f36; border-radius:8px; padding:8px; margin:3px 0; text-align:center; height:100%;">
                                <div style="font-size:0.8em; color:#61dafb;">30-Day Trend</div>
                                <div style="font-size:1.6em; font-weight:bold; margin:3px 0; color:{trend_color};">{trend_arrow} {abs(trend_pct):.1f}%</div>
                                <div style="font-size:0.7em;">Forecast direction</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                error_col1, error_col2 = st.columns(2)
                
                with error_col1:
                    st.markdown("#### Forecast Error Distribution")
                    
                    if product in test_df.columns and product in historical_forecast.columns:
                        errors = test_df[product].values - historical_forecast[product].values
                        
                        errors = errors[~np.isnan(errors)]
                        
                        if len(errors) > 0:
                            error_df = pd.DataFrame({'Error': errors})
                            
                            mean_error = np.mean(errors)
                            
                            hist_chart = alt.Chart(error_df).mark_bar().encode(
                                alt.X('Error:Q', bin=True, title='Forecast Error'),
                                alt.Y('count()', title='Frequency'),
                                color=alt.value('#4285F4'),
                                tooltip=['count()', alt.Tooltip('Error:Q', aggregate='mean')]
                            ).properties(
                                height=180
                            )
                            
                            zero_rule = alt.Chart(pd.DataFrame({'x': [0]})).mark_rule(
                                color='green',
                                strokeWidth=2
                            ).encode(x='x:Q')
                            
                            mean_rule = alt.Chart(pd.DataFrame({'x': [mean_error]})).mark_rule(
                                color='red',
                                strokeWidth=2
                            ).encode(x='x:Q')
                            
                            st.altair_chart(hist_chart + zero_rule + mean_rule, use_container_width=True)
                            
                            st.markdown(f"""<div style='font-size:0.8em; color:#cccccc;'>
                            Mean Error: {mean_error:.2f} | Green line: Perfect prediction | Red line: Mean error
                            </div>""", unsafe_allow_html=True)
                
                with error_col2:
                    st.markdown("#### Cumulative Forecast Error")
                    
                    if product in test_df.columns and product in historical_forecast.columns:
                        errors = test_df[product].values - historical_forecast[product].values
                        
                        errors = errors[~np.isnan(errors)]
                        
                        if len(errors) > 0:
                            cum_error_df = pd.DataFrame({
                                'Date': test_df.index,
                                'Error': test_df[product].values - historical_forecast[product].values,
                                'Cumulative Error': np.cumsum(test_df[product].values - historical_forecast[product].values)
                            })
                            
                            cum_chart = alt.Chart(cum_error_df).mark_line(
                                color='#FF5252',
                                strokeWidth=3
                            ).encode(
                                x='Date:T',
                                y='Cumulative Error:Q',
                                tooltip=['Date:T', 'Error:Q', 'Cumulative Error:Q']
                            ).properties(
                                height=180
                            )
                            
                            zero_line = alt.Chart(pd.DataFrame({
                                'Date': test_df.index,
                                'Zero': [0] * len(test_df.index)
                            })).mark_line(
                                color='#4CAF50',
                                strokeWidth=1,
                                strokeDash=[5, 5]
                            ).encode(
                                x='Date:T',
                                y='Zero:Q'
                            )
                            
                            st.altair_chart(cum_chart + zero_line, use_container_width=True)
                            
                            st.markdown(f"""<div style='font-size:0.8em; color:#cccccc;'>
                            Rising line indicates consistent under-prediction, falling line indicates over-prediction.
                            </div>""", unsafe_allow_html=True)
                    else:
                        st.info("Insufficient data for error distribution chart")
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    with st.expander("View All Products Metrics Table"):
        if all_metrics:
            metrics_data = []
            
            for product, metrics in all_metrics.items():
                metrics_data.append({
                    'Product': product,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'R²': metrics['R2']
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            if not metrics_df.empty:
                metrics_df = metrics_df.sort_values('MAPE', ascending=True)
                
                formatted_df = metrics_df.copy()
                formatted_df['MAE'] = formatted_df['MAE'].round(2)
                formatted_df['RMSE'] = formatted_df['RMSE'].round(2)
                formatted_df['MAPE'] = formatted_df['MAPE'].round(2).astype(str) + '%'
                formatted_df['R²'] = formatted_df['R²'].round(3)
                
                st.dataframe(
                    formatted_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Product": "Product",
                        "MAE": st.column_config.NumberColumn(
                            "MAE",
                            help="Mean Absolute Error",
                            format="%.2f"
                        ),
                        "RMSE": st.column_config.NumberColumn(
                            "RMSE",
                            help="Root Mean Squared Error",
                            format="%.2f"
                        ),
                        "MAPE": st.column_config.TextColumn(
                            "MAPE",
                            help="Mean Absolute Percentage Error"
                        ),
                        "R²": st.column_config.NumberColumn(
                            "R²",
                            help="Coefficient of Determination",
                            format="%.3f"
                        )
                    }
                )
    
    with st.expander("Download Forecast Data"):
        if 'forecast_df' in locals():
            st.download_button(
                "Download Complete Forecast CSV",
                forecast_df.to_csv(),
                file_name=f"demand_forecast_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with st.expander("About the Metrics"):
        st.markdown("""
        ### Understanding Forecast Accuracy Metrics
        
        - **MAE (Mean Absolute Error)**: Average of the absolute differences between the predicted and actual values. Lower is better. Units are the same as the forecast.
        
        - **RMSE (Root Mean Squared Error)**: Square root of the average of squared differences between predicted and actual values. More sensitive to large errors. Lower is better.
        
        - **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values. Provides a relative measure of accuracy. Lower is better.
        
        - **R² (R-squared)**: Proportion of variance in the actual values explained by the model. Ranges from 0 to 1, with 1 being perfect prediction. Higher is better.
        
        ### How Accuracy is Evaluated
        
        The historical accuracy metrics compare the model's predictions against known actual values from the last 30 days. This gives us a measure of how well the model performs on data it hasn't seen during training.
        
        The future forecast shows the model's predictions for the next 30 days based on all available historical data.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

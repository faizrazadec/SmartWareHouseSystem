import streamlit as st
import altair as alt
import pandas as pd
from pages.Defect_Analysis import get_defect_statistics
from pages.Forecasting import load_and_prepare_data

# --- Page Config ---
st.set_page_config(page_title="FF WMS", layout="wide")

# --- Dark Theme Custom CSS ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0f1117;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    # .main-frame {
    #     background-color: #1e1e1e;
    #     border-radius: 12px;
    #     padding: 30px;
    #     margin: 20px;
    #     box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    # }
    .metric-box {
        background-color: #2c2f36;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 10px;
        color: white;
        text-align: center;
    }
    h2 {
        color: #61dafb;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# --- Main Dashboard Frame ---
st.markdown('<div class="main-frame">', unsafe_allow_html=True)
st.title("FF Smart Warehouse Dashboard")

# --- Defect Metrics --- # Added section
st.markdown("### Defect Overview")
defect_stats = get_defect_statistics(days=30)
if not defect_stats.empty:
    total_defects = defect_stats["Count"].sum()
    top_defect = defect_stats.iloc[0]["Defect Class"] if len(defect_stats) > 0 else "None"
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f'''<div class="metric-box">Total Defects (30 days)<br><strong>{total_defects}</strong></div>''', unsafe_allow_html=True)
    with colB:
        st.markdown(f'''<div class="metric-box">Top Defect Type<br><strong>{top_defect}</strong></div>''', unsafe_allow_html=True)
    with colC:
        st.markdown(f'''<div class="metric-box">Unique Defect Types<br><strong>{len(defect_stats)}</strong></div>''', unsafe_allow_html=True)
else:
    st.info("No defect data available for the past 30 days.")


# --- Demand Forecast ---
st.markdown("### Demand Forecast") # Changed heading
_, _, test_df, historical_forecast = load_and_prepare_data() # Load forecast data

if test_df is not None and historical_forecast is not None:
    # Assuming 'Product A' for simplicity, modify as needed
    # You might want to select a product dynamically or use the first available one
    available_products = [p for p in test_df.columns if p in historical_forecast.columns]
    if available_products:
        product_to_display = available_products[0]
        chart_data = []
        for date, value in zip(test_df.index, test_df[product_to_display]):
            chart_data.append({
                'Date': date,
                'Value': value,
                'Type': 'Actual'
            })
        for date, value in zip(historical_forecast.index, historical_forecast[product_to_display]):
            chart_data.append({
                'Date': date,
                'Value': value,
                'Type': 'Forecast'
            })
        
        chart_df = pd.DataFrame(chart_data)

        if not chart_df.empty:
            st.markdown(f"#### Historical Forecast vs. Actual for {product_to_display}")
            product_chart = alt.Chart(chart_df).mark_line().encode(
                x='Date:T',
                y='Value:Q',
                color='Type:N',
                strokeDash=alt.condition(
                    alt.datum.Type == 'Actual',
                    alt.value([0]),  # solid line for actual
                    alt.value([5, 5])  # dashed line for forecast
                ),
                tooltip=['Date:T', 'Value:Q', 'Type:N']
            ).properties(
                height=300 
            )
            st.altair_chart(product_chart, use_container_width=True)
        else:
            st.info(f"No forecast data available for {product_to_display}.")
    else:
        st.info(f"No common products found in forecast data to display.")

else:
    st.error("Failed to load forecast data.")

import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from datetime import datetime, timedelta

def get_defect_statistics(days=30):
    """
    Get defect statistics from the database for the past specified days.
    Returns a DataFrame with defect class and count.
    """
    try:
        date_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect('db.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT defect_class, COUNT(*) as count
            FROM defect_detections
            WHERE timestamp > ?
            GROUP BY defect_class
            ORDER BY count DESC
        ''', (date_threshold,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return pd.DataFrame(columns=["Defect Class", "Count"])
        
        df = pd.DataFrame(results, columns=["Defect Class", "Count"])
        return df
    except Exception as e:
        st.error(f"Error getting defect statistics: {e}")
        return pd.DataFrame(columns=["Defect Class", "Count"])

def get_defect_trend(days=30):
    """
    Get daily defect counts for trend analysis over the specified days.
    Returns a DataFrame with date and count.
    """
    try:
        date_threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect('db.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM defect_detections
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (date_threshold,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            date_range = pd.date_range(end=datetime.now(), periods=days)
            return pd.DataFrame({"Date": date_range, "Count": 0})
        
        df = pd.DataFrame(results, columns=["Date", "Count"])
        df["Date"] = pd.to_datetime(df["Date"])
        
        date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max())
        date_df = pd.DataFrame({"Date": date_range})
        df = pd.merge(date_df, df, on="Date", how="left").fillna(0)
        
        return df
    except Exception as e:
        st.error(f"Error getting defect trend data: {e}")
        date_range = pd.date_range(end=datetime.now(), periods=days)
        return pd.DataFrame({"Date": date_range, "Count": 0})

st.set_page_config(
    page_title="Defect Analysis Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("Defect Analysis Dashboard")

defect_stats = get_defect_statistics(days=30)

if not defect_stats.empty:
    total_defects = defect_stats["Count"].sum()
    top_defect = defect_stats.iloc[0]["Defect Class"] if len(defect_stats) > 0 else "None"
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Defects (30 days)", total_defects)
    with col2:
        st.metric("Top Defect Type", top_defect)
    with col3:
        st.metric("Unique Defect Types", len(defect_stats))
    
    defect_trend = get_defect_trend(days=30)
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        defect_chart = alt.Chart(defect_stats).mark_bar().encode(
            x=alt.X('Defect Class:N', sort='-y', axis=alt.Axis(labelAngle=-45)), 
            y='Count:Q',
            color=alt.Color('Defect Class:N', legend=None),
            tooltip=['Defect Class:N', 'Count:Q']
        ).properties(
            title='Defect Counts by Type',
            height=300
        )
        st.altair_chart(defect_chart, use_container_width=True)
    
    with col_right:
        if not defect_trend.empty:
            trend_chart = alt.Chart(defect_trend).mark_line(point=True).encode(
                x='Date:T',
                y='Count:Q',
                tooltip=['Date:T', 'Count:Q']
            ).properties(
                title='Daily Defect Counts (Last 30 Days)',
                height=300
            )
            st.altair_chart(trend_chart, use_container_width=True)
        
    with st.expander("📋 Defect Data", expanded=False):
        st.dataframe(defect_stats, use_container_width=True)
else:
    st.info("No defect data available for the past 30 days.")
    
    placeholder_data = pd.DataFrame({
        "Defect Class": ["No Data"],
        "Count": [0]
    })
    placeholder_chart = alt.Chart(placeholder_data).mark_bar().encode(
        x='Defect Class:N',
        y='Count:Q'
    ).properties(
        title='No Defect Data Available'
    )
    st.altair_chart(placeholder_chart, use_container_width=True)

st.sidebar.header("Time Period Filter")
days_filter = st.sidebar.slider("Analysis Period (days)", min_value=7, max_value=90, value=30, step=1)

if days_filter != 30:
    filtered_stats = get_defect_statistics(days=days_filter)
    filtered_trend = get_defect_trend(days=days_filter)
    
    st.subheader(f"🔍 Defect Analysis - Last {days_filter} Days")
    
    if not filtered_stats.empty:
        total_defects = filtered_stats["Count"].sum()
        top_defect = filtered_stats.iloc[0]["Defect Class"] if len(filtered_stats) > 0 else "None"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔎 Total Defects", total_defects)
        with col2:
            st.metric("⚠️ Top Defect Type", top_defect)
        with col3:
            st.metric("🧩 Unique Defect Types", len(filtered_stats))
        
        filtered_col_left, filtered_col_right = st.columns(2)
        
        with filtered_col_left:
            filtered_chart = alt.Chart(filtered_stats).mark_bar().encode(
                x=alt.X('Defect Class:N', sort='-y', axis=alt.Axis(labelAngle=-45)), 
                y='Count:Q',
                color=alt.Color('Defect Class:N', legend=None),
                tooltip=['Defect Class:N', 'Count:Q']
            ).properties(
                title=f'Defect Counts by Type (Last {days_filter} Days)',
                height=300
            )
            st.altair_chart(filtered_chart, use_container_width=True)
        
        with filtered_col_right:
            if not filtered_trend.empty:
                trend_chart = alt.Chart(filtered_trend).mark_line(point=True).encode(
                    x='Date:T',
                    y='Count:Q',
                    tooltip=['Date:T', 'Count:Q']
                ).properties(
                    title=f'Daily Defect Counts (Last {days_filter} Days)',
                    height=300
                )
                st.altair_chart(trend_chart, use_container_width=True)
    else:
        st.info(f"No defect data available for the past {days_filter} days.")
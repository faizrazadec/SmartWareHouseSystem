import streamlit as st
import pandas as pd
import sqlite3

DB_PATH = 'db.db'

def get_alerts_from_new_table():
    """Fetches all alerts from the new 'alerts' table."""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, product_id, type, message, created_at FROM alerts ORDER BY created_at DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

st.set_page_config(page_title="System Alerts", layout="wide")
st.title("System Alerts")

st.markdown("---")

alerts_df = get_alerts_from_new_table()

if alerts_df.empty:
    st.info("No alerts available at the moment.")
else:
    st.subheader("Current Alerts")
    st.dataframe(
        alerts_df,
        use_container_width=True,
        column_config={
            "id": st.column_config.NumberColumn("Alert ID", format="%d"),
            "product_id": st.column_config.NumberColumn("Product ID", format="%d", help="ID of the product related to the alert, if applicable."),
            "type": st.column_config.TextColumn("Alert Type"),
            "message": st.column_config.TextColumn("Message", width="large"),
            "created_at": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
        },
        hide_index=True,
    )

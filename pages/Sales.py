import streamlit as st
import pandas as pd
import sqlite3
from datetime import date
from typing import TypedDict, List

DB_PATH = 'db.db'

def get_all_sales_from_new_table():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, product_name, category, quantity, price, sale_date, (quantity * price) as total_price FROM sales ORDER BY sale_date DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
    return df

def get_sales_summary_from_new_table():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT sale_date, SUM(quantity * price) as total_sales FROM sales GROUP BY sale_date"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if 'sale_date' in df.columns:
        df['sale_date'] = pd.to_datetime(df['sale_date'])
    return df

def get_top_products_from_new_table(limit=5):
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT product_name, SUM(quantity) as total_sold FROM sales GROUP BY product_name ORDER BY total_sold DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_category_sales_from_new_table():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT category as category_name, SUM(quantity) as total_sold FROM sales GROUP BY category ORDER BY total_sold DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_all_products_for_form():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        ROW_NUMBER() OVER (ORDER BY product_name) as id,
        product_name
    FROM (SELECT DISTINCT product_name FROM sales)
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

class ProductItem(TypedDict):
    product_id: int
    quantity: int
    price: float

st.set_page_config(
    page_title="Sales",
    page_icon="",
    layout="wide"
)

st.title("Sales Dashboard")

st.markdown("---")
st.header("Sales Overview")

col1, col2 = st.columns(2)

sales_summary_df = get_sales_summary_from_new_table()

with col1:
    if not sales_summary_df.empty:
        sales_summary_df['sale_date'] = pd.to_datetime(sales_summary_df['sale_date'])
        today_sales = sales_summary_df[sales_summary_df['sale_date'].dt.date == date.today()][
            'total_sales'].sum()
        st.metric("Sales Today", f"{today_sales:,.2f} OMR" if today_sales else "0.00 OMR")

        start_of_week = pd.to_datetime(date.today()) - pd.DateOffset(days=date.today().weekday())
        this_week_sales = sales_summary_df[sales_summary_df['sale_date'] >= start_of_week][
            'total_sales'].sum()
        st.metric("This Week", f"{this_week_sales:,.2f} OMR" if this_week_sales else "0.00")

        this_month_sales = sales_summary_df[sales_summary_df['sale_date'].dt.month == date.today().month][
            'total_sales'].sum()
        st.metric("This Month", f"{this_month_sales:,.2f} OMR" if this_month_sales else "0.00")
    else:
        st.metric("Sales Today", "N/A")
        st.metric("This Week", "N/A")
        st.metric("This Month", "N/A")

with col2:
    total_revenue = sales_summary_df['total_sales'].sum() if not sales_summary_df.empty else 0
    st.metric("Total Revenue", f"{total_revenue:,.2f} OMR" if total_revenue else "0.00")

    top_product_df = get_top_products_from_new_table(1)
    st.metric("Top Product", top_product_df['product_name'].values[0] if not top_product_df.empty else "N/A")

    categories_df = get_category_sales_from_new_table()
    st.metric("Categories", f"{categories_df['category_name'].nunique() if not categories_df.empty else 0}")

with st.expander("Recent Sales", expanded=True):
    sales_df = get_all_sales_from_new_table()
    if not sales_df.empty:
        st.dataframe(sales_df[['sale_date', 'product_name', 'category', 'quantity', 'price', 'total_price']],
                     use_container_width=True)
    else:
        st.info("No sales data available.")

st.markdown("---")
st.header("Sales Analytics")

start_date_input = st.date_input("Start Date", value=date.today().replace(month=1, day=1))
end_date_input = st.date_input("End Date", value=date.today())
if start_date_input > end_date_input:
    st.error("Start date must be before end date")
else:
    analytics_sales_df = get_all_sales_from_new_table()
    if analytics_sales_df.empty:
        st.warning("No sales data found for the selected period.")
    else:
        analytics_sales_df['sale_date'] = pd.to_datetime(analytics_sales_df['sale_date'])
        mask = (analytics_sales_df['sale_date'].dt.date >= start_date_input) & (analytics_sales_df['sale_date'].dt.date <= end_date_input)
        filtered_sales_df = analytics_sales_df.loc[mask]

        if filtered_sales_df.empty:
            st.warning("No sales data for the selected date range.")
        else:
            daily_sales = filtered_sales_df.groupby(filtered_sales_df['sale_date'].dt.date)['total_price'].sum()

            st.subheader("Daily Sales Revenue")
            st.line_chart(daily_sales)

            col_top_products, col_category_sales = st.columns(2)

            with col_top_products:
                st.subheader("Top Products (by quantity sold)")
                top_products_filtered = filtered_sales_df.groupby('product_name')['quantity'].sum().nlargest(5).reset_index()
                if not top_products_filtered.empty:
                    st.bar_chart(top_products_filtered.set_index('product_name')['quantity'])
                else:
                    st.info("No top products data for this period.")

            with col_category_sales:
                st.subheader("Category Sales (by quantity sold)")
                category_sales_filtered = filtered_sales_df.groupby('category')['quantity'].sum().reset_index()
                if not category_sales_filtered.empty:
                    st.bar_chart(category_sales_filtered.set_index('category')['quantity'])
                else:
                    st.info("No category sales data for this period.")

st.markdown("---")
st.header("➕ Add New Sale")

with st.form("sale_form", clear_on_submit=True):
    form_sale_date = st.date_input("Sale Date", value=date.today(), key="form_sale_date")

    products_for_form = get_all_products_for_form()
    selected_products: List[ProductItem] = []

    if products_for_form.empty:
        st.warning("No products found in sales data to populate form. Add sales first or define products elsewhere.")
    else:
        product_options = products_for_form[['id', 'product_name']].values.tolist()

        for i in range(5):
            col1, col2 = st.columns([3, 1])
            with col1:
                selected = st.selectbox(f"Product {i+1}", options=["None"] + [p[1] for p in product_options], key=f"product_{i}")
            with col2:
                qty = st.number_input(f"Qty {i+1}", min_value=0, step=1, key=f"qty_{i}")

            if selected != "None" and qty > 0:
                pid = next(p[0] for p in product_options if p[1] == selected)
                price = 10.0
                selected_products.append({"product_id": pid, "quantity": qty, "price": price})

    submitted = st.form_submit_button("Add Sale")

    if submitted:
        if not selected_products:
            st.warning("Please select at least one product with quantity.")
        else:
            add_new_sale({"sale_date": form_sale_date, "items": selected_products})
            st.success("Sale submitted (Note: 'Add New Sale' functionality is a placeholder).")
            st.rerun()

st.markdown("---")
st.header("🛠️ Manage Sales")

manage_sales_df = get_all_sales_from_new_table()

if manage_sales_df.empty:
    st.warning("No sales records found")
else:
    sales_grouped = manage_sales_df.groupby(['id', 'sale_date']).agg(
        total_quantity=('quantity', 'sum'),
        unique_products=('product_name', 'nunique')
    ).reset_index()

    sales_grouped.columns = ['Sale ID', 'Date', 'Total Items', 'Unique Products']

    st.dataframe(sales_grouped, use_container_width=True)

    if not sales_grouped.empty:
        selected_sale_id = st.selectbox(
            "Select a sale to view details",
            options=sales_grouped['Sale ID'].tolist(),
            format_func=lambda x: f"Sale #{x} - {pd.to_datetime(sales_grouped[sales_grouped['Sale ID'] == x]['Date'].iloc[0]).strftime('%Y-%m-%d') if not sales_grouped[sales_grouped['Sale ID'] == x]['Date'].empty else 'N/A Date'}"
        )

        if selected_sale_id:
            st.subheader(f"Sale #{selected_sale_id} Details")
            sale_items = manage_sales_df[manage_sales_df['id'] == selected_sale_id]

            display_items = sale_items[['product_name', 'category', 'quantity', 'price', 'total_price']].rename(columns={
                'product_name': 'Product',
                'category': 'Category',
                'quantity': 'Quantity',
                'price': 'Price',
                'total_price': 'Total Price'
            })

            st.dataframe(display_items, use_container_width=True)
    else:
        st.info("No grouped sales data to display.")
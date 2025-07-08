import streamlit as st
import pickle
import pandas as pd
import joblib

# Load Pickle Files
kmeans = joblib.load("Downloads/Datascience   Guvi/Miniproject 4/rfm_kmeans_model.joblib")
scaler = joblib.load("Downloads/Datascience   Guvi/Miniproject 4/rfm_scaler.joblib")
similarity_df = joblib.load("Downloads/Datascience   Guvi/Miniproject 4/product_similarity.joblib")


try:
    product_names = joblib.load("Downloads/Datascience   Guvi/Miniproject 4/product_names.joblib")
except:
    product_names = {}


# Function: Recommend Products
def recommend_products(product_code):
    if product_code not in similarity_df.columns:
        return ["Product Not Found"]
    sim_scores = similarity_df[product_code].sort_values(ascending=False)
    return sim_scores.iloc[1:6].index.tolist()

def get_product_name(code):
    return product_names.get(code, "Name Not Found")



# Function: Predict Cluster Segment
def predict_segment(recency, frequency, monetary):
    scaled = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(scaled)[0]

    # Manual segment interpretation
    if recency <= 30 and frequency >= 15 and monetary >= 1000:
        return "High-Value"
    elif frequency >= 5:
        return "Regular"
    elif recency >= 90:
        return "At-Risk"
    else:
        return "Occasional"

# --- Streamlit App ---
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum: Segmentation & Recommendations")

tab1, tab2 = st.tabs(["📦 Product Recommendation", "🧍 Customer Segmentation"])

# 🔶 Module 1: Product Recommender
with tab1:
    st.sidebar.subheader("🎯 Product Recommendation")
    prod_code = st.text_input("Enter Product Code (e.g., 85123A):")

    if st.button("Get Recommendations"):
        if prod_code.strip() == "":
            st.warning("Please enter a product code.")
        else:
            recs = recommend_products(prod_code)
            st.success("Top 5 Similar Products:")
            for idx, code in enumerate(recs, 1):
                st.markdown(f"**{idx}.** `{code}` — *{get_product_name(code)}*")

# 🔷 Module 2: Customer Segmentation
with tab2:
    st.sidebar.subheader("🎯 Customer Segment Prediction")
    r = st.number_input("Recency (in days)", min_value=0, value=30)
    f = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    m = st.number_input("Monetary (total spend)", min_value=1.0, value=500.0)

    if st.button("Predict Cluster"):
        segment = predict_segment(r, f, m)
        st.success(f"Predicted Segment: **{segment}**")



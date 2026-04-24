import streamlit as st
import joblib
import pickle
import os

# Print current directory and files (untuk debug di Streamlit Cloud)
st.write("Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir('.'))

# Page configuration
st.set_page_config(page_title="Sales Volume Predictor", layout="wide")

# Load model and features with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load('sales_volume_model.pkl')
        with open('reg_features1.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, features
    except FileNotFoundError:
        st.error("Model files not found. Please ensure sales_volume_model.pkl and reg_features1.pkl are in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, features = load_model()

if model is None:
    st.stop()

# Title
st.title("Fashion Boutique Sales Volume Predictor")
st.caption("Predict sales volume per brand based on product attributes")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product Attributes")
    
    season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x-1])
    size = st.selectbox("Size", [1, 2, 3, 4, 5, 6], format_func=lambda x: ["XS", "S", "M", "L", "XL", "XXL"][x-1])
    category = st.selectbox("Category", ['Accessories', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Tops'])
    
    col1_1, col1_2 = st.columns(2)
    with col1_1:
        original_price = st.number_input("Original Price (USD)", min_value=10.0, max_value=300.0, value=100.0, step=5.0)
        current_price = st.number_input("Current Price (USD)", min_value=5.0, max_value=300.0, value=80.0, step=5.0)
        markdown_pct = st.number_input("Markdown Percentage (%)", min_value=0.0, max_value=70.0, value=10.0, step=1.0)
    with col1_2:
        stock_qty = st.number_input("Stock Quantity", min_value=0, max_value=100, value=30, step=5)
        customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, 0.1)
        purchase_month = st.selectbox("Purchase Month", list(range(1, 13)), format_func=lambda x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1])

with col2:
    st.subheader("Prediction Result")
    
    # Create one-hot encoded category features
    category_cols = {
        'category_Accessories': 1 if category == 'Accessories' else 0,
        'category_Bottoms': 1 if category == 'Bottoms' else 0,
        'category_Dresses': 1 if category == 'Dresses' else 0,
        'category_Outerwear': 1 if category == 'Outerwear' else 0,
        'category_Shoes': 1 if category == 'Shoes' else 0,
        'category_Tops': 1 if category == 'Tops' else 0,
    }
    
    # Prepare input array
    is_discounted = 1 if current_price < original_price else 0
    
    input_features = [
        season, size, original_price, current_price, markdown_pct,
        stock_qty, customer_rating, purchase_month, is_discounted,
        category_cols['category_Accessories'], category_cols['category_Bottoms'],
        category_cols['category_Dresses'], category_cols['category_Outerwear'],
        category_cols['category_Shoes'], category_cols['category_Tops']
    ]
    
    # Display input summary
    st.info(
        f"**Input Summary:**\n"
        f"- Season: {['Spring', 'Summer', 'Fall', 'Winter'][season-1]}\n"
        f"- Size: {['XS', 'S', 'M', 'L', 'XL', 'XXL'][size-1]}\n"
        f"- Category: {category}\n"
        f"- Original Price: ${original_price:.2f}\n"
        f"- Current Price: ${current_price:.2f}\n"
        f"- Markdown: {markdown_pct:.1f}%\n"
        f"- Customer Rating: {customer_rating:.1f}"
    )
    
    if st.button("Predict Sales Volume", type="primary", use_container_width=True):
        with st.spinner("Predicting..."):
            prediction = model.predict([input_features])[0]
        
        st.success(f"### Predicted Sales Volume: **{prediction:.0f} units**")
        
        if prediction > 300:
            st.success("High sales potential. Consider increasing stock.")
        elif prediction > 250:
            st.info("Good sales potential. Maintain current strategy.")
        else:
            st.warning("Low sales potential. Consider adjusting pricing or promotion.")

# Footer
st.divider()
st.caption("PT Data Analytics Ritel | BNSP Associate Data Scientist Certification")
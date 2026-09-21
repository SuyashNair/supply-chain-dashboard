import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Supply Chain Dashboard", layout="wide")

st.title("📦 Supply Chain Delay Prediction Dashboard")
st.markdown("This dashboard provides insights into supply chain delays and model explainability using SHAP.")

# Load Model and Data
@st.cache_data
def load_data():
    model = joblib.load("model/delay_model.joblib")
    data = pd.read_csv("model/test_data.csv")
    return model, data

model, data = load_data()

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Explainability (SHAP)"])

if page == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Sample of the test dataset:")
    st.dataframe(data.head(50))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quantity Distribution")
        fig, ax = plt.subplots()
        ax.hist(data['quantity'], bins=20, color='skyblue')
        st.pyplot(fig)
        
if page == "Model Explainability (SHAP)":
    st.header("Why did the model make these predictions?")
    st.write("SHAP values explain the impact of each feature on the final prediction.")
    
    # Process data through the pipeline's preprocessor
    X_processed = model.named_steps["pre"].transform(data)
    # Convert sparse matrix to dense array if necessary, and ensure floats
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = X_processed.astype(float)
    feature_names = model.named_steps["pre"].get_feature_names_out()
    
    # Generate SHAP values (using a sample to save memory)
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    shap_values = explainer.shap_values(X_processed[:100])
    
    st.subheader("Feature Importance Summary")
    # For binary classification in older shap versions, shap_values is a list. We take [1] for the positive class.
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                      features=X_processed[:100], 
                      feature_names=feature_names, 
                      show=False)
    st.pyplot(plt.gcf())

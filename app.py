# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-safe {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #28a745;
        text-align: center;
    }
    .prediction-risk {
        background-color: #f8d7da;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #dc3545;
        text-align: center;
    }
    .feature-importance {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .best-model-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = r"models\loan_default_model.pkl"  # ✅ Correct absolute path
        model_artifacts = joblib.load(model_path)
        return model_artifacts
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

def predict_default_probability(features, model_artifacts):
    """Make prediction using the loaded SVM model"""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    feature_columns = model_artifacts['feature_columns']
    
    # Create feature vector
    feature_vector = np.array([features[col] for col in feature_columns]).reshape(1, -1)
    
    # Scale features for SVM
    feature_vector = scaler.transform(feature_vector)
    
    # For SVM, we need to use decision function and convert to probability
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(feature_vector)[0, 1]
    else:
        # If SVM doesn't have predict_proba, use decision function and sigmoid
        decision_score = model.decision_function(feature_vector)[0]
        probability = 1 / (1 + np.exp(-decision_score))
    
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Loan Default Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load model
    model_artifacts = load_model()
    
    if model_artifacts is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["🏠 Home", "📊 Predict Default", "📈 Model Analysis", "ℹ️ About"])
    
    if app_mode == "🏠 Home":
        show_home_page(model_artifacts)
    elif app_mode == "📊 Predict Default":
        show_prediction_page(model_artifacts)
    elif app_mode == "📈 Model Analysis":
        show_analysis_page(model_artifacts)
    elif app_mode == "ℹ️ About":
        show_about_page()

def show_home_page(model_artifacts):
    """Home page with overview and metrics"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Welcome to Loan Default Prediction System</h2>', unsafe_allow_html=True)
        st.markdown('<div class="best-model-badge">🏆 BEST MODEL: Support Vector Machine</div>', unsafe_allow_html=True)
        st.write("""
        This intelligent system helps financial institutions predict the likelihood of loan defaults 
        using machine learning. Our **Support Vector Machine (SVM)** model achieved the highest accuracy
        among 10 different algorithms tested.
        
        ### Model Performance Highlights:
        - **🏆 Best Performing Model**: Support Vector Machine
        - **📈 Accuracy**: 70.0% (highest among all models)
        - **🔢 Features Used**: 14 carefully engineered features
        - **📊 Dataset**: Trained on 399 samples with balanced classes
        
        ### Key Predictive Features:
        - **Demographic information** (Age, Young borrower status)
        - **Financial behavior** (Cash inflow patterns, Low cash flow indicators)
        - **Location stability** (GPS movement data, Location stability)
        - **Application timing** (Hour, Day, Month of application)
        
        ### How it works:
        1. Input customer information in the **Predict Default** page
        2. Our SVM model analyzes the data in real-time
        3. Get instant risk assessment with detailed explanations
        4. Make informed lending decisions
        """)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Accuracy", "70.0%")
        st.metric("Best Model", "SVM")
        st.metric("Features Used", "14")
        st.metric("Training Samples", "399")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model comparison chart
        st.markdown("#### Model Performance Comparison")
        models = ['SVM', 'Gradient Boost', 'Extra Trees', 'Random Forest', 'Naive Bayes']
        accuracy = [0.700, 0.688, 0.688, 0.675, 0.663]
        
        fig = px.bar(
            x=accuracy, 
            y=models,
            orientation='h',
            title='Top 5 Model Accuracies',
            labels={'x': 'Accuracy', 'y': 'Model'},
            color=accuracy,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_prediction_page(model_artifacts):
    """Page for making predictions"""
    
    st.markdown('<h2 class="sub-header">🔍 Loan Default Prediction</h2>', unsafe_allow_html=True)
    st.markdown('<div class="best-model-badge">Powered by Support Vector Machine (70.0% Accuracy)</div>', unsafe_allow_html=True)
    
    # Create input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Personal Information")
            age = st.slider("Age", 18, 80, 30)
            cash_incoming = st.number_input("Monthly Cash Incoming ($)", 
                                          min_value=0, 
                                          max_value=100000, 
                                          value=5000, 
                                          step=500)
        
        with col2:
            st.subheader("Behavioral Data")
            gps_fix_count = st.slider("GPS Activity (number of locations)", 
                                    min_value=0, 
                                    max_value=500, 
                                    value=50)
            location_stability = st.slider("Location Stability (lower = more stable)", 
                                         min_value=0.0, 
                                         max_value=1.0, 
                                         value=0.1, 
                                         step=0.01)
            gps_accuracy = st.slider("GPS Accuracy (lower = better)", 
                                   min_value=0.0, 
                                   max_value=100.0, 
                                   value=20.0, 
                                   step=5.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Application Details")
            application_hour = st.slider("Application Hour", 0, 23, 12)
            application_day = st.selectbox("Application Day", 
                                         ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            application_month = st.slider("Application Month", 1, 12, 6)
        
        with col4:
            st.subheader("Risk Flags")
            young_borrower = st.checkbox("Young Borrower (<25 years)")
            low_cash_flow = st.checkbox("Low Cash Flow (<$2000/month)")
            unstable_location = st.checkbox("Unstable Location Patterns")
        
        # Convert day to numerical
        day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                      "Friday": 4, "Saturday": 5, "Sunday": 6}
        application_dayofweek = day_mapping[application_day]
        
        # Age group encoding
        if age < 25:
            age_group_encoded = 0
        elif age < 35:
            age_group_encoded = 1
        elif age < 45:
            age_group_encoded = 2
        else:
            age_group_encoded = 3
            
        # Cash category encoding
        if cash_incoming < 1000:
            cash_category_encoded = 0
        elif cash_incoming < 2000:
            cash_category_encoded = 1
        elif cash_incoming < 5000:
            cash_category_encoded = 2
        else:
            cash_category_encoded = 3
        
        # Submit button
        submitted = st.form_submit_button("Predict Default Risk")
    
    if submitted:
        # Prepare features
        features = {
            'age': age,
            'cash_incoming_30days': cash_incoming,
            'gps_fix_count': gps_fix_count,
            'longitude_std': location_stability,
            'latitude_std': location_stability,
            'accuracy_mean': gps_accuracy,
            'young_borrower': 1 if young_borrower else 0,
            'low_cash_flow': 1 if low_cash_flow else 0,
            'unstable_location': 1 if unstable_location else 0,
            'application_dayofweek': application_dayofweek,
            'application_hour': application_hour,
            'application_month': application_month,
            'age_group_encoded': age_group_encoded,
            'cash_category_encoded': cash_category_encoded
        }
        
        # Make prediction
        prediction, probability = predict_default_probability(features, model_artifacts)
        
        # Display results
        st.markdown("---")
        st.markdown('<h3 class="sub-header">Prediction Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Probability"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk assessment
            if probability < 0.3:
                st.markdown('<div class="prediction-safe">', unsafe_allow_html=True)
                st.success("✅ LOW RISK")
                st.write(f"Default Probability: {probability:.1%}")
                st.write("This application appears to be low risk based on the provided information.")
                st.markdown('</div>', unsafe_allow_html=True)
            elif probability < 0.7:
                st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
                st.warning("⚠️ MEDIUM RISK")
                st.write(f"Default Probability: {probability:.1%}")
                st.write("Recommend further review and additional verification.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-risk">', unsafe_allow_html=True)
                st.error("🚨 HIGH RISK")
                st.write(f"Default Probability: {probability:.1%}")
                st.write("Strong indication of potential default. Consider declining or requesting collateral.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Key factors
            st.markdown("#### Key Risk Factors:")
            risk_factors = []
            if young_borrower:
                risk_factors.append("Young borrower (<25 years)")
            if low_cash_flow:
                risk_factors.append("Low cash flow")
            if unstable_location:
                risk_factors.append("Unstable location patterns")
            if location_stability > 0.5:
                risk_factors.append("High location variability")
            if cash_incoming < 2000:
                risk_factors.append("Low monthly income")
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("• No major risk factors identified")

def show_analysis_page(model_artifacts):
    """Page for model analysis and insights"""
    
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="best-model-badge">Support Vector Machine - Best Performing Model</div>', unsafe_allow_html=True)
    
    # Model performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "0.700")
    with col2:
        st.metric("Best Model", "SVM")
    with col3:
        st.metric("Features Used", "14")
    with col4:
        st.metric("Training Samples", "399")
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Performance Comparison")
    
    models_data = {
        'Model': ['Support Vector Machine', 'Gradient Boosting', 'Extra Trees', 
                 'Random Forest', 'Gaussian Naive Bayes', 'XGBoost', 
                 'K-Nearest Neighbors', 'Decision Tree', 'Logistic Regression', 'AdaBoost'],
        'Accuracy': [0.700, 0.688, 0.688, 0.675, 0.663, 0.663, 0.650, 0.638, 0.625, 0.625],
        'Type': ['SVM', 'Ensemble', 'Ensemble', 'Ensemble', 'Traditional', 
                'Ensemble', 'Traditional', 'Traditional', 'Traditional', 'Ensemble']
    }
    
    df_models = pd.DataFrame(models_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # All models comparison
        fig = px.bar(df_models, 
                    x='Accuracy', 
                    y='Model',
                    orientation='h',
                    title='All Models Accuracy Comparison',
                    color='Accuracy',
                    color_continuous_scale='Viridis')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Performance by model type
        type_performance = df_models.groupby('Type')['Accuracy'].mean().reset_index()
        fig = px.pie(type_performance, 
                    values='Accuracy', 
                    names='Type',
                    title='Average Accuracy by Model Type',
                    color='Type',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    # SVM specific information
    st.markdown("---")
    st.subheader("Support Vector Machine (SVM) Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **SVM Advantages:**
        - Effective in high-dimensional spaces
        - Works well with clear margin of separation
        - Memory efficient
        - Versatile through kernel functions
        """)
    
    with col2:
        st.info("""
        **SVM in This Context:**
        - Best performance: 70.0% accuracy
        - Handles 14 features effectively
        - Robust to overfitting
        - Good generalization capability
        """)
    
    # Feature information
    st.markdown("---")
    st.subheader("Feature Set Information")
    
    feature_categories = {
        'Demographic': ['age', 'young_borrower', 'age_group_encoded'],
        'Financial': ['cash_incoming_30days', 'low_cash_flow', 'cash_category_encoded'],
        'Behavioral': ['gps_fix_count', 'longitude_std', 'latitude_std', 'accuracy_mean', 'unstable_location'],
        'Temporal': ['application_dayofweek', 'application_hour', 'application_month']
    }
    
    for category, features in feature_categories.items():
        with st.expander(f"{category} Features ({len(features)} features)"):
            for feature in features:
                st.write(f"• {feature}")

def show_about_page():
    """About page with information about the project"""
    
    st.markdown('<h2 class="sub-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    st.markdown('<div class="best-model-badge">Support Vector Machine - 70.0% Accuracy</div>', unsafe_allow_html=True)
    
    st.write("""
    ## Loan Default Prediction System
    
    This machine learning application uses a **Support Vector Machine (SVM)** model to help 
    financial institutions make better lending decisions by predicting the likelihood of loan defaults.
    
    ### Model Development:
    - **Best Performance**: SVM achieved 70.0% accuracy
    - **Feature Engineering**: 14 carefully selected features
    - **Balanced Dataset**: 399 samples with equal class distribution
    
    ### Top Performing Models:
    1. **Support Vector Machine** - 70.0% accuracy
    2. **Gradient Boosting** - 68.8% accuracy  
    3. **Extra Trees** - 68.8% accuracy
    4. **Random Forest** - 67.5% accuracy
    5. **Gaussian Naive Bayes** - 66.3% accuracy
    """)

if __name__ == "__main__":
    main()
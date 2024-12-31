import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('preXGBoost(1).pkl')
scaler = joblib.load('prescaler.pkl') 


# Define feature names
feature_names = ["Female age", "Primary infertility", "BMI", "FSH", "E2", "AFC"]

## Streamlit user interface
st.title("PreIVF Predictor")

female_age = st.number_input("Female age:", min_value=18, max_value=100, value=30)
yuanfa = st.selectbox("Primary infertility (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
bmi = st.number_input("BMI:", min_value=10.00, max_value=100.00, value=24.00)
FSH = st.number_input("FSH (U/L):", min_value=0.00, max_value=50.00, value=20.00)
E2 = st.number_input("E2 (pg/ml):", min_value=0.00, max_value=500.00, value=150.00)
AFC = st.number_input("AFC:", min_value=0, max_value=100, value=10)

# Process inputs and make predictions
feature_values = [female_age,yuanfa,bmi,FSH,E2,AFC]
features = np.array([feature_values])

# 分离连续变量和分类变量
continuous_features = [female_age, bmi,FSH,E2,AFC]
categorical_features=[yuanfa]

# 对连续变量进行标准化
continuous_features_array = np.array(continuous_features).reshape(1, -1)


# 关键修改：使用 pandas DataFrame 来确保列名
continuous_features_df = pd.DataFrame(continuous_features_array, columns=["female_age", "bmi","FSH", "E2", "AFC"])

# 标准化连续变量
continuous_features_standardized = scaler.transform(continuous_features_df)

# 将标准化后的连续变量和原始分类变量合并
# 确保连续特征是二维数组，分类特征是一维数组，合并时要注意维度一致
categorical_features_array = np.array(categorical_features).reshape(1, -1)


# 将标准化后的连续变量和原始分类变量合并
final_features = np.hstack([continuous_features_standardized, categorical_features_array])

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(final_features, columns=feature_names)



if st.button("Predict"):    
    # Predict class and probabilities    
    predicted_class = model.predict(final_features_df)[0]   
    predicted_proba = model.predict_proba(final_features_df)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class}(0: No Disease,1: Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:       
        advice = (
            f"According to our model, your probability of having a successful pregnancy is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your health, "            
            "and to seek medical advice promptly if you experience any symptoms." 
        )

              
    else:        
        advice = (
            f"According to our model, your probability of not having a successful pregnancy is {probability:.1f}%. "            
            "While this is just an estimate,it suggests that your probability of successful pregnancy are low."              
            "I recommend that you consult a specialist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )   
    st.write(advice)

# Calculate SHAP values and display force plot 
    st.subheader("SHAP Force Plot Explanation") 
    explainer = shap.TreeExplainer(model) 
   
    shap_values = explainer.shap_values(pd.DataFrame(final_features_df, columns=feature_names))
   # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    shap.force_plot(explainer.expected_value, shap_values[0], original_feature_values, matplotlib=True)   
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
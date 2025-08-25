
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ---- Load Model Safely ----
@st.cache_resource
def load_model():
    try:
        return joblib.load('fraud_detection_model.pkl')
    except Exception as e:
        st.error(f"❌ Model load nahi ho saka. File check karein: {e}")
        st.stop()

model = load_model()

# ---- Page Setup ----
st.set_page_config(page_title="🛡️ FraudGuard AI", page_icon="🛡️", layout="centered")

# ---- Animated Header: Built by Tafseer Haider ----
st.markdown("""
<style>
@keyframes pulse { 0% { transform: scale(1); } 50% { transform: scale(1.1); } 100% { transform: scale(1); } }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
.animated-by {
    font-family: 'Courier New', monospace;
    font-size: 18px;
    text-align: center;
    margin: 20px 0;
    animation: pulse 2s infinite, blink 1.5s infinite;
    color: #FF4081;
    font-weight: bold;
}
</style>
<div class="animated-by">✨ Built by Tafseer Haider ✨</div>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("🛡️ AI-Powered Fraud Detection System")
st.markdown("🔍 Real-time fraud detection using machine learning.")

# ---- Sidebar: Quick Actions ----
st.sidebar.header("⚡ Quick Actions")

if st.sidebar.button("🎯 Load Fraud Sample"):
    sample = [-2.3, 1.5, -1.8, 3.2, -0.5, 2000.0]
    for i in range(1, 6):
        st.session_state[f"v{i}"] = sample[i-1]
    st.session_state.amount = sample[5]

if st.sidebar.button("✅ Load Normal Sample"):
    sample = [0.0, 0.1, -0.2, 0.3, 0.0, 50.0]
    for i in range(1, 6):
        st.session_state[f"v{i}"] = sample[i-1]
    st.session_state.amount = sample[5]

if st.sidebar.button("🗑️ Clear Inputs"):
    for i in range(1, 6):
        st.session_state[f"v{i}"] = 0.0
    st.session_state.amount = 0.0

# ---- Current Time ----
st.sidebar.markdown("---")
st.sidebar.write("🕒 Current Time:")
st.sidebar.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# ---- Input Fields ----
st.subheader("📋 Enter Transaction Details")
cols = st.columns(3)
for i in range(1, 6):
    key = f"v{i}"
    if key not in st.session_state:
        st.session_state[key] = 0.0
    with cols[(i-1) % 3]:
        st.number_input(f"V{i}", value=st.session_state[key], format="%.6f", step=0.01, key=key)

st.subheader("💰 Transaction Amount")
if "amount" not in st.session_state:
    st.session_state.amount = 0.0
st.number_input("", value=st.session_state.amount, format="%.2f", step=1.0, key="amount")

# ---- Predict Button ----
st.markdown("---")
if st.button("🔮 Predict Fraud", key="predict"):
    try:
        # Collect inputs
        input_data = [st.session_state[f"v{i}"] for i in range(1, 6)]
        input_data.append(st.session_state.amount)
        
        # Convert to 2D array
        input_array = np.array([input_data])  # Shape: (1, 6)

        # Predict
        pred = model.predict(input_array)[0]
        prob = model.predict_proba(input_array)[0]

        # Show result
        st.markdown("### 📊 Prediction Result")
        if pred == 1:
            st.error("🚨 **FRAUD DETECTED!**", icon="🚨")
        else:
            st.success("✅ **LEGITIMATE TRANSACTION**", icon="✅")

        # Confidence chart
        fig, ax = plt.subplots(figsize=(6, 2.5))
        classes = ['Not Fraud', 'Fraud']
        colors = ['#4CAF50', '#F44336']
        bars = ax.bar(classes, prob, color=colors, alpha=0.8)
        ax.set_ylim(0, 1)
        for bar, p in zip(bars, prob):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{p:.4f}", ha='center', fontsize=12)
        st.pyplot(fig)

        st.write(f"🔢 **Not Fraud:** {prob[0]:.4f} | **Fraud:** {prob[1]:.4f}")

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
else:
    st.info("👉 Enter values or use Quick Actions to predict.")

# ---- Footer ----
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #BB86FC; font-weight: bold;'>"
    "🌈 Designed with ❤️ by Tafseer Haider</p>",
    unsafe_allow_html=True
)

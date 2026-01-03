import streamlit as st
import joblib
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

# ==========================================================
# 🌟 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# ==========================================================
# 📦 LOAD MODEL
# ==========================================================
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# ==========================================================
# 🎨 UI DESIGN
# ==========================================================
st.markdown(
    """
    <h1 style='text-align:center; color:#4CAF50;'>📩 SMS Spam Detection App</h1>
    <p style='text-align:center;'>AI Powered Machine Learning Application</p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.image(
    "https://cdn-icons-png.flaticon.com/512/2950/2950710.png",
    width=150
)

# ==========================================================
# ✍ USER INPUT
# ==========================================================
user_input = st.text_area(
    "✉ Enter your SMS message:",
    height=150,
    placeholder="Type your message here..."
)

# ==========================================================
# 🔍 PREDICTION
# ==========================================================
if st.button("🔍 Check Spam"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a message first!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized).max() * 100

        if prediction == 1:
            st.error(f"🚨 SPAM MESSAGE\n\nConfidence: {probability:.2f}%")
        else:
            st.success(f"✅ NOT SPAM MESSAGE\n\nConfidence: {probability:.2f}%")

# ==========================================================
# 📌 FOOTER
# ==========================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center;'>
    Made with ❤️ using Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)

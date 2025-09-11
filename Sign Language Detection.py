import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# --- Custom Dark Theme CSS ---
st.markdown("""
<style>
    .stApp { background-color: #1E1E1E; color: #FFFFFF; }
    h1 { color: #00A9B7; }
    .stAlert, .st-emotion-cache-1629p8f { background-color: #2D2D2D; border-radius: 10px; padding: 15px; }
    .stButton>button { background-color: #00A9B7; color: white; border-radius: 10px; border: none; padding: 10px 24px; }
    p, .stMarkdown { color: #E0E0E0; }
    .stImage > img { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


def check_time():
    """Checks if the current time is between 6 PM (18:00) and 10 PM (22:00)."""
    now = datetime.now().time()
    start_time = datetime.strptime("18:00", "%H:%M").time()
    end_time = datetime.strptime("22:00", "%H:%M").time()
    return start_time <= now <= end_time


# --- Main Application ---
st.title("Sign Language Detection 🤟")
st.write("This application is only operational between 6 PM and 10 PM.")

is_active_time = check_time()

if is_active_time:
    st.success("The system is currently active!")

    if 'stop' not in st.session_state:
        st.session_state.stop = False

    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        st.session_state.stop = False
    if stop_button:
        st.session_state.stop = True

    if start_button and not st.session_state.stop:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        while not st.session_state.stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break

            # Simulate sign language detection
            # In a real model, this is where you would process the frame
            signs = ["Hello", "Thank You", "I Love You", "Yes", "No"]
            detected_sign = np.random.choice(signs)

            # Display the detected sign on the frame
            cv2.putText(frame, f"Sign: {detected_sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
else:
    st.error("The system is currently offline. Please check back between 6 PM and 10 PM.")

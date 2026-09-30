import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from google import genai
from vision_processor import process_image

# 1. Environment & API Setup
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or "AQ.Ab8RN6JZVGGbRQ5z6J1JAchvM3iNyMlq072CgYB39QH_GhyWbQ"

client = genai.Client(api_key=api_key)

# 2. UI Configuration
st.set_page_config(page_title="VisionChat Assistant", layout="wide")
st.title("👁️ Multimodal Visual & Image Processing Assistant")

# 3. Sidebar: Image Upload & Computer Vision Transformations
st.sidebar.header("Image Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
filter_option = st.sidebar.selectbox(
    "CV Operation",
    ["Original", "Edge Detection (Canny)", "Thresholding (Otsu)", "Contour Highlighting"]
)

active_img = None
if uploaded_file:
    raw_img = Image.open(uploaded_file)
    active_img = process_image(raw_img, filter_option)
    st.sidebar.image(active_img, caption=f"Processed: {filter_option}", use_container_width=True)

# 4. Chat History Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. User Query & Gemini Inference
if prompt := st.chat_input("Ask a question about the image or CV task..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        inputs = [prompt]
        if active_img:
            # Ensure the image is in RGB format for the Gemini vision encoder
            rgb_img = active_img.convert("RGB")
            inputs.append(rgb_img)

        # Attempt generation using confirmed active model identifiers
        candidate_models = ["gemini-flash-latest", "gemini-3.7-flash", "gemini-3.5-flash"]
        response_text = None

        with st.spinner("Analyzing image..."):
            for model_name in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=inputs
                    )
                    response_text = response.text
                    break
                except Exception:
                    continue

        if response_text:
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            st.error("Failed to generate response. Check your network or API quota.")
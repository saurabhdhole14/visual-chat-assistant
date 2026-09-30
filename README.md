# 👁️ Multimodal Visual & Image Processing Assistant

An interactive conversational AI assistant that bridges classical Computer Vision (OpenCV) with state-of-the-art multimodal Large Language Models (LLMs). Users can upload images, apply computer vision transformations (edge detection, thresholding, contour extraction), and inspect or query the visual artifacts conversationally using the Google Gemini API.

---

## 🚀 Features

- **Classical Computer Vision Pipeline (OpenCV)**:
  - **Canny Edge Detection**: Detects structural outlines and high-frequency boundaries.
  - **Otsu's Thresholding**: Automated binarization for contrast segmentation and feature isolation.
  - **Contour Extraction**: Boundary tracing and morphological shape highlighting.
- **Multimodal Conversational Intelligence**:
  - Direct integration with Google GenAI (`gemini-flash-latest`, `gemini-3.7-flash`).
  - Supports contextual visual Q&A regarding both original inputs and processed feature maps.
- **Dynamic Streamlit Interface**:
  - Instant image filtering preview.
  - Persistent multi-turn chat session state.
  - Built-in multi-model error handling and automatic endpoint fallback.

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: Streamlit
- **Vision & Image Processing**: OpenCV (`opencv-python-headless`), Pillow, NumPy
- **Generative AI & LLM**: Google GenAI SDK (`google-genai`)
- **Environment Management**: Python Dotenv (`python-dotenv`)

---

## 📁 Project Structure

```text
visual-chat-assistant/
├── .env                  # API keys and secret configuration (ignored by Git)
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Pinned project dependencies
├── vision_processor.py   # OpenCV transformation algorithms
├── app.py                # Streamlit UI & conversational reasoning engine
└── README.md             # Project documentation
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import requests  # <-- This is the new part that does the download

# --- Configuration ---

# --- NEW: Model Download Configuration ---
# I have put your GitHub Releases URL here:
MODEL_URL = "https://github.com/geshu-data-scientist/apple_count_detector/releases/download/model_apple/unet_apple_best.keras"
LOCAL_MODEL_PATH = "unet_apple_best.keras" # The name to save it as locally

IMG_HEIGHT = 256
IMG_WIDTH = 256
MIN_APPLE_AREA = 50
SMALL_AREA_THRESH = 300
MEDIUM_AREA_THRESH = 800

# --- Model Loading (Updated) ---
# This function now downloads the file *before* loading it
@st.cache_resource
def load_apple_model(url, local_path):
    """
    Downloads the model from a URL if not present, then loads it.
    """
    # 1. Check if model already exists locally
    if not os.path.exists(local_path):
        with st.spinner(f"Downloading model from {url} (this may take a minute)..."):
            try:
                # 2. Download the file
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                st.success("Model downloaded.")
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                st.error("Please check the MODEL_URL in the script.")
                return None
    
    # 3. Load the model from the local file
    try:
        model = tf.keras.models.load_model(local_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model from local file '{local_path}': {e}")
        return None

# --- Image Processing Functions (No Changes Needed) ---

def preprocess_image(img_cv2, target_size=(256, 256)):
    orig_img_resized = cv2.resize(img_cv2, target_size, interpolation=cv2.INTER_LINEAR)
    img_normalized = orig_img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch, orig_img_resized

def process_and_count(model, orig_img_cv2):
    img_batch, orig_img_resized = preprocess_image(orig_img_cv2, (IMG_HEIGHT, IMG_WIDTH))
    pred_batch = model.predict(img_batch)
    pred_mask = pred_batch[0] 
    pred_mask_binary = (pred_mask > 0.5).astype(np.float32)

    mask_uint8 = (pred_mask_binary * 255).astype(np.uint8).squeeze()
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    size_counts = {'Small': 0, 'Medium': 0, 'Large': 0}
    apple_areas = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_APPLE_AREA:
            apple_areas.append(area)
            if area < SMALL_AREA_THRESH:
                size_counts['Small'] += 1
            elif area < MEDIUM_AREA_THRESH:
                size_counts['Medium'] += 1
            else:
                size_counts['Large'] += 1
                
    apple_count = len(apple_areas)
    avg_area = np.mean(apple_areas) if apple_count > 0 else 0

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle("Apple Detection Results", fontsize=20, y=1.02)
    original_img_rgb = cv2.cvtColor(orig_img_resized, cv2.COLOR_BGR2RGB)
    
    ax1.imshow(original_img_rgb); ax1.set_title("Original (Resized)"); ax1.axis("off")
    ax2.imshow(pred_mask_binary.squeeze(), cmap='gray'); ax2.set_title("Predicted Mask (Raw)"); ax2.axis("off")
    ax3.imshow(original_img_rgb); ax3.imshow(pred_mask_binary.squeeze(), cmap='jet', alpha=0.5); ax3.set_title(f"Mask Overlay (Count: {apple_count})"); ax3.axis("off")
    plt.tight_layout()
    
    return apple_count, fig, size_counts, avg_area

# --- Streamlit App UI (Starts the app) ---

st.title("🍎 Apple Counter & Grader (U-Net)")
st.write("This app uses a U-Net model to segment, count, and size apples in an image.")

# 1. Load the model (now calls the new download function)
model = load_apple_model(MODEL_URL, LOCAL_MODEL_PATH)

if model is None:
    st.error("Model could not be loaded. App cannot continue.")
    st.stop()

# (Rest of the Streamlit UI is identical)

uploaded_file = st.file_uploader("Upload an image of an apple tree:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    orig_img_np = np.array(image)
    orig_img_cv2 = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing image, counting, and sizing apples..."):
        apple_count, fig, size_counts, avg_area = process_and_count(model, orig_img_cv2)
        
        st.metric(label="Total Apples Detected", value=apple_count)
        
        st.subheader("Size Distribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("Large Apples (> " + str(MEDIUM_AREA_THRESH) + " px)", size_counts['Large'])
        col2.metric("Medium Apples (" + str(SMALL_AREA_THRESH) + "-" + str(MEDIUM_AREA_THRESH) + " px)", size_counts['Medium'])
        col3.metric("Small Apples (< " + str(SMALL_AREA_THRESH) + " px)", size_counts['Small'])
        
        st.pyplot(fig)
        
        with st.expander("ℹ️ Calibration & Sizing Info"):
            st.warning("These size thresholds are guesses!")
            st.write(f"**Average Apple Area in this image:** `{avg_area:.2f}` pixels.")
            st.code(f"""
MIN_APPLE_AREA = {MIN_APPLE_AREA}
SMALL_AREA_THRESH = {SMALL_AREA_THRESH}
MEDIUM_AREA_THRESH = {MEDIUM_AREA_THRESH}
            """, language="python")

else:
    st.info("Please upload an image to begin.")


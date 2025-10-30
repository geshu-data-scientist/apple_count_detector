import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
MODEL_PATH = "unet_apple_best.keras"
IMG_HEIGHT = 256
IMG_WIDTH = 256
MIN_APPLE_AREA = 50  # Filters out noise

# --- NEW: Sizing Configuration (IN PIXELS) ---
# *** IMPORTANT ***
# You MUST calibrate these values yourself!
# These are based on the (256, 256) resized image.
# 1. Run the app on a few images
# 2. See the "Average Pixel Area" in the output
# 3. Decide on your thresholds for small, medium, and large.
SMALL_AREA_THRESH = 300   # e.g., anything under 300 pixels is 'Small'
MEDIUM_AREA_THRESH = 800  # e.g., anything 300-800 is 'Medium', >800 is 'Large'


# --- Model Loading ---
# Use @st.cache_resource to load the model only once
@st.cache_resource
def load_apple_model(model_path):
    """
    Loads the trained U-Net model.
    """
    st.write(f"Attempting to load model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Fatal Error: Could not load 'unet_apple_best.keras'.")
        st.error("Please make sure the .keras file is in the same directory as this app.")
        return None

# --- Image Processing Functions ---

def preprocess_image(img_cv2, target_size=(256, 256)):
    """
    Resizes and normalizes an image for model prediction.
    - img_cv2: An image loaded with OpenCV (BGR format).
    - target_size: A tuple (height, width).
    """
    # Resize original for display
    orig_img_resized = cv2.resize(img_cv2, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Preprocess for model
    img_normalized = orig_img_resized / 255.0
    
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, orig_img_resized

def process_and_count(model, orig_img_cv2):
    """
    Runs the full prediction, counting, AND SIZING pipeline.
    Returns: apple_count, fig, size_counts, avg_area
    """
    # 1. Preprocess the image for the model
    img_batch, orig_img_resized = preprocess_image(orig_img_cv2, (IMG_HEIGHT, IMG_WIDTH))

    # 2. Predict the mask
    pred_batch = model.predict(img_batch)
    pred_mask = pred_batch[0] 
    pred_mask_binary = (pred_mask > 0.5).astype(np.float32)

    # 3. Find Contours
    mask_uint8 = (pred_mask_binary * 255).astype(np.uint8).squeeze()
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. --- NEW: Sizing and Counting Logic ---
    size_counts = {'Small': 0, 'Medium': 0, 'Large': 0}
    apple_areas = []
    
    for cnt in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(cnt)
        
        # Filter out tiny noise blobs
        if area > MIN_APPLE_AREA:
            apple_areas.append(area)
            
            # Sort into bins
            if area < SMALL_AREA_THRESH:
                size_counts['Small'] += 1
            elif area < MEDIUM_AREA_THRESH:
                size_counts['Medium'] += 1
            else:
                size_counts['Large'] += 1
                
    apple_count = len(apple_areas)
    avg_area = np.mean(apple_areas) if apple_count > 0 else 0

    # 5. Create the visualization plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plt.suptitle("Apple Detection Results", fontsize=20, y=1.02)
    
    original_img_rgb = cv2.cvtColor(orig_img_resized, cv2.COLOR_BGR2RGB)
    
    # Panel 1: Original Image
    ax1.imshow(original_img_rgb)
    ax1.set_title("Original (Resized)")
    ax1.axis("off")
    
    # Panel 2: Predicted Mask
    ax2.imshow(pred_mask_binary.squeeze(), cmap='gray')
    ax2.set_title("Predicted Mask (Raw)")
    ax2.axis("off")
    
    # Panel 3: Overlay
    ax3.imshow(original_img_rgb)
    ax3.imshow(pred_mask_binary.squeeze(), cmap='jet', alpha=0.5)
    ax3.set_title(f"Mask Overlay (Count: {apple_count})")
    ax3.axis("off")
    
    plt.tight_layout()
    
    return apple_count, fig, size_counts, avg_area

# --- Streamlit App UI ---

st.title("🍎 Apple Counter & Grader (U-Net)")
st.write("This app uses a U-Net model to segment, count, and size apples in an image.")

# 1. Load the model
model = load_apple_model(MODEL_PATH)

if model is None:
    st.stop()

# 2. Create the file uploader
uploaded_file = st.file_uploader("Upload an image of an apple tree:", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 3. Process the uploaded file
    image = Image.open(uploaded_file).convert("RGB")
    orig_img_np = np.array(image)
    orig_img_cv2 = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
    
    with st.spinner("Analyzing image, counting, and sizing apples..."):
        
        # 4. Run the main processing function
        apple_count, fig, size_counts, avg_area = process_and_count(model, orig_img_cv2)

        # 5. Display the results
        
        # --- Total Count ---
        st.metric(label="Total Apples Detected", value=apple_count)
        
        # --- NEW: Size Distribution ---
        st.subheader("Size Distribution")
        col1, col2, col3 = st.columns(3)
        col1.metric("Large Apples (> " + str(MEDIUM_AREA_THRESH) + " px)", size_counts['Large'])
        col2.metric("Medium Apples (" + str(SMALL_AREA_THRESH) + "-" + str(MEDIUM_AREA_THRESH) + " px)", size_counts['Medium'])
        col3.metric("Small Apples (< " + str(SMALL_AREA_THRESH) + " px)", size_counts['Small'])
        
        # --- Visualization ---
        st.pyplot(fig)
        
        # --- NEW: Calibration Note ---
        with st.expander("ℹ️ Calibration & Sizing Info"):
            st.warning("These size thresholds are guesses!")
            st.write(f"The `Small`, `Medium`, and `Large` categories are based on pixel area in the resized (256x256) image. You must calibrate them.")
            st.write(f"**Average Apple Area in this image:** `{avg_area:.2f}` pixels.")
            st.write(f"**Current Thresholds (in `streamlit_app.py`):**")
            st.code(f"""
MIN_APPLE_AREA = {MIN_APPLE_AREA}
SMALL_AREA_THRESH = {SMALL_AREA_THRESH}
MEDIUM_AREA_THRESH = {MEDIUM_AREA_THRESH}
            """, language="python")
            st.write("To calibrate, run this app on several images, note the 'Average Apple Area', and adjust the threshold variables in the script to match your real-world grades.")

else:
    st.info("Please upload an image to begin.")


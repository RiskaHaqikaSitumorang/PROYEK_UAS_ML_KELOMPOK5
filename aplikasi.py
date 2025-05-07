import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Set page configuration
st.set_page_config(
    page_title="Pizza vs Steak Classifier",
    page_icon="🍕",
    layout="centered"
)

# Custom model loading function to handle version mismatches
@st.cache_resource
def load_custom_model(model_path):
    try:
        # Try loading with custom objects first
        model = load_model(model_path, compile=False)
        
        # Recompile the model with appropriate settings
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Model path and settings
MODEL_PATH = 'food_CNN.h5'
CLASS_LABELS = {0: 'Pizza', 1: 'Steak'}
CONFIDENCE_THRESHOLD = 70  # 70% confidence threshold

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    try:
        # Convert to numpy array
        image = np.array(image)
        
        # Convert to RGB if needed
        if image.shape[-1] == 4:  # RGBA image
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 1:  # Grayscale image
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] != 3:  # Unknown format
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize and normalize (224x224 for our model)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

def main():
    st.title("🍕 Pizza vs Steak Classifier")
    st.write("Upload an image of pizza or steak, and our AI will classify it!")
    
    # Load model with custom loader
    model = load_custom_model(MODEL_PATH)
    
    if model is None:
        st.warning("""
            Couldn't load the model. Please ensure:
            1. model_3.h5 exists in this directory
            2. You have TensorFlow 2.x installed
            """)
        return
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            with st.spinner('Analyzing...'):
                processed_img = preprocess_image(image)
                
                if processed_img is not None:
                    prediction = model.predict(processed_img)
                    confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else 1 - float(prediction[0][0])
                    confidence *= 100
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.warning("❓ Unknown Food")
                        st.write("This doesn't look like pizza or steak")
                    else:
                        predicted_class = CLASS_LABELS[int(prediction > 0.5)]
                        if predicted_class == "Pizza":
                            st.success(f"🍕 Pizza Detected! ({confidence:.1f}% confidence)")
                        else:
                            st.success(f"🥩 Steak Detected! ({confidence:.1f}% confidence)")
                        st.progress(int(confidence))
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.info("Please upload an image of pizza or steak")

if __name__ == "__main__":
    main()
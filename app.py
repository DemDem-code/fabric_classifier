import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fabric Classifier", page_icon="🧶")

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('fabric_classifier_final.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

st.title("🧶 AI Fabric Scanner")
st.write("Upload a close-up photo of fabric to identify if it is **Cotton**, **Denim**, or **Silk**.")

model = load_model()

if model is None:
    st.error("❌ Model failed to load. Please check the logs.")
    st.stop()

st.success("✅ Model loaded successfully!")

uploaded_file = st.file_uploader("Choose a fabric image...", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Fabric', width=300)
    
    with st.spinner('🔍 Analyzing fabric texture...'):
        # Preprocess image
        img = image.resize((180, 180))
        img_array = np.array(img)
        img_array = (img_array / 127.5) - 1  # Same preprocessing as training
        img_array = np.expand_dims(img_array, 0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        score = predictions[0]
        
        class_names = ['cotton', 'denim', 'silk']
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
    
    # Display results
    st.header(f"🎯 Result: {predicted_class.upper()}")
    
    # Confidence indicator
    if confidence > 80:
        st.success(f"✅ Confidence: {confidence:.2f}% (High)")
    elif confidence > 50:
        st.warning(f"⚠️ Confidence: {confidence:.2f}% (Medium)")
    else:
        st.error(f"❌ Confidence: {confidence:.2f}% (Low)")
    
    # Progress bar
    st.progress(int(confidence) / 100)
    
    # Show all predictions
    with st.expander("📊 See detailed predictions"):
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name.capitalize()}: {score[i]*100:.2f}%")
```

**File 2: `requirements.txt`** (Copy this - UPDATED):
```
tensorflow==2.15.0
streamlit==1.31.0
numpy==1.24.3
pillow==10.2.0

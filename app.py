import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('models/samsa_model_1ver20.h5')

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

def prepare_image(image):
    img = image.resize((299, 299))  
    img = img_to_array(img)            # Convert PIL Image to NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)        # Preprocess using Xception's preprocessing function
    img = base_model.predict(img)      # Get the features from the base model
    return img


st.image('assets/cat_samsa.jpg', width=250)
st.title('Samsa or not Samsa?')
st.markdown("**made by @hexerty**")

url = 'https://youtu.be/vIci3C4JkL0?si=mWWkBIazs1pQoW1E'
st.markdown("inspired by [this](%s)"% url)
st.divider()

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    
    # Display 
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Thinking🤔...")

    # Preprocess the image
    processed_image = prepare_image(image)

    #  predictions
    try:
        predictions = model.predict(processed_image)
        samsa_prob = predictions[0][1] 

        # Display result
        if samsa_prob > 0.5:
            st.write(f'Yummy!... This looks like samsa with a confidence of {samsa_prob * 100:.2f}%')
        else:
            st.write(f'Oh... This is NOT samsa, confidence: {samsa_prob * 100:.2f}%')
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import streamlit as st
import kagglehub

# ========== STREAMLIT HEADER ==========
st.title("🌸 Flower Classifier - Using Olga Belitskaya's Dataset")
st.write("Upload a flower image and get a prediction.")

# ========== SETUP KAGGLE CREDENTIALS ==========
os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'  # Replace with your Kaggle username
os.environ['KAGGLE_KEY'] = 'your_kaggle_key'            # Replace with your Kaggle API key

# ========== DOWNLOAD DATA ==========
@st.cache_resource
def download_and_prepare_data():
    path = kagglehub.dataset_download("olgabelitskaya/flower-color-images")
    st.write("📁 Downloaded dataset path:", path)
    
    # Try to find the directory that contains class folders
    for root, dirs, files in os.walk(path):
        if all(os.path.isdir(os.path.join(root, d)) for d in dirs) and len(dirs) > 1:
            st.write("✅ Found flower classes in:", root)
            return root

    st.error("❌ Could not find class folders in the dataset.")
    return None

data_dir = download_and_prepare_data()

# ========== LOAD DATA ==========
if data_dir:
    img_height, img_width = 180, 180
    batch_size = 32

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names

    # ========== DEFINE AND TRAIN MODEL ==========
    @st.cache_resource
    def train_model():
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(len(class_names), activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_ds, validation_data=val_ds, epochs=3)
        model.save("flower_model.h5")
        return model

    model = train_model()

    # ========== STREAMLIT APP INTERFACE ==========
    uploaded_file = st.file_uploader("📷 Upload a flower image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_uploaded = Image.open(uploaded_file).convert("RGB").resize((img_width, img_height))
        st.image(image_uploaded, caption="Uploaded Image", use_column_width=False)

        # Preprocess uploaded image
        img_array = tf.keras.utils.img_to_array(image_uploaded) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"🌼 Predicted Flower Type: **{predicted_class}**")
else:
    st.error("Dataset not available. Please check your Kaggle credentials and dataset download.")

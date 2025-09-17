import os
from datetime import datetime
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import pdf_report
from processor import (
    detect_animal_bbox,
    extract_silhouette,
    compute_body_length_and_height,
    detect_visible_disease,
)
from save_and_notify import save_record


# -------------- Page config (dark, icon, wide) --------------
st.set_page_config(
    page_title="NETRA 👁 — Cattle Classifier",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",

)
import streamlit as st
import pandas as pd

data = {
    "Breed": [
        
        "Holstein Friesian",
        "Jersey",
        
        "Sahiwal"
    ],
    "Milk per Day (Liters)": [
        
        "15-35",
        "15-20",
        
        "8-20"
    ],
    "Fat Content (%)": [
        
        "3.5-4",
        "4.5-5",
        
        "4-5"
    ],
    "Average Price per Liter (₹)": [
        
        "60",
        "65",
        
        "62"
    ]
}

df = pd.DataFrame(data)

st.subheader("Cattle Breed Milk Production Details")
st.table(df)




# Cattle breeds sirf inhi list me allowed hain
cattle_breeds = [
    "Sorry it's Currently not in our Dataset",
    "Holstein Friesian    \t   15-35 Liters Milk/Day    \t Fat 3.5-4% \t   Avg Price ₹60 per Liter",
    "Jersey   \t       15-20 Liters Milk/Day    \t       Fat 4.5-5%     \t   Avg Price ₹65 per Liter",
    "RedDane \t       12 Liters Milk/Day \t               Fat 3.2-4% \t    Avg Price ₹58 per Liter",
    "Sahiwal \t       8-20 Liters Milk/Day \t           Fat 4-5% \t    Avg Price ₹62 per Liter"
]


st.title("Animal Type Classification System")

uploaded_file = st.file_uploader("Upload an image of Cattle or Buffalo (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        # Pehle uploaded image show karo for user feedback
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Save temporary image for YOLO input
        temp_path = "temp_img.jpg"
        cv2.imwrite(temp_path, image_cv)

        # Load YOLO model
        model = YOLO("best.pt")

        # YOLO prediction
        results = model.predict(source=temp_path)

        if results and len(results) > 0:
            pred_class_id = int(results[0].boxes.cls[0].item())
            confidence = float(results[0].boxes.conf[0].item())

            if pred_class_id < len(cattle_breeds):
                predicted_breed = cattle_breeds[pred_class_id]
            else:
                predicted_breed = "Unknown"

            if predicted_breed not in cattle_breeds:
                predicted_breed = "Unknown"
                confidence = 0.0

            st.subheader("Results")
            st.write(f"Predicted Animal: {predicted_breed}")
            st.write(f"Confidence: {confidence:.2f}")

            # Detect bounding box on image_cv
            bbox, _ = detect_animal_bbox(image_cv)

            if bbox is not None:
                # Extract silhouette mask for body measurement
                _, silhouette_mask = extract_silhouette(image_cv, bbox)
                body_metrics = compute_body_length_and_height(silhouette_mask)

                if body_metrics:
                    st.write(f"- Body Length: {body_metrics.get('body_length', 0):.2f} cm")
                    st.write(f"- Height at Withers: {body_metrics.get('height_at_withers', 0):.2f} cm")
                else:
                    st.write("Body measurements could not be calculated.")
            else:
                st.write("Animal bounding box not detected.")

            # Option to save the result
            if st.sidebar.checkbox("Save record?"):
                record = {
                    "breed": predicted_breed,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                }
                save_record(record, outfolder="records")
        else:
            st.warning("No prediction results.")

        # Remove temporary image after processing
        if os.path.exists(temp_path):
            os.remove(temp_path)

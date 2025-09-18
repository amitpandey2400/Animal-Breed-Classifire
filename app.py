import os
import re
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
import pandas as pd

def sanitize_filename(name):
    sanitized = re.sub(r'[^\w\-]', '_', name)
    sanitized = sanitized.replace('₹', 'INR')
    return sanitized

st.set_page_config(
    page_title="NETRA 👁 — Cattle Classifier",
    page_icon="🐂",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

cattle_breeds = [
    "Sorry it's Currently not in our Dataset",
    "Holstein Friesian    15-35 Liters Milk/Day  Fat 3.5-4%    Avg Price INR60 per Liter",
    "Jersey           15-20 Liters Milk/Day        Fat 4.5-5%   Avg Price INR65 per Liter",
    "Red Dane (Not covered)",
    "Sahiwal          8-20 Liters Milk/Day         Fat 4-5%     Avg Price INR62 per Liter"
]

st.title("Animal Type Classification System")
uploaded_file = st.file_uploader("Upload an image of Cattle or Buffalo (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_path = "temp_img.jpg"
        cv2.imwrite(temp_path, image_cv)
        model = YOLO("best.pt")
        results = model.predict(source=temp_path)

        predicted_breed = "Unknown"
        confidence = 0.0
        body_metrics = None

        if results and len(results) > 0:
            pred_class_id = int(results[0].boxes.cls[0].item())
            confidence = float(results[0].boxes.conf[0].item())
            if pred_class_id < len(cattle_breeds):
                predicted_breed = cattle_breeds[pred_class_id]
            else:
                predicted_breed = "Unknown"

            # Hide Red Dane from display if needed
            if predicted_breed == "Red Dane (Not covered)":
                predicted_breed = "Unknown"
                confidence = 0.0

            st.subheader("Results")
            st.write(f"Predicted Animal: {predicted_breed}")
            st.write(f"Confidence: {confidence:.2f}")

            bbox, _ = detect_animal_bbox(image_cv)
            if bbox is not None:
                _, silhouette_mask = extract_silhouette(image_cv, bbox)
                body_metrics = compute_body_length_and_height(silhouette_mask)
                if body_metrics:
                    st.write(f"- Body Length: {body_metrics.get('body_length', 0):.2f} cm")
                    st.write(f"- Height at Withers: {body_metrics.get('height_at_withers', 0):.2f} cm")
                else:
                    st.write("Body measurements could not be calculated.")
            else:
                st.write("Animal bounding box not detected.")

            if st.sidebar.checkbox("Save record?"):
                record = {
                    "breed": predicted_breed,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                }
                save_record(record, outfolder="records")

            if st.sidebar.button("Generate PDF Report"):
                pdf_filename = f"{sanitize_filename(predicted_breed)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_data = {
                    "breed": predicted_breed,
                    "confidence": confidence,
                    "body_length": body_metrics.get('body_length', 0) if body_metrics else 0,
                    "height_at_withers": body_metrics.get('height_at_withers', 0) if body_metrics else 0,
                    "chest_width": body_metrics.get('chest_width', 0) if body_metrics else 0,
                    "rump_angle_deg": body_metrics.get('rump_angle_deg', 0) if body_metrics else 0,
                    "atc_score": "N/A"
                }
                pdf_report.generate_report(pdf_filename, pdf_data)
                st.success(f"PDF report generated: {pdf_filename}")

        else:
            st.warning("No prediction results.")
        if os.path.exists(temp_path):
            os.remove(temp_path)

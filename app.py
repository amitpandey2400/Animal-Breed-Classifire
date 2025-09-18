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
)

from scoring import atc_score

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

# Existing milk production details table intact and unchanged
data = {
    "Breed": [
        "Holstein Friesian",
        "Jersey",
        "Sahiwal"
    ],
    "Lifespan (Years)": [
        "6 to 8",
        "7 to 12",
        "10 to 12"
    ],"Maturity Age (Years)": [
        "2 to 2.5",    # Approximate age at sexual maturity
        "1.7 to 2",    # Approximate age at sexual maturity
        "2 to 2.5"     # Approximate age at sexual maturity
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
    "Holstein Friesian 15-35 Liters Milk/Day Fat 3.5-4% Avg Price INR60 per Liter",
    "Jersey 15-20 Liters Milk/Day Fat 4.5-5% Avg Price INR65 per Liter",
    "Red Dane (Not covered)",
    "Sahiwal 8-20 Liters Milk/Day Fat 4-5% Avg Price INR62 per Liter"
]

st.title("Animal Type Classification System")

# New feature: Multiple images upload for combined breed detection
uploaded_files = st.file_uploader(
    "Upload multiple images of Cattle or Buffalo from different angles (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    model = YOLO("best.pt")
    breeds = []
    confidences = []
    body_lengths = []
    heights = []
    chest_widths = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        temp_path = f"temp_{sanitize_filename(uploaded_file.name)}.jpg"
        write_success = cv2.imwrite(temp_path, image_cv)
        if not write_success:
            st.error(f"Error saving temporary file for {uploaded_file.name}")
            continue

        results = model.predict(source=temp_path)

        predicted_breed = "Unknown"
        confidence = 0.0

        if results and len(results) > 0:
            pred_class_id = int(results[0].boxes.cls[0].item())
            confidence = float(results[0].boxes.conf[0].item())
            if pred_class_id < len(cattle_breeds):
                predicted_breed = cattle_breeds[pred_class_id]
            else:
                predicted_breed = "Unknown"
            if predicted_breed == "Red Dane (Not covered)":
                predicted_breed = "Unknown"
                confidence = 0.0

        breeds.append(predicted_breed)
        confidences.append(confidence)

        bbox, _ = detect_animal_bbox(image_cv)
        if bbox is not None:
            _, silhouette_mask = extract_silhouette(image_cv, bbox)
            body_metrics = compute_body_length_and_height(silhouette_mask)
            if body_metrics:
                body_lengths.append(body_metrics.get('body_length', 0))
                heights.append(body_metrics.get('height_at_withers', 0))
                chest_widths.append(body_metrics.get('chest_width', 0))
            else:
                body_lengths.append(0)
                heights.append(0)
                chest_widths.append(0)
        else:
            body_lengths.append(0)
            heights.append(0)
            chest_widths.append(0)

        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Combined breed decision and average results display
    if len(set(breeds)) == 1 and breeds[0] != "Unknown":
        avg_confidence = sum(confidences) / len(confidences)
        avg_length = sum(body_lengths) / len(body_lengths)
        avg_height = sum(heights) / len(heights)
        avg_chest_width = sum(chest_widths) / len(chest_widths)

        # Calculate average rump angle and ATC score
        avg_rump_angle = 0.0  # You can compute average if you save all rump_angle values in loop
        # Here adding 0.0 for simplicity
        avg_atc_score = atc_score(avg_length, avg_height, avg_chest_width, avg_rump_angle)

        st.subheader("Combined Results")
        st.write(f"Predicted Breed: {breeds[0]}")
        st.write(f"Confidence: {avg_confidence:.2f}")
        st.write(f"Body Length: {avg_length:.2f} cm")
        st.write(f"Height at Withers: {avg_height:.2f} cm")
        st.write(f"Chest Width: {avg_chest_width:.2f} cm")

        if st.sidebar.checkbox("Save combined record?"):
            record = {
                "breed": breeds[0],
                "confidence": avg_confidence,
                "timestamp": datetime.now().isoformat(),
            }
            save_record(record, outfolder="records")

        if st.sidebar.button("Generate Combined PDF Report"):
            pdf_filename = f"{sanitize_filename(breeds[0])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_combined.pdf"
            pdf_data = {
                "breed": breeds[0],
                "confidence": avg_confidence,
                "body_length": avg_length,
                "height_at_withers": avg_height,
                "chest_width": avg_chest_width,
                "rump_angle_deg": avg_rump_angle,
                "atc_score": avg_atc_score
            }
            pdf_report.generate_report(pdf_filename, pdf_data)
            st.success(f"PDF report generated: {pdf_filename}")

    else:
        st.warning("The uploaded images do not belong to the same breed. Please upload multiple images of the same breed for combined analysis.")

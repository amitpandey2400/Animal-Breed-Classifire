import os
import io
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import cv2

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
    initial_sidebar_state="expanded"
)

# -------------- Theme: background and colors via inlined CSS in markdown --------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #2e003e 0%, #14062d 100%)!important;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Emoji and Style Palette
icon_bold = "🌟"
icon_upload = "📤"
icon_metrics = "📏"
icon_breed = "🧬"
icon_disease = "🦠"
icon_pdf = "📄"
icon_success = "✅"
icon_fail = "🛑"
icon_warning = "⚠️"
icon_save = "💾"
icon_setting = "⚙️"

badge_style = "background:linear-gradient(90deg,#fc466b,#3f5efb 100%);padding:0.2em 0.9em;color:white;border-radius:12px;font-weight:900;border:2px solid #fff;font-size:1.15em;box-shadow:0 3px 12px 0 rgba(0,0,30,0.16);"

# -------------- Side Panel (bold, badge) --------------
with st.sidebar:
    st.markdown(f"{icon_setting} <span style='{badge_style}'>SETTINGS</span>", unsafe_allow_html=True)
    auto_save_local = st.checkbox(f"{icon_save} Auto-save record locally", value=True)
    st.markdown("---")
    st.markdown(
        f"<span style='{badge_style};background:linear-gradient(90deg,#095a5a,#4636a0);'>Powered by NETRA</span>",
        unsafe_allow_html=True)

# -------------- Header Section --------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"<h1 style='font-size:2.75em;font-weight:900;color:#ff289b;text-align:center;'>🐄 <span style='{badge_style}background:linear-gradient(90deg,#ff9371,#865dff);'>NETRA - The Cattle Classifier</span> 👁</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<div style='text-align:center;font-size:1.3em;font-weight:700;color:#fff;'>Upload images for superfast <span style='color:#ffe340;'>breed</span> classification, <span style='color:#14df1b;'>body measurements</span>, <span style='color:#ff2874;'>ATC score</span> & instant PDF reports.</div><br>",
    unsafe_allow_html=True
)

# -------------- Model setup --------------
MODEL_WEIGHTS = "last.pt"
model = YOLO(MODEL_WEIGHTS)
class_names = [
    "Cross Cattle", "Holstein_Friesian", "Jersey", "Red_Dane", "Sahiwal",
    # Add more classes if your model has more
]

def atc_score(body_length, height_at_withers, chest_width, rump_angle):
    score = 0
    score += 30 if body_length > 150 else 25 if body_length > 140 else 10
    score += 30 if height_at_withers > 120 else 10
    score += 20 if chest_width > 50 else 10
    score += 20 if (rump_angle is not None and 70 <= rump_angle <= 90) else 10
    return score

# -------------- Image Uploader (badge label) --------------
st.markdown(
    f"<span style='{badge_style};background:linear-gradient(90deg,#ffe140,#fa709a);font-size:1.18em;'>{icon_upload} Upload 1–6 images (Different Angles)</span>",
    unsafe_allow_html=True
)
uploaded_files = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload max 6 images for comprehensive analysis"
)

last_report_payload = None

if uploaded_files:
    if len(uploaded_files) > 6:
        st.error(f"{icon_warning} Max 6 images allowed!")
    else:
        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown(
                f"<div style='{badge_style};background:linear-gradient(90deg,#53ffcd,#23a6d5);color:#222;text-align:left;font-size:1.05em;'>Image {idx + 1}</div>",
                unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.image(uploaded_file, caption="Original Image", use_column_width=True)
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            bbox, _ = detect_animal_bbox(cv_img)
            if bbox:
                crop, silhouette = extract_silhouette(cv_img, bbox)
                with col2:
                    st.image(crop, caption="Detected Animal Crop", use_column_width=True)
                with st.spinner("Calculating physical measurements..."):
                    result = compute_body_length_and_height(silhouette)
                atc_score_value = None
                if result:
                    atc_score_value = atc_score(
                        result["body_length"],
                        result["height_at_withers"],
                        result["chest_width"],
                        result["rump_angle_deg"],
                    )
                with st.spinner("Classifying breed..."):
                    tmp_path = f"_temp_{datetime.now().strftime('%H%M%S%f')}.jpg"
                    cv2.imwrite(tmp_path, crop)
                    yolo_results = model.predict(tmp_path, verbose=False)
                    os.remove(tmp_path)
                    predicted_breed = "Unknown"
                    confidence_pct = 0.0
                    if yolo_results and len(yolo_results) > 0:
                        r = yolo_results[0]
                        if r.boxes is not None and len(r.boxes) > 0:
                            top_idx = int(np.argmax(r.boxes.conf.cpu().numpy()))
                            cls_id = int(r.boxes.cls[top_idx])
                            conf = float(r.boxes.conf[top_idx])
                            predicted_breed = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
                            confidence_pct = round(conf * 100.0, 2)
                with st.spinner("Detecting visible diseases..."):
                    diseases = detect_visible_disease(crop)

                st.markdown(
                    f"<span style='{badge_style}background:linear-gradient(90deg,#fdc830,#f37335);font-size:1.08em;'>{icon_metrics} Physical Measurements</span>",
                    unsafe_allow_html=True,
                )
                if result:
                    measurements_df = pd.DataFrame({
                        "Measurement": ["Body Length (cm)", "Height at Withers (cm)", "Chest Width (cm)", "Rump Angle (deg)"],
                        "Value": [
                            f"**{result['body_length']:.2f}**",
                            f"**{result['height_at_withers']:.2f}**",
                            f"**{result['chest_width']:.2f}**",
                            f"**{result['rump_angle_deg']:.2f}**" if result["rump_angle_deg"] is not None else "**N/A**",
                        ],
                    })
                    st.dataframe(measurements_df, use_container_width=True)
                    st.metric(label="🏅 ATC Score", value=f"{atc_score_value}" if atc_score_value is not None else "N/A")
                else:
                    st.info("Measurements not detected.")

                st.markdown(
                    f"<div style='font-size:1.3em;font-weight:900;color:#fff;background:linear-gradient(90deg,#f7971e 15%,#ffd200 100%);padding:0.33em 0.7em;border-radius:9px;display:inline-block;'>"
                    f"{icon_breed} <span style='color:#222;'>Predicted Breed:</span> <span style='color:#081996'>{predicted_breed}</span></div>",
                    unsafe_allow_html=True
                )
                st.metric(label="Confidence", value=f"{confidence_pct:.2f}%")
                st.markdown("<hr>", unsafe_allow_html=True)

                with st.expander(f"{icon_disease} Visible Swelling Detection »", expanded=False):
                    if diseases:
                        crop_disp = crop.copy()
                        for d in diseases:
                            x, y, w, h = d["bbox"]
                            cv2.rectangle(crop_disp, (x, y), (x + w, y + h), (255, 69, 58), 3)
                            cv2.putText(crop_disp, d["type"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 69, 58), 2)
                        st.image(crop_disp, caption="Swelling Areas Marked", use_column_width=True)
                        st.markdown(
                            f"{', '.join('<span style=\"font-weight:700;color:#ff2874;\">' + d['type'] + '</span>' for d in diseases)}",
                            unsafe_allow_html=True)
                    else:
                        st.write("No visible Swelling detected.")

                last_report_payload = {
                    "timestamp": datetime.now().isoformat(),
                    "predicted_breed": predicted_breed,
                    "confidence_pct": confidence_pct,
                    "body_length_cm": float(result["body_length"]) if result else None,
                    "height_at_withers_cm": float(result["height_at_withers"]) if result else None,
                    "chest_width_cm": float(result["chest_width"]) if result else None,
                    "rump_angle_deg": float(result["rump_angle_deg"]) if (result and result["rump_angle_deg"] is not None) else None,
                    "atc_score": int(atc_score_value) if atc_score_value is not None else None,
                }
                # Auto save locally if enabled
                if auto_save_local:
                    try:
                        saved_path = save_record(last_report_payload, out_folder="records")
                        st.success(f"{icon_success} Saved record: {os.path.basename(saved_path)}")
                    except Exception as e:
                        st.warning(f"{icon_warning} Save failed: {e}")
            else:
                with col2:
                    st.info(f"{icon_warning} Animal not detected in this image.")

# -------------- PDF Generation (Heading fixed to use markdown) --------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<h2>{icon_pdf} <span style='background:linear-gradient(90deg,#43cea2 0,#185a9d 100%);padding:0.19em 0.6em;border-radius:10px;color:#fff;font-weight:900;'>Generate PDF Report (Last Image)</span></h2>",
    unsafe_allow_html=True
)
if st.button(f"{icon_pdf} Generate PDF"):
    if last_report_payload and last_report_payload.get("predicted_breed"):
        os.makedirs("reports", exist_ok=True)
        pdf_path = f"reports/Report_{last_report_payload['predicted_breed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_report.generate_report(
            pdf_path,
            {
                "breed": last_report_payload["predicted_breed"],
                "confidence": last_report_payload["confidence_pct"],
                "body_length": last_report_payload["body_length_cm"] or 0.0,
                "height_at_withers": last_report_payload["height_at_withers_cm"] or 0.0,
                "chest_width": last_report_payload["chest_width_cm"] or 0.0,
                "rump_angle_deg": last_report_payload["rump_angle_deg"] or 0.0,
                "atc_score": last_report_payload["atc_score"] if last_report_payload["atc_score"] is not None else "N/A",
            },
        )
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                data=f.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
    else:
        st.warning(f"{icon_fail} Please process at least one image before generating a report.")

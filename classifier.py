import streamlit as st
import tempfile
import os
from processor import process_image
from save_and_notify import save_results
from classifier import ml_classifier  # 👈 Assuming you refactor/use ML classifier for other analysis

st.title("🐄 Animal Type Classification System")

uploaded_file = st.file_uploader("Upload an image of Cattle or Buffalo", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    results = process_image(tmp_path)
    cls = ml_classifier(tmp_path)

    results["Predicted Animal"] = cls["label"]
    results["Confidence"] = round(cls["confidence"] * 100, 2)

    st.subheader("Results")
    st.write(f"**Predicted Animal:** {results['Predicted Animal']}")
    st.write(f"**Confidence:** {results['Confidence']}%")

    st.subheader("📏 Body Measurements")
    st.write(f"- Body Length: {results.get('Body Length', 'N/A')}")
    st.write(f"- Height at Withers: {results.get('Height at Withers', 'N/A')}")
    st.write(f"- Chest Width: {results.get('Chest Width', 'N/A')}")
    st.write(f"- Rump Angle: {results.get('Rump Angle', 'N/A')}°")

    save_results(results)
    os.remove(tmp_path)

# pdf_report.py

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_report(filename, data):
    """
    Generate PDF report for animal type classification

    Parameters:
        filename (str): Path to save PDF
        data (dict): Dictionary containing keys like breed, confidence, body measurements, atc_score etc.
    """
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Animal Type Classification Report")

    c.setFont("Helvetica", 12)
    y = height - 80
    line_height = 18

    c.drawString(50, y, f"Breed: {data.get('breed', 'N/A')}")
    y -= line_height
    c.drawString(50, y, f"Confidence: {data.get('confidence', 0):.2f}")
    y -= line_height

    c.drawString(50, y, "Physical Measurements:")
    y -= line_height
    c.drawString(70, y, f"Body Length: {data.get('body_length', 0):.2f} cm")
    y -= line_height
    c.drawString(70, y, f"Height at Withers: {data.get('height_at_withers', 0):.2f} cm")
    y -= line_height
    c.drawString(70, y, f"Chest Width: {data.get('chest_width', 0):.2f} cm")
    y -= line_height
    c.drawString(70, y, f"Rump Angle: {data.get('rump_angle_deg', 0):.2f} degrees")
    y -= line_height

    c.drawString(50, y, f"ATC Score: {data.get('atc_score', 'N/A')}")
    y -= line_height

    c.save()

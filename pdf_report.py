from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def safe_format(value, unit=""):
    """Format numbers to 2 decimal places if number, else return as string."""
    if isinstance(value, (int, float)):
        return f"{value:.2f}{unit}"
    return str(value) + (f" {unit}" if unit else "")

def generate_report(filename, data):
    """
    Generate PDF report for animal type classification.
    Parameters:
        filename (str): Path to save PDF
        data (dict): Dictionary containing keys like breed, confidence, body measurements, atc_score, etc.
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
    c.drawString(50, y, f"Confidence: {safe_format(data.get('confidence', 'N/A'))}")
    y -= line_height

    c.drawString(50, y, "Physical Measurements:")
    y -= line_height
    c.drawString(70, y, f"Body Length: {safe_format(data.get('body_length', 'N/A'), ' cm')}")
    y -= line_height
    c.drawString(70, y, f"Height at Withers: {safe_format(data.get('height_at_withers', 'N/A'), ' cm')}")
    y -= line_height
    c.drawString(70, y, f"Chest Width: {safe_format(data.get('chest_width', 'N/A'), ' cm')}")
    y -= line_height

    rump_angle = data.get('rump_angle_deg', 'N/A')
    if isinstance(rump_angle, (int, float)):
        rump_angle_text = f"{rump_angle:.2f} degrees"
    else:
        rump_angle_text = str(rump_angle)
    c.drawString(70, y, f"Rump Angle: {rump_angle_text}")
    y -= line_height

    atc_score = data.get('atc_score', 'N/A')
    c.drawString(50, y, f"ATC Score: {atc_score}")
    y -= line_height

    c.save()

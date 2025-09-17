# save_and_notify.py

import json
import pandas as pd
import os
import requests
from datetime import datetime

def save_record(record, out_folder="records"):
    os.makedirs(out_folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(out_folder, f"record_{ts}.json")
    with open(filename, "w") as f:
        json.dump(record, f, indent=2)

    # Also append to CSV summary
    csv_path = os.path.join(out_folder, "summary.csv")
    df = pd.DataFrame([record])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
    return filename

def post_to_bpa_api(record, api_url, api_key, timeout=10):
    """
    Post classification record to Bharat Pashudhan App (BPA) API
    
    Parameters:
        record (dict): The classification data to send
        api_url (str): BPA API endpoint URL
        api_key (str): Authorization token or API key
        timeout (int): Request timeout in seconds
    
    Returns:
        status_code (int or None), response_text (str)
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    try:
        resp = requests.post(api_url, json=record, headers=headers, timeout=timeout)
        return resp.status_code, resp.text
    except Exception as e:
        return None, str(e)
    from twilio.rest import Client

def send_sms_alert(to_number, message, account_sid, auth_token, from_number):
    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            to=to_number,
            from_=from_number,
            body=message
        )
        return True, msg.sid
    except Exception as e:
        return False, str(e)


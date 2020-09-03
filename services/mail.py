import requests
import json
import base64
import datetime
import time
import os
from services import keys
# from dotenv import load_dotenv
# load_dotenv()
# SEND_MAIL_URL = os.getenv('SEND_MAIL_URL')
# BEARER_TOKEN = os.getenv('BEARER_TOKEN')

def send_approval_mail(publicIP,macAddress,hostName,image,empId):
    try:
        SEND_MAIL_URL = keys.get_send_mail_url()['SEND_MAIL_URL']
        BEARER_TOKEN = keys.get_bearer_token()['BEARER_TOKEN']
        ts = time.time()
        utc_time = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')

        base64string=""
        with open(image, "rb") as image_file:
            base64string = base64.b64encode(image_file.read())
            
        base64string = base64string.decode('utf8')

        url = SEND_MAIL_URL
    
        payload = {
        "publicIP":publicIP,
        "macAddress":macAddress,
        "hostName":hostName,
        "image": "data:image/jpeg;base64,"+base64string,
        "timestamp": utc_time,
        "empId": empId
        }
        headers = {
            "Content-Type":"application/json",
            'Authorization': BEARER_TOKEN
        }
    
        response = requests.request("POST", url, headers=headers, data = json.dumps(payload))
        payload = json.loads(response.text)   
        return None, payload
    
    except Exception as e:
        return e, None

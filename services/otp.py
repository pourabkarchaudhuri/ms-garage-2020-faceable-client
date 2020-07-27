import requests
import json
import os
import base64
from dotenv import load_dotenv
load_dotenv()
SEND_OTP_URL = os.getenv('SEND_OTP_URL')
VERIFY_OTP_URL = os.getenv('VERIFY_OTP_URL')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

def send_otp(emp_id):
    try:
        url = SEND_OTP_URL+emp_id
    
        payload = {}
        headers = {
        'Authorization': BEARER_TOKEN
        }
    
        response = requests.request("GET", url, headers=headers, data = payload)
        payload = json.loads(response.text)   
        return None, payload
    
    except Exception as e:
        return e, None

def verify_otp(otp_cont,emp_id):
    try:
        url = VERIFY_OTP_URL+"otp="+otp_cont+"&empId="+emp_id
    
        payload = {}
        headers = {
            'Authorization': BEARER_TOKEN
        }
    
        response = requests.request("GET", url, headers=headers, data = payload)
        payload = json.loads(response.text)   
        return None, payload
    
    except Exception as e:
        return e, None
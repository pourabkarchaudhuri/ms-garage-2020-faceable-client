import json
import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()
AZURE_FACE_RECOGNIZE_URL = os.getenv('AZURE_FACE_RECOGNIZE_URL')
AZURE_FACE_RETRAIN_URL = os.getenv('AZURE_FACE_RETRAIN_URL')
BEARER_TOKEN= os.getenv('BEARER_TOKEN')

def recognize(data):
    
     try:
        base64string=""
        with open(data, "rb") as image_file:
            base64string = base64.b64encode(image_file.read())
            
        base64string = base64string.decode('utf8')
        
        url = AZURE_FACE_RECOGNIZE_URL

        ##print(base64string)
    
        payload = {"imageString": "data:image/jpeg;base64,"+base64string}
        
        headers = {
            "Content-Type":"application/json",
            "Authorization": BEARER_TOKEN
        }

        response = requests.request("POST", url, headers=headers, data = json.dumps(payload))
        payload = json.loads(response.text)
        ##print("response")
        return None, payload
     except Exception as e:
        return e, None

def retrain(image, emp_id):
     try:
        base64string=""
        with open(image, "rb") as image_file:
            base64string = base64.b64encode(image_file.read())
            
        base64string = base64string.decode('utf8')

        url = AZURE_FACE_RETRAIN_URL
    
        payload = {"empId":emp_id, "imageString": "data:image/jpeg;base64,"+base64string}
        # f = open("t.txt","a")
        # f.write(format(payload))
        # f.close()
        headers = {
            "Content-Type":"application/json",
            'Authorization': BEARER_TOKEN
        }
    
        response = requests.request("POST", url, headers=headers, data = json.dumps(payload))
        payload = json.loads(response.text)   
        return None, payload
     except Exception as e:
        return e, None

import requests
import json
import base64
import os
import socket
from services import keys
# from dotenv import load_dotenv
# load_dotenv()
#CHECK_WFH_STATUS_URL = os.getenv('CHECK_WFH_STATUS_URL')
#BEARER_TOKEN = os.getenv('BEARER_TOKEN')



#print(socket.gethostname())

def check_status(hostname):
    try:
        CHECK_WFH_STATUS_URL = keys.get_check_wfh_status_url()['CHECK_WFH_STATUS_URL']
        BEARER_TOKEN = keys.get_bearer_token()['BEARER_TOKEN']
        #print(BEARER_TOKEN)
        url = CHECK_WFH_STATUS_URL+hostname
        payload = {}
        headers = {
    'Authorization': BEARER_TOKEN
        }
    
        response = requests.request("GET", url, headers=headers, data = payload)
        payload = json.loads(response.text)   
        return None, payload
    
    except Exception as e:
        return e, None


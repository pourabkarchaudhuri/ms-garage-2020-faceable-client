from requests import get
import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()

IP_INFO_URL= os.getenv("IP_INFO_URL")


def get_ip_info():
    try:
        url = IP_INFO_URL
        headers = {}
        response = requests.request("GET", url)
        payload = json.loads(response.text)   
        return None, payload
    
    except Exception as e:
        return e, None
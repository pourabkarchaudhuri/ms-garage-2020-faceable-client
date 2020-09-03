import requests
import json
import base64
import datetime
import time
import glob
import os
import uuid, re
from services import keys
#from dotenv import load_dotenv
#load_dotenv()

#EVENT_FAILURE_LOG_URL = os.getenv('EVENT_FAILURE_LOG_URL')
#BEARER_TOKEN = os.getenv('BEARER_TOKEN')

def log_failure_event(public_ip,hostName,city,region,country,org,loc,event_state):
    try:
        EVENT_FAILURE_LOG_URL  = keys.get_event_failure_log_url()['EVENT_FAILURE_LOG_URL']
        BEARER_TOKEN = keys.get_bearer_token()['BEARER_TOKEN']
        lat_long = loc.split(",")
        ##print(lat_long[0])
        mac_address = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        ts = time.time()
        utc = datetime.datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S')
        #event_arr = convert_to_base64(images)   

        url = EVENT_FAILURE_LOG_URL
        payload = {
        "publicIP":public_ip,
        "macAddress":mac_address,
        "hostName":hostName,
        "event": event_state,
        "loc": {"lat":lat_long[0], "long":lat_long[1]},
        "city": city,
        "region": region,
        "country": country,
        "broadband_org":org
        
}
        # f = open("payload.txt","a")
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

# def convert_to_base64(images_path):
#     event_dict ={}

#     event_arr =[]
    
#     base64string = ""
#     count = 0
#     #print('Converting to base 64 the event failure images')
#     ##print(os.getcwd())
#     for path in glob.glob(images_path+'*'): 
#         ##print(path)
#         count = count + 1
#         file_name = os.path.basename(path)
#         ##print(file_name)
#         if (count < 6):
#             #print(file_name)
#             with open(path, "rb") as image_file:
#                 unix_epoch_time = os.path.getmtime(path)
#                 base64string = base64.b64encode(image_file.read())
#                 base64string = base64string.decode('utf8')
#                 event_dict["image"] = "data:image/jpeg;base64,"+base64string
#                 event_dict["timestamp"] = datetime.datetime.utcfromtimestamp(unix_epoch_time).strftime('%Y-%m-%dT%H:%M:%S')
#                 event_arr.append(event_dict)
#     ##print(format(event_dict))
#     return event_arr

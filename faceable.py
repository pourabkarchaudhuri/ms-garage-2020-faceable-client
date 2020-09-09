import os
os.environ["KIVY_NO_CONSOLELOG"] = "1"
from kivy.app import App
from kivy.config import Config
Config.set('kivy', 'window_icon', 'static/faceable_logo.png')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'width', '640')
Config.set('graphics', 'height', '480')
Config.set('graphics', 'resizable', False)
#Config.set('graphics', 'window_state', 'hidden')
#Config.set('graphics', 'borderless', '1')
#Config.set('kivy', 'exit_on_escape', '0')
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.camera import Camera
from services import wfh, azure_face, otp, ip, mail, logs
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import BooleanProperty, ObjectProperty, StringProperty
from kivy.uix.image import Image
from kivy.core.window import Window
import socket
import time
import uuid, re
from kivy.uix.button import Button
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import shutil
from kivy.uix.image import Image
import imutils
from kivy.graphics.texture import Texture
import train_model
import extract_embeddings as embeddings
import numpy as np
import pickle
import ctypes
import sys
from collections import Counter
from kivy.cache import Cache
import subprocess
# from pystray import Icon as icon, Menu as menu, MenuItem as item
# from PIL import Image, ImageDraw, ImageFont
# import time
# from threading import Thread
import time
import win32gui
import win32con

#Window.clearcolor = (.0, .3, .9, 0)
Window.clearcolor = (.0,0.6,0.9,0)
Cache.register('cache', limit=10)
mac_address = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
GLOBAL_THRESHOLD = 0.90

def create_folder_if_not_present(path):
    dir = os.path.dirname(path)  #if it doesn't exist this function will create
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

# def delete_or_create_events_folder(path):
#     folder = path
#     dir = os.path.dirname(path)  #if it doesn't exist this function will create
#     if not os.path.exists(dir):
#         os.makedirs(dir)
#     else:
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))


#function to confirm whether the given path exists or not
#if it doesn't exist this function will create

def check_path(path):           
    dir = os.path.dirname(path)  
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        shutil.rmtree(dir)
        os.makedirs(dir)

def delete_output_folder(path):
    folder = path
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

class SplashScreen(Screen):
    def on_enter(self, *args):
        #print("on Enter Splash Screen")
        #Clock.schedule_once(self.initial_req_check, 3)
        # Clock.schedule_interval(self.is_user_logged_in, 10)
        Clock.schedule_once(self.initial_req_check, 3)
    
    # def is_user_logged_in(self, dt):
    #     process_name='LogonUI.exe'
    #     callall='TASKLIST'
    #     outputall=subprocess.check_output(callall)
    #     outputstringall=str(outputall)
    #     if process_name in outputstringall:
    #         Cache.append('cache','is_locked', True)
    #         print("Locked.")
    #     else: 
    #         Cache.append('cache','is_locked', False)
    #         print("Unlocked.")
    #         self.manager.current='lock_screen_status'

    # def move_to_lock_screen_status(self, dt):
    #     self.manager.current = 'lock_screen_status'
        


    def initial_req_check(self, dt):
        is_there_internet = checkInternetSocket()
        cam_present = is_cam_present(0)
        if(cam_present):
            #print("Camera present")
            if is_there_internet:
                #print("Connected to the internet" )
                self.manager.current = "wfh"
            else:
                #print("No internet")
                self.manager.current= "No internet"
        else:
            #print("No camera detected")
            self.manager.current="no_camera"
        ##print(self.manager.current)
        ##print(self.manager.next())
        #self.manager.current = 'wfh'


class InternetCheck(Screen):
    pass

class WFHScreen(Screen):
    def on_enter(self):
        create_folder_if_not_present("Output/")
        #delete_or_create_events_folder('events/')
        #create_folder_if_not_present("events/")
        self.is_status_active = False
        #print("WFH Screen")
        self.wfh_text_1 = self.ids["'wfhscreen_text1'"]
        self.wfh_loading_img = self.ids["'wfh_loading_img'"]
        #self.wfh_text_2 = self.ids["'wfhscreen_text2'"]
        #wfh_text.text = "Checking if this machine is allowed wfh..."
        Clock.schedule_once(self.check_for_approval_ip,4)

    def check_for_approval_ip(self,dt):
        #mac_address = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
        #print("Here")
        HOSTNAME = socket.gethostname()
        ip_retrieval_error, my_ip = ip.get_ip_info()
        if (ip_retrieval_error is not None):
            print("IP retrieval error")
        else:
            my_ip = my_ip["ip"]
            #print("My IP : " + my_ip)
            Cache.append('cache','ip',my_ip)
        wfh_service_error, wfh_status_response = wfh.check_status(HOSTNAME)
        #print(format(wfh_status_response))
        if(wfh_service_error is not None):
            print("Error in service : ", wfh_service_error)
        else:
            if(wfh_status_response['success']):
                #print(wfh_status_response)
                #print("status: "+format(wfh_status_response['data']['isActive']))
                ##Test###
                #self.is_status_active = False
                #self.approval_pending = True
                ###Test ends #####
                # machine's wfh status
                self.is_status_active = wfh_status_response['data']['isActive']
                #print(self.is_status_active+" approved")
                self.registered_public_ip = wfh_status_response['data']['publicIP']
                #employee's wfh status
                self.isApproved = wfh_status_response['data']['isApproved']
                self.approval_pending = not(self.isApproved)
                # #print(format(self.approval_pending) +": approval")
                Cache.append('cache', 'approval_pending', self.approval_pending)
                Cache.append('cache','reg_ip',self.registered_public_ip)
            if self.is_status_active:
                self.manager.current = "opencv_screen"
            else:
                #print("Not approved for wfh")
                ##print(self.ids.wfhscreen)
                self.wfh_text_1.text = "Machine unauthorized for remote work"
                self.wfh_loading_img.opacity=0
                self.wfh_loading_img.reload()
                #self.wfh_text_2.text = ""
                Clock.schedule_once(self.wfh_not_approved, 10)
        
    def wfh_not_approved(self, dt):
        ctypes.windll.user32.LockWorkStation()
        Faceable().stop()
            
            #lockscreen

class OpenCVScreen(Screen):

    def on_enter(self):
        self.opencv_status_text = self.ids["'opencv_status'"]
        self.image = self.ids["'open_cv_image'"]
        #print(format(self.image.texture))
        from_flow = StringProperty('')
        #print("OpenCV Screen")
        #print(format(self.ids))
        #image = self.ids["'open_cv_image'"]
        self.from_flow_cache = Cache.get('cache', 'from_flow')
        

        # notepad_handle = ctypes.windll.user32.FindWindowW(u"Faceable", None) 
        # ctypes.windll.user32.ShowWindow(notepad_handle, 6)  

        if (self.from_flow_cache == "continuous_recognition"):
            #print("Switching to Continuous recognition")
            Clock.schedule_once(self.recognize_employee, 0.1)
            self.opencv_status_text.text="Continuous recognition"
        else:
            if(len(os.listdir('Output/'))==0):
                Clock.schedule_once(self.generate_model, 0.1)
                self.opencv_status_text.text="Running Pre-recognition"
            else:
                Clock.schedule_once(self.recognize_employee, 0.1)
                #print("Calling here")

    def recognize_employee(self, dt):
        self.HOSTNAME = socket.gethostname()
        self.my_ip = Cache.get('cache','ip')
        self.recog_count = 0
        self.recognition_result = []
        self.no_detection = 0
        self.no_face_detection_pop = Popup(title='No faces detected',
                  content=Label(text='Please make sure only your face is cleary visible before the camera'), auto_dismiss=True,
                  size_hint=(None, None), size=(500, 500))
        self.many_face_detection_pop = Popup(title='Too many faces detected',
                  content=Label(text='Please make sure only your face is cleary visible before the camera'), auto_dismiss=True,
                  size_hint=(None, None), size=(500, 500))
        self.CURRENT_DIR = os.getcwd()
        self.OUTPUT_DIR = os.path.join(os.getcwd(), 'Output')
        self.EMBEDDINGS_PATH = os.path.join(self.OUTPUT_DIR, 'embeddings.pickle')
        self.RECOGNIZER_PATH = os.path.join(self.OUTPUT_DIR, 'recognizer.pickle')
        self.LABEL_ENCODER_PATH = os.path.join(self.OUTPUT_DIR, 'le.pickle')
        self.EMBEDDING_MODEL_PATH = os.path.join(self.CURRENT_DIR, 'openface_nn4.small2.v1.t7')
        self.LOG_FILE_PATH = os.path.join(self.CURRENT_DIR, 'log', 'app.log')
        self.GLOBAL_FACE_DETECTION_THRESHOLD = 0.7
        #self.GLOBAL_FACE_RECOGNITION_ACCURACY_THRESHOLD = 1.75
        self.GLOBAL_TRIGGER_DELAY = 3

        # load our serialized face detector from disk
        #print("[INFO] loading face detector...")
        self.protoPath = os.path.join(os.getcwd(), 'caffe', 'deploy.prototxt.txt')
        self.modelPath = os.path.join(os.getcwd(), 'caffe', 'res10_300x300_ssd_iter_140000.caffemodel')
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)

        # load our serialized face embedding model from disk
        #print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.EMBEDDING_MODEL_PATH)

        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(self.RECOGNIZER_PATH, "rb").read())
        self.le = pickle.loads(open(self.LABEL_ENCODER_PATH, "rb").read())

        VIDEO_SOURCE=0
        self.fps = FPS().start()
        self.counter_correct = 0  #counter variable to count number of times loop runs
        self.counter_wrong = 0
        self.no_of_faces= 0
        self.fps_counter = 0
        self.frame_num = 0
        self.frame_num_trigger = 0
        #print("Recognizing employee")
        self.total_faces = 0
        # initialize the video stream, then allow the camera sensor to warm up
        #print("[INFO] starting video stream...")
        # vs = VideoStream(src=VIDEO_SOURCE).start()
        #time.sleep(2.0)  # Time delay in seconds

        self.vs = VideoStream(src=VIDEO_SOURCE).start()
        time.sleep(1)
        if (self.from_flow_cache == "continuous_recognition"):
            #print("Moving to continuous recognition on update")
            time.sleep(2)
            Minimize = win32gui.FindWindow('SDL_app','Faceable')
            win32gui.ShowWindow(Minimize, win32con.SW_MINIMIZE)
            Clock.schedule_interval(self.continuous_recognition_on_update, 1)
        else:
            self.recognition_clock = Clock.schedule_interval(self.update_recognition, 1.0 / 33.0)
            self.opencv_status_text.text="Running Pre-recognition"

    def continuous_recognition_on_update(self, dt):
        self.image = self.ids["'open_cv_image'"]
        frame = self.vs.read()
        start_time = time.time()

        frame = imutils.resize(frame, width=1024)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        self.detections = self.detector.forward()


        # loop over the detections
        for i in range(0, self.detections.shape[2]):
            # extract the confidence (i.e., probability) associated with

            # the prediction
            confidence = self.detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.GLOBAL_FACE_DETECTION_THRESHOLD:
                self.no_of_faces += 1
                self.recog_count+=1
                # compute the (x, y)-coordinates of the bounding box for
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue



                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                ##print(j)
                name = self.le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                # if (self.no_of_faces == 1):
                    #print("In")
                    #cv2.imwrite("events/"+ self.HOSTNAME +"/user." + str(1) + '.' + str(self.recog_count) + ".jpg", frame)
                    #cv2.imwrite(os.path.join(os.getcwd()+'/events',"user."+str(1)+'.'+str(self.recog_count)+".jpg"),face)
                self.frame_num += 1
                self.no_detection=0
                #frame_num_temp += 1
                if (name == 'unknown'):
                    self.recognition_result.append(0)
                    #print("Stranger Detected")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #cv2.imwrite(os.path.join(os.getcwd()+'/events',"stranger."+str(1)+'.'+str(self.recog_count)+".jpg"),face)
                    #cv2.putText(frame, 'Stranger Frame Count: {0}'.format(self.counter_wrong), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #self.counter_wrong += 1
                    #self.counter_correct = 0
                else:
                    self.recognition_result.append(1)
                    #print("Host Detected")
                    #cv2.imwrite(os.path.join(os.getcwd()+'/events',"user."+str(1)+'.'+str(self.recog_count)+".jpg"),face)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #cv2.putText(frame, 'Host Frame Count: {0}'.format(self.counter_correct), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #self.counter_correct += 1
                    #self.counter_wrong = 0


        #self.recognition_result.clear()
        # update the FPS counter
        self.fps.update()
        #cv2.putText(frame, 'Face Count: {0}'.format(self.no_of_faces), (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # #print("Frame Number Temp {}".format(frame_num_trigger))
        if (self.no_of_faces > 1):
            #lockscreen
            self.total_faces = self.no_of_faces + self.total_faces
            #cv2.imwrite(os.getcwd()+"dataset/Stranger"+"/event.jpg", frame)
            ##print("Lock the screen")
        elif (self.no_of_faces == 0):
            self.no_detection += 1
            #self.recognition_result.append(2)
            pass
            #print("No of faces " + format(self.no_of_faces))
            #cv2.imwrite("dataset/Stranger/event.jpg", frame)

        #elif (self.no_of_faces == 1):
            #print("No of faces " + format(self.no_of_faces))
            #self.one_face_path = os.path.join(os.getcwd(), 'events', 'event.jpg')
            #cv2.imwrite(self.one_face_path,frame)
            #cv2.imwrite("dataset/Stranger"+"/event.jpg", frame)

        # Reset number of faces
        self.no_of_faces = 0

        end_time = time.time()
        self.fps_counter = self.fps_counter * 0.91 + 1/(end_time - start_time) * 0.1
        start_time = end_time
        frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(self.frame_num, self.fps_counter)

        #cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.image.texture = texture1
        #print("Frame count " + format(self.frame_num))
        #print("No detection count"+format(self.no_detection))

        # if self.frame_num % 30 == 0:
        #     recognition_count = Counter(self.recognition_result)
        #     recogniton_count = recognition_count[2]
        #     if (recognition_count == 30):
                # ctypes.windll.user32.LockWorkStation()
                # self.vs.stream.release()
                # cv2.destroyAllWindows()
                # self.recognition_result.clear()
        if(self.no_detection==30):
                ctypes.windll.user32.LockWorkStation()
                self.vs.stream.release()
                cv2.destroyAllWindows()
                self.recognition_result.clear()
                Faceable().stop()
        
        if self.frame_num % 10 == 0:
            #self.HOSTNAME = socket.gethostname()
            ##print(self.count)
            #if (self.total_faces == 0):
                # ctypes.windll.user32.LockWorkStation()
                # self.vs.stream.release()
                # cv2.destroyAllWindows()
                # self.recognition_result.clear()
                # #delete_output_folder('events/')
                # Faceable().stop()
            
            if (self.total_faces / 10 > 1):
                #print(len(recognition_result))
                #event_image_path = "events/"
                #print("Locking the screen - Too many faces in 10 consecutive frames")
                # logging_failure_event_error, logging_failure_event_body = logs.log_failure_event(self.my_ip,mac_address,self.HOSTNAME, "More than 1 face detected in 10 consecutive frames", event_image_path)
                # if (logging_failure_event_error is not None):
                #     #print("Failure in logging the failure event (more than one face)")
                # else:
                #     if(logging_failure_event_body["success"]):
                #         #print("Successfully logged failure event(more than one face)")
                #         ctypes.windll.user32.LockWorkStation()
                #         self.vs.stream.release()
                #         cv2.destroyAllWindows()
                #         self.recognition_result.clear()
                #         delete_output_folder('events/')
                #         Faceable().stop()
                ip__info_error, ip_info = ip.get_ip_info()
                if ip__info_error is not None:
                    print("IP info error"+str(ip__info_error))
                else:
                    #print(format(ip_info))
                    #print(ip_info["ip"])
                    logging_failure_event_error,logging_failure_event_body = logs.log_failure_event(ip_info["ip"], self.HOSTNAME , ip_info["city"], ip_info["region"], ip_info["country"], ip_info["org"], ip_info["loc"], "multi_unrecognized")
                    if (logging_failure_event_error is not None):
                        print("Failure in logging the failure event (more than one face)")
                    else:
                        if(logging_failure_event_body["success"]):
                            print("Successfully logged failure event(multi_unrecognized)")
                            ctypes.windll.user32.LockWorkStation()
                            self.vs.stream.release()
                            cv2.destroyAllWindows()
                            self.recognition_result.clear()
                            #delete_output_folder('events/')
                            Faceable().stop()
            if (len(self.recognition_result) == 0):
                #print("Recognition result is empty")
                # delete_output_folder('events/')
                # ctypes.windll.user32.LockWorkStation()
                # self.vs.stream.release()
                # cv2.destroyAllWindows()
                # self.recognition_result.clear()
                # #delete_output_folder('events/')
                # Faceable().stop()
                pass
            else:
                recognition_count = Counter(self.recognition_result)
                self.accuracy = recognition_count[1] / len(self.recognition_result)
                #print("Accuracy " + format(self.accuracy))
                if (self.accuracy > 0.50):
                    #print(format(self.recognition_result))
                    self.recognition_result.clear()
                    #delete_output_folder('events/')
                else:
                    #print("Accuracy failed for 10 consecutive frames - Locking Screen")
                    #event_image_path = "events/"
                    ip__info_error, ip_info = ip.get_ip_info()
                    if ip__info_error is not None:
                        print("IP info error"+str(ip__info_error))
                    else:
                        #print(format(ip_info))
                        #print(ip_info["ip"])
                        logging_failure_event_error,logging_failure_event_body = logs.log_failure_event(ip_info["ip"], self.HOSTNAME , ip_info["city"], ip_info["region"], ip_info["country"], ip_info["org"], ip_info["loc"], "single_unrecognized")
                        if (logging_failure_event_error is not None):
                            print("Failure in logging the failure event (single_unrecognized) "+self.accuracy)
                        else:
                            if (logging_failure_event_body["success"]):
                                #print("Successfully logged failure event(single_unrecognized)")
                                ctypes.windll.user32.LockWorkStation()
                                #delete_output_folder('events/')
                                self.vs.stream.release()
                                cv2.destroyAllWindows()
                                self.recognition_result.clear()
                                Faceable().stop()
                    # logging_failure_event_error, logging_failure_event_body = logs.log_failure_event(self.my_ip,mac_address, self.HOSTNAME,"Accuracy lower than threshold for 10 consecutive frames",event_image_path)
                    # if (logging_failure_event_error is not None):
                    #     #print("Failure in logging the failure event (accuracy < threshold)")
                    # else:
                    #     if(logging_failure_event_body["success"]):
                    #         #print("Successfully logged failure event(accuracy < threshold)")
                    #         ctypes.windll.user32.LockWorkStation()
                    #         delete_output_folder('events/')
                    #         self.vs.stream.release()
                    #         cv2.destroyAllWindows()
                    #         self.recognition_result.clear()
                    #         Faceable().stop()
            #delete_output_folder("events/")

    def update_recognition(self, dt):
        self.image = self.ids["'open_cv_image'"]
        frame = self.vs.read()
        start_time = time.time()
        self.frame_num += 1

        frame = imutils.resize(frame, width=1024)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        self.detections = self.detector.forward()

        # loop over the detections
        for i in range(0, self.detections.shape[2]):
            # extract the confidence (i.e., probability) associated with

            # the prediction
            confidence = self.detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > self.GLOBAL_FACE_DETECTION_THRESHOLD:
                self.no_of_faces += 1
                # compute the (x, y)-coordinates of the bounding box for
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue


                self.recog_count+=1
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                ##print(j)
                name = self.le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if (self.no_of_faces == 1):
                    self.HOSTNAME = socket.gethostname()
                    cv2.imwrite("dataset/"+ self.HOSTNAME +"/user." + str(1) + '.' + str(self.recog_count) + ".jpg", frame)
                # frame_num_temp += 1
                if (name == 'unknown'):
                    self.recognition_result.append(0)
                    #print("Stranger Detected")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 1)
                    #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    #cv2.putText(frame, 'Stranger Frame Count: {0}'.format(self.counter_wrong), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #self.counter_wrong += 1
                    #self.counter_correct = 0
                else:
                    self.recognition_result.append(1)
                    #print("Host Detected")
                    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 1)
                    #cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #cv2.putText(frame, 'Host Frame Count: {0}'.format(self.counter_correct), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    #self.counter_correct += 1
                    #self.counter_wrong = 0


        # update the FPS counter
        self.fps.update()
        #cv2.putText(frame, 'Face Count: {0}'.format(self.no_of_faces), (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # #print("Frame Number Temp {}".format(frame_num_trigger))
        if (self.no_of_faces > 1):
            self.many_face_detection_pop.open()
            self.no_face_detection_pop.dismiss()
            #self.frame_num_trigger += 1
            # #print("Number of faces : {0}".format(no_of_faces))
        elif (self.no_of_faces == 0):
            self.no_face_detection_pop.open()
            self.many_face_detection_pop.dismiss()
        elif (self.no_of_faces == 1):
            self.many_face_detection_pop.dismiss()
            self.no_face_detection_pop.dismiss()
            #self.frame_num_trigger = 0

        # Reset number of faces
        self.no_of_faces = 0

        end_time = time.time()
        self.fps_counter = self.fps_counter * 0.91 + 1/(end_time - start_time) * 0.1
        start_time = end_time
        frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(self.frame_num, self.fps_counter)

        # cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.image.texture = texture1
        #print("Frame count "+format(self.recog_count))
        if self.recog_count > 24:
            ##print(self.count)
            recognition_count = Counter(self.recognition_result)
            self.accuracy = recognition_count[1] / len(self.recognition_result)
            #print("Accuracy " + format(self.accuracy))
            self.recognition_clock.cancel()
            self.vs.stop()
            self.vs.stream.release()
            self.image.source = 'static/black.png'
            self.image.reload()
            self.send_to_azure_face_api("from_recognition", self.accuracy)
            cv2.destroyAllWindows()
            # ##print("from")
            # #Call Azure Face API here
            # #if recognized
            # #else
            # if self.accuracy < self.GLOBAL_FACE_RECOGNITION_ACCURACY_THRESHOLD:
            #     #print("unschedule")
            #     # #print(self.recognition_clock)
            #     # self.vs = VideoStream(0).stop()
            #     # self.recognition_clock.cancel()
            #     # self.image.source = "splash.jpg"
            #     #WindowManager().send_to_azure_face_api("from_pretrained_model")
            #     #Clock.schedule_once(self.train_model, 0.1)
            # else:
            #     #print("No training required")
            #Faceable().stop()

    def send_to_azure_face_api(self, status, accuracy):
        face_count=0
        Cache.append('cache', 'from_flow', status)
        Cache.append('cache','accuracy',accuracy)
        HOSTNAME = socket.gethostname()
        #print("Sending to Azure face API")
        self.filename = "dataset/"+HOSTNAME+"/user.1.12.jpg"
        no_faces_pop = Popup(title='No faces detected',
                  content=Label(text='Please make sure your face is cleary visible before the camera and well lit'), auto_dismiss=True,
                  size_hint=(None, None), size=(400, 400))
        more_than_one_face_pop = Popup(title='More than one faces detected',
                  content=Label(text='Make sure no one else is in the frame before capturing the photo'), auto_dismiss=True,
                  size_hint=(None, None), size=(400, 400))
        recognition_error, recognition_response = azure_face.recognize(self.filename)
        if (recognition_error is not None):
            print("recognition_error " +format(recognition_error))
        else:
            #print(format(recognition_response))
            if(recognition_response["success"]):
                #print("recognition successful :" + format(recognition_response))
                #face_count = 1
                # if face_count == 1:
                #     #print("One face detected")
                #     #print("Person recognized response "+format(recognition_response))
                is_recognized = recognition_response["data"]["isRecognized"]
                if is_recognized:
                    #print("Person is recognized")
                    #print("Person recognized response "+format(recognition_response))
                    emp_id = recognition_response["data"]["empId"]
                    #Cache.remove('cache', 'from_flow')
                    Cache.append('cache', 'recognized', True)
                    Cache.append('cache','emp_id',emp_id)
                    self.manager.current = "training_screen"
                    #mac_address = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
                    # approval_error, approval_response = wfh.check_status(HOSTNAME)
                    # if approval_error is not None:
                    #     #print("Approval error "+approval_error)
                    # else:
                    #     #print("Approval body " + format(approval_body))
                    #     is_approved = approval_response['data']['isApproved']
                    #     if (is_approved):
                    #         #print("approved")
                    #         self.manager.current="training_screen"
                    #     else:
                    #         #print("Need to request for approval")
                else:
                    #print("Person not recognized")
                    #print(self.manager.screens)
                    ##print(self.screen_names)
                    Cache.append('cache','recognized',False)
                    self.manager.current = "OTPWindow"
            else:
                face_count = recognition_response["data"]["noOfFaces"] 
                if face_count == 0:
                    #print("No faces detected")
                    self.manager.current = "OTPWindow"
                    #self.no_faces_pop.open()
                    #Clock.schedule_once(self.hide_pop,5)
                elif face_count > 1:
                    #print("More than one face detected")
                    self.manager.current = "OTPWindow"
                    #self.more_than_one_face_pop.open()
                    ##print("More than one faces detected")
                    #Clock.schedule_once(self.hide_pop, 5)


    def generate_model(self, dt):

        self.image = self.ids["'open_cv_image'"]
        self.fps = FPS().start()

        HOSTNAME = socket.gethostname()

        self.face_id = 1  # For each person,there will be one face id
        self.count = 0    # Initialize sample face image

        check_path("dataset/" + HOSTNAME + "/")

        # load our serialized model from disk
        #print("[INFO] loading model...")
        self.net = cv2.dnn.readNetFromCaffe(os.path.join(os.getcwd(), 'caffe', 'deploy.prototxt.txt'), os.path.join(os.getcwd(), 'caffe', 'res10_300x300_ssd_iter_140000.caffemodel'))
        self.face_detection_error_pop = Popup(title='Error in face detection',
                  content=Label(text='Please make sure only your face is cleary visible before the camera'), auto_dismiss=True,
                  size_hint=(None, None), size=(500, 500))
        # initialize the video stream and allow the cammera sensor to warmup
        #print("[INFO] starting video stream...")

        time.sleep(2.0)
        self.fps = 0
        self.frame_num = 0
        self.HOSTNAME = socket.gethostname()

        # self.img1 = Image()
        # layout = BoxLayout()
        # layout.add_widget(self.img1)
        #opencv2 stuffs
        self.vs = VideoStream(src=0).start()
        self.generate_model_clock = Clock.schedule_interval(self.update_generate_model, 1.0/33.0)

        #return layout
        #self.capture = cv2.VideoCapture(1)
    def update_generate_model(self, dt):
        # display image from cam in opencv window
        # time.sleep(1)
        #no_of_faces=0
        frame = self.vs.read()
        start_time = time.time()
        self.frame_num += 1

        frame = imutils.resize(frame, width=1024)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.net.setInput(blob)
        detections = self.net.forward()
        no_of_faces = 0
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with

            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.5:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                self.count += 1  # Increment face image
                no_of_faces+=1

                cv2.imwrite("dataset/"+ self.HOSTNAME +"/user." + str(self.face_id) + '.' + str(self.count) + ".jpg", frame)
                # draw the bounding box of the face along with the associated
                # probability
                text = "{:.2f}%".format(confidence * 100)
                # y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 1)
                # cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                end_time = time.time()
                self.fps = self.fps * 0.91 + 1/(end_time - start_time) * 0.1
                start_time = end_time
                frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(self.frame_num, self.fps)
                #cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.imshow("CV2 Image", frame)
        # convert it to texture
        buf1 = cv2.flip(frame, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        #if working on RASPBERRY PI, use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer.
        texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # display image from the texture
        self.image.texture = texture1
        

        #print(no_of_faces)

        if (no_of_faces == 0 or no_of_faces > 1):
            self.face_detection_error_pop.open()
        else:
            self.face_detection_error_pop.dismiss()

        if self.count > 24:
            #Clock.schedule_once(self.send_to_azure_face_api,0.1,"first_time")
            self.generate_model_clock.cancel()
            self.vs.stream.release()
            cv2.destroyAllWindows()
            self.image.source = 'static/black.png'
            self.image.reload()
            self.send_to_azure_face_api("first_time", 'NA')
            #embeddings.create_embeddings()
            #train_model.train()
            #Faceable().stop()


class OTPWindow(Screen):
    emp_id = ObjectProperty(None)
    otp_cont = ObjectProperty(None)
    isSent = BooleanProperty(False)

    def on_enter(self):
        #print("Entered OTP Window")
        self.emp_id_invalid=self.ids["'emp_id_invalid'"]
        self.otp_sent_status = self.ids["'otp_sent_status'"]
        self.otp_invalid = self.ids["'otp_invalid'"]

    def send_otp(self):
        self.send_otp_btn = self.ids["'send_otp'"]
        # send_otp_btn.background_color=(1,1,1,1)
        #print(self.emp_id.text)
        if(self.emp_id.text==""):
            self.otp_sent_status.text=""
            self.emp_id_invalid.text="The employee field is blank"
            #self.send_otp_btn.background_color=(.0,.3,.9,.0)
        else:
            otp_error, otp_body = otp.send_otp(self.emp_id.text)
            #print(format(otp_body))
            #print("otp_sent")
            self.isSent = True
            if (otp_error is not None):
                print("OTP API error")
                #self.send_otp_btn.background_color=(.0,.3,.9,.0)
            else:
                if(otp_body["success"]):
                    #print("OTP sent successfully")
                    #send_otp_btn.background_color=(.0,.3,.9,.0)
                    self.otp_sent_status.text="Sent an OTP to your email address"
                    self.emp_id_invalid.text=""
                    self.isSent=False
                else:
                    #print("Employee ID doesn't exist")
                    self.emp_id_invalid.text="This employee id does not exist"
                ##print("OTP body " + otp_body)
                #self.manager.current = self.manager.next()
                    #send_otp_btn.background_color=(.0,.3,.9,.0)        

    def verify_otp(self):
        #verify_otp_btn = self.ids["'verify_otp'"]
        # self.verify_otp_btn.background_color=(1,1,1,1)
        if(self.otp_cont.text ==""):
            self.otp_invalid.text = "Entered OTP is not valid"
        else:
            verification_error, verification_body = otp.verify_otp(self.otp_cont.text,self.emp_id.text)
            #print("verifying otp")
            #self.isSent = True
            if (verification_error is not None):
                print("OTP verification API error")
            else:
                if(verification_body["success"]):
                    #print("OTP verified successfully")
                    Cache.append('cache', 'emp_id', self.emp_id.text)
                    self.emp_id.text = ""
                    self.otp_cont.text = ""
                    self.otp_sent_status.text=""
                    self.manager.current = "training_screen"
                else:
                    #print("Entered OTP is not the same as sent one")
                    self.otp_invalid.text = "Entered OTP is not valid"
                    self.otp_cont.text = "" 
        #verify_otp_btn.background_color=(.0,.3,.9,0)       

class TrainingScreen(Screen):
    def on_enter(self):
        #print("Entering training screen")
        #self.train_pop = Popup(title='Training',content=Label(text='Training...'), auto_dismiss=True,size_hint=(None, None), size=(400, 400))
        self.from_flow = Cache.get('cache', 'from_flow')
        self.accuracy = Cache.get('cache', 'accuracy')
        self.recognized = Cache.get('cache','recognized')
        if (self.recognized):
            Clock.schedule_once(self.train_on_device_model,2)
        else:
            Clock.schedule_once(self.retrain,2)

    def retrain(self, dt):
        #print("Inside retrain")
        emp_id = Cache.get('cache','emp_id')
        HOSTNAME = socket.gethostname()
        self.filename = "dataset/"+HOSTNAME+"/user.1.12.jpg"
        retrain_error, retrain_body = azure_face.retrain(self.filename,emp_id)
        if (retrain_error is not None):
            print("Azure Face API Retrain error")
            #print(self.from_flow)
            #print("Test")
        else:
            if(self.from_flow=="from_recognition"):
                #print("Azure Face API re-trained")
                emp_id = '47466'
                Cache.append('cache','emp_id',emp_id)
                #print(self.accuracy)
                Clock.schedule_once(self.checking_for_local_training_neccessity,0.1)

            if (self.from_flow == "first_time"):
                #print("Face API retrained first time")
                Clock.schedule_once(self.train_the_model_first_time,0.1)

    def checking_for_local_training_neccessity(self,dt):
        if (self.accuracy < GLOBAL_THRESHOLD):
            #delete_output_folder("Output/")
            #Cache.remove('cache', 'from_flow')
            #Cache.append('cache','from_flow','from')
            #self.manager.current = "opencv_screen"
            embeddings.create_embeddings()
            train_model.train()
            #print("On device accuracy failed...Flow Complete")
            #call approval from manager
            Clock.schedule_once(self.check_for_approval_from_manager,0.1)
        else:
            #print("On device accuracy passed. No training required")
            #call approval from manager
            Clock.schedule_once(self.check_for_approval_from_manager,0.1)


    def train_on_device_model(self, dt):
        #print( Cache.get('cache', 'from_flow'))
        #print("Skipping face api retrain...recognized")
        #print(self.from_flow)
        #print(self.accuracy)
        if(self.from_flow=="from_recognition"):
            ##print("Azure Face API re-trained")
            #print(self.accuracy)
            Clock.schedule_once(self.checking_for_local_training_neccessity,0.1)
        if (self.from_flow == "first_time"):
            Clock.schedule_once(self.train_the_model_first_time,0.1)
            #call approval from manager


    def train_the_model_first_time (self,dt):
        #self.train_pop.open()
        embeddings.create_embeddings()
        train_model.train()
        #self.train_pop.dismiss()
        #print("First time model trained")
        Clock.schedule_once(self.check_for_approval_from_manager,0.1)

    def check_for_approval_from_manager(self, dt):
        #self.approval_pending_pop = Popup(title='Approval Pending',content=Label(text='Manager approval pending'), auto_dismiss=True,size_hint=(None, None), size=(500, 500))
        self.approval_from_manager_pending = Cache.get('cache', 'approval_pending')
        self.registered_ip = Cache.get('cache','reg_ip')
        self.emp_id = Cache.get('cache','emp_id')
        if(self.approval_from_manager_pending):
            #self.approval_pending_pop.open()
            #print("Here")
            self.manager.current = "Approval pending"
        else:
            my_ip = Cache.get('cache','ip')
            # if (self.registered_ip is not None and self.registered_ip is not my_ip ):
            #     #print("trigger mail")
            #     #print("Current IP " + my_ip)
            #     #print("Registered IP"+self.registered_ip)
            #     HOSTNAME = socket.gethostname()
            #     image = "dataset/" + HOSTNAME + "/user.1.55.jpg"
            #     #print("Image path :" + "dataset/" + HOSTNAME + "/user.1.55.jpg")
            #     #print("Employee ID : " + self.emp_id)
            #     mail_error, mail_body = mail.send_approval_mail(my_ip, self.emp_id, image)
            #     if mail_error is not None:
            #         #print("Error in sending the approval mail")
            #     else:
            #         #print("Approval mail sent")
            #         #Cache.remove('cache', 'ip')
            #         Cache.append('cache','ip_mismatch',True)
            if(self.registered_ip == my_ip):
                #print("Start Recognition")
                #self.manager.add_widget(ContinuousRecognitionScreen(name="continuous_recognition"))
                #self.manager.current = "continuous_recognition"
                Cache.remove('cache', 'from_flow')
                Cache.append('cache','from_flow','continuous_recognition')
                self.manager.current = "opencv_screen"
            else:
                #print("trigger mail")
                #print("Current IP " + my_ip)
                #print("Registered IP"+self.registered_ip)
                HOSTNAME = socket.gethostname()
                image = "dataset/" + HOSTNAME + "/user.1.12.jpg"
                #print("Image path :" + "dataset/" + HOSTNAME + "/user.1.27.jpg")
                #print("Employee ID : " + format(self.emp_id))
                mac_address = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
                mail_error, mail_body = mail.send_approval_mail(my_ip,mac_address,socket.gethostname(),image,self.emp_id)
                if mail_error is not None:
                    print("Error in sending the approval mail")
                else:
                    #print("Approval mail sent")
                    #Cache.remove('cache', 'ip')
                    Cache.append('cache','ip_mismatch',True)
                    self.manager.current = "Approval pending"

class ApprovalPendingScreen(Screen):
    def on_enter(self):
        self.ip_mismatch = Cache.get('cache', 'ip_mismatch')
        self.my_ip = Cache.get('cache','ip')
        #print("Entered approval pending screen")

        if (self.ip_mismatch):
            Clock.schedule_once(self.ip_mismatch_fun, 0.1)
        else:
            Clock.schedule_once(self.approval_pending, 0.1)

    def ip_mismatch_fun(self,dt):
        #print("There is an IP Mismatch")
        self.approval_pending_text_1 = self.ids["'approval_pending_text1'"]
        #approval_pending_text_2 = self.ids["'approval_pending_text2'"]
        self.approval_pending_text_1.text = "Current IP " + self.my_ip + " doesn't match organization records."
        #approval_pending_text_2.text = "Sending a mail to your manager for approval"
        Clock.schedule_once(self.approval_pending, 10)

    def approval_pending(self, dt):
        self.approval_pending_text_1 = self.ids["'approval_pending_text1'"]
        #approval_pending_text_2 = self.ids["'approval_pending_text2'"]
        self.start_over_btn = self.ids["'start_over_btn'"]
        self.approval_pending_text_1.text = "Sent email to manager for approval"
        #approval_pending_text_2.text = "You can start over, when its approved"
        self.start_over_btn.opacity=1

    def start_over(self):
        self.approval_pending_text_1.text = ""
        self.manager.current = "splash"
        self.start_over_btn.opacity = 0
        self.start_over_btn.background_color = (.0,.3,.9,0)

class NoCamera(Screen):
    def on_enter(self, *args):
        # Clock.schedule_once(self.no_camera_lock_kill_app,7)
        pass
    
    def no_camera_lock_kill_app(self, dt):
            ctypes.windll.user32.LockWorkStation()
            Faceable().stop()


 #The below class checks if the Windows is locked or not. This can be removed as the application now closes on locking of Windows

class LockScreenStatus(Screen):
    def on_enter(self):
        #print("Inside on_enter of LockScreenStatus")
        # Clock.schedule_once(self.set_status_screen,1)
        self.is_user_logged_in_clock = Clock.schedule_interval(self.is_user_logged_in, 10)
    
    def set_status_screen(self,dt):
        self.lock_screen_status = self.ids["'lock_screen_status'"]
        is_locked = Cache.get('cache', 'is_locked')
        if (is_locked):
            #print("Is system locked"+format(is_locked))
            self.lock_screen_status.text= 'Windows is locked'
        else:
            #print("Is system locked"+format(is_locked))
            self.lock_screen_status.text= 'Windows is unlocked.Moving to the next step'
            self.is_user_logged_in_clock.cancel()
            Clock.schedule_once(self.initial_req_check, 1)
    
    def is_user_logged_in(self, dt):
        process_name='LogonUI.exe'
        callall='TASKLIST'
        outputall=subprocess.check_output(callall)
        outputstringall=str(outputall)
        if process_name in outputstringall:
            Cache.append('cache','is_locked', True)
            #print("Locked.")
            Clock.schedule_once(self.set_status_screen,1)
        else: 
            Cache.append('cache','is_locked', False)
            #print("Unlocked.")
            self.manager.current='lock_screen_status'
            Clock.schedule_once(self.set_status_screen,1)

    def initial_req_check(self, dt):
        is_there_internet = checkInternetSocket()
        cam_present = is_cam_present(0)
        if(cam_present):
            #print("Camera present")
            if is_there_internet:
                #print("Connected to the internet" )
                self.manager.current = "wfh"
            else:
                #print("No internet")
                self.manager.current= "No internet"
        else:
            #print("No camera detected")
            self.manager.current="no_camera"
        ##print(self.manager.current)
        ##print(self.manager.next())
        #self.manager.current = 'wfh'

    
        

class WindowManager(ScreenManager):
    pass

class Faceable(App):
    visible = False
    def build(self):
        HOSTNAME = socket.gethostname()
        check_path("dataset/" + HOSTNAME + "/")
        #self.capture = cv2.VideoCapture(1)
        kv = Builder.load_file("screens.kv")
        sm = WindowManager()
        sm.add_widget(SplashScreen(name="splash"))
        ##print("Checking for Internet....")
        sm.add_widget(WFHScreen(name="wfh"))
        sm.add_widget(ApprovalPendingScreen(name="Approval pending"))
        sm.add_widget(OpenCVScreen(name="opencv_screen"))
        sm.add_widget(OTPWindow(name="Send OTP"))
        sm.add_widget(TrainingScreen(name="training_screen"))
        sm.add_widget(InternetCheck(name="No internet"))
        sm.add_widget(NoCamera(name="no_camera"))
        sm.add_widget(LockScreenStatus(name="lock_screen_status"))
        return sm
    
    def on_start(self):
        self.root.focus = True    

    # def alternate(self):
    #     if self.visible:
    #         self.root.get_root_window().hide()
    #     else:
    #         self.root.get_root_window().show()

    #     self.visible = not self.visible

        
def checkInternetSocket(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        #print(ex)
        return False

def is_cam_present(source):
    #print("Checking for camera")
    cam = cv2.VideoCapture(source)
    if cam is None or not cam.isOpened():
       #print('Warning: unable to open video source: ', source)
       return False
    else:
        cam.release()
        return True

if __name__ == '__main__':
    Faceable().run()
    # ma = Faceable()       
    # kivy_app = Thread(target=ma.run)
    # kivy_app.start()
    # my_menu = menu(item(text="Show Main Window", action=ma.alternate, default = True, visible=False))
    # image = Image.open("static/faceable_logo.png")
    # icon = icon('EasyBright', image, menu=my_menu)
    # icon.run()


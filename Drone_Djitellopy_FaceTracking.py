import os # (os.path.splitext(os.path.basename(fileName)))[0] #分離出路徑名的主/副檔名
inpath = os.path.join("..", "Input")
outpath = os.path.join("..", "Output")
imgVidPath = os.path.join("..", "ImageVideo")
codeFileName = os.path.splitext(os.path.basename(__file__))[0] #[1] is extension
########## Program Code ##########
##### Import #####
import time
import numpy as np
from djitellopy import tello
from time import sleep
from threading import Thread

from datetime import datetime
import cv2
##### Face detection
def findFace(img):
    #-- load the cascades of pretrained model for face
    
    face_cascade_name = "haarcascade_frontalface_default.xml" 
    faceCascade = cv2.CascadeClassifier()    
    if not faceCascade.load(cv2.samples.findFile(face_cascade_name)):
        print('----- Error loading face cascade!!')
        return img, [[0,0], 0]
    
    #print("face", face_cascade_name)
    
    #-- find facee
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # color to gray
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 8) # 1.2 scale factor, 8 nighbors
   
    #-- detect faces
    faceListC = [] # record center [x,y]
    faceListA = [] # record face area
    for (fX, fY, fW, fH) in faces:
        cx = fX + fW // 2
        cy = fY + fH // 2
        faceArea = fW * fH

        cv2.rectangle(img, (fX, fY), (fX+fW, fY+fH), (0, 255, 0), 2) # face area
        cv2.circle(img, (cx,cy), 5, (255, 0,0), cv2.FILLED) # center of face,  5 radius
                
        faceListC.append([cx, cy])
        faceListA.append(faceArea)
        
    #-- find max face area to avoid detection error when have multiple faces in the frame
    if len(faceListA) > 0: # more than one face detected
        ii = faceListA.index(max(faceListA))
        return img, [faceListC[ii], faceListA[ii]]
    else:
        return img, [[0, 0], 0] # center [0,0], area 0

##### Trace face: update drone according to pid 
""" info: center and area
    fbRange: tracking range of drone
    pid: values to compute rotate 
    ww: width of face/object
"""
def trackFace(drone, info, fbRange, pid, ww, pError): # w denote width of image/face/object
    #-- initial info
    area = info[1]
    x, y = info[0]
    fb = 0
    
    #-- compute drone moving info
    error = x - ww//2 # how far away face is from the center, erro is the deviation
    speed = pid[0] * error + pid[1] * (error-pError)    
    speed = int(np.clip(speed, -100, 100)) # speed between -100~100 
    
    #-- adjust drone according position of face in the drone frame    
    if fbRange[0] < area and area < fbRange[1]: # face in detecting area, drone stop
       fb = 0    
    elif area > fbRange[1]: # face too close/big, drone backward
        fb = -20
    elif area < fbRange[0] and area != 0: # face too far/small, drone forward
        fb = 20
    if x == 0:
        speed = 0
        error = 0
    #print("fb {}, speed {}".format(fb, speed))    

    drone.send_rc_control(0, fb, 0, speed) # consider forward/backward only
    return error

if (__name__ == "__main__"):
    width, height = 360,240
    fbRange = [6200,6800]
    pid = [0.4,0.4,0]
    pError = 0
    
    try:
        drone = tello.Tello()
        drone.connect()
        print("Battery --> {}%".format(drone.get_battery()))
        
        drone.streamon()
        droneFrame = drone.get_frame_read()
        
        drone.takeoff()
        drone.send_rc_control(0,0,10,0)
        sleep(1)
        
        while True:
            img = drone.get_frame_read().frame
            
            img = cv2.resize(img, (width, height))
            img, info = findFace(img)
            pError = trackFace(drone, info, fbRange, pid, width, pError)
            cv2.imshow("Tracking face", img)
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
        #cv2.destroyAllWindows()
        
    except Exception as exMessage:
       print("Tracking error -->", exMessage)
 
    finally:
       cv2.destroyAllWindows()
       drone.streamoff()
       drone.land()
       drone.reboot()
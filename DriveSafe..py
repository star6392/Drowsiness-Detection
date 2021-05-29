import cv2
import dlib
from scipy.spatial import distance
import time
from playsound import playsound
from threading import Thread

#Function which returns the Eye Aspect Ratio value
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

#The dlib trained model
cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
timer=0
flag=0
start_time=time.time()

def play_sound():
    playsound('alert.mp3')

while True:
    _, frame = cap.read()
    #Converting image to grey for efficiency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        
        #Landmark location for left eye
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        #Landmark location for right eye
        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)
        
        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        #Taking average EAR value of both eyes
        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        cv2.putText(frame,f"EAR={str(EAR)}",(450,390),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        
        #Detection of closed eyes
        if EAR<0.23:
            end_time=time.time()
            timer=(start_time-end_time)*-1
            cv2.putText(frame,"Closed Eyes Detected",(20,460),
                cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
            cv2.putText(frame,f"{str(timer)[0:3]}s",(20,420),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(50,50,255),2)
            
            #Alerting the user
            if timer>3:
                thread = Thread(target=play_sound)
                thread.start()
                cv2.putText(frame,"ALERT",(200,100),
                cv2.FONT_HERSHEY_SIMPLEX,3,(50,50,255),7)
                
                print("Drowsy")
        else:
            start_time=time.time()
        print(EAR)

        cv2.imshow("DriveSafe", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
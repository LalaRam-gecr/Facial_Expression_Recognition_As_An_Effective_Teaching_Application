#Importing Libraries
import tkinter as tk
import deepface as DeepFace
import cv2
from time import sleep
import matplotlib.pyplot as plt

window = tk.Tk()
window.title("Face Expression Recognizer")
window.configure(background="white")
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.label(window, text="Real-Time Face Expression Recognition", bg="green", fg="white", width=50,height=3, font=('times', 30, 'bold'))

message.place(x=200, y=20)
second = 60
dict = {}

def TakeImages():
    cascade = 'haarcascade_frontalface_default.xml'  # Loading HaarCascade Model
    faceCascade = cv2.CascadeClassifier(cascade)

    cap = cv2.VideoCapture(0)

#Check if webcam is not open correctly
    if not cap.isOpened():
         cap = cv2.VideoCapture(1)
         if not cap.isOpened():
             raise IOError("Can't open webcam")
         while True:
             ret, frame = cap.read()
             result = DeepFace.analyze(frame, actions = ['emotion'])
             print(result['dominant_emotion'])
             temp = result['dominant_emotion']
             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             print(faceCascade.empty())
             faces = faceCascade.detectMultiScale(gray, 1.1, 4)

             if(temp not in dict.key()):
                 dict[temp] = 1
             else:
                 dict[temp] = dict[temp] + 1

                 a = dict.keys()
                 b = dict.values()

                 plt.bar(a, b)
                 plt.show()

                 #Draw a rectangle around the face
                 for(x, y, w, h) in faces:
                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                     font = cv2.FONT_HERSHEY_SIMPLEX

                 #Use putText( method for inserting text on video
                 cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 3, (0, 0, 255), 2, cv2.LINE_4
                             )
                 cv2.imshow('Original video', frame)
                 sleep(second)
                 if cv2.waitKey(2) & 0xFF == ord('q'):
                     break

                 cap.release()
                 cv2.destryAllWindows()

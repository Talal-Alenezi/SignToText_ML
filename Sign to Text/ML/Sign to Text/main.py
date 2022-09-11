# Importing dependencies
from unittest import result
import cv2 # webcam stuff
import numpy as np # arrays and things
import os # arranges files ig
from matplotlib import pyplot as plt # graphics and so
import time 
import mediapipe as mp
from pyparsing import results # the cool stuff


# media pipe stuff
mp_holistic = mp.solutions.holistic # holistic model
mp_drawing = mp.solutions.drawing_utils # drawing utilities

#  
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make Prediction 
    image.flags.writeable = True                    # Image is writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion
    return image, results

# Drawing Function
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # USE FACEMESH_TESSELATION if u want connections between all the land marks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
# Drawing Function but stylish
def draw_styled_landmarks(image, results):
    # FACE
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # POSE
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # LEFT HAND
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # RIGHT HAND
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# Accessing a webcam
cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read feed from webcam [reads each frame alone but look like a video]
        ret, frame = cap.read() 
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # Draaw Landmarks
        draw_styled_landmarks(image, results)
        # show to screen
        cv2.imshow('OpenCV Feed', image)
        # Breaking
        if cv2.waitKey(10) & 0xFF == ord('q'): # press q to quit 
            break
    cap.release()
    cv2.destroyAllWindows()

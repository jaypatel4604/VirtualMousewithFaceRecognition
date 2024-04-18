import cv2
import face_recognition
from cvzone.HandTrackingModule import HandDetector
import mouse
import threading
import numpy as np
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import os

# Intialization of camera and its properties
frameR = 100
cam_w, cam_h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)
detector = HandDetector(detectionCon=0.9, maxHands=1)
pTime = 0


# Variables for volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Initialization of Delay Variables
l_delay = 0
r_delay = 0
double_delay = 0


# Function for Adjusting Volume
def adjust_volume(direction):
    current_volume = volume.GetMasterVolumeLevelScalar()
    if direction == 'up':
        new_volume = min(1.0, current_volume + 0.1)  # Increase volume by 0.1 (10%)
    elif direction == 'down':
        new_volume = max(0.0, current_volume - 0.1)  # Decrease volume by 0.1 (10%)
    volume.SetMasterVolumeLevelScalar(new_volume, None)

# Functions for Delays
def l_clk_delay():
    global l_delay
    global l_clk_thread
    time.sleep(1)
    l_delay = 0
    l_clk_thread = threading.Thread(target=l_clk_delay)


def r_clk_delay():
    global r_delay
    global r_clk_thread
    time.sleep(1)
    r_delay = 0
    r_clk_thread = threading.Thread(target=r_clk_delay)


def double_clk_delay():
    global double_delay
    global double_clk_thread
    time.sleep(2)
    double_delay = 0
    double_clk_thread = threading.Thread(target=double_clk_delay)


l_clk_thread = threading.Thread(target=l_clk_delay)
r_clk_thread = threading.Thread(target=r_clk_delay)
double_clk_thread = threading.Thread(target=double_clk_delay)


# Intialization of Variables to Store Known Face Encodings and Locations
known_face_encodings = []
known_face_names = []

# Loop for Gathering Encoding and Locations of Known Images in Image Directory
KNOWN_FACES_DIR =  "images"
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Loop for Face Recognition and Hand Gesture based Mouse Control
while True:
    # capture frame by frame
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # Loop through each faces found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # check if any matches are there
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If Face Found in Current frame with any face in the images directory then Hand Gestures will Run
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            hands, img = detector.findHands(img, flipType=False)
            cv2.rectangle(img, (frameR, frameR), (cam_w - frameR, cam_h - frameR), (255, 0, 255), 2)
            if hands:
                lmlist = hands[0]['lmList']
                ind_x, ind_y = lmlist[8][0], lmlist[8][1]
                mid_x, mid_y = lmlist[12][0], lmlist[12][1]
                pin_x, pin_y = lmlist[20][0], lmlist[20][1]
                rin_x, rin_y = lmlist[16][0], lmlist[16][1]
                cv2.circle(img, (ind_x, ind_y), 5, (0, 255, 255), 2)  # index finger
                cv2.circle(img, (mid_x, mid_y), 5, (0, 255, 255), 2)  # middle finger
                cv2.circle(img, (pin_x, pin_y), 5, (0, 255, 255), 2)  # pinky finger
                cv2.circle(img, (rin_x, rin_y), 5, (0, 255, 255), 2)  # ring finger
                fingers = detector.fingersUp(hands[0])

                # Mouse movement
                if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 1:
                    conv_x = int(np.interp(ind_x, (frameR, cam_w - frameR), (0, 1536)))
                    conv_y = int(np.interp(ind_y, (frameR, cam_h - frameR), (0, 864)))
                    mouse.move(conv_x, conv_y)
                    print(conv_x, conv_y)

                # Mouse Button Clicks
                if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 1:
                    if abs(ind_x - mid_x) < 25:
                        # Left Click
                        if fingers[4] == 0 and l_delay == 0:
                            mouse.click(button="left")
                            l_delay = 1
                            l_clk_thread.start()
                            print("Left Click")
                        # Right Click
                        if fingers[4] == 1 and r_delay == 0:
                            mouse.click(button="right")
                            r_delay = 1
                            r_clk_thread.start()
                            print("Right Click")
                # Mouse Scrolling
                if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[4] == 0:
                    if abs(ind_x - mid_x) < 25:
                        mouse.wheel(delta=-1)
                        print("Scroll Down")
                if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[4] == 1:
                    if abs(ind_x - mid_x) < 25:
                        mouse.wheel(delta=1)
                        print("Scroll Up")

                # Double Mouse Click
                if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0 and fingers[4] == 0 and double_delay == 0:
                    double_delay = 1
                    mouse.double_click(button="left")
                    double_clk_thread.start()
                    print("Double Click")

                # Volume Up
                if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0 and fingers[4] == 1 and fingers[3] == 0:
                    if abs(ind_x - pin_x) < 40:
                        if ind_y < mid_y:
                            adjust_volume('up')
                            print("Volume Up")
                # Volume Down
                if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0:
                    if abs(ind_x - rin_x) < 25:
                        if ind_y < mid_y:
                            adjust_volume('down')
                            print("Volume Down")

        # draw a box around the face and label with the name
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display the resulting frame and FPS of Current Frame

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)
    cv2.imshow("Video", img)


    cv2.waitKey(1)

# release the webcam and close Opencv windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import mediapipe as mp
from pywinauto import keyboard
import time
import pydirectinput

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def Tpose():
    print("Tpose")
    pydirectinput.keyDown('w')
    time.sleep(0.5)
    pydirectinput.keyUp('w')
    pass

def botharmup():

    pass

def leftarmup():
    print("Left arm is up")
    pydirectinput.keyDown('a')
    time.sleep(0.5)
    pydirectinput.keyUp('a')
    pass

def rightarmup():
    print("Right arm is up")
    pydirectinput.keyDown('d')
    time.sleep(0.5)
    pydirectinput.keyUp('d')
    pass


while True:
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark

        Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        Lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        Relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

        angle1 = calculate_angle(Relbow, Rshoulder, Rhip)
        angle2 = calculate_angle(Lelbow, Lshoulder, Lhip)
        # print(int(angle1)) # Right
        # print(int(angle2)) # Left

        if (angle1 and angle2 > 70) and (angle1 and angle2 < 130):
            Tpose()

        if angle1 > 130 and angle2 < 70:
            rightarmup()

        if angle2 > 130 and angle1 < 70:
            leftarmup()

        if angle1 > 130 and angle2 > 130:
            botharmup()

        # Visualize angle
        cv2.putText(image, str(int(angle1)),
                    tuple(np.multiply(Rshoulder, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

        cv2.putText(image, str(int(angle2)),
                    tuple(np.multiply(Lshoulder, [640, 480]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

    except:
        pass

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # print(results.pose_landmarks)
    cv2.imshow("Mediapipe Feed", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



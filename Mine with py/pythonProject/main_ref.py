import cv2
import mediapipe as mp
import pydirectinput as pdi
import numpy as np
import subprocess

# For webcam input:
squat = False
jump = False
spacehold = False
click = False
click_hold = False
right_click = False
walk = False
right = False
left = False
mouseup = False
mousedown = False

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                  min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})')

        annotated_image = image.copy()
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        annotated_image = np.where(condition, annotated_image, bg_image)

        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        if results.pose_landmarks:
            image_height, image_width, _ = image.shape
            Lshoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Rshoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            Lelbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            Relbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            Lhip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Rhip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            shoulder_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y +
                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y) * image_height / 2
            elbow_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y +
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y) * image_height / 2
            wrist_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y +
                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y) * image_height / 2
            hip_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y +
                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y) * image_height / 2
            knee_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y +
                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y) * image_height / 2

            normalfactor = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x -
                               results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x)
            hand_distance = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x -
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x)

            # If you want to get the nose landmarks and make that as a mouse tracker
            nose_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x * image_width
            nose_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y * image_height

            if knee_y < (hip_y + 50) and squat is False:
                squat = True
                pdi.keyDown('ctrl')
                click_hold = True

            elif knee_y > hip_y and squat is True:
                squat = False
                pdi.keyUp('ctrl')
                click_hold = False

            angle1 = calculate_angle(Relbow, Rshoulder, Rhip)
            angle2 = calculate_angle(Lelbow, Lshoulder, Lhip)

            if shoulder_y > hip_y:
                exit()

            # Raise your hands to jump
            if elbow_y > wrist_y and jump is False and results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_ELBOW].y and results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_WRIST].y < results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_ELBOW].y and hand_distance > 0.1:
                jump = True
                pdi.press('space')

            elif elbow_y < shoulder_y and jump is True:
                jump = False

            # make a prayer pose and move it up or down
            if hand_distance < 0.06 and elbow_y > wrist_y and mouseup is False:
                mouseup = True
                moveup = subprocess.Popen(
                    ["python", "C:\\Users\\newpassword\\Desktop\\Mine with py\\pythonProject\\mouse_up.py"], stdin=None,
                    stdout=None, stderr=None, close_fds=True)
            elif hand_distance < 0.06 and elbow_y < wrist_y and mousedown is False:
                mousedown = True
                movedown = subprocess.Popen(
                    ["python", "C:\\Users\\newpassword\\Desktop\\Mine with py\\pythonProject\\mouse_down.py"],
                    stdin=None, stdout=None, stderr=None, close_fds=True)
            elif hand_distance > 0.06 and (mouseup is True or mousedown is True):
                mouseup = False
                mousedown = False
                try:
                    pollup = moveup.poll()
                    if pollup is None:
                        moveup.terminate()
                    polldown = movedown.poll()
                    if polldown is None:
                        movedown.terminate()
                except:
                    pass
            '''
            if hand_distance < 0.06 and elbow_y < wrist_y and mousedown == False:
              mousedown = True
              movedown = subprocess.Popen(["python", "mousedown.py"], stdin=None, stdout=None, stderr=None, close_fds=True)  
            elif hand_distance > 0.06 and mousedown == True:
              mouseup = False
              mousedown = False
              try:
                poll = moveup.poll()
                if poll is None:
                  movedown.terminate()
              except:
                pass
            '''

            if angle2 > 130 and angle1 < 70:
                click = True

            elif angle2 > 130 and angle1 < 70 and click is True:
                click = False
                if click_hold == False:
                    # pdi.leftClick()
                    pdi.mouseDown()
                    click_hold = True
                else:
                    pdi.mouseUp()
                    click_hold = False

            # Left arm up
            if angle1 > 130 and angle2 < 70:
                right_click = True

            # Right arm up
            elif angle1 > 130 and angle2 < 70:
                right_click = False
                pdi.rightClick()

            '''
            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y < results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y:
              click = True

            elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y > results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y and click == True:
              click = False
              pdi.click(button=RIGHT)
            '''

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y < (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y) / 2 and walk is False:
                walk = True
                pdi.keyDown('w')
            elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y > (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y) / 2 and walk is True:
                walk = False
                pdi.keyUp('w')

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y < (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2 and spacehold is False:
                spacehold = True
                pdi.keyDown('w')
                pdi.keyDown('shift')
                pdi.keyDown('space')
            elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y > (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y +
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2 and spacehold == True:
                spacehold = False
                pdi.keyUp('w')
                pdi.keyUp('space')
                pdi.keyUp('shift')

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y < (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.RIGHT_SHOULDER].y - normalfactor) and right is False:
                right = True
                moveright = subprocess.Popen(
                    ["python", "C:\\Users\\newpassword\\Desktop\\Mine with py\\pythonProject\\mouse_right.py"],
                    stdin=None, stdout=None, stderr=None, close_fds=True)
            elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > \
                    (results.pose_landmarks.landmark[
                         mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and right is True:
                right = False
                try:
                    poll = moveright.poll()
                    if poll is None:
                        moveright.terminate()
                except:
                    pass

            if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y < (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and \
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y > (
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER].y - normalfactor) and left is False:
                left = True
                moveleft = subprocess.Popen(
                    ["python", "C:\\Users\\newpassword\\Desktop\\Mine with py\\pythonProject\\mouse_left.py"],
                    stdin=None, stdout=None, stderr=None,
                    close_fds=True)
            elif results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y > (results.pose_landmarks.landmark[
                                                                                           mp_pose.PoseLandmark.RIGHT_SHOULDER].y + normalfactor) and left == True:
                left = False
                try:
                    poll = moveleft.poll()
                    if poll is None:
                        moveleft.terminate()
                except:
                    pass

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()


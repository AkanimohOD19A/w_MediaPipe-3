## DEPENDENCIES
import mediapipe as mp
import time
import cv2
import os

## INFRASTRUCTURE
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

input_img = './content/sample.jpg'

## RUN
### STATIC Images
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
) as pose:
    image = cv2.imread(input_img)
    image_height, image_width, _ = image.shape
    ## Convert BGR -> RGB
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ## Draw pose
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imwrite(r'results/sample.png', annotated_image)

### WEBCAM Input:
# cap = cv2.VideoCapture("sample.mp4")
vid_cap = cv2.VideoCapture(0)
#=> Video Input
prevTime = 0 ## Initiate for timer/FramePerSecond(fps)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while vid_cap.isOpened():
        success, image = vid_cap.read()
        if not success:
            print("Ignoring empty camera frame")
            continue

        # Convert BGR => RGB
        image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ### --
        image.flags.writeable = False
        results = pose.process(image)
        ### --/
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        currTime = time.time()
        fps = 1/(currTime - prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_ITALIC, 3, (0, 196, 255), 2)
        cv2.imshow('BlazePose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
vid_cap.release()

## - RealTime
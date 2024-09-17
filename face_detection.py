import cv2
import mediapipe as mp

cap = cv2.VideoCapture("two_face.mp4")


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(0.20)

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = faceDetection.process(imgRGB)
    
    # print(results.detections)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            # print(bboxC)
            h, w, _ = img.shape
            bbox = int(bboxC.xmin*w), int(bboxC.ymin*h), int(bboxC.width*w), int(bboxC.height*h)
            cv2.rectangle(img, bbox, (0,255,255), 8)
            

    img = cv2.resize(img, (0,0), fx = 0.25, fy = 0.25)

    cv2.imshow("img", img)
    cv2.waitKey(1)



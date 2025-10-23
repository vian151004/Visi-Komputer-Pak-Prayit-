import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

detector = PoseDetector()

while True:
    # membaca video dri webcam
    success, img = cap.read()

    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=True, bboxWithHands=False)
    if lmList:
        center = bboxInfo["center"]

        cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

        length, img, info = detector.findDistance(lmList[11][0:2],
            lmList[15][0:2],
            img=img,
            color=(255, 0, 0),
            scale=10
        )

        angle, img = detector.findAngle(lmList[11][0:2], lmList[13][0:2],
            lmList[15][0:2],
            img=img,
            color=(0, 0, 255),
            scale=10
        )

        isCloseAngle50 = detector.angleCheck(myAngle=angle,
            targetAngle=50,
            offset=10
        )
        print(isCloseAngle50)

    cv2.imshow("Pose + Angle ", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
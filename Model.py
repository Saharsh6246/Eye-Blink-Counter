import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)


blinkCounter = 0
color = (255, 0, 255)
eye_closed_state = False  

plotY = LivePlot(640, 480, [20, 45], invert=True)

LeftidList = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RightidList = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
ratioList = []

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]

        for id in LeftidList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)
        for id in RightidList:
            cv2.circle(img, face[id], 5, color, cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lengthVert, _ = detector.findDistance(leftUp, leftDown)
        lengthHorz, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

        ratio = int((lengthVert / lengthHorz) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 30 and not eye_closed_state:
            blinkCounter += 1
            color = (0, 200, 0)
            eye_closed_state = True  
        
        if ratioAvg >= 30 and eye_closed_state:
            color = (255, 0, 255)
            eye_closed_state = False
            
        imgPlot = plotY.update(ratioAvg)
        cv2.imshow("Plot", imgPlot)

    img = cv2.flip(img, 1)
    cvzone.putTextRect(img, f'Blink count : {blinkCounter}', (50, 100), colorR=color)
    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
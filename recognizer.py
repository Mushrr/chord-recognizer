#手指位置的标点
import cv2
import mediapipe as mp
import time
from chord_reconize import ChordRecognizer

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

# chrod_name = "D"
# f = open(f'./data/{chrod_name}.csv', 'w')

chord_recognizer = ChordRecognizer('./data', ["Am", "C", "E", "F", "G", "D"], sal=0.01, max_depth=18, estimators=150)
chord_recognizer.train()

# cv text config
org = (50, 50)
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (0, 0, 255)
thickness = 2
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 2 = to
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)//检查手坐标输出
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            indexs = []
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                indexs.extend([cx, cy])
                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # f.write(' '.join(indexs) + '\n')
            try:
                pred = chord_recognizer.predict(indexs)

                cv2.putText(img, pred, org, fontFace, fontScale, color, thickness)
            except:
                # cv2.putText(img, "Not Found", org, fontFace, fontScale, color, thickness)
                pass
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 255, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

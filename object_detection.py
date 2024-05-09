import numpy as np 
import cv2
import pygame

# รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
    "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
    "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
    "SOFA", "TRAIN", "TVMONITOR"]

# สีตัวกรอบที่วาด random ใหม่ทุกครั้ง
COLORS = np.random.uniform(0, 100, size=(len(CLASSES), 3))

# โหลด model จากแฟ้ม
net = cv2.dnn.readNetFromCaffe("MobileNetSSD.prototxt","MobileNetSSD.caffemodel")


sound_file = "sound_detected.wav"


pygame.mixer.init()
sound = pygame.mixer.Sound(sound_file)


def play_bottle_sound(percent):
    if percent > 0.7:
        sound.play()


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if ret:
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]

            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")


                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                y = startY - 15 if startY-15 > 15 else startY + 15
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)


                if CLASSES[class_index] == "BOTTLE":
                    play_bottle_sound(percent)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()

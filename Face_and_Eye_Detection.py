import cv2 as cv
haarface = cv.CascadeClassifier('haar_face.xml')
haareye = cv.CascadeClassifier('haar_eye.xml')
webcam = cv.VideoCapture(0)
while True:
    # capture the image by the webcam
    _, img = webcam.read()
    
    # detecting face and eye
    face = haarface.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 7)
    eye = haareye.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 8)

    # drawing rectangle around face and eye
    for (x,y,w,z) in face:
        cv.rectangle(img, (x,y), (x+w,y+z), (0,255,0), thickness = 2)
        for (a,s,d,f) in eye:
            cv.rectangle(img, (a,s), (a+d,s+f), (0,255,0), thickness = 2)
    cv.imshow('Face and Eye Detection',img)
    
    if cv.waitKey(1) == ord(' '):
        break

webcam.release()
cv.destroyAllWindows()

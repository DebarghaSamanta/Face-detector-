import cv2
face_cascade = cv2.CascadeClassifier(r"haarcascades/haarcascade_frontalface_alt.xml")
eyes_cascade = cv2.CascadeClassifier(r"haarcascades/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(r"haarcascades/haarcascade_smile.xml")
if face_cascade.empty():
    print("Error loading cascade!")
    exit()

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey,1.1,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
    
        roi_grey = grey[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        
        eye = eyes_cascade.detectMultiScale(roi_grey,1.1,10)
        if len(eye) >0:
            cv2.putText(frame,"Eyes Detected",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.6,(0, 255, 255),2)
        cv2.imshow("webcam feed",frame)

        smile = smile_cascade.detectMultiScale(roi_grey,1.7,25)
        if len(smile) >0:
            cv2.putText(frame,"Smile Detected",(x,y-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(0, 255, 255),2)
        cv2.imshow("webcam feed",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break
cap.release()
cv2.destroyAllWindows()
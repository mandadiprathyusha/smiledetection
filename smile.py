import cv2

#loading the algo
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector =cv2.CascadeClassifier("haarcascade_smile.xml")

#on the webcame
webcam=cv2.VideoCapture(0)

#infinity loop
while True:
    #reads each and every frame
    successful_frame_read,frame=webcam.read()

    #if the frame is not being successfully read then break
    if not successful_frame_read:
        break
    
    #covert the image into black and white 
    frame_grayscale=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #detect the faces
    faces=face_detector.detectMultiScale(frame_grayscale)

    for (x,y,w,h) in faces:
        #drawing a green rectangle
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,200,50),4)
        #slice the face part to detect smile
        the_face=frame[y:y+h,x:x+w]
        #convert the face part into gray color
        face_grayscale=cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY)

        #detect the smiles
        smiles=smile_detector.detectMultiScale(face_grayscale,scaleFactor=1.7,minNeighbors=20)

        #put the text as smiling in the outer box
        if len(smiles)>0:
            cv2.putText(frame,"u r smiling",(x,y+h+40),fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN,color=(255,255,255))


    #show the image
    cv2.imshow("smile",frame)
    cv2.waitKey(1)

#we release the webcam for anyother purpose
webcam.release()
cv2.destroyAllWindows()

print("code is correct")
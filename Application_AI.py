import numpy as np
import cv2 as cv
from keras.utils.image_utils import img_to_array
from tensorflow import keras
model = keras.models.load_model('model_tomat1.h5')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    frame = cv.GaussianBlur(frame, (7,7), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask_red = cv.inRange(hsv,(0,6,26),(10,255,255))
    mask_g = cv.inRange(hsv,(0,120,141),(29,255,255))
    mask_y = cv.inRange(hsv,(19,67,72),(39,255,255))

    mask = cv.bitwise_or(mask_red, mask_y)
    mask = cv.bitwise_or(mask, mask_g)

    # Use morphology to remove noise in the binary image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    b_img = cv.morphologyEx(mask,cv.MORPH_OPEN,kernel,iterations=3)

    # Find contours 
    contours, hierachy = cv.findContours(b_img,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt)>2700 and cv.contourArea(cnt) < 35000:
            # Get the coordinates and dimensions of the bounding box around the digit
            x,y,w,h = cv.boundingRect(cnt)
            # Crop
            roi = frame[y-int(h/5):y+h+int(h/5), x-int(w/5):x+w+int(w/5)]
            if roi.shape != (0, 0, 3):
                # Resize the ROI to 300x400 pixels
                roi = cv.resize(roi,(400,300))
                vat = {1: 'Red',2:'Yellow', 3:'Green' }
                img = img_to_array(roi)
                # Normalize
                img=img.reshape(1,300,400,3)
                img = img.astype('float32')
                img =img/255
                # Predict
                result  = np.argmax(model.predict(img),axis=1)
                text = vat[result[0]]
                cv.putText(frame, text, (x,y-10), cv.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)
                cv.rectangle(frame,(x,y),(x+w+10,y+h+20),(0,255,0),2) 
                  
    # Display the resulting frame
    cv.imshow('frame', frame)
    cv.imshow('frame1', b_img)
    
    # cv.imshow('binary', b_img)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

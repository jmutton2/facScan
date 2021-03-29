import cv2

load='person2.jpg'
image= cv2.imread(load)

facial = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = facial.detectMultiScale(grayscale,1.1,4)

for (x,y,w,h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)

cv2.imshow('image', image)
cv2.waitKey()

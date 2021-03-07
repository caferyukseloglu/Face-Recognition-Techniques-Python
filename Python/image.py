from readfile import ReadFiles
# Importing Libraries
import face_recognition
import cv2
import numpy as np
import argparse
from imagenet_labels import LABEL_MAP
PROTOTXT = "data/bvlc_googlenet.prototxt"
MODEL = "data/bvlc_googlenet.caffemodel"
SIZE = 224

class Image:
  def __init__(self):
    self.rf = ReadFiles()
    self.face_locations = []
    self.encode_list = []

  def color_change(self,image):
    image = face_recognition.load_image_file(image)
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

  def get_face_locations(self,image,type):
    try:
      e1 = cv2.getTickCount()    
      face_locations = face_recognition.face_locations(image,0,type)[0]
      e2 = cv2.getTickCount()
      time = (e2 - e1)/ cv2.getTickFrequency()
      print("I found {} face(s) in this photograph.".format(len(face_locations))+" at "+str(time))
      self.face_locations.append(face_locations)
      return 1
    except IndexError as e:
      print(e)
      return 0
  
  def get_face_location(self,image,type):
    face_locations = face_recognition.face_locations(image,1,type)
    return face_locations
  
  def find_encodings(self):
    directories = self.rf.get_images()
    for key in directories.keys():
      if not isinstance(directories[key], list):
        img = self.color_change(key+"/"+directories[key])
        x = self.get_face_locations(img,"hog")
        if x == 1:
          e1 = cv2.getTickCount()    
          encode = face_recognition.face_encodings(img)[0]
          e2 = cv2.getTickCount()
          time = (e2 - e1)/ cv2.getTickFrequency() 
          print("I found encode in this photograph.".format(len(encode))+" at "+str(time))       
          self.encode_list.append(encode)
      else:
        for item in directories[key]:
          img = self.color_change(key+"/"+item)
          x = self.get_face_locations(img,"hog")
          if x == 1:
            e1 = cv2.getTickCount() 
            encode = face_recognition.face_encodings(img)[0]
            e2 = cv2.getTickCount()
            time = (e2 - e1)/ cv2.getTickFrequency() 
            print("I found encode in this photograph.".format(len(encode))+" at "+str(time))   
            self.encode_list.append(encode)

  def find_encoding(self,image,face):
    encode = face_recognition.face_encodings(image, face)
    return encode
  
  def haar_cascade(self):
    parser = argparse.ArgumentParser(description='HAAR TEST')
    parser.add_argument('--face_cascade', help='Path to face cascade.', default='data/haarcascade_frontalface_alt.xml')
    parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='data/haarcascade_eye_tree_eyeglasses.xml')
    parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
    args = parser.parse_args()
    self.face_cascade_name = args.face_cascade
    self.eyes_cascade_name = args.eyes_cascade
    self.face_cascade = cv2.CascadeClassifier()
    self.eyes_cascade = cv2.CascadeClassifier()
    #-- 1. Load the cascades
    if not self.face_cascade.load(cv2.samples.findFile(self.face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not self.eyes_cascade.load(cv2.samples.findFile(self.eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)
    camera_device = args.camera
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        detectAndDisplay(self,frame)
        if cv2.waitKey(10) == 27:
            break
  def detectAndDisplay(self,frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = self.face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = self.eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv2.imshow('Capture - Face detection', frame)

    if not self.face_cascade.load(cv2.samples.findFile(self.face_cascade_name)):
      print('--(!)Error loading face cascade')
      exit(0)
    if not self.eyes_cascade.load(cv2.samples.findFile(self.eyes_cascade_name)):
      print('--(!)Error loading eyes cascade')
      exit(0)

  def run_caffe(net, image, input_size):
      (h, w) = image.shape[:2]
      blob = cv2.dnn.blobFromImage(cv2.resize(image, (input_size, input_size)), 1,
              (input_size, input_size), (104, 177, 123))

      net.setInput(blob)
      out = net.forward()
      return out

  def dnn(img):
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    preds = run_caffe(net, img, SIZE)

    idxs = np.argsort(preds[0])[::-1][:5]

    for i, idx in enumerate(idxs):
        print("{}. {}: {:.2}".format(i + 1, LABEL_MAP[idx], preds[0][idx]))

  def start_video(self):
    self.find_encodings()
    cap = cv2.VideoCapture(0)

    while True:      
      # Capture frame-by-frame
      ret, frame = cap.read()
      frame = cv2.flip(frame,1)
      face = self.get_face_location(frame,"hog")
      encodesCurFrame =  self.find_encoding(frame, face)

      for encodeFace,faceLoc in zip(encodesCurFrame,face):
        matches = face_recognition.compare_faces(self.encode_list, encodeFace)
        faceDis = face_recognition.face_distance(self.encode_list, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
          name = "FOUND"
          y1,x2,y2,x1=faceLoc
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
          cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

      # Display the resulting frame
      cv2.imshow('frame',frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
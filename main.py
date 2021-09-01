import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras.models import load_model

inp = input('Enter the number of words on image:' )
inp = int(inp)

IMG_DIR = 'captcha.jpg'       # input test image path
model = load_model('model.h5')# input model path


mapping={
1:"क", 2:"ख", 3:"ग", 4:"घ", 5:"ङ",
6:"च", 7:"छ", 8:"ज", 9:"झ", 10:"ञ",
11:"ट", 12:"ठ", 13:"ड", 14:"ढ", 15:"ण",
16:"त", 17:"थ", 18:"द", 19:"ध", 20:"न",
21:"प", 22:"फ", 23:"ब", 24:"भ", 25:"म",
26:"य", 27:"र", 28:"ल", 29:"व", 30:"श", 31:"ष",
32:"स", 33:"ह",34:"क्ष", 35:"त्र", 36:"ज्ञ",
37:"अ", 38:"आ", 39:"इ", 40:"ई", 41:"उ", 42:"ऊ", 43:"ऋ", 44:"ए", 45:"ऐ", 46:"ओ", 47:"औ",
48:"अं" , 49:"अ:"}


num = 0

while(num < inp):

     img = cv2.imread(IMG_DIR)
     img = cv2.GaussianBlur(img,(5,5),0)
     rgb_planes = cv2.split(img)
     result_norm_planes = []
     
     for plane in rgb_planes:
             dilated_img = cv2.dilate(plane,np.ones((7,7),np.uint8))
             bg_img = cv2.medianBlur(dilated_img, 21)
             diff_img = 255 - cv2.absdiff(plane,bg_img)
             norm_img = cv2.normalize(diff_img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8UC1)
             result_norm_planes.append(norm_img)
     img = cv2.merge(result_norm_planes)
     

     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     ret,img1 = cv2.threshold(img,170,255,cv2.THRESH_BINARY_INV)
     kernel = np.ones((2,2),np.uint8)
     img2 = cv2.dilate(img1,kernel,iterations=2)
     img1 = cv2.erode(img2,kernel,iterations=1)

     contours, hierarchy = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
     img_arr = []
     contours = sorted(contours,key=lambda x: cv2.contourArea(x),reverse = True)
     

     for cnt in contours:
          x,y,w,h = cv2.boundingRect(cnt)
          img2 = img1[y:y+h,x:x+w]
          coords = np.column_stack(np.where(img2>0))
          angle = cv2.minAreaRect(coords)[-1]
          if (len(cnt) > 500):
               rotated = imutils.rotate_bound(img2,angle)
          else:
               rotated = img2
          contours1, hierarchy = cv2.findContours(rotated,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
          contours1 = sorted(contours1, key=lambda y: cv2.contourArea(y), reverse = True)
          if (len(contours1) > 0):
               a,b,c,d = cv2.boundingRect(contours1[0])
               rotated = rotated[b:b+d,a:a+c]
          if (rotated.shape[0] > 2* rotated.shape[1]):
               rotated = cv2.rotate(rotated,cv2.ROTATE_90_COUNTERCLOCKWISE)
          if (np.argmax(np.sum(rotated,axis=1)) > 0.75*rotated.shape[0]):
               rotated = cv2.rotate(rotated,cv2.ROTATE_180)
          img_arr.append(rotated)
          cv2.imshow('p',rotated)
          print(cnt)
          cv2.waitKey()

     img_arr[num] = cv2.copyMakeBorder(img_arr[num],top=5,bottom=5,left=10,right=10,borderType=cv2.BORDER_CONSTANT,value=0)
     hist = np.sum(img_arr[num],axis=1,dtype='double')
     MAX = np.argmax(hist)
     new_img = img_arr[num][MAX + 14:,:]
     hist1 = np.sum(new_img,axis=0)



     crop = []
     crop1 = []
     for i in range(1,hist1.size-5):
          if (hist1[i] ==0 and (hist1[i-1] !=0 or hist1[i+1] !=0)and (hist1[i-5] == 0 or hist1[i+5]==0)):
               crop.append(i)

     crop1 = [int(x) for x in crop]

     char_arr = []

     for i in range(0,len(crop1)-1,2):
          char = img_arr[num][:,crop1[i]-3:crop1[i+1]+3]
          char_arr.append(char)
     
     print('\n'+'Word ' + str(num+1) + ':',end = ' ')
     for i in range(len(char_arr)):
          char_arr[i] = cv2.resize(char_arr[i],(28,28))
          char_arr[i] = char_arr[i]/255
          char_arr[i] = char_arr[i].reshape(1,28,28,1)
          index = np.argmax(model.predict(char_arr[i]))+1
          print(mapping.get(index),end=" ")
          
     num = num + 1
     







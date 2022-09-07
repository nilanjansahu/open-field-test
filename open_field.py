import cv2
import numpy as np
import imutils
from collections import deque
import numpy as np
import h5py
import easygui
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import os

with h5py.File('newmodel.h5', 'r') as hf: #import model
    d_high = hf['high'][:]
    d_low = hf['low'][:]
    brt = hf['brt'][:][0]


def define_rect(image): #define region
   
    clone = image.copy()
    rect_pts = [] 
    win_name = "image" 

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]

        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    
    cv2.destroyWindow(win_name)

    return rect_pts

def points_4(image):
    poin=[]
    clone = image.copy()
    rect_pts = [] 
    win_name = "image" 

    def select_points(event, x, y, flags, param):

        nonlocal rect_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]
            poin.append((x, y))
        if event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))

            
            cv2.rectangle(clone, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
            cv2.imshow(win_name, clone)

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, select_points)

    while True:
        
        cv2.imshow(win_name, clone)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("r"): # Hit 'r' to replot the image
            clone = image.copy()

        elif key == ord("c"): # Hit 'c' to confirm the selection
            break

    
    cv2.destroyWindow(win_name)
    return poin

path = easygui.fileopenbox() #give recorded video file

inside=0
outside=0
movement=0
static=0

cap = cv2.VideoCapture(path)
ret, frame = cap.read()
fps=cap.get(cv2.CAP_PROP_FPS)
print('select 4 points in order then press c to confirm')
poin=points_4(frame)
print('define rectangle')
print('Hit r to replot the image')
print('Hit c to confirm the selection')
points = define_rect(frame)
frame = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
print('define rectangle')
bounding_box_points = define_rect(frame)


if (cap.isOpened() == False): 
    print("Unable to read camera feed")

pre_x=0
pre_y=0
pts = deque()



while((inside+outside)/fps<=300):
    ret, frame = cap.read()
    print(inside+outside)
    if ret == True:
        mask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask,np.array([poin], dtype=np.int32),(255,)*frame.shape[2])
        frame = cv2.bitwise_xor(frame, ~mask)
        frame = frame[points[0][1]:points[1][1], points[0][0]:points[1][0]]
        frame=(frame + brt)
        #cv2.imshow('frame',frame)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        #cv2.imshow('blur',blurred)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        #cv2.imshow('hsv',hsv)
        
        mask = cv2.inRange(hsv ,d_low, d_high)
        cv2.imshow('mask',mask)
        mask = cv2.erode(mask, (5,5), iterations=3)
        mask = cv2.dilate(mask,(15,15), iterations=3)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        ret,mask = cv2.threshold(mask,60,255,cv2.THRESH_BINARY)
        cv2.imshow('mask',mask)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = cnts[1]
        center = None
        center_m=(0,0)
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
           
            try:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                              
                
                
                if radius > 15:
                    
                    distance=np.sqrt((int(x)-pre_x)**2 + (int(y)-pre_y)**2)
                    
                    pts.appendleft(center)
                    velocity=distance*fps
                    if distance<1:
                        static+=1
                    else:
                        movement+=1
                    #print(str(velocity))
                    cv2.putText(img = frame, text = 'V='+ str("%.2f" % velocity), org = (int(x),int(y)+20), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5,color = (0, 255, 0))
                    pre_x=int(x)
                    pre_y=int(y)
                    
                    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 1)
                    if bounding_box_points[0][1]<int(x)<bounding_box_points[1][1] and bounding_box_points[0][0]<int(y)<bounding_box_points[1][0]:
                        inside+=1
                    else:
                        outside+=1
                    cv2.circle(frame, center, 1, (0, 0, 255), -1)
                
            except:
                continue
            
        
        
        
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(100 / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        
        cv2.imshow("Frame", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break 

cap.release()

cv2.destroyAllWindows()

inside_per=(inside/(inside+outside))*100
outside_per=(outside/(inside+outside))*100
print('inside %'+str(inside_per)+'% '+str(inside/fps)+'seconds')
print('outside %'+str(outside_per)+'% '+str(outside/fps)+'seconds')

movement_per=movement*100/(movement+static)
static_per=static*100/(movement+static)
print('movement %'+str(movement_per)+'% '+str(movement/fps)+'seconds')
print('static %'+str(static_per)+'% '+str(static/fps)+'seconds')

with open(os.path.splitext(os.path.basename(path))[0]+'.csv','w+') as f:
    f.write('time,inside, outside,movement,static'+'\n')
    f.write('seconds,'+str(inside/fps)+','+str(outside/fps)+','+str(movement/fps)+','+str(static/fps)+'\n')
    f.write('normalised time,'+str(inside_per)+','+str(outside_per)+','+str(movement_per)+','+str(static_per)+'\n')



 
objects = ('inside', 'outside','movement','static')

y_pos = np.arange(len(objects))
performance = [inside_per,outside_per,movement_per,static_per]
 
plt.bar(y_pos, performance, align='center', alpha=0.5) #bar plot of time
plt.xticks(y_pos, objects)
plt.ylabel('normalised time spent')
plt.title('Time Plot')
 
plt.show()
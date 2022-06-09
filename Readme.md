# 车牌识别

## 1 导入素材

```py
#图片
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```py
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 2 灰度转换

```py
#二元阈值 彩色
import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```py
#二元阈值 灰度
import cv2
import numpy as np
img = cv2.imread('car.jpg')
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```py
#自适应阈值
import cv2
import numpy as np
img = cv2.imread('car.jpg')
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
# 大津阈值
import cv2
import numpy as np
img = cv2.imread('car.jpg')
grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('Otsu threshold',threshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3 颜色过滤

```py
#RGB2hsv取色
import cv2
import numpy as np
img = cv2.imread('car3.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow('frame',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 4 模糊

```
# easy模糊
import cv2
import numpy as np
img = cv2.imread('car3.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
cv2.imshow('frame',img)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

kernel = np.ones((15,15),np.float32)/225
smoothed = cv2.filter2D(res,-1,kernel)
cv2.imshow('Original',img)
cv2.imshow('Averaging',smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```py
# 中值模糊
median = cv2.medianBlur(res,15)
cv2.imshow('Median Blur',median)
```

## 5 形态学

```
import cv2
import numpy as np
img = cv2.imread('car3.jpg')


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)

kernel = np.ones((5,5),np.uint8)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original',img)
cv2.imshow('Mask',mask)
cv2.imshow('Opening',opening)
cv2.imshow('Closing',closing)

cv2.waitKey(0)

cv2.destroyAllWindows()
```

## 6 边缘检测

```
import cv2
import numpy as np
img = cv2.imread('car3.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
#关计算更好

import cv2
import numpy as np
img = cv2.imread('try1.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
import cv2
import numpy as np
img = cv2.imread('try4.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
newImg = cv2.drawContours(img,contours,-1,(0,0,255),2)
cv2.imshow('newImg',newImg)


cv2.waitKey(0)
cv2.destroyAllWindows()
```

example

```
import cv2
import numpy as np

img =cv2.imread("car.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


gs_img = cv2.GaussianBlur(gray_img, (5,5), 0, 0, cv2.BORDER_DEFAULT)

kernel = np.ones((23, 23), np.uint8)
open_img = cv2.morphologyEx(gs_img ,cv2.MORPH_OPEN ,kernel)
open_img = cv2.addWeighted(gs_img, 1, open_img, -1, 0)
ret, edge_img = cv2.threshold(open_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edge_img = cv2.Canny(edge_img, 100, 200)
kernel = np.ones((10, 10), np.uint8)
edge_img1 = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)
edge_img2 = cv2.morphologyEx(edge_img1, cv2.MORPH_OPEN, kernel)
contours = cv2.findContours(edge_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# car_plates=[]
# for contour in contours:
#     if cv2.contourArea( contour ) >500 : #这个500参数和图片大小有关，本文经验值
#         rect_tupple = cv2.minAreaRect( contour )
#         rect_width, rect_height = rect_tupple[1]
#         if rect_width < rect_height:
#             rect_width, rect_height = rect_height, rect_width
#         aspect_ratio = rect_width / rect_height
#         # 车牌正常情况下宽高比在2 - 5.5之间
#         if aspect_ratio > 2 and aspect_ratio < 5.5:
#             car_plates.append(contour)

cv2.imshow('gs',gs_img)
cv2.imshow('img',img)
cv2.imshow('open_img',open_img)
cv2.imshow('edge_img',edge_img)
cv2.imshow('edge_img1',edge_img1)
cv2.imshow('edge_img2',edge_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
import cv2
import numpy as np
img = cv2.imread('car3.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
# newImg = cv2.drawContours(img,contours,-1,(0,0,255),2)
# cv2.imshow('newImg',newImg)
# 遍历轮廓
for c in contours:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当恰好是 4 个角点的时候，获取轮廓。
    if len(approx) == 4:
        screen_cnt = approx
        break

# 结果显示
anoImg = cv2.drawContours(img, [screen_cnt], -1, (0, 0, 255), 2)
cv2.imshow('anoImg',anoImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
import cv2
import numpy as np
img = cv2.imread('try1.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((6,6),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")


cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
import cv2
import numpy as np
img = cv2.imread('try1.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Opening',opening)
cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
# newImg = cv2.drawContours(img,contours,-1,(0,0,255),2)
# cv2.imshow('newImg',newImg)
# 遍历轮廓
for c in contours:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当恰好是 4 个角点的时候，获取轮廓。
    if len(approx) == 4:
        screen_cnt = approx
        break

# 结果显示
anoImg = cv2.drawContours(img, [screen_cnt], -1, (0, 0, 255), 2)
cv2.imshow('anoImg',anoImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

保存图片

```
import cv2
import numpy as np
img = cv2.imread('car3.jpg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Opening',opening)
#cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
#cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
# newImg = cv2.drawContours(img,contours,-1,(0,0,255),2)
# cv2.imshow('newImg',newImg)

for c in contours:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

# 遍历轮廓
for c in contours:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当恰好是 4 个角点的时候，获取轮廓。
    if len(approx) == 4:
        screen_cnt = approx
        break

# 结果显示
anoImg = cv2.drawContours(img, [screen_cnt], -1, (0, 0, 255), 2)
#cv2.imshow('anoImg',anoImg)
print(screen_cnt)
x1 = screen_cnt[0][0][0]
x2 = screen_cnt[1][0][0]
x3 = screen_cnt[2][0][0]
x4 = screen_cnt[3][0][0]
y1 = screen_cnt[0][0][1]
y2 = screen_cnt[1][0][1]
y3 = screen_cnt[2][0][1]
y4 = screen_cnt[3][0][1]
xMin = min(x1,x2,x3,x4)
xMax = max(x1,x2,x3,x4)
yMin = min(y1,y2,y3,y4)
yMax = max(y1,y2,y3,y4)

roi = img[yMin:yMax,xMin:xMax]
cv2.imshow('roi',roi)
cv2.imwrite('save15.jpg', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

```
import tkinter as tk
import cv2
import numpy as np

top = tk.Tk()
top.title("Hello OpenCV")

w = tk.Label(top, text="请点击按钮")
w.pack()
imageName = ['try10.jpg','try2.jpg','try1.jpg','try4.jpg','try5.jpg','try8.jpg','try13.jpg','car3.jpg']
imageIndex = 0
def nextIndex(imageIndex):
    imageIndex += 1
    if imageIndex == 8 :
        imageIndex = 0
#_______________________按钮实例1
b = tk.Button(top, text="下一张图片",command=lambda :nextIndex(imageIndex))  # this is placed in 1 0
b.pack()



#_______________________按钮实例1
b = tk.Button(top, text="查看图片")  # this is placed in 1 0
b.pack()

def left_click(event):

    img = cv2.imread(imageName[imageIndex], cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

b.bind("<Button-1>", left_click)

#_______________________按钮实例
b = tk.Button(top, text="Button")  # this is placed in 1 0
b.pack()

def left_click(event):
    top.title("Welcome to Tkinter Windows with a Button.")
    img = cv2.imread('watch.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

b.bind("<Button-1>", left_click)
#_______________________按钮实例
# 进入消息循环
top.mainloop()
```

```
import cv2
import numpy as np

imageName = ['try13.jpg','try2.jpg','try1.jpg','try4.jpg','try5.jpg','try8.jpg','try10.jpg','car3.jpg','try3.jpg']
imageIndex = 0

img = cv2.imread(imageName[imageIndex])
imgCopy = img

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100,43,46])
upper_blue = np.array([124,255,255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img,img, mask= mask)
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('Opening',opening)
#cv2.imshow('Original',img)
edges = cv2.Canny(opening,100,200)
#cv2.imshow('Edges',edges)

tempImg,contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓数量：{len(contours)}")
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
#newImg = cv2.drawContours(img,contours,-1,(0,0,255),2)
#cv2.imshow('newImg',newImg)

# 遍历轮廓
for c in contours:
    # 计算轮廓近似
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # 当恰好是 4 个角点的时候，获取轮廓。
    if len(approx) == 4:
        screen_cnt = approx
        break

try:
    # 结果显示
    #anoImg = cv2.drawContours(img, [screen_cnt], -1, (0, 0, 255), 2)
    #cv2.imshow('anoImg',anoImg)
    #print(screen_cnt)
    x1 = screen_cnt[0][0][0]
    x2 = screen_cnt[1][0][0]
    x3 = screen_cnt[2][0][0]
    x4 = screen_cnt[3][0][0]
    y1 = screen_cnt[0][0][1]
    y2 = screen_cnt[1][0][1]
    y3 = screen_cnt[2][0][1]
    y4 = screen_cnt[3][0][1]
    xMin = min(x1,x2,x3,x4)
    xMax = max(x1,x2,x3,x4)
    yMin = min(y1,y2,y3,y4)
    yMax = max(y1,y2,y3,y4)

    roi = imgCopy[yMin:yMax,xMin:xMax]
    cv2.imshow('roi',roi)
    #保存图片
    temp = 'save' + str(imageIndex)+'.jpg'
    print(temp)
    #cv2.imwrite(temp, roi)
except:
    print("no")
cv2.waitKey(0)
cv2.destroyAllWindows()
```
# @Author  : Ryan
# @Email   : ryan1057@csu.edu.cn
# @File    : test.py
# @Software: PyCharm
# @Time    : 2022/6/7 16:04
# @Github  : https://github.com/Ryan-dodo/Chinese_License_plate_recognition
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
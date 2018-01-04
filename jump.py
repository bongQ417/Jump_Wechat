# -*- coding:utf-8 -*-

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import wda
import time

wdaClient = wda.Client('http://169.254.91.126:8100')
wdaSession = wdaClient.session()


# 获取16进制的颜色
def getColor(BGR):
    b, g, r = BGR
    color = str(hex(r))[2:] + str(hex(g))[2:] + str(hex(b))[2:]
    return color


def isYellowBG(HSV):
    hh, ss, vv = HSV
    if (int(hh) >= 11 and int(hh) <= 25):
        return True
    return False


def display(image, code=-1):
    cv_rgb = image
    if not (str(code) == '-1'):
        cv_rgb = cv2.cvtColor(image, code)
    plt.figure()
    plt.imshow(cv_rgb, animated=True)
    plt.show()


# 获取棋子位置
def getChessPosition(image, hsv):
    # 设定紫色阈值，HSV空间
    purpleLower = np.array([119, 60, 60])
    purpleUpper = np.array([130, 120, 120])
    # 根据阈值构建掩膜
    mask = cv2.inRange(hsv, purpleLower, purpleUpper)
    # 腐蚀操作
    # mask = cv2.erode(mask, None, iterations=3)
    # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
    mask = cv2.dilate(mask, None, iterations=7)
    # display(mask, cv2.COLOR_GRAY2RGB)
    # 轮廓检测
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    c = max(cnts, key=cv2.contourArea)
    # 确定面积最大的轮廓的外接圆
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    ((x, y), radius) = ((int(x), int(y)), int(radius))

    # 抹除棋子
    bgcolor = image[y][0]
    for my in range(y - radius, y + radius):
        for mx in range(x - radius, x + radius):
            if (mask[my, mx] == 255):
                image[my][mx] = bgcolor
    # display(image, cv2.COLOR_BGR2RGB)
    return (x, y + 65)


# 获取棋盘位置
def getBoardPosition(image, topY, endY):
    imageCopy = image.copy()
    # 截取中间部分
    img_gray = imageCopy[topY:endY, 0:image.shape[1]]
    # display(img_gray, cv2.COLOR_GRAY2RGB)
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    edges = cv2.Canny(img_gray, 50, 100)
    # display(edges, cv2.COLOR_GRAY2RGB)
    # 初始化参数
    h, w = edges.shape
    maxsize = 0
    topPoint = (-1, -1)
    startPoint = (-1, -1)
    endPoint = (-1, -1)
    saveY = 0
    for i in range(h):
        isStart = True
        if (endPoint == (-1, -1)):
            # 计算顶点
            for j in range(w):
                if (int(edges[i][j]) == 255):
                    # 记录一整行
                    if (isStart):
                        startPoint = (j, i)
                        isStart = False
                    else:
                        endPoint = (j, i)
        else:
            # 计算中心点
            k = 1
            x = topPoint[0]
            size = 0
            while (x - k - 10 >= 0 and x + k + 10 < w):
                if (int(edges[i][x - k]) == 255):
                    for z in range(x + k, x + k + 10):
                        if (int(edges[i][z]) == 255):
                            size = k
                            break
                elif (int(edges[i][x + k]) == 255):
                    for z in range(x - k - 10, x - k):
                        if (int(edges[i][z]) == 255):
                            size = k
                            break
                k += 1
            if (maxsize < size):
                maxsize = size
                saveY = i
            elif (maxsize > 20
                  and (maxsize >= size or saveY - topPoint[1] > maxsize)):
                saveY = i
                break
        # 计算顶点
        if (endPoint == (-1, -1)):
            continue
        if (topPoint == (-1, -1)):
            topPoint = (int(startPoint[0] / 2) + int(endPoint[0] / 2), i)
            continue
    return (topPoint[0], saveY + topY)


def jump(distance):
    # 0.00196根据io手机修改
    press_time = distance * 0.00196
    press_time = press_time
    wdaSession.tap_hold(200, 200, press_time)
    return press_time


# 目前测试机为iPhone8
if __name__ == '__main__':
    i = 1
    while True:
        wdaClient.screenshot('screen/screen' + str(i) + '.png')
        image = cv2.imread('screen/screen' + str(i) + '.png')
        # image = cv2.imread('screen.png')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # 获取棋子位置
        (chessX, chessY) = getChessPosition(image, hsv)
        cv2.circle(image, (chessX, chessY), 4, (0, 0, 0), 2)
        # 获取棋盘位置  
        # 210  可以根据屏幕像素修改
        (boardX, boardY) = getBoardPosition(image, 210, chessY)
        cv2.circle(image, (boardX, boardY), 4, (0, 0, 0), 2)
        dx = chessX - boardX
        dy = chessY - boardY
        distance = int(math.sqrt(dx * dx + dy * dy))
        spTime = jump(distance)
        time.sleep(spTime + 1)
        i += 1
        # display(image)

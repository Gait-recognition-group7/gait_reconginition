import numpy as np
import cv2
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()  # 背景分割
while (1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)  # 应用
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()







import cv2
import numpy as np

cap = cv2.VideoCapture("001-bg-01-090.avi")

# 取出视频的第一帧
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# 创造一个全零列表，维度大小等同于frame1
hsv = np.zeros_like(frame1)  # 为绘制创建掩码图片
hsv[..., 1] = 255

while (1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # 计算光流以获取点的新位置

    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # 色调范围：0°~360°
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

cap.release()
cv2.destroyAllWindows()




# 来自opencv\sources\samples\python\lk_track.py
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
# params for ShiTomasi corner detection 设置 ShiTomasi 角点检测的参数
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow 设置 lucas kanade 光流场的参数
# maxLevel 为使用的图像金字塔层数
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors 产生随机的颜色值
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it 获取第一帧，并寻找其中的角点
(ret, old_frame) = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes 创建一个掩膜为了后面绘制角点的光流轨迹
mask = np.zeros_like(old_frame)

# 视频文件输出参数设置
out_fps = 12.0  # 输出文件的帧率
fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
sizes = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
out = cv.VideoWriter('E:/video/v5.avi', fourcc, out_fps, sizes)

while True:
    (ret, frame) = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow 能够获取点的新位置
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points 取好的角点，并筛选出旧的角点对应的新的角点
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # draw the tracks 绘制角点的轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    cv.imshow('frame', img)
    out.write(img)
    k = cv.waitKey(200) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points 更新当前帧和当前角点的位置
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

out.release()
cv.destroyAllWindows()
cap.release()
#
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
capture = cv2.VideoCapture('001-bg-01-090.avi')  # 视频名称
print(capture.isOpened())
num = 0

frame_count = 0
all_frames = []
while True:
    ret, frame = capture.read()
    if ret is False:
        break
    all_frames.append(frame)
    frame_count = frame_count + 1

# The value below are both the number of frames
print(frame_count)
print(len(all_frames))
capture.release()
'''

# 导出中间的图像
capture = cv2.VideoCapture('001-bg-01-090.avi')  # 视频名称
print(capture.isOpened())
num = 0
while True:
    ret, img = capture.read()
    if not ret:
        break
    if num == 41:  # 导出视频的第41帧图像
        cv2.imwrite('%s.jpg' % ('pic_' + str(num)), img)  # 写出视频图片.jpg格式
        break
    num = num + 1

capture.release()

img = cv2.imread('pic_41.jpg', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 显示直方图
hist_full = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist_full)
plt.show()


# 灰度变换
Imax = np.max(gray)
Imin = np.min(gray)
MAX = 255
MIN = 0
gray_cs = (gray - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
cv2.imshow("gray_cs", gray_cs.astype("uint8"))
cv2.waitKey()
cv2.imwrite("canny.jpg", cv2.Canny(gray_cs.astype("uint8"), 200, 310))

# 显示变换后的直方图
hist_full1 = cv2.calcHist([gray_cs.astype("uint8")], [0], None, [256], [0, 256])
plt.plot(hist_full1)
plt.show()

cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()



# 腐蚀
# 读取图片
src = cv2.imread('canny.jpg', cv2.IMREAD_UNCHANGED)

# 设置卷积核
kernel = np.ones((7, 7), np.uint8)

# 图像腐蚀处理
erosion = cv2.erode(src, kernel)

# 显示图像
cv2.imshow("src", src)
cv2.imshow("result", erosion)

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()

# 设置卷积核
kernel2 = np.ones((3, 3), np.uint8)

# 图像膨胀处理
expand = cv2.dilate(src, kernel2)
# 图像腐蚀处理
erosion2 = cv2.erode(expand, kernel2)

# 显示图像
# cv2.imshow("src", src2)
cv2.imshow("result2", erosion2)
cv2.imwrite("result2.jpg", erosion2)
# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()


# def FillHole(imgPath, SavePath):
# 以灰度模式读入图片
im_in = cv2.imread('result2.jpg', cv2.IMREAD_GRAYSCALE);

# 复制 im_in 图像
im_floodfill = im_in

# Mask 用于 floodFill，官方要求长宽+2
h, w = im_in.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)

# floodFill函数中的seedPoint必须是背景
# 思路：遍历一下整张图片，按行扫描
isbreak = False
for i in range(im_floodfill.shape[0]):
    for j in range(im_floodfill.shape[1]):
        if im_floodfill[i][j] == 0:
            seedPoint = (i, j)
            isbreak = True
            break
    if (isbreak):
        break
# 得到im_floodfill
cv2.floodFill(im_floodfill, mask, seedPoint, 255)

# 得到im_floodfill的逆im_floodfill_inv
im_floodfill_inv = cv2.bitwise_not(im_floodfill)
im_in_inv = cv2.bitwise_not(im_in)
cv2.imshow(" inv_mask ", im_floodfill_inv)
# 把im_in、im_floodfill_inv这两幅图像结合起来得到前景
cv2.imshow(" input_image ", erosion2)
im_out = erosion2 | im_floodfill_inv

# 保存结果
cv2.imwrite("result3.jpg", im_out)
cv2.imshow("final_result", cv2.imread("result3.jpg"))

# 等待显示
cv2.waitKey(0)
cv2.destroyAllWindows()





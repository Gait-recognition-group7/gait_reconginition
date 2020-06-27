# coding:utf-8
"""
计算帧之间的差异 考虑背景帧与其他帧之间的差异
"""
import cv2
import numpy as np

print("hello world")
# 打开视频文件，命名为video
video = cv2.VideoCapture('001-bg-01-090.avi')
# 定义核为kernel
kernel = np.ones((5, 5), np.uint8)
background = None

while True:
    # 读入摄像头的帧
    ret, frame = video.read()
    # ret作为布尔值，Ture代表有读取到图片
    # 把第一帧作为背景
    if background is None:
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = cv2.GaussianBlur(background, (5, 5), 0)
        continue
    # 读入帧
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯平滑 模糊处理 减小光照 震动等原因产生的噪声影响
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

    # 检测背景和帧的区别
    diff = cv2.absdiff(background, gray_frame)
    # 将区别转为二值，定义阈值为25，最大值为255，判断方式为大于阈值设置为255，小于则设置为0
    diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    # 定义结构元素,定义结构元素为椭圆形状，内核尺寸为（9，4）
    # es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,9))
    # 膨胀运算
    # diff = cv2.dilate(diff, es, iterations=2)
    # 搜索轮廓
    cnts, hierarcchy= cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    """
    cv.findContours()
        参数：
            1 要寻找轮廓的图像 只能传入二值图像，不是灰度图像
            2 轮廓的检索模式，有四种：
                cv2.RETR_EXTERNAL表示只检测外轮廓
                cv2.RETR_LIST检测的轮廓不建立等级关系
                cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，
                    里面的一层为内孔的边界信息。
                    如果内孔内还有一个连通物体，这个物体的边界也在顶层
                cv2.RETR_TREE建立一个等级树结构的轮廓
            3 轮廓的近似办法
                cv2.CHAIN_APPROX_NONE存储所有的轮廓点，
                    相邻的两个点的像素位置差不超过1，
                    即max（abs（x1-x2），abs（y2-y1））==1
                cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，
                    只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        返回值:
            contours:一个列表，每一项都是一个轮廓， 不会存储轮廓所有的点，只存储能描述轮廓的点
            hierarchy:一个ndarray, 元素数量和轮廓数量一样，
                每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，
                分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数
    """

    for c in cnts:
        # 轮廓太小忽略 有可能是斑点噪声
        if cv2.contourArea(c) < 2000:
            continue
        # 将轮廓画出来
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("contours", frame)
    cv2.imshow("diff", diff)
    if cv2.waitKey(5) & 0xff == ord("q"):
        break

cv2.destroyAllWindows()
camera.release()
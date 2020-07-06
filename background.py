# coder: 李航
import cv2 as cv


def extract_silhouette(input_video):
    """
    生成步态能量图
    :param input_video: 输入视频名，或摄像头编号
    :return: all_binary:每一帧的处理结果
    """
    capture = cv.VideoCapture(input_video)
    # 构造高斯混合模型
    mog = cv.createBackgroundSubtractorMOG2(varThreshold=200, detectShadows=True)
    # 选择矩形内核，大小为5*5
    se = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    # 初始化列表
    all_binary = []
    while True:
        ret, image = capture.read()
        if ret is True:
            fgmask = mog.apply(image)
            ret, binary = cv.threshold(fgmask, 150, 255, cv.THRESH_BINARY)
            binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, se)
            all_binary.append(binary)
            # bgimage = mog.getBackgroundImage()
            """
            cv.imshow("bgimage", bgimage)
            cv.imshow("frame", image)
            cv.imshow("fgmask", binary)
            c = cv.waitKey(100)
            if c == 27:
                break
            """
        else:
            break
    return all_binary

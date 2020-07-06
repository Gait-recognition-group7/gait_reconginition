# coder: 李栾、李歆哲
import os
from PIL import Image
import numpy as np


def find_the_largest_contour(binary):
    """
    寻找输出周期，确定剪切图片大小
    :param binary: 输入二值化图片
    :return: size_max: 剪切图片的输出大小
    """

    size_max = 0
    for image in binary:
        image = np.array(image)  # 转化为np对象
        # 找到人的最小最大高度与宽度
        height_min = (image.sum(axis=1) != 0).argmax()
        height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
        # 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
        size = height_max-height_min
        if size > size_max:
            size_max = size
    pass
    return size_max


def cut_image(binary, size):
    """
    剪切图片
    :param binary: 输入帧构成列表
    :param size: 要剪切的图片大小
    :return: image_list 剪切后的图片构成的列表
    """
    image_list = []
    for b_img in binary:
        image_object = Image.fromarray(b_img)
        image, flag = cut(image_object)  # 调用下面的cut方法
        if not flag:
            image_list.append(Image.fromarray(image).convert('L').resize((size, size)))
    return image_list


def cut(image):
    """
    通过找到人的最小最大高度与宽度把人的轮廓分割出来，、
    因为原始轮廓图为二值图，因此头顶为将二值图像列相加后，形成一列后第一个像素值不为0的索引。
    同理脚底为形成一列后最后一个像素值不为0的索引。
    人的宽度也同理。
    :param image: 需要裁剪的图片 N*M的矩阵
    :return: temp:裁剪后的图片 size*size的矩阵。flag：是否是符合要求的图片
    """
    image = np.array(image)  # 转化为np对象

    # 找到人的最小最大高度与宽度
    height_min = (image.sum(axis=1) != 0).argmax()
    height_max = ((image.sum(axis=1) != 0).cumsum()).argmax()
    width_min = (image.sum(axis=0) != 0).argmax()
    width_max = ((image.sum(axis=0) != 0).cumsum()).argmax()
    head_top = image[height_min, :].argmax()
    # 设置切割后图片的大小，为size*size，因为人的高一般都会大于宽
    size = height_max-height_min
    temp = np.zeros((size, size))

    # 将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    # l = (width_max-width_min)//2
    # r = width_max-width_min-l
    # 以头为中心，将将width_max-width_min（宽）乘height_max-height_min（高，szie）的人的轮廓图，放在size*size的图片中央
    l1 = head_top-width_min
    r1 = width_max-head_top
    # 若宽大于高，或头的左侧或右侧身子比要生成图片的一半要大。则此图片为不符合要求的图片
    flag = False
    if size <= width_max-width_min or size//2 < r1 or size//2 < l1:
        flag = True
        return temp, flag
    # centroid = np.array([(width_max+width_min)/2,(height_max+height_min)/2], dtype = 'int')
    temp[:, (size//2-l1):(size//2+r1)] = image[height_min:height_max, width_min:width_max]
    # rat = (width_max-width_min)/(height_max-height_min)  # 我加的，宽高比
    return temp, flag


def gei_synthesis(image_list, out_path, size, file):
    """
    生成步态能量图
    :param image_list: 作为剪切后图片的列表
    :param out_path: 生成gei图片的路径
    :param size: 生成能量图大小
    :param file: 视频文件名
    :return: None
    """
    file_name = file.replace(".avi", ".png")
    gei_array = np.zeros([size, size])
    if len(image_list) != 0:
        for image in image_list:
            gei_array += np.array(image)
        gei_array /= len(image_list)
        Image.fromarray(gei_array).resize((size, size)).convert('L').save(os.path.join(out_path, file_name))
        pass


# if __name__=='__main_':
def get_gei_from_one_cycle(path_out, binary, file):
    size = find_the_largest_contour(binary)
    image_list = cut_image(binary, size)
    gei_synthesis(image_list, path_out, size, file)

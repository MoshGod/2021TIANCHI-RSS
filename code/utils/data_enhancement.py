
from osgeo import gdal
import os
import cv2
import numpy as np
from matplotlib import pylab as plt

'''光照调整'''


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


'''模糊处理'''


def blur(img):
    img = cv2.blur(img, (3, 3))
    return img


'''遍历整个数据集进行数据增强'''


def transform(image_path, label_path):
    image_list = os.listdir(image_path)
    label_list = os.listdir(label_path)
    file_idx = len(image_list) + 1
    for i, (img, lab) in enumerate(zip(image_list, label_list)):

        ori_width, ori_height, ori_bands, ori_data, ori_geotrans, ori_proj = readTif(image_path + '/' + img)
        lab_width, lab_height, lab_bands, lab_data, lab_geotrans, lab_proj = readTif(label_path + '/' + lab)
        # print(np.sum(lab_data==1), np.sum(lab_data==0))

        img_save_path = image_path + '/' + str(file_idx + i) + img[-4:]
        label_save_path = label_path + '/' + str(file_idx + i) + lab[-4:]

        # 基于Gamma变换调整光照
        if 0 <= np.random.random() < 0.2:
            ori_data = ori_data.swapaxes(1, 0)
            ori_data = ori_data.swapaxes(1, 2)
            ori_data = np.array(ori_data, dtype='uint8')
            gamma_data = random_gamma_transform(ori_data, np.random.uniform(0.7, 1.3))
            gamma_data = gamma_data.swapaxes(1, 2)
            gamma_data = gamma_data.swapaxes(0, 1)
            gamma_data = np.array(gamma_data, dtype='uint16')
            writeTiff(gamma_data, ori_geotrans, ori_proj, img_save_path)

            writeTiff(lab_data, lab_geotrans, lab_proj, label_save_path)
        # 图像模糊化
        elif 0.2 <= np.random.random() < 0.4:
            ori_data = ori_data.swapaxes(1, 0)
            ori_data = ori_data.swapaxes(1, 2)
            ori_data = np.array(ori_data, dtype='uint8')
            bl_data = blur(ori_data)
            bl_data = bl_data.swapaxes(1, 2)
            bl_data = bl_data.swapaxes(0, 1)
            bl_data = np.array(bl_data, dtype='uint16')
            writeTiff(bl_data, ori_geotrans, ori_proj, img_save_path)

            writeTiff(lab_data, lab_geotrans, lab_proj, label_save_path)
        # 水平翻转
        elif 0.4 <= np.random.random() < 0.6:
            # image
            hor_data = np.flip(ori_data, axis=2)
            writeTiff(hor_data, ori_geotrans, ori_proj, img_save_path)
            # label
            hor_label = np.flip(lab_data, axis=1)
            writeTiff(hor_label, lab_geotrans, lab_proj, label_save_path)
        # 垂直翻转
        elif 0.6 <= np.random.random() < 0.8:
            # image
            vec_data = np.flip(ori_data, axis=1)
            writeTiff(vec_data, ori_geotrans, ori_proj, img_save_path)
            # label
            vec_label = np.flip(lab_data, axis=0)
            writeTiff(vec_label, lab_geotrans, lab_proj, label_save_path)
        # 对角镜像翻转
        else:
            # image
            vec_data = np.flip(ori_data, axis=1)
            dia_data = np.flip(vec_data, axis=2)
            writeTiff(dia_data, ori_geotrans, ori_proj, img_save_path)
            # label
            dia_label = np.flip(lab_data, -1)
            writeTiff(dia_label, lab_geotrans, lab_proj, label_save_path)


if __name__ == '__main__':
    transform(r'../data/validation2/image', r'../data/validation2/label')

#!/user/bin/env python#-*- coding:utf-8 -*-import jsonfrom osgeo import gdalfrom matplotlib import pylab as pltfrom PIL import Imageimport numpy as npdef print_tif(file_name):    dataset = gdal.Open(file_name)    print(dataset)    width = dataset.RasterXSize    height = dataset.RasterYSize    print(width, height)    img_np = dataset.ReadAsArray(0, 0, width, height)    print(type(img_np), img_np.shape)    img_np = img_np.swapaxes(1, 0)    img_np = img_np.swapaxes(1, 2)    plt.imshow(img_np)    plt.axis('off')    plt.show()    return img_npdef print_png(file_name):    img = Image.open(file_name)    plt.imshow(img)    plt.show()    img_np = np.array(img)    return img_np# 原图可视化# print_tif(r'..\data\train\img\000010.tif')# 标签图可视化path = r'..\data\train\label\016008.png'img = print_png('../test.png')print(img.shape, img)# 猫一眼各地物分类所占比例idx_to_class = json.load(open('../data/class_index.json', encoding='UTF-8'))['classes']total = 0for i in range(0, 10):    print('类别: {0:{1}^7}'.format(idx_to_class[str(i)], chr(12288)), ' 灰度值个数:', '{:^5}'.format(np.sum(img == i)), ' 所占比例: %.4f' % round(np.sum(img == i) / (256*256), 4))    total += round(np.sum(img == i) / (256*256), 4)print(total)"""Tips:     PIL: RGB Image    cv2: BGR ndarray    skimage: RGB ndarray"""
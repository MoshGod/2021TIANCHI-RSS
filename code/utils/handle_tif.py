from osgeo import gdal
# import gdal
import numpy as np


def read_tif(img_path):

    dataset = gdal.Open(img_path)  # (band_num, width, height)
    # 获取栅格矩阵列数
    img_width = dataset.RasterXSize
    # 获取栅格矩阵行数
    img_height = dataset.RasterYSize
    # 获取波段数
    img_bands = dataset.RasterCount

    '''实际上上方三个参数都可以从下方的 data_np.shape 中获得'''

    # 获取投影信息
    img_proj = dataset.GetProjection()
    # 获取仿射矩阵信息
    img_geotrans = dataset.GetGeoTransform()
    # 转换成numpy数组格式
    data_np = dataset.ReadAsArray(0, 0, img_width, img_height)

    del dataset

    return data_np, img_geotrans, img_proj


def write_tif(img_data, img_geotrans, img_proj, img_path):

    if 'int8' in img_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(img_data.shape) == 2:
        img_data = np.array([img_data])

    img_bands, img_height, img_width = img_data.shape

    # 创建驱动
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(img_path, img_width, img_height, img_bands, datatype)

    # 写入仿射变换参数
    dataset.SetGeoTransform(img_geotrans)
    # 写入投影信息
    dataset.SetProjection(img_proj)


    for i in range(img_bands):
        dataset.GetRasterBand(i + 1).WriteArray(img_data[i])

    del dataset


if __name__ == '__main__':
    img_data, img_geotrans, img_proj = read_tif(r'D:\_2020Winter\SkyPool\data\suichang_round1_train_210120\suichang_round1_train_210120\000002.tif')
    # 仿射矩阵
    print(type(img_geotrans), img_geotrans)
    # 投影信息
    print(type(img_proj), img_proj)
    print(type(img_data.dtype), type(img_data.dtype.name))
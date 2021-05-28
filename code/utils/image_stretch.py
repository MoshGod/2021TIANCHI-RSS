import numpy as np


def image_stretch(image, rate, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, rate)
        truncated_up = np.percentile(gray, 100 - rate)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = np.uint8(gray)
        elif(max_out <= 65535):
            gray = np.uint16(gray)
        return gray
    
    image_stretch = []
    #  对RGB进行拉伸
    for i in range(image.shape[0] - 1):
        gray = gray_process(image[i])
        image_stretch.append(gray)
        
    image_stretch.append(image[3])  # Nir
    image_stretch = np.array(image_stretch)
    
    return image_stretch
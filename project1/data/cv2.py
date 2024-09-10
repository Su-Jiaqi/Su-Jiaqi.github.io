import numpy as np
import skimage as sk
import cv2
import skimage.io as skio
from time import time
from numba import njit, prange
def remove_border(image):
    """
    去除图像中的边框
    
    参数:
    image: 输入图像矩阵 (numpy array)
    
    返回:
    cropped_image: 去除边框后的图像矩阵 (numpy array)
    """
    image = np.uint8(image * 255)
    # 提取蓝色通道
    blue_channel = image[:, :, 0]

    # 应用自适应阈值处理
    binary = cv2.adaptiveThreshold(blue_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 反转二值图像
    binary = cv2.bitwise_not(binary)

    # 膨胀操作，填补边框中的小孔洞
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)

    # 获取边界框
    x, y, w, h = cv2.boundingRect(max_contour)

    # 裁剪图像
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image

imname = 'F:/UCB/CS180/project1/data/train.tif'
im = skio.imread(imname)
im = sk.img_as_float(im)
height = np.floor(im.shape[0] / 3.0).astype(np.int64)
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

b = b / np.max(b)
g = g / np.max(g)
r = r / np.max(r)

# Align the images
def align(im1, ref):
    # Normalize images to the range [0, 255]
    im1_n = (255 * (im1 - np.min(im1)) / (np.max(im1) - np.min(im1))).astype(np.uint8)
    ref_n = (255 * (ref - np.min(ref)) / (np.max(ref) - np.min(ref))).astype(np.uint8)
    
    # Apply Gaussian Blur
    im1_n = cv2.GaussianBlur(im1_n, (5, 5), 0)
    ref_n = cv2.GaussianBlur(ref_n, (5, 5), 0)
    
    # Detect edges using Canny
    im1_edges = cv2.Canny(im1_n, 50, 150)
    ref_edges = cv2.Canny(ref_n, 50, 150)
    
    # Find the optimal translation using phase correlation
    shift = cv2.phaseCorrelate(np.float32(im1_edges), np.float32(ref_edges))
    shift_x, shift_y = shift[0]
    
    # Create translation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Translate im1 to align with ref
    # Use borderMode=cv2.BORDER_REPLICATE to replicate the border pixels
    aligned_im1 = cv2.warpAffine(im1, M, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    return aligned_im1

# Align the images
before_time = time()
ag = align(g, b)
ar = align(r, b)

min_height = min(ag.shape[0], ar.shape[0], b.shape[0])
min_width = min(ag.shape[1], ar.shape[1], b.shape[1])

ag = ag[:min_height, :min_width]
ar = ar[:min_height, :min_width]
b = b[:min_height, :min_width]

im_out = np.dstack([ar, ag, b])
im_out = remove_border(im_out)
fname = './out_fname2.jpg'
# skio.imsave(fname, np.uint8(255 * im_out))
print(f'spend: {time() - before_time}')
skio.imshow(im_out)
skio.show()

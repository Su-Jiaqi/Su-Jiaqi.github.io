import cv2
import numpy as np
import skimage.io as skio
from skimage import img_as_float
import matplotlib.pyplot as plt

def remove_border(image):
    """
    去除图像中的边框
    """
    image = np.uint8(image * 255)
    blue_channel = image[:, :, 0]
    binary = cv2.adaptiveThreshold(blue_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_not(binary)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def preprocess_channel(channel):
    return cv2.GaussianBlur(channel, (5, 5), 0)

def normalized_cross_correlation(im1, im2):
    im1 = (im1 - np.mean(im1)) / (np.std(im1) + 1e-8)
    im2 = (im2 - np.mean(im2)) / (np.std(im2) + 1e-8)
    return np.sum(im1 * im2) / (im1.size - 1)

def align_channel(channel, reference, max_shift=5):
    best_ncc = -np.inf
    best_shift = (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(channel, dy, axis=0), dx, axis=1)
            ncc = normalized_cross_correlation(shifted, reference)
            if ncc > best_ncc:
                best_ncc = ncc
                best_shift = (dy, dx)
    return best_shift

def apply_shift(channel, shift):
    dy, dx = shift
    return np.roll(np.roll(channel, dy, axis=0), dx, axis=1)

def pyramid_align(channel, reference, max_shift=5, pyramid_levels=4):
    if pyramid_levels == 0:
        return align_channel(channel, reference, max_shift)
    
    scaled_channel = cv2.resize(channel, (channel.shape[1] // 2, channel.shape[0] // 2))
    scaled_reference = cv2.resize(reference, (reference.shape[1] // 2, reference.shape[0] // 2))
    
    shift = pyramid_align(scaled_channel, scaled_reference, max_shift, pyramid_levels - 1)
    
    shift = (shift[0] * 2, shift[1] * 2)
    
    dy, dx = shift
    shifted_channel = np.roll(np.roll(channel, dy, axis=0), dx, axis=1)
    fine_shift = align_channel(shifted_channel, reference, max_shift=2)
    
    final_shift = (shift[0] + fine_shift[0], shift[1] + fine_shift[1])
    
    return final_shift

# 读取图像
imname = 'F:/UCB/CS180/project1/data/train.tif'
im = skio.imread(imname)
im = img_as_float(im)

# 分离通道
height = np.floor(im.shape[0] / 3.0).astype(np.int64)
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# 归一化和预处理
b = preprocess_channel(b)
g = preprocess_channel(g)
r = preprocess_channel(r)

# 对齐通道 (使用多尺度金字塔)
pyramid_levels = 3  # 减少金字塔层级
print("Aligning G channel to B channel using pyramid...")
shift_g = pyramid_align(g, b, max_shift=10, pyramid_levels=pyramid_levels)
print(f"Shift for G: {shift_g}")

print("Aligning R channel to B channel using pyramid...")
shift_r = pyramid_align(r, b, max_shift=10, pyramid_levels=pyramid_levels)
print(f"Shift for R: {shift_r}")

# 应用位移
aligned_g = apply_shift(g, shift_g)
aligned_r = apply_shift(r, shift_r)

# 裁剪到最小公共尺寸
min_height = min(aligned_g.shape[0], aligned_r.shape[0], b.shape[0])
min_width = min(aligned_g.shape[1], aligned_r.shape[1], b.shape[1])
aligned_g = aligned_g[:min_height, :min_width]
aligned_r = aligned_r[:min_height, :min_width]
b = b[:min_height, :min_width]

# 合成图像
im_out = np.dstack([aligned_r, aligned_g, b])
im_out = remove_border(im_out)  

# 显示结果
plt.figure(figsize=(5, 5))
plt.imshow(im_out)
plt.title(f'Shift for G: {shift_g}, Shift for R: {shift_r}')
plt.axis('off')
plt.show()

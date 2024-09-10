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

def canny_edge_detection(channel):
    """ 使用Canny边缘检测来提取图像关键特征 """
    channel = np.uint8(channel * 255)
    edges = cv2.Canny(channel, 50, 150)
    return edges

def normalized_cross_correlation(im1, im2):
    im1 = (im1 - np.mean(im1)) / (np.std(im1) + 1e-8)
    im2 = (im2 - np.mean(im2)) / (np.std(im2) + 1e-8)
    return np.sum(im1 * im2) / (im1.size - 1)

def align_channel(channel, reference, max_shift=5):
    best_ncc = -np.inf
    best_shift = (0, 0)
    
    h, w = channel.shape
    crop_fraction = 0.5  # 对齐时裁剪至中心区域
    h_crop = int(h * crop_fraction)
    w_crop = int(w * crop_fraction)
    
    # 仅对中心区域进行对比
    channel_center = channel[h//4:h//4 + h_crop, w//4:w//4 + w_crop]
    reference_center = reference[h//4:h//4 + h_crop, w//4:w//4 + w_crop]

    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted = np.roll(np.roll(channel_center, dy, axis=0), dx, axis=1)
            ncc = normalized_cross_correlation(shifted, reference_center)
            if ncc > best_ncc:
                best_ncc = ncc
                best_shift = (dy, dx)
    
    return best_shift

def apply_shift(channel, shift):
    dy, dx = shift
    return np.roll(np.roll(channel, dy, axis=0), dx, axis=1)

# 读取图像
imname = 'F:/UCB/CS180/project1/data/.jpg'
im = skio.imread(imname)
im = img_as_float(im)

# 分离通道
height = np.floor(im.shape[0] / 3.0).astype(np.int64)
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# 提取图像边缘
b_edges = canny_edge_detection(b)
g_edges = canny_edge_detection(g)
r_edges = canny_edge_detection(r)

# 对齐通道
print("Aligning G channel to B channel...")
shift_g = align_channel(g_edges, b_edges, max_shift=10)
print(f"Shift for G: {shift_g}")

print("Aligning R channel to B channel...")
shift_r = align_channel(r_edges, b_edges, max_shift=10)
print(f"Shift for R: {shift_r}")

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
# plt.title('Final Aligned Image')
plt.title(f'Shift for G: {shift_g}, Shift for R: {shift_r}')
plt.axis('off')
plt.show()

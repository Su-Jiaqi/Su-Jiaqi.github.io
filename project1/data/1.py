import numpy as np
import skimage.io as skio
import skimage as sk
from skimage.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 对齐函数，使用 L2 范数 (也可以实现 NCC 作为替代)
def align(img1, img2, max_offset=15):
    # 初始化最佳偏移量和最小误差
    best_offset = (0, 0)
    min_error = float('inf')
    
    # 遍历所有可能的偏移量
    for dx in range(-max_offset, max_offset+1):
        for dy in range(-max_offset, max_offset+1):
            # 平移图像
            shifted_img1 = np.roll(np.roll(img1, dx, axis=1), dy, axis=0)
            # 计算两张图像之间的 L2 范数（或其他度量）
            error = mean_squared_error(shifted_img1, img2)
            # 更新最佳偏移量
            if error < min_error:
                min_error = error
                best_offset = (dx, dy)
                
    # 返回最佳偏移量
    return best_offset

# 输入图像名称
imname = 'cathedral.jpg'

# 读取图像并转换为浮点数
im = skio.imread(imname)
im = sk.img_as_float(im)

# 计算每个通道的高度
height = np.floor(im.shape[0] / 3.0).astype(int)

# 分离颜色通道 (B, G, R)
b = im[:height]
g = im[height:2*height]
r = im[2*height:3*height]

# 对齐 G 通道到 B 通道
g_offset = align(g, b)
g_aligned = np.roll(np.roll(g, g_offset[0], axis=1), g_offset[1], axis=0)

# 对齐 R 通道到 B 通道
r_offset = align(r, b)
r_aligned = np.roll(np.roll(r, r_offset[0], axis=1), r_offset[1], axis=0)

# 叠加通道生成彩色图像
im_out = np.dstack([r_aligned, g_aligned, b])

# 保存生成的图像
output_filename = 'output_color_image.jpg'
skio.imsave(output_filename, im_out)

# 显示生成的彩色图像
plt.imshow(im_out)
plt.axis('off')  # 不显示坐标轴
plt.show()

# 输出偏移量
print(f"G 通道偏移量: {g_offset}")
print(f"R 通道偏移量: {r_offset}")

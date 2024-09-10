1. **导入库**：使用 `cv2` 进行图像处理。使用 `numpy` 进行数组操作。使用 `skimage.io` 读取图像。使用 `matplotlib.pyplot` 显示图像。
2. **边缘检测**：`canny_edge_detection` 函数使用Canny算法提取图像的边缘特征，以便更好地对齐。
3. **归一化互相关 (NCC)**：`normalized_cross_correlation` 函数计算两个图像中心区域的相似度，用于评估对齐效果。
4. **通道对齐**：`align_channel` 函数在指定的最大偏移范围内搜索最佳对齐位置。仅使用图像的中心区域进行对齐，以减少边缘对齐误差。
5. **应用偏移**：`apply_shift` 函数根据计算出的偏移对通道进行平移。
6. **图像读取和处理**：读取图像并转换为浮点格式。将图像分成红、绿、蓝三个通道。
7. **对齐过程**：对绿、红通道分别与蓝通道对齐。打印对齐的偏移量。
8. **合成和显示图像**：将对齐后的通道合成为一个彩色图像。使用 `matplotlib` 显示结果。



The code is designed to align the RGB channels of an image to produce a color image. First, `imread` a picture and covert it into floating-point format. After getting its height, splits it into blue, green, and red channels. To get better results, use `canny_edge_detection` to extract edges from an image channel using the Canny algorithm. Then aligns the green and red channels to the blue channel using the previously defined functions `normalized_cross_correlation` to measure similarity between two images for alignment purposes, and `align_channel` to find the best shift for aligning one channel to a reference channel by comparing their central regions and maximizing the normalized cross-correlation score. After that, applies the calculated shifts. After aligning, the channels are cropped to the smallest common size and combined into a final color image. Finally, the result is displayed using `matplotlib`, with the title 'Final Aligned Image' and the axis turned off for better visualization.

It defines several functions: `canny_edge_detection` to extract edges from an image channel using the Canny algorithm, `normalized_cross_correlation` to measure similarity between two images for alignment purposes, and `align_channel` to find the best shift for aligning one channel to a reference channel by comparing their central regions and maximizing the normalized cross-correlation score. The `apply_shift` function applies the calculated shift to the channel. 

The main part of the code reads an image, converts it to floating-point format, and splits it into blue, green, and red channels. It then extracts the edges from each channel, aligns the green and red channels to the blue channel using the previously defined functions, and applies the calculated shifts. After aligning, the channels are cropped to the smallest common size and combined into a final color image. Finally, the result is displayed using `matplotlib`, with the title 'Final Aligned Image' and the axis turned off for better visualization.


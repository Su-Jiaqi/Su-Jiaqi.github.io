import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numba import jit
import time
import random
import cv2
import numpy as np
from skimage.feature import corner_harris, peak_local_max

# Constants
ransac_max_dist = 2  # RANSAC distance threshold
n = 12  # Number of points to calculate the homography

# Paths
cwd = os.path.dirname(os.path.abspath(__file__))  # Current script path
figures_path = os.path.join(cwd, 'result')
img_path = os.path.join(cwd, 'figures')
img_to_warp_path = os.path.join(img_path, '18.jpg')
img_base_path = os.path.join(img_path, '19.jpg')
src_pts_file = os.path.join(cwd, './result/18_points.txt')
dst_pts_file = os.path.join(cwd, './result/19_points.txt')

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords

def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """
    
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    if dimx != dimc:
        raise ValueError('Data dimension does not match dimension of centers')

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

# 绘制角点并保存彩色图像
def plot_save_corners(image, corners, save_path):
    # 绘制角点在彩色图像上
    for y, x in zip(corners[0], corners[1]):
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)  # 使用红色点
    
    # 保存带有角点的图像
    cv2.imwrite(save_path, image)
    print(f"保存成功：{save_path}")

# ANMS 实现，使用 dist2 函数计算距离
def adaptive_non_maximal_suppression(corner_strength, corners, num_points=500, crobust=0.9):
    # 存储 (y, x, 强度) 形式的兴趣点
    interest_points = [(y, x, corner_strength[y, x]) for y, x in zip(corners[0], corners[1])]
    # 按强度降序排序
    interest_points = sorted(interest_points, key=lambda point: point[2], reverse=True)

    # 将兴趣点坐标转换为数组，并按顺序保存
    points = np.array([[y, x] for y, x, _ in interest_points])

    # 计算兴趣点的抑制半径
    radii = np.full(len(points), float('inf'))

    # 依次计算抑制半径
    for i in range(len(points)):
        # 选择当前点及其所有强度更高的点，计算距离
        current_point = points[i].reshape(1, -1)
        stronger_points = points[:i]
        
        # 仅当存在强度更大的邻居时，才计算抑制半径
        if stronger_points.size > 0:
            # 计算当前点与所有更强点之间的距离
            distances = np.sqrt(dist2(current_point, stronger_points)).flatten()
            radii[i] = distances.min()

    # 按半径降序选择前 num_points 个兴趣点
    selected_points = sorted(zip(radii, interest_points), key=lambda x: x[0], reverse=True)[:num_points]
    selected_points = [(int(x), int(y)) for _, (x, y, _) in selected_points]
    
    return selected_points

# 绘制角点并保存彩色图像
def plot_and_save_corners(image, points, save_path):
    # 绘制兴趣点在彩色图像上
    for y, x in points:
        cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # 使用红色点，半径 3
    
    # 保存带有兴趣点的图像
    cv2.imwrite(save_path, image)
    print(f"保存成功：{save_path}")

# 特征描述符提取函数
def extract_feature_descriptors(image, points, window_size=40, patch_size=8, spacing=5):
    descriptors = []
    half_window = window_size // 2

    # 遍历每个兴趣点
    for y, x in points:
        # 提取中心点周围的 40x40 窗口，确保不超出边界
        if (y - half_window < 0 or y + half_window >= image.shape[0] or 
            x - half_window < 0 or x + half_window >= image.shape[1]):
            continue  # 跳过边界处的兴趣点

        # 提取 40x40 的大窗口
        window = image[y - half_window:y + half_window, x - half_window:x + half_window]

        # 缩小采样，获得 8x8 的采样点
        patch = cv2.resize(window, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

        # 归一化：偏置-增益归一化，设均值为 0，标准差为 1
        patch = (patch - np.mean(patch)) / (np.std(patch) + 1e-5)

        # 将归一化后的描述符添加到列表
        descriptors.append(patch.flatten())

    return np.array(descriptors)

# 特征匹配函数，添加调试信息
def match_features(descriptors1, descriptors2, threshold=0.5):
    matches = []
    print(f"匹配的阈值：{threshold}\n")
    # print("匹配细节：")

    # 遍历 descriptors1 的每个描述符
    for i, desc1 in enumerate(descriptors1):
        # 计算当前描述符与 descriptors2 中所有描述符的距离
        distances = np.linalg.norm(descriptors2 - desc1, axis=1)
        
        # 找到第一和第二最近邻的距离和索引
        nearest_idx = np.argmin(distances)
        sorted_indices = np.argsort(distances)
        second_nearest_idx = sorted_indices[1]
        
        # 计算误差比值
        ratio = distances[nearest_idx] / distances[second_nearest_idx]

        # 输出调试信息
        # print(f"描述符 {i} 与最近邻距离：{distances[nearest_idx]:.4f}, 次近邻距离：{distances[second_nearest_idx]:.4f}, 比值：{ratio:.4f}")

        # 应用 Lowe 的阈值筛选
        if ratio < threshold:
            matches.append((i, nearest_idx))
            # print(f"匹配成功：描述符 {i} -> {nearest_idx} (比值满足条件)\n")
        # else:
            # print(f"匹配失败：描述符 {i} -> {nearest_idx} (比值不满足条件)\n")
    
    return matches

def save_matched_points(anms_points1, anms_points2, matches, src_file="./result/19_points.txt", dst_file="./result/18_points.txt"):
    # 根据 matches 中的对应关系，获取点的坐标，并反转 x 和 y 顺序
    src_points = [(anms_points1[i][1], anms_points1[i][0]) for i, j in matches]  # 第一个图片的匹配点
    dst_points = [(anms_points2[j][1], anms_points2[j][0]) for i, j in matches]  # 第二个图片的匹配点

    # 保存源点和目标点到对应的 txt 文件
    np.savetxt(src_file, src_points, fmt="%.4f", comments="")
    np.savetxt(dst_file, dst_points, fmt="%.4f", comments="")

    print(f"匹配的源点保存在 {src_file}")
    print(f"匹配的目标点保存在 {dst_file}")

# 绘制匹配的点对
def plot_and_save_matched_points(image1, src_points, image2, dst_points, save_path):
    # 创建一个新图像以并排显示两张图片
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])
    new_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # 确保图像是三通道
    if len(image1.shape) == 2:  # 如果 image1 是单通道灰度图像
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:  # 如果 image2 是单通道灰度图像
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # 将两张图像放到一起
    new_image[:image1.shape[0], :image1.shape[1]] = image1
    new_image[:image2.shape[0], image1.shape[1]:] = image2
    
    # 绘制匹配的点对
    for (x1, y1), (x2, y2) in zip(src_points, dst_points):
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2) + image1.shape[1], int(y2)  # 将第二张图像的坐标向右平移

        # 画点和连线
        cv2.circle(new_image, (x1, y1), 3, (0, 0, 255), -1)
        cv2.circle(new_image, (x2, y2), 3, (0, 0, 255), -1)
        cv2.line(new_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 保存图像
    cv2.imwrite(save_path, new_image)
    print(f"匹配点图像已保存到: {save_path}")

def load_images():
    """Load images from the specified paths."""
    print("Loading images...")
    img_base = cv2.imread(img_base_path, 0)  # Base gray image
    img_to_warp = cv2.imread(img_to_warp_path, 0)  # Image to be warped
    img_base_rgb = cv2.cvtColor(cv2.imread(img_base_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img_to_warp_rgb = cv2.cvtColor(cv2.imread(img_to_warp_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    print("Images loaded successfully.")
    return img_base, img_to_warp, img_base_rgb, img_to_warp_rgb

def select_corresponding_points(image1, image2, num_points):
    """Manually select corresponding points from two images using mouse clicks."""
    print(f"Manually selecting {num_points} corresponding points...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax1, ax2 = axes
    ax1.imshow(image1)
    ax1.set_title("Image 1")
    ax1.axis('on')
    
    ax2.imshow(image2)
    ax2.set_title("Image 2")
    ax2.axis('on')
    
    points1 = []
    points2 = []

    def onclick(event):
        if event.inaxes == ax1 and len(points1) < num_points:  # If click on Image 1
            points1.append([event.xdata, event.ydata])
            print(f"Point on Image 1: {event.xdata}, {event.ydata}")
            ax1.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
        elif event.inaxes == ax2 and len(points2) < num_points:  # If click on Image 2
            points2.append([event.xdata, event.ydata])
            print(f"Point on Image 2: {event.xdata}, {event.ydata}")
            ax2.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
        
        if len(points1) == num_points and len(points2) == num_points:
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("Point selection complete.")
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)

def draw_correspondences(image1, image2, points1, points2):
    # Create a combined image to display both images side by side
    combined_width = image1.shape[1] + image2.shape[1]
    combined_height = max(image1.shape[0], image2.shape[0])
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Place the two images side by side
    combined_image[:image1.shape[0], :image1.shape[1]] = image1
    combined_image[:image2.shape[0], image1.shape[1]:] = image2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(combined_image)
    ax.set_title("Corresponding Points with Lines")
    ax.axis('off')

    # Draw the points
    for p1, p2 in zip(points1, points2):
        # Draw points on image 1
        ax.plot(p1[0], p1[1], 'ro')
        # Draw points on image 2 (shifted by image1's width)
        ax.plot(p2[0] + image1.shape[1], p2[1], 'ro')
        
        # Draw a line connecting the corresponding points
        line = plt.Line2D([p1[0], p2[0] + image1.shape[1]], [p1[1], p2[1]], color='blue')
        ax.add_line(line)
        
    # Save the combined image with correspondences
    correspondence_image_path = os.path.join(figures_path, 'correspondences.jpg')
    fig.savefig(correspondence_image_path)
    
    plt.show()

def calculate_homography(src_pts, dst_pts):
    """Calculate the homography matrix manually without using cv2.findHomography."""
    # print("Calculating homography...")
    A = []
    
    for i in range(len(src_pts)):
        x, y = src_pts[i][0], src_pts[i][1]
        x_prime, y_prime = dst_pts[i][0], dst_pts[i][1]
        
        # Add equations for this pair of points
        A.append([x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime, -x_prime])
        A.append([0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime, -y_prime])
    
    # Convert A to a numpy array and use SVD to solve for H
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    
    # Normalize so that H[2,2] is 1
    H = H / H[2, 2]
    # print("Homography calculated.")
    return H

def ransac_homography(src_pts, dst_pts, num_iterations=5000, threshold=2.0, min_inliers=4,n=0):
    """RANSAC algorithm to find a robust homography matrix."""
    max_inliers = 0
    best_H = None

    for i in range(num_iterations):
        # Step 1: 随机选择 4 对点
        indices = random.sample(range(len(src_pts)), 4)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # Step 2: 计算单应性矩阵 H
        try:
            H = calculate_homography(src_sample, dst_sample)
        except np.linalg.LinAlgError:
            # 如果计算 H 失败（例如矩阵不可逆），跳过该次迭代
            n += 1
            continue
        
        # Step 3: 将 src_pts 投影到目标空间并计算误差
        src_pts_homogeneous = np.hstack((src_pts, np.ones((len(src_pts), 1))))
        projected_pts = (H @ src_pts_homogeneous.T).T
        projected_pts /= projected_pts[:, 2].reshape(-1, 1)  # 归一化

        # Step 4: 计算每个投影点与目标点之间的欧氏距离
        distances = np.linalg.norm(projected_pts[:, :2] - dst_pts, axis=1)
        
        # Step 5: 计算内点数量
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Step 6: 检查是否找到新的最佳模型
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers_src = src_pts[inliers]
            best_inliers_dst = dst_pts[inliers]
            best_H = H
            print(f"Iteration {i+1}: Found new best with {num_inliers} inliers")
            
    print(f"Failed to calculate H {n} times.")
    # Step 7: 如果找到足够的内点，重新基于所有内点计算单应性矩阵
    if max_inliers >= min_inliers:
        best_H = calculate_homography(best_inliers_src, best_inliers_dst)
        print(f"RANSAC completed with {max_inliers} inliers.")
    else:
        raise ValueError("RANSAC could not find a sufficient number of inliers.")

    return best_H

def compute_canvas_size(H, img_base_shape):
    """Compute the canvas size needed to contain both images."""
    print("Computing canvas size...")
    (y, x) = img_base_shape
    pts = [
        np.array([0, 0, 1], dtype=np.float32),  # Top-left
        np.array([x, 0, 1], dtype=np.float32),  # Top-right
        np.array([0, y, 1], dtype=np.float32),  # Bottom-left
        np.array([x, y, 1], dtype=np.float32)   # Bottom-right
    ]
    
    min_x, min_y, max_x, max_y = None, None, None, None
    
    for pt in pts:
        hp = np.linalg.inv(H) @ pt
        hp /= hp[2]  # Normalize
        x, y = hp[0], hp[1]
        min_x = min(min_x, x) if min_x is not None else x
        min_y = min(min_y, y) if min_y is not None else y
        max_x = max(max_x, x) if max_x is not None else x
        max_y = max(max_y, y) if max_y is not None else y
    
    min_x, min_y = min(0, min_x), min(0, min_y)
    max_x, max_y = max(max_x, img_base_shape[1]), max(max_y, img_base_shape[0])
    
    T = np.identity(3, dtype=np.float32)
    if min_x < 0:
        T[0, 2] = -min_x
    if min_y < 0:
        T[1, 2] = -min_y
    
    print("Canvas size computed.")
    return T, int(math.ceil(max_x - min_x)), int(math.ceil(max_y - min_y))

@jit(nopython=True)
def warp_pixel(src_x, src_y, img):
    if 0 <= src_x < img.shape[1] - 1 and 0 <= src_y < img.shape[0] - 1:
        x0, y0 = int(src_x), int(src_y)
        x1, y1 = x0 + 1, y0 + 1
        
        wa = (x1 - src_x) * (y1 - src_y)
        wb = (src_x - x0) * (y1 - src_y)
        wc = (x1 - src_x) * (src_y - y0)
        wd = (src_x - x0) * (src_y - y0)
        
        return (wa * img[y0, x0] + wb * img[y0, x1] + 
                wc * img[y1, x0] + wd * img[y1, x1]).astype(np.uint8)
    return np.zeros(3, dtype=np.uint8)

def manual_warp_perspective(img, M, output_size):
    height, width = output_size[::-1]
    warped_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    M_inv = np.linalg.inv(M)
    
    start_time = time.time()
    for y in range(height):
        if y % 100 == 0:
            print(f"Processing row {y}/{height}, Time elapsed: {time.time() - start_time:.2f}s")
        for x in range(width):
            src = M_inv.dot([x, y, 1])
            src = src[:2] / src[2]
            warped_img[y, x] = warp_pixel(src[0], src[1], img)
    
    return warped_img

def warp_images(img_base_rgb, img_to_warp_rgb, H, T, img_w, img_h):
    print("Warping base image...")
    img_base_translated = manual_warp_perspective(img_base_rgb, T, (img_w, img_h))
    print("Base image warped.")
    
    base_translated_image_path = os.path.join(figures_path, 'base_translated_image.jpg')
    cv2.imwrite(base_translated_image_path, cv2.cvtColor(img_base_translated, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(img_base_translated)
    plt.title('Warped Base Image')
    plt.show()
    
    print("Warping second image...")
    M_inv = T @ np.linalg.inv(H)
    warped_img = manual_warp_perspective(img_to_warp_rgb, M_inv, (img_w, img_h))
    print("Second image warped.")
    
    warped_image_path = os.path.join(figures_path, 'warped_image.jpg')
    cv2.imwrite(warped_image_path, cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(warped_img)
    plt.title('Warped Image to Be Mapped')
    plt.show()
 
    return img_base_translated, warped_img

def rectify_image(image, pts_source):
    # 根据选择的点自动计算目标矩形
    width_top = np.linalg.norm(pts_source[0] - pts_source[1])
    width_bottom = np.linalg.norm(pts_source[2] - pts_source[3])
    width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(pts_source[0] - pts_source[2])
    height_right = np.linalg.norm(pts_source[1] - pts_source[3])
    height = max(int(height_left), int(height_right))
    
    # 自动生成 "横平竖直" 的目标矩形
    pts_rectified = np.array([
        [0, 0], 
        [width - 1, 0], 
        [width - 1, height - 1], 
        [0, height - 1]
    ], dtype=np.float32)
    
    # 计算单应性矩阵
    H = calculate_homography(pts_source, pts_rectified)
    rectified = warp_images(image, H)
    return rectified

def blend_images(img_base_translated, warped_img):
    """Blend the two images together using masking."""
    canvas = np.zeros_like(img_base_translated)
    _, mask = cv2.threshold(warped_img, 0, 255, cv2.THRESH_BINARY_INV)
    pre_final_img = cv2.add(canvas, img_base_translated, mask=mask[:, :, 0], dtype=cv2.CV_8U)
    return cv2.add(pre_final_img, warped_img, dtype=cv2.CV_8U)

def save_and_show_result(img_final):
    """Save and display the resulting blended image."""
    plt.figure()
    plt.imshow(img_final)
    plt.title('Result')
    result_path = os.path.join(figures_path, 'result.jpg')
    cv2.imwrite(result_path, cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB))
    plt.show()

def main():    
    # 加载彩色图像并转换为灰度
    image1 = cv2.imread("./figures/18.jpg")
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.imread("./figures/19.jpg")
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 检测角点并使用 ANMS 选择兴趣点
    corner_strength1, corners1 = get_harris_corners(gray_image1, edge_discard=20)
    # 显示图像并保存
    # plot_save_corners(image1, corners1, "result/18_corners.jpg")
    anms_points1 = adaptive_non_maximal_suppression(corner_strength1, corners1, num_points=500)
    
    corner_strength2, corners2 = get_harris_corners(gray_image2, edge_discard=20)
    # 显示图像并保存
    # plot_save_corners(image2, corners2, "result/19_corners.jpg")
    anms_points2 = adaptive_non_maximal_suppression(corner_strength2, corners2, num_points=500)

    # 显示图像并保存带有 ANMS 的兴趣点
    plot_and_save_corners(image1, anms_points1, "result/18_corners_anms.jpg")
    plot_and_save_corners(image2, anms_points2, "result/19_corners_anms.jpg")
    
    # 提取特征描述符
    descriptors1 = extract_feature_descriptors(gray_image1, anms_points1)
    descriptors2 = extract_feature_descriptors(gray_image2, anms_points2)

    # 匹配特征点
    matches = match_features(descriptors1, descriptors2)

    # 输出匹配结果
    print(f"找到 {len(matches)} 对匹配点。")
    
    # 保存匹配的点到txt文件
    save_matched_points(anms_points1, anms_points2, matches)

    # 读取matches_src_points.txt和matches_dst_points.txt文件，绘制匹配的点对
    src_points = np.loadtxt("./result/18_points.txt")
    dst_points = np.loadtxt("./result/19_points.txt")

    # 显示匹配的点对
    plot_and_save_matched_points(image1, src_points, image2, dst_points, "result/18_19_matched_points.jpg")
    
    # Step 1: Load images
    print("Step 1: Load images.")
    img_base, img_to_warp, img_base_rgb, img_to_warp_rgb = load_images()
    
    # Step 2: Load corresponding points from txt files
    print("Step 2: Load corresponding points from txt files.")
    if os.path.exists(src_pts_file) and os.path.exists(dst_pts_file):
        src_pts = np.loadtxt(src_pts_file)
        dst_pts = np.loadtxt(dst_pts_file)
        print("Loaded saved points from txt files.")
    else:
        raise FileNotFoundError("Source or destination points file not found. Please create '12_points.txt' and '13_points.txt' with corresponding points.")

    # Step 3: Calculate homography
    print("Step 3: Calculate homography.")
    # H = calculate_homography(src_pts, dst_pts)
    H = ransac_homography(src_pts, dst_pts)
    print("Homography matrix:")
    print(H)
    
    # Display correspondences and save image
    print("Drawing and saving correspondence image.")
    draw_correspondences(img_base_rgb, img_to_warp_rgb, src_pts, dst_pts)
    
    # Step 4: Compute canvas size
    print("Step 4: Compute canvas size.")
    T, img_w, img_h = compute_canvas_size(H, img_base.shape)
    
    # Step 5: Warp images
    print("Step 5: Warp images.")
    img_base_translated, warped_img = warp_images(img_base_rgb, img_to_warp_rgb, H, T, img_w, img_h)
    
    # 将img_base_translated 和 warped_img 取最大的大小，reisze,空出的部分用padding补齐
    h1,w1 = img_base_translated.shape[:2]
    h2,w2 = warped_img.shape[:2]
    nWidth = max(w1,w2)
    nHeight = max(h1,h2)
    img_base_translated = cv2.resize(img_base_translated,(nWidth,nHeight))
    warped_img = cv2.resize(warped_img,(nWidth,nHeight))
    
    # Step 6: Blend images
    print("Step 6: Blend images.")
    img_final = blend_images(img_base_translated, warped_img)
    
    # Step 7: Save and display result
    print("Step 7: Save and display result.")
    save_and_show_result(img_final)
    print("Processing completed.")

if __name__ == "__main__":
    main()

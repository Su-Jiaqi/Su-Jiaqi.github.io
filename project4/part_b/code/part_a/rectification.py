import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    if img.shape[2] == 4:  # If image has 4 channels (i.e., RGBA), convert to 3-channel RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def select_points_for_rectification(image, num_points):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_title("Select Points for Rectification")
    ax.axis('on')
    
    points = []

    # Mouse click event handler
    def onclick(event):
        if event.inaxes == ax and len(points) < num_points:  # If click on Image
            points.append([event.xdata, event.ydata])
            print(f"Point: {event.xdata}, {event.ydata}")
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
        
        if len(points) == num_points:
            plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return np.array(points)

def computeH(im1_pts, im2_pts):
    n = im1_pts.shape[0]
    A = []
    
    for i in range(n):
        x, y = im1_pts[i, 0], im1_pts[i, 1]
        x_prime, y_prime = im2_pts[i, 0], im2_pts[i, 1]
        
        A.append([x, y, 1, 0, 0, 0, -x * x_prime, -y * x_prime, -x_prime])
        A.append([0, 0, 0, x, y, 1, -x * y_prime, -y * y_prime, -y_prime])
    
    A = np.array(A)
    U, S, V = np.linalg.svd(A)
    h = V[-1, :] / V[-1, -1]
    H = h.reshape(3, 3)
    
    return H

def compute_output_shape(im, H):
    h, w = im.shape[:2]
    corners = np.array([
        [0, 0, 1],          # Top-left
        [w, 0, 1],          # Top-right
        [0, h, 1],          # Bottom-left
        [w, h, 1]           # Bottom-right
    ]).T

    transformed_corners = H @ corners
    transformed_corners /= transformed_corners[2, :]  # Normalize

    min_x, min_y = np.min(transformed_corners[0]), np.min(transformed_corners[1])
    max_x, max_y = np.max(transformed_corners[0]), np.max(transformed_corners[1])
    
    output_width = int(np.ceil(max_x - min_x))
    output_height = int(np.ceil(max_y - min_y))
    
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0

    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    
    adjusted_H = translation_matrix @ H
    return (output_height, output_width), adjusted_H

def warpImage(im, H):
    output_shape, adjusted_H = compute_output_shape(im, H)
    h, w = output_shape
    warped_image = cv2.warpPerspective(im, adjusted_H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped_image

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
    H = computeH(pts_source, pts_rectified)
    rectified = warpImage(image, H)
    return rectified

def main():
    image1_path = './figures/origin/10.jpg'
    image1 = load_image(image1_path)
    
    # 手动选择四个点进行校正
    print("Please select four points for rectification...")
    pts_source = select_points_for_rectification(image1, 4)
    
    # 进行图像校正
    rectified_image = rectify_image(image1, pts_source)
    
    # 保存校正后的图像
    cv2.imwrite('./figures/result_rec/10.jpg', cv2.cvtColor(rectified_image, cv2.COLOR_RGB2BGR))
    
    # 修复颜色顺序：使用 cv2 转换 BGR 到 RGB
    plt.imshow(rectified_image)
    plt.title("Rectified Image")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()

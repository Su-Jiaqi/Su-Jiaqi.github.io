import numpy as np
import cv2
import math
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from numba import jit
import time

# Constants
ransac_max_dist = 2  # RANSAC distance threshold
n = 12  # Number of points to calculate the homography

# Paths
cwd = os.path.dirname(os.path.abspath(__file__))  # Current script path
figures_path = os.path.join(cwd, 'figures')
img_to_warp_path = os.path.join(figures_path, '1.jpg')
img_base_path = os.path.join(figures_path, '2.jpg')
src_pts_file = os.path.join(cwd, 'src_pts.npy')
dst_pts_file = os.path.join(cwd, 'dst_pts.npy')

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
    print("Calculating homography...")
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
    print("Homography calculated.")
    return np.linalg.inv(H)

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
        hp = H @ pt
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
    M_inv = T @ H
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
    # Step 1: Load images
    print("Step 1: Load images.")
    img_base, img_to_warp, img_base_rgb, img_to_warp_rgb = load_images()
    
    # Step 2: Manually select corresponding points if not already saved
    print("Step 2: Select corresponding points or load from file.")
    if not os.path.exists(src_pts_file) or not os.path.exists(dst_pts_file):
        print(f"Please select {n} corresponding points between Image 1 and Image 2...")
        src_pts, dst_pts = select_corresponding_points(img_base_rgb, img_to_warp_rgb, n)
        # Save the points to file for future use
        np.save(src_pts_file, src_pts)
        np.save(dst_pts_file, dst_pts)
        print("Points saved to file.")
    else:
        # Load previously saved points
        src_pts = np.load(src_pts_file)
        dst_pts = np.load(dst_pts_file)
        print("Loaded saved points.")

    # Step 3: Calculate homography
    print("Step 3: Calculate homography.")
    H = calculate_homography(src_pts, dst_pts)
    
    # 显示对应点并保存图片
    print("Drawing and saving correspondence image.")
    draw_correspondences(img_base_rgb, img_to_warp_rgb, src_pts, dst_pts)
    
    # Step 4: Compute canvas size
    print("Step 4: Compute canvas size.")
    T, img_w, img_h = compute_canvas_size(H, img_base.shape)
    
    # Step 5: Warp images
    print("Step 5: Warp images.")
    img_base_translated, warped_img = warp_images(img_base_rgb, img_to_warp_rgb, H, T, img_w, img_h)
    
    # Step 6: Blend images
    print("Step 6: Blend images.")
    img_final = blend_images(img_base_translated, warped_img)
    
    # Step 7: Save and display result
    print("Step 7: Save and display result.")
    save_and_show_result(img_final)
    print("Processing completed.")

if __name__ == "__main__":
    main()

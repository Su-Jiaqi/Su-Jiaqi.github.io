# Image Warping and Blending Project

This project implements an image processing pipeline that performs image warping, rectification, and blending. The program allows users to manually select corresponding points between two images, calculate the homography matrix, warp images to correct perspective, and blend them into a seamless mosaic.

## Features

1. **Load Images**: Load two images from the specified paths for processing.
2. **Select Corresponding Points**: Manually select corresponding points between the two images to aid in calculating the homography matrix.
3. **Calculate Homography**: Derive the transformation matrix to map points from one image to another.
4. **Warp Images**: Adjust the perspective of images using the homography matrix and correct for any distortions.
5. **Rectify Images**: Automatically calculate the target rectangle to rectify a distorted image.
6. **Blend Images**: Seamlessly combine two images into a mosaic using masking techniques.
7. **Save and Display Results**: Save the final blended image and display it to the user.

## Prerequisites

Make sure you have the following libraries installed:

- `numpy`
- `opencv-python`
- `matplotlib`
- `scipy`
- `numba`



## How It Works

### Step-by-Step Process:

1. **Load Images**: The program reads two images from the paths defined at the beginning of the script.
2. **Select Points**: If not already saved, users can manually select corresponding points in the two images using mouse clicks. These points will be saved for future use.
3. **Homography Calculation**: The selected points are used to calculate the homography matrix, which determines how to transform one image to align with the other.
4. **Canvas Size Computation**: The program determines the appropriate canvas size to accommodate the warped images.
5. **Image Warping**: Using the homography matrix, the program warps the images to adjust their perspectives.
6. **Image Blending**: After warping, the images are blended using masking techniques to create a smooth mosaic.
7. **Save and Display**: The final result is saved as an image file and displayed to the user.

## Code Explanation

### Main Functions

- **`load_images()`**: Reads and returns grayscale and RGB versions of the base and target images.
- **`select_corresponding_points(image1, image2, num_points)`**: Allows manual selection of corresponding points from two images.
- **`calculate_homography(src_pts, dst_pts)`**: Computes the homography matrix from the selected points.
- **`compute_canvas_size(H, img_base_shape)`**: Determines the canvas size needed to accommodate the warped images.
- **`manual_warp_perspective(img, M, output_size)`**: Warps the image based on the provided transformation matrix.
- **`warp_images(img_base_rgb, img_to_warp_rgb, H, T, img_w, img_h)`**: Applies warping to the base and target images, saving the results.
- **`blend_images(img_base_translated, warped_img)`**: Combines two warped images into a single mosaic.
- **`rectify_image(image, pts_source)`**: Rectifies a distorted image by mapping it to a standard rectangle.
- **`save_and_show_result(img_final)`**: Saves and displays the final blended image.

## Running the Program

1. Place the images you want to process in the `figures` directory. Rename them as `1.jpg` (image to warp) and `2.jpg` (base image).

2. Run the script:

   ```bash
   python main.py
   ```

3. Follow the instructions to manually select corresponding points if required.

4. The program will output the following files:

   - Warped images and correspondence images in the `figures` directory.
   - The final blended mosaic image as `result.jpg` in the `figures` directory.
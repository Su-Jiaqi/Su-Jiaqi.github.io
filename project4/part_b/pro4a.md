# CS180 Project 4

## Part A: Image Warping and Mosaicing

### 1. Images 

| Image 1                                                      | Image 2                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![2](F:\UCB\CS180\project4\part_a\code\figures\result1\2.jpg) | ![1](F:\UCB\CS180\project4\part_a\code\figures\result1\1.jpg) |
| ![2](F:\UCB\CS180\project4\part_a\code\figures\result2\2.jpg) | ![1](F:\UCB\CS180\project4\part_a\code\figures\result2\1.jpg) |
| ![2](F:\UCB\CS180\project4\part_a\code\figures\result3\2.jpg) | ![1](F:\UCB\CS180\project4\part_a\code\figures\result3\1.jpg) |

| Image 1                                                      | Image 2                                                      | Image 3                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![7](F:\UCB\CS180\project4\part_a\code\figures\origin\7.jpg) | ![8](F:\UCB\CS180\project4\part_a\code\figures\origin\8.jpg) | ![9](F:\UCB\CS180\project4\part_a\code\figures\origin\9.jpg) |



### 2. Recovering Homographies

#### (1) Understanding Homography

A homography matrix $H$ is a $3×3$ matrix used to transform a point $ p=(x,y,1)^T $  in image 1 to a corresponding point $p′=(x′,y′,1)^T$ in image 2:
$$
p′=H⋅p
$$
where $H$ is:
$$
H = \begin{bmatrix}
h_{11} & h_{12} & h_{13} \\
h_{21} & h_{22} & h_{23} \\
h_{31} & h_{32} & h_{33}
\end{bmatrix}
$$
Since the matrix $H$ has a scale-invariant degree of freedom, we can set $h_{33} = 1$ , leaving 8 unknowns.



#### (2) Setting Up the Linear Equations

To compute $H$, you need at least 4 pairs of corresponding points, which will allow you to set up a linear system. By selecting feature points in both images manually or automatically, you can obtain these corresponding points.

For each point pair $(p,p′)$, the relationship is:
$$
\begin{bmatrix}
x' \\
y' \\
1
\end{bmatrix} 
= H 
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
$$
This expands to:
$$
x' = \frac{h_{11}x + h_{12}y + h_{13}}{h_{31}x + h_{32}y + h_{33}}
\\
y' = \frac{h_{21}x + h_{22}y + h_{23}}{h_{31}x + h_{32}y + h_{33}}
$$

We can rearrange these equations to get:
$$
x'(h_{31}x + h_{32}y + h_{33}) = h_{11}x + h_{12}y + h_{13}
\\
y'(h_{31}x + h_{32}y + h_{33}) = h_{21}x + h_{22}y + h_{23}
$$
Further expanding and rearranging gives:
$$
h_{11}x + h_{12}y + h_{13} - x'(h_{31}x + h_{32}y + h_{33}) = 0
\\
h_{21}x + h_{22}y + h_{23} - y'(h_{31}x + h_{32}y + h_{33}) = 0
$$
Each point pair $ (x, y) $ and $(x', y')$ generates two linear equations. By reorganizing, you can express this as:
$$
A \cdot \mathbf{h} = 0
$$
where $\mathbf{h} $  is a vector containing the 8 unknowns from the matrix $H$.

#### (3) Creating Matrix $A$ and Vector $\mathbf{b}$

Each point pair provides two equations in the following format:
$$
\begin{bmatrix}
x & y & 1 & 0 & 0 & 0 & -x \cdot x' & -y \cdot x' \\
0 & 0 & 0 & x & y & 1 & -x \cdot y' & -y \cdot y'
\end{bmatrix}
\begin{bmatrix}
h_{11} \\
h_{12} \\
h_{13} \\
h_{21} \\
h_{22} \\
h_{23} \\
h_{31} \\
h_{32}
\end{bmatrix}
=
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
$$
By doing this for $n$ pairs of points, you will create a $2n×8$ matrix $A$ and a $2n×1$ vector $\mathbf{b}$, which can be solved using least squares.

#### (4) Solve Using Least Squares

Express the system as $A\mathbf{h} = \mathbf{b}$ and use `numpy`'s `np.linalg.lstsq()` to solve.

When there are more than $n>4$ pairs of points, the system of equations we obtain has more equations than unknowns, creating an **overdetermined system**. Since the data may contain noise, directly solving this system can be unstable, so we use the **least squares method**.

The goal of the least squares method is to find a vector $\mathbf{h}$ such that the equation $A \mathbf{h} \approx \mathbf{b}$ has minimal error. This means we want to minimize the following objective function:
$$
\min_{\mathbf{h}} \| A \mathbf{h} - \mathbf{b} \|^2
$$
To stably solve for $\mathbf{h}$, we use **Singular Value Decomposition (SVD)** to decompose the matrix $A$:
$$
A = U \Sigma V^{T}
$$
where:

- $U$ and $V$ are orthogonal matrices,
- $\Sigma$ is a diagonal matrix containing the singular values.

#### （6）Result

<img src="F:\UCB\CS180\project4\part_a\code\figures\result1\correspondences.jpg" alt="correspondences" style="zoom: 200%;" />

![correspondences](F:\UCB\CS180\project4\part_a\code\figures\result2\correspondences.jpg)

![correspondences](F:\UCB\CS180\project4\part_a\code\figures\result3\correspondences.jpg)



| Image 1                                                      | Correspondence                                               | Image 2                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="F:\UCB\CS180\project4\part_a\code\figures\result1\2.jpg" alt="2" style="zoom: 40%;" /> | ![corres](F:\UCB\CS180\project4\part_a\code\figures\result1\corres.png) | <img src="F:\UCB\CS180\project4\part_a\code\figures\result1\1.jpg" alt="1" style="zoom:40%;" /> |
| ![2](F:\UCB\CS180\project4\part_a\code\figures\result2\2.jpg) | ![corre](F:\UCB\CS180\project4\part_a\code\figures\result2\corre.png) | ![1](F:\UCB\CS180\project4\part_a\code\figures\result2\1.jpg) |
| ![2](F:\UCB\CS180\project4\part_a\code\figures\result3\2.jpg) | ![corre](F:\UCB\CS180\project4\part_a\code\figures\result3\corre.png) | ![1](F:\UCB\CS180\project4\part_a\code\figures\result3\1.jpg) |



### 3. Warp the Image

Using inverse mapping, we determine the source image position $(x, y)$ for each pixel in the target image. Since the computed $(x, y)$ are usually floating-point values, interpolation is required.

#### (1) Bilinear Interpolation

When $(x, y)$ are floating-point coordinates, **bilinear interpolation** is used to determine the pixel value from the source image. The formula is as follows:

Suppose $(x, y)$ falls between integer grid points $(x_0, y_0)$ and $(x_1, y_1)$, where:
$$
x_0 = \lfloor x \rfloor, \quad x_1 = x_0 + 1, \quad y_0 = \lfloor y \rfloor, \quad y_1 = y_0 + 1
$$
Weights are calculated as:
$$
w_a = (x_1 - x) \cdot (y_1 - y), \quad w_b = (x - x_0) \cdot (y_1 - y)
$$

$$
w_c = (x_1 - x) \cdot (y - y_0), \quad w_d = (x - x_0) \cdot (y - y_0)
$$

The interpolated pixel value is then:
$$
I(x, y) = w_a \cdot I(x_0, y_0) + w_b \cdot I(x_1, y_0) + w_c \cdot I(x_0, y_1) + w_d \cdot I(x_1, y_1)
$$

#### (2) Mapping and Transformation

Combining inverse mapping and interpolation, the process for each pixel $(x', y')$ in the target image involves:

1. Using the inverse matrix $H^{-1}$ to find the corresponding source coordinates $(x, y)$.
2. Applying bilinear interpolation to compute the pixel value from the source image.
3. Assigning the computed pixel value to the position $(x', y')$ in the target image.

#### (5) Adjusting Canvas and Translation

Sometimes, the transformation causes parts of the image to fall outside the original canvas. To ensure that the entire content fits, we adjust the canvas size and use a **translation matrix** $T$:
$$
T = \begin{bmatrix} 1 & 0 & \Delta x \\ 0 & 1 & \Delta y \\ 0 & 0 & 1 \end{bmatrix}
$$


The final transformation matrix becomes $M = T \cdot H$, ensuring that all content can be displayed on the new canvas.

#### (4) Result

| Warpeed Image 1                                              | Warpeed Image 2                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![base_translated_image](F:\UCB\CS180\project4\part_a\code\figures\result1\base_translated_image.jpg) | ![warped_image](F:\UCB\CS180\project4\part_a\code\figures\result1\warped_image.jpg) |
| ![base_translated_image](F:\UCB\CS180\project4\part_a\code\figures\result2\base_translated_image.jpg) | ![warped_image](F:\UCB\CS180\project4\part_a\code\figures\result2\warped_image.jpg) |
| ![base_translated_image](F:\UCB\CS180\project4\part_a\code\figures\result3\base_translated_image.jpg) | ![warped_image](F:\UCB\CS180\project4\part_a\code\figures\result3\warped_image.jpg) |



### 4.  Image Rectification

1. **Calculate Width and Height**: Determine the width and height of the area to be rectified by measuring distances between the selected source points.
2. **Generate Target Rectangle**: Create standard coordinates for the target rectangle based on the calculated dimensions.
3. **Calculate Homography Matrix**: Obtain the transformation matrix that maps the source area to the target rectangle.
4. **Perform Image Warping**: Use the homography matrix to transform the source image, resulting in a rectified rectangular image.

| Original                                                     | Result                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![10](F:\UCB\CS180\project4\part_a\code\figures\origin\10.jpg) | <img src="F:\UCB\CS180\project4\part_a\code\figures\result_rec\10.jpg" alt="10" style="zoom: 25%;" /> |
| ![11](F:\UCB\CS180\project4\part_a\code\figures\origin\11.jpg) | <img src="F:\UCB\CS180\project4\part_a\code\figures\result_rec\11.jpg" alt="11" style="zoom:25%;" /> |



### 5. **Blend the images into a mosaic**

#### (1) Masking and blending

The core idea is to use a mask to separate regions of the base image and `warped_img`, ensuring that they are blended together seamlessly. Areas that are black (empty) in `warped_img` will show `img_base_translated`, while the rest of the canvas will display `warped_img`.

#### (2) Smooth combination

 This approach effectively merges the two images into a single, coherent mosaic, with each image appearing in its proper place without harsh edges or overlaps.

#### (3) Result

![result](F:\UCB\CS180\project4\part_a\code\figures\result1\result.jpg)



![result](F:\UCB\CS180\project4\part_a\code\figures\result2\result.jpg)





![result](F:\UCB\CS180\project4\part_a\code\figures\result3\result.jpg)



### 5. Blend 3  Images Into a Mosaic

First，blend two images:

| Image 1                                                      | Corresponding                                                | Image 2                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![8](F:\UCB\CS180\project4\part_a\code\figures\origin\8.jpg) | ![corre](F:\UCB\CS180\project4\part_a\code\figures\result4\corre.png) | ![9](F:\UCB\CS180\project4\part_a\code\figures\origin\9.jpg) |
| **Warped Image 1**                                           | **Result **                                                  | **Warped Image 2**                                           |
| ![base_translated_image](F:\UCB\CS180\project4\part_a\code\figures\result4\base_translated_image.jpg) | ![result](F:\UCB\CS180\project4\part_a\code\figures\result4\result.jpg) | ![warped_image](F:\UCB\CS180\project4\part_a\code\figures\result4\warped_image.jpg) |



Then blend the result of first two images with the third one:

| Image 1                                                      | Correspondence                                               | Image 2                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![7](F:\UCB\CS180\project4\part_a\code\figures\origin\7.jpg) | ![corre](F:\UCB\CS180\project4\part_a\code\figures\result5\corre.png) | ![result](F:\UCB\CS180\project4\part_a\code\figures\result4\result.jpg) |
| **Warped Image 1**                                           | **Result**                                                   | **Warped Image 2**                                           |
| ![base_translated_image](F:\UCB\CS180\project4\part_a\code\figures\result5\base_translated_image.jpg) | ![result](F:\UCB\CS180\project4\part_a\code\figures\result5\result.jpg) | ![warped_image](F:\UCB\CS180\project4\part_a\code\figures\result5\warped_image.jpg) |



**Final result:**

![result](F:\UCB\CS180\project4\part_a\code\figures\result5\result.jpg)
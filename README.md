# Muti-View-Geometry
Some code inspired from reading "Multiple View Geometry in Computer Vision" (By Richard Hartley, Andrew Zisserman).
Code in utils.py from [here](https://github.com/marktao99/python/blob/master/CVP/samples/sfm.py).


## From Uncalibrated Images

1. Compute points pairs.
2. Get fundamental matrix F.
3. Get camera matrices of each image.
4. Triangulate 3D points.


## From Calibrated Images (To-Do)

1. Compute each camera intrinsic matrix K.
2. Calculate the rotation R and translation t from one image to other. 
3. Calculate homography matrix H [as here](https://stackoverflow.com/questions/7836134/get-3d-coordinates-from-2d-image-pixel-if-extrinsic-and-intrinsic-parameters-are/10750648#10750648).
4. Project 3D points using the matrix or [using OpenCV functions](https://stackoverflow.com/questions/22334023/how-to-calculate-3d-object-points-from-2d-image-points-using-stereo-triangulatio).

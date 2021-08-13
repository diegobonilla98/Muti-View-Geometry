import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import utils


img1_org = cv2.imread('./uncalibrated/pairA/P1000943.jpeg')
img2_org = cv2.imread('./uncalibrated/pairA/P1000947.jpeg')


img1 = cv2.cvtColor(img1_org, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_org, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
good = []
matches_points = []
for m, n in matches:
    if m.distance < 0.65 * n.distance:
        pt2 = list(kp2[m.trainIdx].pt) + [1.]
        pt1 = list(kp1[m.queryIdx].pt) + [1.]
        matches_points.append([pt1, pt2])
        good.append([m])
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()

matches_points = np.array(matches_points)  # (n_points, (pt1, pt2), (x, y, z))
x1 = matches_points[:, 0, :]
x2 = matches_points[:, 1, :]

F = utils.compute_fundamental(x1, x2)

P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float32)
P_prime = utils.compute_P_from_fundamental(F)

X = utils.triangulate(x1.T, x2.T, P, P_prime)
X1 = X[0, :]
X2 = X[1, :]
X3 = X[2, :]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X1, X2, X3)
plt.show()

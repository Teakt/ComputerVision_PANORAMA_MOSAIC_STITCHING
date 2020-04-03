import numpy as np
import cv2 as cv2
import random
import math

# img1 = cv2.imread('Rainier1.png', cv2.IMREAD_GRAYSCALE)
# rainier1 = cv2.imread('Rainier1.png')
# img2 = cv2.imread('Rainier2.png', cv2.IMREAD_GRAYSCALE)
# rainier2 = cv2.imread('Rainier2.png')
# img3 = cv2.imread('Rainier3.png', cv2.IMREAD_GRAYSCALE)
# rainier3 = cv2.imread('Rainier3.png')
# img4 = cv2.imread('Rainier4.png', cv2.IMREAD_GRAYSCALE)
# rainier4 = cv2.imread('Rainier4.png')
# img5 = cv2.imread('Rainier5.png', cv2.IMREAD_GRAYSCALE)
# rainier5 = cv2.imread('Rainier5.png')
# img6 = cv2.imread('Rainier6.png', cv2.IMREAD_GRAYSCALE)
# rainier6 = cv2.imread('Rainier6.png')


# Images for the second BONUS , using images taken from my bathroom

img1 = cv2.imread('custom1.png', cv2.IMREAD_GRAYSCALE)
custom1 = cv2.imread('custom1.png')
img2 = cv2.imread('custom2.png', cv2.IMREAD_GRAYSCALE)
custom2 = cv2.imread('custom2.png')
# img3 = cv2.imread('custom3.png', cv2.IMREAD_GRAYSCALE)
# custom3 = cv2.imread('custom3.png')
# img4 = cv2.imread('custom4.png', cv2.IMREAD_GRAYSCALE)
# custom4 = cv2.imread('custom4.png')
# img5 = cv2.imread('custom5.png', cv2.IMREAD_GRAYSCALE)
# custom5 = cv2.imread('custom5.png')



final_board = np.zeros((1000, 1000, 3), np.uint8)

# #We place on the Stiched_Image final board , here rainier 1
# for i in range(len(img1)):
#     for j in range(len(img1[i])):
#         for k in range(3):
#             final_board[i][j][k] = rainier1[i][j][k]


# We started here for our bonus , BATHROOM
for i in range(int(len(img1)/2), len(img1)):
    for j in range(int(len(img1[i])/2), len(img1[i])):
        for k in range(3):
            final_board[i][j][k] = custom1[i][j][k]


#This should project point (x, y) using the homography “H”. Return the projected point (x2, y2).
def project(x, y, H):
    mat = np.matrix([y, x, 1])
    transposed_mat = H.transpose()
    multiplied_mat = np.matmul(mat, transposed_mat)
    return multiplied_mat[0, 1] / multiplied_mat[0, 2], multiplied_mat[0, 0] / multiplied_mat[0, 2]


# is a helper function for RANSAC that computes the number of inlying points given a homography "H".
def computeInlierCount(H, matches, inlierThreshold):
    inliers = 0
    for m in matches:
        projectedPoint = project(m[0][1], m[0][0], H)
        if math.sqrt((projectedPoint[1] - m[1][0]) ** 2 + (projectedPoint[0] - m[1][1]) ** 2) < inlierThreshold:
            inliers += 1
    return inliers

def RANSAC( numIterations, final_board, image1Display, image2Display, source_image):
    #Brute-Force Matching with SIFT Descriptors and Ratio Test
    good = []

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1Display, None)
    kp2, des2 = sift.detectAndCompute(image2Display, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    matches_mat = []

    for el in good:

        matches_mat.append([kp2[el[0].trainIdx].pt, kp1[el[0].queryIdx].pt])

    max_count = 0
    h = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32)
    rpz = []
    for l in range(numIterations):
        src = []
        dst = []

        while len(src) < 4: #Randomly select 4 pairs of potentially matching points from "matches".
            randomPoint = random.choice(matches_mat)
            if randomPoint[0] not in src:
                src.append(randomPoint[0])
                dst.append(randomPoint[1])

        src = np.asarray(src)
        dst = np.asarray(dst)

        hom, s = cv2.findHomography(src, dst)
        c = computeInlierCount(hom, matches_mat, 15)
        rpz.append(c)
        if c > max_count:
            max_count = c
            h = hom

    #offset = 2
    for i in range(len(image2Display)):
        for j in range(len(image2Display[i])):
            u, v = project(i, j, h)
            if u > 0 and v > 0 and u < len(final_board) - 5 and v < len(final_board) - 5:
                u = int(round(u))
                v = int(round(v))


                final_board[u][v] = source_image[i][j]

                for l in range(u - 4, u + 4):
                    for m in range(v - 4, v + 4):
                        final_board[l][m] = source_image[i][j]




    return final_board


good = []

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

A = []

for el in good:
    A.append([kp2[el[0].trainIdx].pt, kp1[el[0].queryIdx].pt])

#display_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# new2 = RANSAC(5, final_board, img1, img2, rainier2)
# new3 = RANSAC(5, final_board, img1, img3, rainier3)
# new4 = RANSAC(5, final_board, img1, img4, rainier4)
# new5 = RANSAC(5, final_board, img1, img5, rainier5)
# new6 = RANSAC(5, final_board, img1, img6, rainier6)

new2 = RANSAC(5, final_board, img1, img2, custom2)
# new3 = RANSAC(5, final_board, img1, img3, custom3)
# new4 = RANSAC(5, final_board, img1, img4, custom4)
# new5 = RANSAC(5, final_board, img1, img5, rainier5)
# new6 = RANSAC(5, final_board, img1, img6, rainier6)

#cv2.imshow("Display the inlier matches using cv::drawMatches(...).", display_matches)
cv2.imshow("Final_Diapo", final_board)
#cv2.imwrite("3.png", display_matches)
#cv2.imwrite("AllStitched_BONUS.png", final_board)

k = cv2.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv2.destroyAllWindows()

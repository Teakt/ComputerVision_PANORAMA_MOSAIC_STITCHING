TEAK GUILLAUME 40157390 COMPUTER VISION PROJECT 03/04/2020

---------------------------------------------------------------

PANORAMA MOSAIC STITCHING : 

In the file "Panorama_Mosaic_Stiching.py" : 

def project(x, y, H): 

	- We use np.matmul() to multiply 
	- It returns the projected point coordinates :" return multiplied_mat[0, 1] / multiplied_mat[0, 2], multiplied_mat[0, 0] / multiplied_ _mat[0, 2]"

def computeInlierCount(H, matches, inlierThreshold):

	- We go through the number of matches, and we check with a if condition with a least squares to see  If the projected point is less than the distance "inlierThreshold" from thesecond point, it is an inlier.
	- Return the total number of inliers.

def RANSAC( numIterations, final_board, image1Display, image2Display, source_image):

	- We first # find the keypoints and descriptors with SIFT and then use the knnMatch to have the matches, apply ratio test 
	- We find out Given the highest scoring homography, once again find all the inliers. Compute a new refined homography using all of the inliers
	- We do it so we can get our h, to have our best homography

	""    for i in range(len(image2Display)):
        for j in range(len(image2Display[i])):
            u, v = project(i, j, best_homo)
            if u > 0 and v > 0 and u < len(final_board) - 5 and v < len(final_board) - 5:
                u = int(round(u))
                v = int(round(v))
                final_board[u][v] = source_image[i][j]


	- Here we apply our image stiching, by projecting points onto the "final_board" through the homography matrix
                for l in range(u - 4, u + 4): 
                    for m in range(v - 4, v + 4):
                        final_board[l][m] = source_image[i][j]""

	- This piece of code was made for correcting when the image sticthing created black lines on the result,  just "splattered" the pixel from the source image onto the board on a size of a 4*4 matrix here 

- 3.png and 4.png are saved at the root of the project

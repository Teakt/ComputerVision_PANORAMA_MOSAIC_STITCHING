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


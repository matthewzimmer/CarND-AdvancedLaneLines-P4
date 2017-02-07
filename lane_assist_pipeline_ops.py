import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from pipeline_ops import PipelineOp


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None


class LaneAssistOp(PipelineOp):
	def __init__(self, img, calibration_op, margin=100, kernel_size=3, sobelx_thresh=(20, 100), sobely_thresh=(20, 100),
	             mag_grad_thresh=(20, 250), dir_grad_thresh=(0., np.pi / 2)):
		PipelineOp.__init__(self)
		self.__img = img
		self.__margin = margin
		self.__calibration_op = calibration_op
		self.__parameters = {
			'kernel_size': kernel_size,
			'sobelx_thresh': sobelx_thresh,
			'sobely_thresh': sobely_thresh,
			'mag_grad_thresh': mag_grad_thresh,
			'dir_grad_thresh': dir_grad_thresh
		}

	def perform(self):
		img = self.__img
		kernel_size = self.__parameters['kernel_size']
		sobelx_thresh = self.__parameters['sobelx_thresh']
		sobely_thresh = self.__parameters['sobely_thresh']
		mag_grad_thresh = self.__parameters['mag_grad_thresh']
		dir_grad_thresh = self.__parameters['dir_grad_thresh']

		# undistort the raw image
		undistorted = UndistortOp(img, self.__calibration_op).perform().output()

		# Convert undistored image to HLS and use the 'S' channel as our gray image.
		hls = ConvertColorSpaceOp(undistorted, color_space='hls', color_channel=2).perform().output()

		# Compute sobel X binary image
		gradx = SobelThreshOp(hls, orient='x', sobel_kernel=kernel_size, thresh=sobelx_thresh).perform().output()

		# Compute sobel Y binary image
		grady = SobelThreshOp(hls, orient='y', sobel_kernel=kernel_size, thresh=sobely_thresh).perform().output()

		# Compute Magnitude Gradient binary image
		mag_binary = MagnitudeGradientThreshOp(hls, sobel_kernel=kernel_size, thresh=mag_grad_thresh).perform().output()

		# Compute Direction Gradient binary image
		dir_binary = DirectionGradientThreshOp(hls, sobel_kernel=kernel_size, thresh=dir_grad_thresh).perform().output()

		# Perform bitwise AND and OR where to create a final binary image where we generate a binary image of
		# all white pixels in (SobelX AND SobelY) and combine it via binary OR with a binary image of all white pixels
		# in (Magnitude AND Direction) gradients.
		combined = np.zeros_like(hls)
		combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

		# Now we're going to warp our combined threshholded binary image
		img_size = (img.shape[1], img.shape[0])
		src_pts = np.float32(
			[[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
			 [((img_size[0] / 6) + 15), img_size[1]],
			 [(img_size[0] * 5 / 6) + 90, img_size[1]],
			 [(img_size[0] / 2 + 85), img_size[1] / 2 + 100]])

		dst_pts = np.float32(
			[[(img_size[0] / 6), 0],
			 [(img_size[0] / 6), img_size[1]],
			 [(img_size[0] * 5 / 6), img_size[1]],
			 [(img_size[0] * 5 / 6), 0]])

		src_pts = np.float32(
		    [[605, 460],
		    [268, 720],
		    [1157, 720],
		    [725, 460]])

		dst_pts = np.float32(
		    [[(img_size[0] / 6), 0],
		    [(img_size[0] / 6), img_size[1]],
		    [(img_size[0] * 5 / 6), img_size[1]],
		    [(img_size[0] * 5 / 6), 0]])

		# define 4 source points for perspective transformation
		src_pts = np.float32([[220,719],[1220,719],[750,480],[550,480]])
		# define 4 destination points for perspective transformation
		dst_pts = np.float32([[240,719],[1040,719],[1040,0],[240,0]])

		warper_op = WarperOp(combined, src_pts, dst_pts).perform().output()
		binary_warped = warper_op['warped']

		if True:
			polyfit_op = PolyfitLineOp(binary_warped).perform().output()
			left_fit = polyfit_op['left_fit']
			right_fit = polyfit_op['right_fit']
			nonzeroy = polyfit_op['nonzeroy']
			nonzerox = polyfit_op['nonzerox']
			left_lane_inds = polyfit_op['left_lane_inds']
			right_lane_inds = polyfit_op['right_lane_inds']

			# Generate x and y values for plotting
			fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
			fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
			fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

			out_img = polyfit_op['out_img']
			out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Assume you now have a new warped binary image
		# from the next frame of video (also called "binary_warped")
		# It's now much easier to find line pixels!
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		#print(len(nonzeroy))
		#print(len(nonzerox))
		margin = self.__margin
		left_lane_inds = (
		(nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
		nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
		right_lane_inds = (
		(nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
		nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

		# Again, extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		# Generate x and y values for plotting
		fity = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
		fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
		fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

		# Create an image to draw on and an image to show the selection window
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
		window_img = np.zeros_like(out_img)
		# Color in left and right line pixels
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		# Generate a polygon to illustrate the search window area
		# And recast the x and y points into usable format for cv2.fillPoly()
		left_line_window1 = np.array([np.transpose(np.vstack([fit_leftx - margin, fity]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_leftx + margin, fity])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([fit_rightx - margin, fity]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fit_rightx + margin, fity])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		# Define y-value where we want radius of curvature
		# I'll choose the maximum y-value, corresponding to the bottom of the image
		y_eval = out_img.shape[0]

		left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
		right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

		#print('left:', left_curverad, '| right:', right_curverad)

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30 / out_img.shape[0]  # meters per pixel in y dimension
		xm_per_pix = 3.7 / (out_img.shape[0] - 20)  # meteres per pixel in x dimension

		left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
		right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

		left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
		                / np.absolute(2 * left_fit_cr[0])
		right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
		                 / np.absolute(2 * right_fit_cr[0])
		# Now our radius of curvature is in meters
		#print('left curve rad: {}m    |     right curve rad: {}m'.format(left_curverad, right_curverad))

		# Create an image to draw the lines on
		warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0, 195, 255))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(color_warp, warper_op['Minv'], (binary_warped.shape[1], binary_warped.shape[0]))

		# Combine the result with the original image
		result = cv2.addWeighted(undistorted, 1, newwarp, 0.5, 0)

		return self._apply_output(result)


class CameraCalibrationOp(PipelineOp):
	MODE_INITIALIZED = 0x0
	MODE_CALIBRATING = 0x1
	MODE_CALIBRATED = 0x2

	# Bitmask of completed camera calibration stages
	COMPLETED_CALIBRATION_STAGES = 0x0

	STAGE_OBTAINED_CALIBRATION_IMAGES = 0x1 << 0
	STAGE_COMPUTED_OBJ_AND_IMG_POINTS = 0x1 << 1
	STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS = 0x1 << 2
	STAGE_UNDISTORED_CALIBRATION_IMAGES = 0x1 << 3
	STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS = 0x1 << 4

	def __init__(self, calibration_images, x_inside_corners=9, y_inside_corners=6,
	             calibration_results_pickle_file="camera_cal/camera_mtx_and_dist_pickle.p"):
		PipelineOp.__init__(self)

		self.__mode = self.MODE_INITIALIZED

		# Images taken by a camera for which this class calibrates by
		# calculating the object and image points used to undistort
		# any image take by the same camera.
		self.__calibration_images = calibration_images

		self.__x_inside_corners = x_inside_corners
		self.__y_inside_corners = y_inside_corners

		# Arrays to store object points and image points from all the images.
		self.__objpoints = []  # 3d points in real world space
		self.__imgpoints = []  # 2d points in image plane

		# Computed using cv2.calibrateCamera() in __compute_camera_matrix_and_distortion_coefficients
		self.__camera_matrix = None
		self.__distortion_coefficients = None

		# The location of the pickle file where our camera calibration matrix and
		# distortion coefficients are persisted to
		self.__calibration_results_pickle_file = calibration_results_pickle_file

		self.__apply_stage(self.STAGE_OBTAINED_CALIBRATION_IMAGES)

	def perform(self):
		self.__mode = self.MODE_CALIBRATING
		self.__compute_obj_and_img_points()
		calibrations = self.__load_calibrations()
		if calibrations is False:
			self.__compute_camera_matrix_and_distortion_coefficients(self.__calibration_images[0])
			self.__save_calibration_mtx_and_dist()
		# self.__undistort_chessboard_images()
		self.__mode = self.MODE_CALIBRATED
		return self._apply_output({
			'matrix': self.__camera_matrix,
			'dist_coefficients': self.__distortion_coefficients,
			'objpoints': self.__objpoints,
			'imgpoints': self.__imgpoints
		})

	def undistort(self, img):
		"""
		A function that takes an image and performs the camera calibration,
		image distortion correction and returns the undistorted image
		"""
		img = np.copy(img)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img.shape[0:2][::-1],
		                                                   None, None)
		return cv2.undistort(img, mtx, dist, None, mtx)

	# PRIVATE

	def __detect_corners(self, img, nx, ny):
		"""
		This function converts an RGB chessboard image to grayscale and finds the
		chessboard corners using cv2.findChessboardCorners.
		"""
		img = np.copy(img)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		return cv2.findChessboardCorners(gray, (nx, ny), None)

	def __compute_obj_and_img_points(self):
		"""
		A function which iterates over all self.calibration_images and detects all
		chessboard corners for each.

		For each image corners are detected, a copy of that image with the corners
		drawn on are saved to camera_cal/corners_found
		"""
		if not self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS):
			nx, ny = self.__x_inside_corners, self.__y_inside_corners

			# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
			objp = np.zeros((nx * ny, 3), np.float32)
			objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

			# Step through the list and search for chessboard corners
			for fname in self.__calibration_images:
				img = mpimg.imread(fname)
				ret, corners = self.__detect_corners(img, nx, ny)

				# If found, add object points, image points
				if ret == True:
					self.__objpoints.append(objp)
					self.__imgpoints.append(corners)

					# print("{} corners detected".format(os.path.basename(fname)))
					calibrated_name = 'camera_cal/corners_found/{}'.format(str(os.path.basename(fname)))
					cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
					cv2.imwrite(calibrated_name, img)

			self.__apply_stage(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS)

	def __undistort_chessboard_images(self):
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_UNDISTORED_CALIBRATION_IMAGES):
			# Step through the list and search for chessboard corners
			for fname in self.__calibration_images:
				img = mpimg.imread(fname)
				undistorted = self.undistort(img)

				# print("{} undistorted".format(os.path.basename(fname)))
				undist_file = 'camera_cal/undistorted/{}'.format(os.path.basename(fname))
				cv2.imwrite(undist_file, undistorted)

			self.__apply_stage(self.STAGE_UNDISTORED_CALIBRATION_IMAGES)

	def __load_calibrations(self):
		if os.path.isfile(self.__calibration_results_pickle_file):
			with open(self.__calibration_results_pickle_file, 'rb') as f:
				pickle_data = pickle.load(f)
				self.__camera_matrix = pickle_data['mtx']
				self.__distortion_coefficients = pickle_data['dist']
				if self.__camera_matrix is not None:
					self.__apply_stage(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS)
					self.__apply_stage(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS)
					return True
		return False

	def __compute_camera_matrix_and_distortion_coefficients(self, distorted_image_path):
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS):
			fname = distorted_image_path
			img = mpimg.imread(fname)
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img.shape[0:2][::-1],
			                                                   None, None)

			self.__camera_matrix = mtx
			self.__distortion_coefficients = dist

			self.__apply_stage(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS)

	def __save_calibration_mtx_and_dist(self):
		"""
		Saves a pickled representation of the camera calibration matrix and
		distortion coefficient results for the provided image for later use
		"""
		if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(
				self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS):
			dist_pickle = {}
			dist_pickle["mtx"] = self.__camera_matrix
			dist_pickle["dist"] = self.__distortion_coefficients
			pickle.dump(dist_pickle, open(self.__calibration_results_pickle_file, "wb"))

			# print('camera matrix and distortion coefficients pickled to "{}" for later use'.format(self.__calibration_results_pickle_file))

			self.__apply_stage(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS)

	def __is_stage_complete(self, flag):
		return self.COMPLETED_CALIBRATION_STAGES & flag == flag

	def __apply_stage(self, flag):
		"""Marks a stage as complete"""
		self.COMPLETED_CALIBRATION_STAGES = self.COMPLETED_CALIBRATION_STAGES | flag

	def __str__(self):
		s = []

		s.append('')
		s.append('')
		s.append('-------------------------------------------------------------')
		s.append('')
		s.append('[ CALIBRATION MODES ]')
		s.append('')
		s.append('   Initialized? {}'.format('YES' if self.__mode == self.MODE_INITIALIZED else 'NO'))
		s.append('   Calibrating? {}'.format('YES' if self.__mode == self.MODE_CALIBRATING else 'NO'))
		s.append('   Calibration complete? {}'.format('YES' if self.__mode == self.MODE_CALIBRATED else 'NO'))
		s.append('')
		s.append('')

		s.append('[ CALIBRATION STAGES - {} ]'.format(self.COMPLETED_CALIBRATION_STAGES))
		s.append('')
		s.append('   Obtained calibration images? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_OBTAINED_CALIBRATION_IMAGES) else 'NO'))
		s.append('   Computed object/image points? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) else 'NO'))
		s.append('   Calculated camera matrix and distortion coefficients? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS) else 'NO'))
		s.append('   Undistored calibration images? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_UNDISTORED_CALIBRATION_IMAGES) else 'NO'))
		s.append('   Persisted camera matrix and distortion coefficients? {}'.format(
			'YES' if self.__is_stage_complete(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS) else 'NO'))
		s.append('')

		s.append('[ PARAMS ]')
		s.append('')
		s.append('Number calibration images: {}'.format(len(self.__calibration_images)))
		s.append('X inside corners = {}'.format(self.__x_inside_corners))
		s.append('Y inside corners = {}'.format(self.__y_inside_corners))
		s.append('')
		# s.append('output = {}'.format(str(self.output())))

		s.append('')
		s.append('')

		return '\n'.join(s)


class ConvertColorSpaceOp(PipelineOp):
    def __init__(self, img, color_space, src_color_space='RGB', color_channel=-1):
        """
        Converts an image to a different color space.
        
        Available color spaces: HSV, HLS, YUV, GRAY
        """
        PipelineOp.__init__(self)
        self.__img = np.copy(img)
        self.__color_space = color_space.upper()
        self.__src_color_space = src_color_space.upper()
        self.__color_channel = color_channel
    
    def perform(self):
        img = cv2.cvtColor(self.__img, eval('cv2.COLOR_{}2{}'.format(self.__src_color_space, self.__color_space))).astype(np.float)
        if self.__color_channel > -1:
            img = img[:,:,self.__color_channel]
        return self._apply_output(img)


class ColorThreshOp(PipelineOp):
	def __init__(self, gray_img, color_thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__color_thresh = color_thresh

	def perform(self):
		# ret, thresholded_img = cv2.threshold(img.astype('uint8'), self._color_thresh[0], self._color_thresh[1], cv2.THRESH_BINARY)
		# self._thresholded_img = thresholded_img
		# self._binary_img = binary_img
		binary = np.zeros_like(self.__img)
		binary[(self.__img > self.__color_thresh[0]) & (self.__img <= self.__color_thresh[1])] = 1
		return self._apply_output(binary)


class UndistortOp(PipelineOp):
	def __init__(self, img, camera_calibration_op):
		"""
		Takes an image and cam and performs image distortion correction
		"""
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__camera_calibration_op = camera_calibration_op

	def perform(self):
		return self._apply_output(self.__camera_calibration_op.undistort(self.__img))


class SobelThreshOp(PipelineOp):
	def __init__(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray)
		self.__orient = orient
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		gray = self.__img
		sobel = cv2.Sobel(gray, cv2.CV_64F, self.__orient == 'x', self.__orient != 'x', ksize=self.__sobel_kernel)
		abs_sobel = np.absolute(sobel)
		scaled_sobel = (255 * abs_sobel / np.max(abs_sobel)).astype(np.uint8)
		binary = np.zeros_like(scaled_sobel)
		binary[(scaled_sobel >= self.__thresh[0]) & (scaled_sobel <= self.__thresh[1])] = 1
		return self._apply_output(binary)


class MagnitudeGradientThreshOp(PipelineOp):
	def __init__(self, gray_img, sobel_kernel=3, thresh=(0, 255)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		gray = self.__img
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
		gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
		gradmag = (255 * gradmag / np.max(gradmag)).astype(np.uint8)
		binary = np.zeros_like(gradmag)
		binary[(gradmag >= self.__thresh[0]) & (gradmag <= self.__thresh[1])] = 1
		return self._apply_output(binary)


class DirectionGradientThreshOp(PipelineOp):
	"""
	Calculates the gradient direction of detected lines
	"""

	def __init__(self, gray_img, sobel_kernel=3, thresh=(0, np.pi / 2)):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__sobel_kernel = sobel_kernel  # Choose a larger odd number to smooth gradient measurements
		self.__thresh = thresh

	def perform(self):
		sobelx = cv2.Sobel(self.__img, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
		sobely = cv2.Sobel(self.__img, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
		with np.errstate(divide='ignore', invalid='ignore'):
			abs_grad_dir = np.absolute(np.arctan(sobely / sobelx))
			binary = np.zeros_like(abs_grad_dir)
			binary[(abs_grad_dir > self.__thresh[0]) & (abs_grad_dir < self.__thresh[1])] = 1
		return self._apply_output(binary)


class WarperOp(PipelineOp):
	def __init__(self, gray_img, src_pts, dst_pts):
		PipelineOp.__init__(self)
		self.__img = np.copy(gray_img)
		self.__src_pts = src_pts
		self.__dst_pts = dst_pts

	def perform(self):
		# Compute the perspective transform, M, given source and destination points:
		M = cv2.getPerspectiveTransform(self.__src_pts, self.__dst_pts)

		# Compute the inverse perspective transform:
		Minv = cv2.getPerspectiveTransform(self.__dst_pts, self.__src_pts)

		# Warp an image using the perspective transform, M:
		warped = cv2.warpPerspective(self.__img, M, self.__img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)

		return self._apply_output({
			'warped': warped,
			'M': M,
			'Minv': Minv
		})

	def __str__(self):
		s = []

		s.append(' source image shape: ')
		s.append('')
		s.append('   ' + str(self.__img.shape))
		s.append('')

		s.append(' source points: ')
		s.append('')
		s.append('   top.L: ' + str(self.__src_pts[0]))
		s.append('   bot.L: ' + str(self.__src_pts[1]))
		s.append('   bot.R: ' + str(self.__src_pts[2]))
		s.append('   top.R: ' + str(self.__src_pts[3]))
		s.append('')

		s.append(' desination points: ')
		s.append('')
		s.append('   top.L: ' + str(self.__dst_pts[0]))
		s.append('   bot.L: ' + str(self.__dst_pts[1]))
		s.append('   bot.R: ' + str(self.__dst_pts[2]))
		s.append('   top.R: ' + str(self.__dst_pts[3]))
		s.append('')
		s.append('')

		return '\n'.join(s)


class PlotImageOp(PipelineOp):
	def __init__(self, img, title='', cmap='gray', interpolation='none', aspect='auto'):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__title = title
		self.__cmap = cmap
		self.__interpolation = interpolation
		self.__aspect = aspect

	def perform(self):
		fig1 = plt.figure(figsize=(16, 9))
		ax = fig1.add_subplot(111)
		ax.imshow(self.__img, cmap=self.__cmap, interpolation=self.__interpolation, aspect=self.__aspect)
		plt.tight_layout()
		ax.set_title(self.__title)
		plt.show()
		return self._apply_output(ax)


class DrawPolyLinesOp(PipelineOp):
	def __init__(self, img, pts, color=(0, 140, 255), thickness=5):
		PipelineOp.__init__(self)
		self.__img = np.copy(img)
		self.__pts = pts
		self.__color = color
		self.__thickness = thickness

	def perform(self):
		return self._apply_output(cv2.polylines(self.__img, [np.array([self.__pts], np.int32)], True, self.__color,
		                              thickness=self.__thickness))


class PolyfitLineOp(PipelineOp):
	def __init__(self, binary_warped):
		PipelineOp.__init__(self)
		self.__binary_warped = binary_warped

	def perform(self):
		binary_warped = self.__binary_warped

		# Assuming you have created a warped binary image called "binary_warped"
		# Take a histogram of the bottom half of the image
		histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

		# Create an output image to draw on and  visualize the result
		out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

		# Find the peak of the left and right halves of the histogram
		# These will be the starting point for the left and right lines
		midpoint = np.int(histogram.shape[0] / 2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		# Choose the number of sliding windows
		nwindows = 9
		# Set height of windows
		window_height = np.int(binary_warped.shape[0] / nwindows)
		# Identify the x and y positions of all nonzero pixels in the image
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Current positions to be updated for each window
		leftx_current = leftx_base
		rightx_current = rightx_base
		# Set the width of the windows +/- margin
		margin = 100
		# Set minimum number of pixels found to recenter window
		minpix = 50
		# Create empty lists to receive left and right lane pixel indices
		left_lane_inds = []
		right_lane_inds = []

		# Step through the windows one by one
		for window in range(nwindows):
			# Identify window boundaries in x and y (and right and left)
			win_y_low = binary_warped.shape[0] - (window + 1) * window_height
			win_y_high = binary_warped.shape[0] - window * window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			# Draw the windows on the visualization image
			cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
			cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
			nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
			nonzerox < win_xright_high)).nonzero()[0]
			# Append these indices to the lists
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			# If you found > minpix pixels, recenter next window on their mean position
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		# Concatenate the arrays of indices
		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		# Extract left and right line pixel positions
		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds]
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		# Fit a second order polynomial to each
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		return self._apply_output({
			'left_fit': left_fit,
			'right_fit': right_fit,
			'out_img': out_img,
			'left_lane_inds': left_lane_inds,
			'right_lane_inds': right_lane_inds,
			'leftx': leftx,
			'lefty': lefty,
			'rightx': rightx,
			'righty': righty,
			'nonzerox': nonzerox,
			'nonzeroy': nonzeroy
		})

class PipelineOp:
    def perform(self):
        raise NotImplementedError
    
    def output(self):
        """
        Returns the result from performing this operation.
        """
        raise NotImplementedError


class CameraCalibrationOp(PipelineOp):
    
    MODE_INITIALIZED = 0x0
    MODE_CALIBRATING = 0x1
    MODE_CALIBRATED = 0x2
    
    # Bitmask of completed camera calibration stages
    COMPLETED_CALIBRATION_STAGES = 0x0    
    
    STAGE_OBTAINED_CALIBRATION_IMAGES = 0x1<<0
    STAGE_COMPUTED_OBJ_AND_IMG_POINTS = 0x1<<1
    STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS = 0x1<<2
    STAGE_UNDISTORED_CALIBRATION_IMAGES = 0x1<<3
    STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS = 0x1<<4
    
    def __init__(self, calibration_images, x_inside_corners=9, y_inside_corners=6, calibration_results_pickle_file="camera_cal/camera_mtx_and_dist_pickle.p"):
        PipelineOp.__init__(self)
        
        self.__mode = self.MODE_INITIALIZED
        
        # Images taken by a camera for which this class calibrates by 
        # calculating the object and image points used to undistort 
        # any image take by the same camera.
        self.__calibration_images = calibration_images
        
        self.__x_inside_corners = x_inside_corners
        self.__y_inside_corners = y_inside_corners
        
        # Arrays to store object points and image points from all the images.
        self.__objpoints = [] # 3d points in real world space
        self.__imgpoints = [] # 2d points in image plane
        
        # Computed using cv2.calibrateCamera() in __compute_camera_matrix_and_distortion_coefficients
        self.__camera_matrix = None
        self.__distortion_coefficients = None
        
        # The location of the pickle file where our camera calibration matrix and 
        # distortion coefficients are persisted to
        self.__calibration_results_pickle_file = calibration_results_pickle_file
        
        self.__apply_stage(self.STAGE_OBTAINED_CALIBRATION_IMAGES)

    def perform(self):
        print(self)
        
        self.__mode = self.MODE_CALIBRATING
        print(self)
        
        self.__compute_obj_and_img_points()
        print(self)
        
        self.__compute_camera_matrix_and_distortion_coefficients(self.__calibration_images[0])
        print(self)
        
        self.__save_calibration_mtx_and_dist()
        print(self)
        
        self.__undistort_chessboard_images()
        print(self)
        
        self.__mode = self.MODE_CALIBRATED
        print(self)
        
        return self
        
    def output(self):
        return {
            'matrix': self.__camera_matrix,
            'dist_coefficients': self.__distortion_coefficients,
            'objpoints': self.__objpoints,
            'imgpoints': self.__imgpoints
        }
    
    def undistort(self, img):
        """
        A function that takes an image and performs the camera calibration, 
        image distortion correction and returns the undistorted image
        """
        img = np.copy(img)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img.shape[0:2][::-1], None, None)
        return cv2.undistort(img, mtx, dist, None, mtx)

    # PRIVATE
    
    def __detect_corners(self, img, nx, ny):
        """
        This function converts an RGB chessboard image to grayscale and finds the 
        chessboard corners using cv2.findChessboardCorners.
        """
        img = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.findChessboardCorners(gray, (nx,ny), None)
    
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
            objp = np.zeros((nx*ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

            # Step through the list and search for chessboard corners
            for fname in self.__calibration_images:
                img = mpimg.imread(fname)
                ret, corners = self.__detect_corners(img, nx, ny)

                # If found, add object points, image points
                if ret == True:
                    self.__objpoints.append(objp)
                    self.__imgpoints.append(corners)

                    print("{} corners detected".format(os.path.basename(fname)))
                    calibrated_name = 'camera_cal/corners_found/{}'.format(str(os.path.basename(fname)))
                    cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
                    cv2.imwrite(calibrated_name, img)
            
            self.__apply_stage(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS)
    
    def __undistort_chessboard_images(self):
        if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(self.STAGE_UNDISTORED_CALIBRATION_IMAGES):
            # Step through the list and search for chessboard corners
            for fname in self.__calibration_images:
                img = mpimg.imread(fname)
                undistorted = self.undistort(img)

                print("{} undistorted".format(os.path.basename(fname)))
                undist_file = 'camera_cal/undistorted/{}'.format(os.path.basename(fname))
                cv2.imwrite(undist_file, undistorted)
            
            self.__apply_stage(self.STAGE_UNDISTORED_CALIBRATION_IMAGES)
    
    def __compute_camera_matrix_and_distortion_coefficients(self, distorted_image_path):
        if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS):
            fname = distorted_image_path
            img = mpimg.imread(fname)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.__objpoints, self.__imgpoints, img.shape[0:2][::-1], None, None)
            
            self.__camera_matrix = mtx
            self.__distortion_coefficients = dist
            
            self.__apply_stage(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS)
            
    def __save_calibration_mtx_and_dist(self):
        """
        Saves a pickled representation of the camera calibration matrix and 
        distortion coefficient results for the provided image for later use
        """
        if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) and not self.__is_stage_complete(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS):
            dist_pickle = {}
            dist_pickle["mtx"] = self.__camera_matrix
            dist_pickle["dist"] = self.__distortion_coefficients
            pickle.dump( dist_pickle, open( self.__calibration_results_pickle_file, "wb" ) )

            print('camera matrix and distortion coefficients pickled to "{}" for later use'.format(self.__calibration_results_pickle_file))
            
            self.__apply_stage(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS)

    def __is_stage_complete(self, flag):
        return self.COMPLETED_CALIBRATION_STAGES&flag==flag
    
    def __apply_stage(self, flag):
        """Marks a stage as complete"""
        self.COMPLETED_CALIBRATION_STAGES = self.COMPLETED_CALIBRATION_STAGES|flag

    def __str__(self):
        s = []
        
        s.append('')
        s.append('')
        s.append('-------------------------------------------------------------')
        s.append('')
        s.append('[ CALIBRATION MODES ]')
        s.append('')
        s.append('   Initialized? {}'.format('YES' if self.__mode==self.MODE_INITIALIZED else 'NO'))
        s.append('   Calibrating? {}'.format('YES' if self.__mode==self.MODE_CALIBRATING else 'NO'))
        s.append('   Calibration complete? {}'.format('YES' if self.__mode==self.MODE_CALIBRATED else 'NO'))
        s.append('')
        s.append('')
        
        s.append('[ CALIBRATION STAGES - {} ]'.format(self.COMPLETED_CALIBRATION_STAGES))
        s.append('')
        s.append('   Obtained calibration images? {}'.format('YES' if self.__is_stage_complete(self.STAGE_OBTAINED_CALIBRATION_IMAGES) else 'NO'))
        s.append('   Computed object/image points? {}'.format('YES' if self.__is_stage_complete(self.STAGE_COMPUTED_OBJ_AND_IMG_POINTS) else 'NO'))
        s.append('   Calculated camera matrix and distortion coefficients? {}'.format('YES' if self.__is_stage_complete(self.STAGE_CALCULATED_CAMERA_MTX_AND_DIST_COEFFICIENTS) else 'NO'))
        s.append('   Undistored calibration images? {}'.format('YES' if self.__is_stage_complete(self.STAGE_UNDISTORED_CALIBRATION_IMAGES) else 'NO'))
        s.append('   Persisted camera matrix and distortion coefficients? {}'.format('YES' if self.__is_stage_complete(self.STAGE_SAVED_MTX_AND_DIST_CALIBRATIONS) else 'NO'))
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


class ColorSpaceConvertOp(PipelineOp):
    def __init__(self, img, color_space, color_channel=-1):
        """
        Converts an image to a different color space.
        
        Available color spaces: HSV, HLS, YUV, GRAY
        """
        PipelineOp.__init__(self)
        self.__img = np.copy(img)
        self.__color_space = color_space
        self.__color_channel = color_channel
        self.__output = None

    def output(self):
        return self.__output
    
    def perform(self):
        if self.__color_space.lower() == 'hsv':
            img = cv2.cvtColor(self.__img, cv2.COLOR_RGB2HSV).astype(np.float)
        if self.__color_space.lower() == 'hls':
            img = cv2.cvtColor(self.__img, cv2.COLOR_RGB2HLS).astype(np.float)
        if self.__color_space.lower() == 'yuv':
            img = cv2.cvtColor(self.__img, cv2.COLOR_RGB2YUV).astype(np.float)
        if self.__color_space.lower() == 'gray':
            img = cv2.cvtColor(self.__img, cv2.COLOR_RGB2GRAY).astype(np.float)
        if self.__color_channel > -1:
            img = img[:,:,self.__color_channel]
        self.__output = img
        return self


class ColorThreshOp(PipelineOp):
    def __init__(self, gray_img, color_thresh=(0, 255)):
        PipelineOp.__init__(self)
        self.__img = np.copy(gray_img)
        self.__color_thresh = color_thresh
        self.__output = None

    def output(self):
        return self.__output
    
    def perform(self):
        # ret, thresholded_img = cv2.threshold(img.astype('uint8'), self._color_thresh[0], self._color_thresh[1], cv2.THRESH_BINARY)
        # self._thresholded_img = thresholded_img
        # self._binary_img = binary_img
        binary = np.zeros_like(self.__img)
        binary[(self.__img > self.__color_thresh[0]) & (self.__img <= self.__color_thresh[1])] = 1
        self.__output = binary
        return self


class UndistortOp(PipelineOp):
    def __init__(self, img, camera_calibration_op):
        """
        Takes an image and cam and performs image distortion correction
        """
        PipelineOp.__init__(self)
        self.__img = np.copy(img)
        self.__camera_calibration_op = camera_calibration_op
        self.__output = None
    
    def output(self):
        return self.__output
    
    def perform(self):
        self.__output = self.__camera_calibration_op.undistort(self.__img)
        return self

        
class SobelThreshOp(PipelineOp):
    def __init__(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        PipelineOp.__init__(self)
        self.__img = np.copy(gray)
        self.__orient = orient
        self.__sobel_kernel = sobel_kernel # Choose a larger odd number to smooth gradient measurements
        self.__thresh = thresh
        self.__output = None
    
    def output(self):
        return self.__output
        
    def perform(self):
        gray = self.__img
        sobel = cv2.Sobel(gray, cv2.CV_64F, self.__orient=='x', self.__orient!='x', ksize=self.__sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = (255*abs_sobel/np.max(abs_sobel)).astype(np.uint8) 
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= self.__thresh[0]) & (scaled_sobel <= self.__thresh[1])] = 1
        self.__output = binary
        return self

        
class MagnitudeGradientThreshOp(PipelineOp):
    def __init__(self, gray_img, sobel_kernel=3, thresh=(0, 255)):
        PipelineOp.__init__(self)
        self.__img = np.copy(gray_img)
        self.__sobel_kernel = sobel_kernel # Choose a larger odd number to smooth gradient measurements
        self.__thresh = thresh
        self.__output = None
    
    def output(self):
        return self.__output
        
    def perform(self):
        gray = self.__img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        gradmag = (255*gradmag/np.max(gradmag)).astype(np.uint8) 
        binary = np.zeros_like(gradmag)
        binary[(gradmag >= self.__thresh[0]) & (gradmag <= self.__thresh[1])] = 1
        self.__output = binary
        return self

        
class DirectionGradientThreshOp(PipelineOp):
    """
    Calculates the gradient direction of detected lines 
    """
    def __init__(self, gray_img, sobel_kernel=3, thresh=(0, np.pi/2)):
        PipelineOp.__init__(self)
        self.__img = np.copy(gray_img)
        self.__sobel_kernel = sobel_kernel # Choose a larger odd number to smooth gradient measurements
        self.__thresh = thresh
        self.__output = None
    
    def output(self):
        return self.__output
        
    def perform(self):
        sobelx = cv2.Sobel(self.__img, cv2.CV_64F, 1, 0, ksize=self.__sobel_kernel)
        sobely = cv2.Sobel(self.__img, cv2.CV_64F, 0, 1, ksize=self.__sobel_kernel)
        with np.errstate(divide='ignore', invalid='ignore'):
            abs_grad_dir = np.absolute(np.arctan(sobely/sobelx))
            binary = np.zeros_like(abs_grad_dir)
            binary[(abs_grad_dir > self.__thresh[0]) & (abs_grad_dir < self.__thresh[1])] = 1
        self.__output = binary
        return self

        
class WarperOp(PipelineOp):
    
    def __init__(self, gray_img, src_pts, dst_pts):
        PipelineOp.__init__(self)
        self.__img = np.copy(gray_img)
        self.__src_pts = src_pts
        self.__dst_pts = dst_pts
        self.__transform_matrix = None
        self.__output = None
    
    def output(self):
        return self.__output
    
    def perform(self):
        # use cv2.getPerspectiveTransform() to get M, the transform matrix
        self.__transform_matrix = cv2.getPerspectiveTransform(self.__src_pts, self.__dst_pts)

        # use cv2.warpPerspective() to warp the image to a top-down view
        self.__output = cv2.warpPerspective(self.__img, self.__transform_matrix, self.__img.shape[0:2][::-1], flags=cv2.INTER_LINEAR)
        return self
    
    def __str__(self):
        s = []

        s.append(' source image shape: ')
        s.append('')
        s.append('   '+str(self.__img.shape))
        s.append('')

        s.append(' source points: ')
        s.append('')
        s.append('   top.L: '+str(self.__src_pts[0]))
        s.append('   bot.L: '+str(self.__src_pts[1]))
        s.append('   bot.R: '+str(self.__src_pts[2]))
        s.append('   top.R: '+str(self.__src_pts[3]))
        s.append('')

        s.append(' desination points: ')
        s.append('')
        s.append('   top.L: '+str(self.__dst_pts[0]))
        s.append('   bot.L: '+str(self.__dst_pts[1]))
        s.append('   bot.R: '+str(self.__dst_pts[2]))
        s.append('   top.R: '+str(self.__dst_pts[3]))
        s.append('')
        
        s.append(' warped image shape: ')
        s.append('')
        if not self.__output is None:
            s.append('   '+str(self.__output.shape))
        else:
            s.append('   '+str(None))
        s.append('')
        
        s.append(' transform matrix: ')
        s.append('')
        s.append('   '+str(self.__transform_matrix))
        s.append('')
        s.append('')
        
        s.append('')
        
        return '\n'.join(s)


class PlotImageOp(PipelineOp):
    def __init__(self, img, cmap='gray', interpolation='none', aspect='auto'):
        PipelineOp.__init__(self)
        self.__img = np.copy(img)
        self.__cmap = cmap
        self.__interpolation = interpolation
        self.__aspect = aspect

    def perform(self):
        fig1 = plt.figure(figsize=(10,8))
        ax = fig1.add_subplot(111)
        ax.imshow(self.__img, cmap=self.__cmap, interpolation=self.__interpolation, aspect=self.__aspect)
        plt.tight_layout()
        plt.show()
        return self
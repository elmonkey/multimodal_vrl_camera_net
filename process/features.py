'''
Ver2 - Update on Jan 10, 2015

Created on Jan 26, 2014

1) Hog  feature from skimage (scikitlearn)
ref: 
http://scikit-image.org/docs/dev/auto_examples/plot_hog.html#example-plot-hog-py

Status - WORKS

2-3) Geometric moments: computes 0-3rd geometric moments on an image that cropped,
resized and sectioned into 6x6 (=36) blocks. Moments are computed on each block 
(starting at block coordinates (1,1)) and concatenated to form a 6x6x10 feature 
vector per image.  
    - Binary images, uses opencv implementation
    - Grayscale images 
        (ver1): naive, see the opencv documentation 
        (ver2): otmized using np.mesh (about 20x faster)

Status - WORKS

4) Hu Moments (Chris Wheat): Uses raw, central, and scale invariant to compute 
                            rotation and scale invariant Hu Moments
                            8-element vector -- added in version 2

5-6) Normalized R tranform (sum of radon transforms along rho)
 ref:
 http://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=radon#skimage.transform.radon
    Uses the scikit.transform.radon implementation.
    - R transform for the binary/mask image
    - R transform for the gray scale/depth image
    R transform is normalized by max(R)
Status - Testing

@author: Carlos
'''
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure  # , data, color
from skimage.transform import radon  # , rescale

# # *****************************************************************************
# # moments


def raw_moments(im, i, j):
    """
        Raw Moments

        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    nx, ny = im.shape
    lx = np.arange(nx)
    ly = np.arange(ny)
    yv, xv = np.meshgrid(lx, ly, indexing='ij')
    return np.sum(im * ((xv)**i) * ((yv)**j))


def central_moments(im, i, j):
    """
        Central Moments

        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    nx, ny = im.shape
    lx = np.arange(nx)
    ly = np.arange(ny)
    yv, xv = np.meshgrid(lx, ly, indexing='ij')
    m00 = np.sum(im)
    m10 = np.sum(xv * im)
    m01 = np.sum(yv * im)
    xavg = m10 / (m00)
    yavg = m01 / (m00)
    xavg = np.ones([nx, ny]) * xavg
    yavg = np.ones([nx, ny]) * yavg
    return np.sum(im * ((xv - xavg)**i) * ((yv - yavg)**j))


def scale_invarient_moments(im, i, j):
    """
        Scale Invariant Moments

        @param im - image numpy array
        @param i - x parameter
        @param j - y parameter
    """
    return central_moments(im, i, j) / (np.sum(im)**(1 + ((i + j) / 2)))


def hu(im):
    """
    A few scale, rotation and position invariant moment features
    input:
        im, numpy array 
    output:
        numpy array, 8-element long Hu Moment Vector
        @param im - image numpy array
    """
    # CentralMoments: Scale Invariant
    n11 = scale_invarient_moments(im, 1, 1)
    n02 = scale_invarient_moments(im, 0, 2)
    n20 = scale_invarient_moments(im, 2, 0)
    n30 = scale_invarient_moments(im, 3, 0)
    n03 = scale_invarient_moments(im, 0, 3)
    n21 = scale_invarient_moments(im, 2, 1)
    n12 = scale_invarient_moments(im, 1, 2)
    # HuMoments: Scale (Central) + Rotation Invariant
    I0 = n20 + n02
    I1 = (n20 - n02)**2 + 4 * n11**2
    I2 = (n30 - 3 * n12)**2 + (3 * n21 - n03)**2
    I3 = (n30 + n12)**2 + (n21 + n03)**2
    I4 = (n30 - 3 * n12) * (n30 + n12) * (((n30 + n12)**2 - 3 * (n21 + n30)**2) +
                                          (3 * n21 - n03) * (n21 - n03) * (3 * (n30 + n12)**2 - (n21 + n30)**2))
    I5 = (n20 - n02) * ((n30 + n12)**2 - (n21 + n03)**2) + 4 * n11 * (n30 + n12) * (n21 + n03)
    I6 = (3 * n21 - n03) * (n30 + n21) * (((n30 + n12)**2 - 3 * (n21 + n03)**2) -
                                          (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12)**2 - (n21 + n03)**2))
    I7 = n11 * ((n30 + n21)**2 - (n03 + n21)**2) - (n20 - n02) * (n30 + n21) * (n03 + n21)
    return np.array([I0, I1, I2, I3, I4, I5, I6, I7])


# # ****************************************************************************
def compute_hog(image, mask=None, vis=False):
    '''Computes HOG on a grayscale image.
    Inputs: 
        image:= numpy array (2D or 3D)
        mask := single channel 2D image 
        vis:= visualization flag (bool)
    Outputs: 
        fd: = numpy array, hog feature descriptor, length depends on parameters 
              and image dimensions (1L, ?L)
        hog_image_rescaled: = 2D single channel image used for displaying the 
                              hog histogram pixel intensities [0,255]
    Pre-Process image:
    1) Set image range [0, 1] and dtype = float64
    2) IF 3 channel image(color): convert to gray, ELSE gray = image
    >> (numpyarray, bool) -> 1D numpyarray, 2D numpy array [0, 255]
    '''
    # Check if image properties: type & range
    if image.dtype == "uint8":
        image = np.float64(image.astype(float) / 255)
    # Number of channels: rgb to grayscale if needed
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Check if mask is provided. If not, then use the whole image
    if mask == None:
        mask = np.ones(gray.shape, gray.dtype)

    def apply_mask(gray, mask):
        '''applies a mask to a grayscale image
        1. ensure mask is binary and same # of channels as the image
        '''
        mask = np.uint8(mask.astype(float) / np.max(mask))  # binary [0,1]
        # Apply the mask via bit-wise AND:
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        return masked
    # apply mask

    gray_masked = apply_mask(gray, mask)
    # fd,hog_image = hog(gray, orientations=8, pixels_per_cell=(16, 16),
    #                    cells_per_block=(1,1),visualise=True)
    fd, hog_image = hog(gray_masked, orientations=4, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualise=True)  # , normalise=True)
    fd = fd.reshape(1, fd.shape[0])
    # Re-scale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    hog_image_rescaled = np.uint8(hog_image_rescaled / np.max(hog_image_rescaled) * 255)
    if vis:
        winname = 'HOG: Masked input | HOG Vis'
        cv2.imshow(winname, np.hstack((gray_masked, hog_image_rescaled)))
        #cv2.moveWindow(winname, 20, 20)
        cv2.waitKey(10)
    return fd, hog_image_rescaled
# compute_hog


# #############################################################################
# ********************************************
# GEOMETRIC MOMENTS:
#     BINARY IMAGE
# ********************************************
def binary_moments(mask, vis=False, cropflag=False):
    """ 
    Calls the function that executes the image geometric moments on each of the
    36 (6x6) blocks
    inputs: 
        mask:= the 2D single channel black and white image in range [0,1]
        vis := boolean, display ea of the 6x6 blocks & image w overlayed grid
    output: 
        moms:= np.array, 10 elements from ea of the 36 blocks (36x10 =360)
               shape:1Lx(10momentelements x 6rowblocks x 6colblocks)=(1L,360L)
               L2-normalized
    >> (numpyarray(h,w,1L), bool) -> numpyarray(1L,360L)
    NOTE:
    Checks the input image properties and attempts to adjust them to uint8 
    single channel 2D array, [0,1]
    """
    # Initialize the moment array to zeros:
    moms = np.zeros((1, 10 * 37), dtype=np.float)

    if vis:
        cv2.imshow('input', mask)
    if len(mask.shape) > 2:
        # print 'convert to single channel'
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if len(np.unique(mask)) > 2:
        # print "Warning: Binary moments require binary image"
        mask[mask > 0] == 255
    if np.max(mask) > 1:
        # print "Bin-Moms Warning: Image is not in [0,1] range. Attempting to normalize: dividing image by 255"
        mask = np.uint8(mask.astype(float) / np.max(mask))  # uint8 & [0,1]

    # Normalize the image:
    mask = np.uint8(mask / np.max(mask))  # binary [0,1]

    def drawblocks(image):
        """draws lines over the 6x6 blocks that will be used to compute the image geometric moments
        input: image
        output: image with the blocks """
        im = cv2.cvtColor(image * 255, cv2.COLOR_GRAY2BGR)
        w, h = image.shape
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))

        x = np.arange(hb, h, hb)
        y = np.arange(wb, w, wb)

        for i in range(len(x)):
            cv2.line(im, (x[i], 0), (x[i], w), (255, 0, 0))  # blue lines
        for i in range(len(y)):
            cv2.line(im, (0, y[i]), (h, y[i]), (0, 0, 255))  # red lines
        cv2.imshow('grid', im)
        return
# drawblocks

    def cropAndResizeMask(mask):
        """Crops and resizes mask to 100x300 pixels. 
        Input:
            mask - uint8 2D single channel binary[0,1] image (mask)
        Output:
            imcrop - silhouette cropped image
            k - flag that indicates when something is detected in the mask """
        rows, cols = np.nonzero(mask == 1)
        # basic check for when there is nothing to mask
        if ((not(rows.size and cols.size)) or ((rows.min() == rows.max()) and (cols.min() == cols.max()))):
            crop = mask
            k = False
            print "WARNING: Nothing to crop - returning black (empty) image"
        else:
            crop = mask[rows.min():rows.max(), cols.min():cols.max()]
            k = True
        # (100,300)) # paper uses these dimensions but nicer for 6x6 block processes
        imcrop = cv2.resize(crop, (96, 300))
        return imcrop, k
# cropresizeMask

    def computeMoments(mask):
        """Uses opencv built-in to return geometric moments for a binary [0,1] image (up to 3rd order)
        input: single channel binary image with values in range[0,1]
        output: geometric moments"""
        m = np.zeros((1, 10), dtype=float)
        M = cv2.moments(mask, 1)  # moment-dictionary {spatial, central, central normalized}
        m00 = M['m00']
        m10 = M['m10']
        m01 = M['m01']
        m20 = M['m20']
        m11 = M['m11']
        m02 = M['m02']
        m30 = M['m30']
        m21 = M['m21']
        m12 = M['m12']
        m03 = M['m03']
        m = np.array([m00, m10, m01, m11, m20, m02, m21, m12, m30, m03])
#         m = [m00, m10, m01, m20, m02, m30, m03]
        return m
# computeMoments

    def extracMoments(img, vis):
        """Calls the function that executes the image geometric moments on each
           of the 36 (6x6) blocks
        inputs: 
            img:= the 2D single channel black and white image in range [0,1]
            display:= boolean to display each of the 6x6 blocks and the image w 
                      the overlayed grid
        output: 
            moms:= a vector of length 360: 10 elements from ea of the 36 blocks 
                   (36x10 =360)
            moms has a form: [row1(col1-6), row2(col1-6),..., row6(col1-6)]"""
        if vis:
            drawblocks(img)
        w, h = img.shape
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))
        moms = np.zeros((1, 37 * 10), dtype=np.float)
        r = 1  # rows counter
        c = 1  # cols counter
        count = 1
        moms[0, 0:10] = computeMoments(img)
        # for all the blocks in the mask
        for i in range(0, w, wb):  # row
            for j in range(0, h, hb):  # column
                iend = (wb * r)
                jend = (hb * c)
                b = img[i:iend, j:jend]
                bshape = b.shape
                a = np.zeros((bshape[0], bshape[1], 3), dtype=np.uint8)
                if b.max() > 0:
                    cm = computeMoments(b)
                    moms[0, count * 10:(count + 1) * 10] = cm  # matrix
                if vis:
                    print "{} 0-3 mnts: {}; Block({},{}) ".format(count, moms[0, count * 10:(count + 1) * 10], r, c)
                    a = cv2.cvtColor(b * 255, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('current block', a)
                    cv2.waitKey(10)
                c += 1
                count += 1
            r += 1
            c = 1  # reset cols counter
        return moms
# extracMoments

# Crop and resize the mask
#     if cropflag == True:
    mask, _ = cropAndResizeMask(mask)
    moms = extracMoments(mask, vis)
    moms = moms / np.linalg.norm(moms)
    return moms  # moms.flatten(1)
# binary_moments


# #############################################################################
# ********************************************
# GEOMETRIC MOMENTS:
#     GRAYSCALE IMAGE
# ********************************************
# #############################################################################
def grayscale_moments(gray, mask=None, vis=False, cropflag=False):
    """ Calls the function that executes the image geometric moments on each of 
        the 36 (6x6) blocks
    inputs: 
        gray:= numpyarray, 2D single channel grayscale in range [0 255]
        mask:= numpyarray, 2D single channel binary image in range [0,1]
        vis := boolean, display ea of the 6x6 blocks & image w overlayed grid
    output: 
        moms:= np.array, 10 elements from ea of the 36 blocks (36x10 =360)
               shape:1Lx(10momentelements x 6rowblocks x 6colblocks)=(1L,360L)
               L2-normalized.
    >> (numpyarray(h,w,1L), bool) -> numpyarray(1L,360L)
    """
    # Initialize the moment array to zeros:
    # moms = np.zeros((1, 37*10), dtype=np.float)

    # Check if mask is provided. If not, then use the whole image
    if mask == None:
        mask = np.ones(gray.shape, gray.dtype)

    def drawblocks(image):
        """draws lines over the 6x6 blocks that will be used to compute the image geometric moments
        input: image
        output: image with the blocks """
        # cv2.imshow('test',image)
        # cv2.waitKey(0)
        im = cv2.cvtColor(np.uint8(image / np.max(image) * 255), cv2.COLOR_GRAY2BGR)
        w, h = image.shape
        #  w=s[0]
        #  h=s[1]
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))

        x = np.arange(hb, h, hb)
        y = np.arange(wb, w, wb)

        for i in range(len(x)):
            cv2.line(im, (x[i], 0), (x[i], w), (255, 0, 0))  # blue lines
        for i in range(len(y)):
            cv2.line(im, (0, y[i]), (h, y[i]), (0, 0, 255))  # red lines
        cv2.imshow('grid', im)
        return
    # drawblocks

    def cropAndResizeMask(mask, image):
        """Crops and resizes both (mask & image) based on mask values to 96x300 
           pixels. 
        Input:
            mask - uint8 2D single channel binary[0,1] image (mask)
            image - uint8 2D single channel gray-scale image 
        Output:
            imcrop - silhouette cropped image
            k - flag that indicates when something is detected in the mask """
        # ensure that the mask is binary in the [0,1] range
        mask = mask / np.max(mask)
        rows, cols = np.nonzero(mask == 1)
        # basic check for when there is nothing to mask
        if ((not(rows.size and cols.size)) or ((rows.min() == rows.max()) and (cols.min() == cols.max()))):
            msk_crop = mask
            img_crop = image
            k = False
            print "WARNING: Nothing to crop - returning black (empty) image"
        else:
            msk_crop = mask[rows.min():rows.max(), cols.min():cols.max()]
            img_crop = image[rows.min():rows.max(), cols.min():cols.max()]
            k = True
        # (100,300)) # paper uses these dimensions but nicer for 6x6 block processes
        imcrop = cv2.resize(img_crop, (96, 300))
        maskcrop = cv2.resize(msk_crop, (96, 300))
        return imcrop, maskcrop, k
    # cropresizeMask

    def geometricmoments(gray):
        """computes the 0-3rd geometric moments of a 2-dimensional grayscale 
        images and returns a dictionary. Grayscale image pixel values are 
        normalized to [0,1] 
        >> Note: See dictionary definition for dictionary keys (ms_keys)
        >> (numpy.array) -> dictionary
        >> input: 
            gray := 2D numpy array of size M,N
        >> output: 
            moments:= 10-element dictionary, 0 to 3rd geometric moments 'pq'; 
                      p+q = moment order
        """
        # *********** THE DICTIONARY THAT TILL CONTAIN THE MOMENTS!
        moments = {}
        ms_list = [[1, 0], [0, 1], [2, 0], [1, 1], [0, 2], [3, 0], [2, 1], [1, 2], [0, 3]]
        ms_keys = ['m10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']
        # =============== Using Opencv expression for geo moments (see opencv docs):
        # Image dimensions:
        M, N = gray.shape
        # Zeroth moment = max(integral image):
        integral_image = cv2.integral(gray)
        moments['m00'] = np.max(integral_image)
        # The remaining moments:
        k = 0  # index for the dictionary keys
        for l in ms_list:
            p, q = l
            m = raw_moments(gray, p, q)
            # for xi in np.arange(0,M):
            #    xp = xi**p
            #    for yj in np.arange(0,N):
            #        m += xp * (yj**q) * (gray[xi,yj])
            # #   end yj
            # # end xi
            moments[ms_keys[k]] = m
            k += 1
        # end l
        return moments
    # grayscale_geometric_moments

    def computeMoments(mask):
        """Uses opencv built-in to return geometric moments for a binary [0,1] image (up to 3rd order)
        input: single channel binary image with values in range[0,1]
        output: geometric moments"""
        M = geometricmoments(mask)  # moment-dictionary {spatial, central, central normalized}
        m = np.zeros((1, 10), dtype=float)
        m00 = M['m00']
        m10 = M['m10']
        m01 = M['m01']
        m20 = M['m20']
        m11 = M['m11']
        m02 = M['m02']
        m30 = M['m30']
        m21 = M['m21']
        m12 = M['m12']
        m03 = M['m03']
        m = np.array([m00, m10, m01, m11, m20, m02, m21, m12, m30, m03])
    #         m = [m00, m10, m01, m20, m02, m30, m03]
        return m
    # computeMoments

    def extracMoments(img, display=False):
        """Calls the function that executes the image geometric moments on each
           of the 36 (6x6) blocks.
        inputs: 
            img    := the 2D single channel black and white image in range [0,1]
            display:= boolean to display each of the 6x6 blocks and the image w 
                      the overlayed grid
        output: 
            moms:= vector of length 360: 10 elements from ea of the 36 blocks 
                   (36x10 =360)
            moms has a form: [row1(col1-6), row2(col1-6),..., row6(col1-6)]
        """
        # Change the range of the image to [0,1]:
        img = img.astype(float)
        img = img / np.max(img)
        # if display then draw the blocks on the image and display it
        if display:
            drawblocks(img)
        w, h = img.shape
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))
        moms = np.zeros((1, 37 * 10), dtype=np.float)
        r = 1  # rows counter
        c = 1  # cols counter
        count = 1
        moms[0, 0:10] = computeMoments(img)
        # for all the blocks in the array (image):
        for i in range(0, w, wb):  # row
            for j in range(0, h, hb):  # column
                iend = (wb * r)
                jend = (hb * c)
                b = img[i:iend, j:jend]
                if b.max() > 0:
                    moms[0, count * 10:(count + 1) * 10] = computeMoments(b)
                if display:
                    print "0-3-mnts: ", moms[0, count * 10:(count + 1) * 10], ', for block(r,c): ', r, c
                    cv2.imshow('current block', b)
                    # cv2.waitKey(50)
                c += 1
                count += 1
            r += 1
            c = 1  # reset cols counter
        return moms
    # extracMoments

    def apply_mask(gray, mask):
        '''applies a mask to a grayscale image
        mask = single channel binary numpy array
        gray = single channel numpy array
        '''
        # Apply the mask via bit-wise AND:
        mask = np.uint8(mask.astype(float) / np.max(mask))  # binary [0,1]
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        return masked
    # apply_mask

# Pre-Process the images:
#     if cropflag == True:
    im, mask, _ = cropAndResizeMask(mask, gray)
    im = apply_mask(im, mask)
#     else: im = gray
    im = np.uint8(im.astype(float) / np.max(im) * 255)

    # if not f1: # if the mask exists, extract the image Geo-moments
    moms = extracMoments(im, vis)
    if vis:
        cv2.imshow('masked image', im)
        cv2.imshow('mask', mask * 255)
        # cv2.waitKey(0)
    return moms / np.linalg.norm(moms)  # moms.flatten(1)
# grayscale_moments
# #############################################################################


# #############################################################################
# ********************************************
# HU MOMENTS: SCALE, SHIFT & ROTATION INVARIANT
#    GRAY or BINARY IMAGE
# ********************************************
# #############################################################################
def hu_moments(gray, mask=None, vis=False, cropflag=False):
    """ 
    Calls the function that executes the image geometric moments on the whole 
    image and on each of the 36 (6x6) blocks
    inputs: 
        gray:= numpyarray, 2D single channel grayscale in range [0 255]
        mask:= numpyarray, 2D single channel binary image in range [0,1]
        vis := boolean, display ea of the 6x6 blocks & image w overlayed grid
    output: 
        moms:= np.array, 8 elements from the whole image and ea of the 36 blocks
         (1 + 36) x 8 = 296
               shape:1Lx(10momentelements x 6rowblocks x 6colblocks)=(1L,360L)
               L2-normalized.
    >> (numpyarray(h,w,1L), bool) -> numpyarray(1L,360L)
    """
    # Initialize the moment array to zeros:
    #moms = np.zeros((1, 37*10), dtype=np.float)

    # Check if mask is provided. If not, then use the whole image
    if mask == None:
        mask = np.ones(gray.shape, gray.dtype)

    def drawblocks(image):
        """draws lines over the 6x6 blocks that will be used to compute the image geometric moments
        input: image
        output: image with the blocks """
    #     cv2.imshow('test',image)
    #     cv2.waitKey(0)
        im = cv2.cvtColor(np.uint8(image / np.max(image) * 255), cv2.COLOR_GRAY2BGR)
        w, h = image.shape
#         w=s[0]
#         h=s[1]
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))

        x = np.arange(hb, h, hb)
        y = np.arange(wb, w, wb)

        for i in range(len(x)):
            cv2.line(im, (x[i], 0), (x[i], w), (255, 0, 0))  # blue lines
        for i in range(len(y)):
            cv2.line(im, (0, y[i]), (h, y[i]), (0, 0, 255))  # red lines
        cv2.imshow('grid', im)
        return
    # drawblocks

    def cropAndResizeMask(mask, image):
        """Crops and resizes both (mask & image) based on mask values to 96x300 
           pixels. 
        Input:
            mask - uint8 2D single channel binary[0,1] image (mask)
            image - uint8 2D single channel gray-scale image 
        Output:
            imcrop - silhouette cropped image
            k - flag that indicates when something is detected in the mask """
        # ensure that the mask is binary in the [0,1] range
        mask = mask / np.max(mask)
        rows, cols = np.nonzero(mask == 1)
        # basic check for when there is nothing to mask
        if ((not(rows.size and cols.size)) or ((rows.min() == rows.max()) and (cols.min() == cols.max()))):
            msk_crop = mask
            img_crop = image
            k = False
            print "WARNING: Nothing to crop - returning black (empty) image"
        else:
            msk_crop = mask[rows.min():rows.max(), cols.min():cols.max()]
            img_crop = image[rows.min():rows.max(), cols.min():cols.max()]
            k = True
        # (100,300)) # paper uses these dimensions but nicer for 6x6 block processes
        imcrop = cv2.resize(img_crop, (96, 300))
        maskcrop = cv2.resize(msk_crop, (96, 300))
        return imcrop, maskcrop, k
    # cropresizeMask

    def computeHu(im):
        """
        A few scale, rotation and position invariant moment features
        input:
            im, numpy array 
        output:
            numpy array, 8-element long Hu Moment Vector
            @param im - image numpy array
        """
        # CentralMoments: Scale Invariant
        n11 = scale_invarient_moments(im, 1, 1)
        n02 = scale_invarient_moments(im, 0, 2)
        n20 = scale_invarient_moments(im, 2, 0)
        n30 = scale_invarient_moments(im, 3, 0)
        n03 = scale_invarient_moments(im, 0, 3)
        n21 = scale_invarient_moments(im, 2, 1)
        n12 = scale_invarient_moments(im, 1, 2)
        # HuMoments: Scale (Central) + Rotation Invariant
        I0 = n20 + n02
        I1 = (n20 - n02)**2 + 4 * n11**2
        I2 = (n30 - 3 * n12)**2 + (3 * n21 - n03)**2
        I3 = (n30 + n12)**2 + (n21 + n03)**2
        I4 = (n30 - 3 * n12) * (n30 + n12) * (((n30 + n12)**2 - 3 * (n21 + n30)**2) +
                                              (3 * n21 - n03) * (n21 - n03) * (3 * (n30 + n12)**2 - (n21 + n30)**2))
        I5 = (n20 - n02) * ((n30 + n12)**2 - (n21 + n03)**2) + 4 * n11 * (n30 + n12) * (n21 + n03)
        I6 = (3 * n21 - n03) * (n30 + n21) * (((n30 + n12)**2 - 3 * (n21 + n03)**2) -
                                              (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12)**2 - (n21 + n03)**2))
        I7 = n11 * ((n30 + n21)**2 - (n03 + n21)**2) - (n20 - n02) * (n30 + n21) * (n03 + n21)
        return np.array([I0, I1, I2, I3, I4, I5, I6, I7])
    # hu

    def extracMoments(img, display=False):
        """Calls the function that executes the image geometric moments on each
           of the 36 (6x6) blocks.
        inputs: 
            img    := the 2D single channel black and white image in range [0,1]
            display:= boolean to display each of the 6x6 blocks and the image w 
                      the overlayed grid
        output: 
            moms:= vector of length 360: 10 elements from ea of the 36 blocks 
                   (36x10 =360)
            moms has a form: [row1(col1-6), row2(col1-6),..., row6(col1-6)]
        """
        # Change the range of the image to [0,1]:
        img = img.astype(float)
        img = img / np.max(img)
        # if display then draw the blocks on the image and display it
        if display:
            drawblocks(img)
        w, h = img.shape
        n = 6  # given by paper 6x6 = 36 blocks
        wb = np.uint8(np.floor(w / n))
        hb = np.uint8(np.floor(h / n))
        moms = np.zeros((1, 37 * 8), dtype=np.float)
        r = 1  # rows counter
        c = 1  # cols counter
        count = 1
        moms[0, 0:8] = computeHu(img)
        # for all the blocks in the array (image):
        for i in range(0, w, wb):  # row
            for j in range(0, h, hb):  # column
                iend = (wb * r)
                jend = (hb * c)
                b = img[i:iend, j:jend]
                if b.max() > 0:
                    moms[0, count * 8:(count + 1) * 8] = hu(b)
                if display:
                    print "0-3-mnts: ", moms[0, count * 8:(count + 1) * 8], ', for block(r,c): ', r, c
                    cv2.imshow('current block', b)
                    # cv2.waitKey(50)
                c += 1
                count += 1
            r += 1
            c = 1  # reset cols counter
        return moms
    # extracMoments

    def apply_mask(gray, mask):
        '''applies a mask to a grayscale image
        mask = single channel binary numpy array
        gray = single channel numpy array
        '''
        # Apply the mask via bit-wise AND:
        mask = np.uint8(mask.astype(float) / np.max(mask))  # binary [0,1]
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        return masked
    # apply_mask

# Pre-Process the images:
#     if cropflag == True:
    im, mask, _ = cropAndResizeMask(mask, gray)
    im = apply_mask(im, mask)
#     else: im = gray
    im = np.uint8(im.astype(float) / np.max(im) * 255)

    # if not f1: # if the mask exists, extract the image Geo-moments
    moms = extracMoments(im, vis)
    if vis:
        cv2.imshow('masked image', im)
        cv2.imshow('mask', mask * 255)
        # cv2.waitKey(0)
        print moms / np.linalg.norm(moms)
    return moms / np.linalg.norm(moms)  # moms.flatten(1)
# grayscale_moments
# #############################################################################


# ********************************************
# ORIENTATION MATRIX
# ********************************************
def orientationMatrix(joints):
    """Given a set of joints computes the 3D orientation vectors in 180 modulus.
    input: 
        joints:=, np.array, [joint#, (x, y, z)]
        joints.shape = 15 x 3
        joint expected order:
            0) head            5) R wrist        10) L elbow
            1) neck            6) R hip          11) L wrist
            2) torso           7) R knee         12) L hip
            3) R shoulder      8) R foot         13) L knee
            4) R elbow         9) L shoulder     14) L foot
    1 Torso: origin
    1 Neck: vertical
    2 Shoulders: horizontal (Right & Left)
    z = N = cross_product(x = Torso_left shoulder, y = Torso_Neck)
    i = [1;0;0]; j=[0;1;0]; k=[0;0;1]
    orientation_vector = [X,Y,Z]
              horizontal:[     ]
                vertical:[     ]
                   depth:[     ]
    """
    neck = joints[1, :]
    torso = joints[2, :]
    shoulder = joints[9, :]
#
    # Compute and normalize the 3D vectors
    x_hat = (shoulder - torso) / np.linalg.norm(shoulder - torso)
    y_hat = (neck - torso) / np.linalg.norm(neck - torso)
#     depth = np.cross(x_hat, y_hat)
    z_hat = np.cross(x_hat, y_hat) / np.linalg.norm(np.cross(x_hat, y_hat))
#
    orientation_matrix = np.zeros((4, 3), dtype=np.float)
    orientation_matrix[0, :] = torso  # also the origin
    orientation_matrix[1, :] = x_hat
    orientation_matrix[2, :] = y_hat
    orientation_matrix[3, :] = z_hat
#
    return orientation_matrix
# orientationMatrix


# #############################################################################
# ********************************************
# R TRANSFORM (vaiation of the radon transform)
# ********************************************
def RT(image, mask, N=180, vis=False):
    """Computes the radon transform and the R-transform of a given single 2D 
    image. Sinogram access: row_index=rho_index, col_index=theta_index.
    The images are cropped by the user pixels mask and resize to 96,100.

    inputs:
        image:= numpyarray,depth image; range [0,1]; shape=2D(w,h); dtype=float64
        mask := numpyarray, shape=2D single channel; binary image in range [0,1]
        vis  := boolean, display ea of the 6x6 blocks & image w overlayed grid
        N    := int, # samples in theta for the space [0:180]

    outputs: L2-normalized
        R_depth:= numpyarray, dtype=float64; shape=len(theta); range[0,1]
                 R tranforms for the depth image
        R_mask:= numpyarray, dtype=float64; shape=len(theta); range[0,1]
                 R tranforms for the mask/binary image
    """
    # ===== Radon Parameters
    # h,w = gray.shape
    # N = 180 # np.max([h, w, 180]) # at least 180 samples
    theta = np.linspace(0., 180., N, endpoint=True)

    # auxiliary function
    def applyMask(gray, mask):
        '''applies a binary mask to a grayscale image
        1. ensure mask & binary images have the same # of channels
        '''
        mask = np.uint8(mask.astype(float) / np.max(mask))  # binary [0,1]
        # Apply the mask via bit-wise AND:
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        return masked
    # applymask

    def cropandResize_MaskandImage(mask, image):
        """Crops and resizes both (mask & image) based on mask values to 100x300 pixels. 
        Input:
            mask - uint8 2D single channel binary[0,1] image (mask)
            image - uint8 2D single channel gray-scale image 
        Output:
            imcrop - silhouette cropped image
            k - flag that indicates when something is detected in the mask 
        >> (2D_numpyarray, 2D_numpyarray) -> (1D_numpyarray, 1D_numpyarray)
        """
        # ensure that the mask is binary in the [0,1] range
        mask = mask / np.max(mask)
        rows, cols = np.nonzero(mask == 1)
        # basic check for when there is nothing to mask
        if ((not(rows.size and cols.size)) or ((rows.min() == rows.max()) and (cols.min() == cols.max()))):
            msk_crop = mask
            img_crop = image
            k = False
            print "WARNING: Nothing to crop - returning black (empty) image"
        else:
            msk_crop = mask[rows.min():rows.max(), cols.min():cols.max()]
            img_crop = image[rows.min():rows.max(), cols.min():cols.max()]
            k = True
        # (100,300)) # paper uses these dimensions but nicer for 6x6 block processes
        imcrop = cv2.resize(img_crop, (96, 300))
        maskcrop = cv2.resize(msk_crop, (96, 300))
        return imcrop, maskcrop, k
    # cropMaskandImage

    # === radon transforms
    def computeR(image, vis):
        #  === Check if image properties: type & range
        if image.dtype == "uint8":
            image = image.astype(float) / np.max(image)
        # Number of channels: 3channels to 1channel if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        sinogram = radon(gray, theta=theta)  # , circle=True)#
        # === R_transform
        R = np.zeros(N)  # R_transform
        for i in xrange(N):
            a = sinogram[:, i]
            R[i] = a.sum()
        Rn = R / np.max(R)  # normalized R transform

        if vis:
            print "Displaying image and its r and R transforms"
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.subplot(121)
            plt.title("Radon(r) transform\n(Sinogram)")
            plt.xlabel("Projection Angle (theta) [degree]")
            plt.ylabel("Projection Position (rho) [pixels]")
            plt.imshow(sinogram, cmap=plt.cm.Greys_r,  # hot,
                       extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
            plt.subplot(122)
            plt.title("R-transform\nR(normalized)")
            plt.xlabel("Projection Angle (theta) [degree]")
            plt.ylabel("Projection Position (rho) [pixels]")
            plt.plot(theta, Rn)
            plt.subplots_adjust(hspace=0.4, wspace=0.5)

            cv2.imshow("image to R transform", np.uint8(image * 255))
            plt.show()
            cv2.waitKey(10)
            plt.close()

        return Rn
    # computeR

# === PRE-PROCESS the images
    gray, mask, f1 = cropandResize_MaskandImage(mask, image)
    gray = applyMask(gray, mask)

#
# === R transforms
    if f1:  # if the mask exists, compute the R transform
        R_gray = computeR(gray, vis)
        R_mask = computeR(mask, vis)
    else:
        print "Nothing to R-transform"
        R_gray = np.zeros(N)
        R_mask = R_gray

    return R_gray, R_mask
# R_Transform


# ##############################################################################
# TESTING!
#
# dif __name__ == "__main__":
# ######## LOAD SOME IMAGES TO TEST:

#     image=color.rgb2gray(data.lena()) # float & [0,1]
#     sc_image = np.uint8( image/np.max(image) * 255) # uint8 & [0,255]
#
#     depthname = '../waveL/carlos/View_0_ALL_devs/seq1/pose1_dev2_DepthMap_13.png'
#     maskname  = '../waveL/carlos/View_0_ALL_devs/seq1/pose1_dev2_UserPixels_13.png'
#     gray = cv2.cvtColor(cv2.imread(depthname), cv2.COLOR_BGR2GRAY)
#     mask = cv2.cvtColor(cv2.imread(maskname),  cv2.COLOR_BGR2GRAY)
#
#
#     # call hog function
#     hog_f,hog_image = compute_hog(sc_image, vis=True)
#     print hog_image.shape, hog_image.dtype, type(hog_image)
#
#     # THE MOMENTS:
#     bin_moms = binary_moments(mask, vis=False) # binary moments
#     gray_moms = grayscale_moments(mask,gray, vis=False) # grayscale moments
#     # Print some info about the extracted features:
#     print 'hog vector: ', hog_f.shape
#     print 'bin vector: ', bin_moms.shape #np.max(bin_moms)
#     print 'gry vector: ', gray_moms.shape
# #main
# ##############################################################################
# main()

"""
Created on Mon Jul  3 09:54:53 2017

@author: Julian

Use to record with the primesense camera RGB and depth cameras and the seek thermal camera
"""
import numpy as np
import cv2
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from seek_camera import thermal_camera

#############################################################################
# set-up primesense camera
dist = '/home/julian/Install/OpenNI2-x64/Redist'
# Initialize openni and check
openni2.initialize(dist)
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"
# Register the device
prime = openni2.Device.open_any()
# Create the streams
rgb_stream = prime.create_color_stream()
depth_stream = prime.create_depth_stream()
# Configure the depth_stream -- changes automatically based on bus speed
# print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))
# Start the streams
rgb_stream.start()
depth_stream.start()
# Synchronize the streams
prime.set_depth_color_sync_enabled(True)
# IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
prime.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),
                        dtype=np.uint8).reshape(240, 320, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def get_depth():
    """
    Returns numpy ndarrays representing the raw and ranged depth images.
    Outputs:
        dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
        d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255    
    Note1: 
        fromstring is faster than asarray or frombuffer
    Note2:     
        .reshape(120,160) #smaller image for faster response 
                OMAP/ARM default video configuration
        .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
                Requires .set_video_mode
    """
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),
                         dtype=np.uint16).reshape(240, 320)  # Works & It's FAST
    # Correct the range. Depth images are 12bits
    d4d = np.uint8(dmap.astype(float) * 255 / 2**12 - 1)
    d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
    return dmap, d4d


def get_8bit(frame):
    h, w = frame.shape
    output = np.zeros((h, w, 3), dtype='uint8')
    temp1 = frame / 256
    temp2 = frame - temp1 * 256
    output[:, :, 1] = temp1.astype('uint8', casting='unsafe')
    output[:, :, 0] = temp2.astype('uint8', casting='unsafe')
    output[:, :, 2] = output[:, :, 0]
    output = output.astype('uint8')
    return output


# ==============================================================================
# Video .avi output setup
# ==============================================================================
#############################################################################
# setup thermal camera
therm = thermal_camera()
# setup needed inormation for display and
rgb_frame = get_rgb()
dmap, depth_frame = get_depth()
ir_frame = therm.get_frame()
rgb_h, rgb_w, channels = rgb_frame.shape
ir_h, ir_w = ir_frame.shape
depth_h, depth_w, depth_channels = depth_frame.shape
ir_place = np.zeros((rgb_h, ir_w, channels), dtype='uint8')
depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
place_ir = rgb_h / 2 - ir_h / 2
place_depth = rgb_h / 2 - depth_h / 2
fps = 8.0

# ==============================================================================
# THE CODECS
# ==============================================================================
fourcc = cv2.cv.CV_FOURCC('M', 'P', 'E', 'G')
video_location = '/home/julian/Videos/'
rgb_vid = cv2.VideoWriter(video_location + 'rgb_vid.avi', fourcc, fps, (rgb_w, rgb_h), 1)
ir_vid = cv2.VideoWriter(video_location + 'ir_vid.avi', fourcc, fps, (ir_w, ir_h), 1)
ir_full_vid = cv2.VideoWriter(video_location + 'ir_full_vid.avi', fourcc, fps, (ir_w, ir_h), 1)
depth_vid = cv2.VideoWriter(video_location + 'depth_vid.avi', fourcc, fps, (depth_w, depth_h), 1)
depth_full_vid = cv2.VideoWriter(video_location + 'depth_full_vid.avi',
                                 fourcc, fps, (depth_w, depth_h), 1)

###############################################################################


def nothing(x):
    pass


drawing = False
ix = 0
iy = 0
rgb_pts = np.zeros((2, 2), dtype='int')
ir_pts = rgb_pts.copy()


def get_rgb_pts(event, x, y, flags, param):
    global drawing, ix, iy, rgb_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        rgb_pts[0, 0] = x
        rgb_pts[0, 1] = y
        ix, iy = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        rgb_pts[1, 0] = x
        rgb_pts[1, 1] = y
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(rgb_find, (ix, iy), (x, y), (0, 255, 0), 1)


def get_ir_pts(event, x, y, flags, param):
    global drawing, ix, iy, ir_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        ir_pts[0, 0] = x
        ir_pts[0, 1] = y
        ix, iy = x, y
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        ir_pts[1, 0] = x
        ir_pts[1, 1] = y
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(ir_find, (ix, iy), (x, y), (0, 255, 0), 1)


# pattern dimensions for the grid detection and drawing
r = 4
c = 4
pattern_size = (r, c)  # circles grid


def detectGrid(img):
    found, corners = cv2.findCirclesGrid(
        img, pattern_size, None, cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING)
    detected = False
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)  # subpixel accuracy
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        detected = True

    return img, corners, detected


# set-up homography collection
rgb_thresh = 125
rgb_blur = 1
ir_thresh = 100
ir_blur = 1
cv2.namedWindow('vid')
cv2.namedWindow('ir')
cv2.namedWindow('rgb')
cv2.createTrackbar('RGB Threshold', 'vid', rgb_thresh, 255, nothing)
cv2.createTrackbar('RGB Blur', 'vid', rgb_blur, 21, nothing)
cv2.createTrackbar('IR Threshold', 'vid', rgb_thresh, 255, nothing)
cv2.createTrackbar('IR Blur', 'vid', rgb_blur, 21, nothing)
cv2.setMouseCallback('rgb', get_rgb_pts)
cv2.setMouseCallback('ir', get_ir_pts)
###############################################################################
print ("Press 'esc' to terminate")
f = 0   # frame counter
num = 0
times = 0
done = False
while not done:
    k = cv2.waitKey(1) & 255
    # capture frames
    rgb_frame = get_rgb()
    full_ir = therm.get_frame()
    full_depth, depth_frame = get_depth()
    rgb_frame = cv2.flip(rgb_frame, 1)
    full_ir = cv2.flip(full_ir, 1)
    full_depth = cv2.flip(full_depth, 1)
    depth_frame = cv2.flip(depth_frame, 1)

    # make visible
    ir_frame = therm.get_8bit_frame(full_ir)
    ir_place[place_ir:place_ir + ir_h, :, :] = ir_frame
    depth_place[place_depth:place_depth + depth_h, :, :] = depth_frame

    times += 1
    if times == 80:  # space
        times = 0
        rgb_img = rgb_frame.copy()
        ir_img = ir_frame.copy()
        rgb_flag = False
        ir_flag = False
        leave = False
        rgb_find = rgb_img.copy()
        ir_find = ir_img.copy()
        while ((not rgb_flag) & (not leave)):
            g = cv2.waitKey(1) & 255
            cv2.imshow('rgb', rgb_find)
            if g == 27:
                leave = True
            elif g == 115:
                rgb_flag = True
        while ((not ir_flag) & (not leave)):
            g = cv2.waitKey(1) & 255
            cv2.imshow('ir', ir_find)
            if g == 27:
                leave = True
            elif g == 115:
                ir_flag = True
        rgb_pts += rgb_pts

        while not leave:
            g = cv2.waitKey(1) & 255

            rgb_temp = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
            rgb_temp = cv2.resize(rgb_temp, (640, 480), interpolation=cv2.INTER_AREA)
            rgb_temp = cv2.GaussianBlur(rgb_temp, (rgb_blur, rgb_blur), 0)
            ret, rgb_temp = cv2.threshold(rgb_temp, rgb_thresh, 255, cv2.THRESH_BINARY)
            rgb_temp = cv2.cvtColor(rgb_temp, cv2.COLOR_GRAY2BGR)

            ir_temp = cv2.GaussianBlur(ir_img, (ir_blur, ir_blur), 0)
            ret, ir_temp = cv2.threshold(ir_temp, ir_thresh, 255, cv2.THRESH_BINARY_INV)

            # detect pattern
            rgb_viz, rgb_corners, rgb_detected = detectGrid(
                rgb_temp[rgb_pts[0, 1]:rgb_pts[1, 1], rgb_pts[0, 0]:rgb_pts[1, 0], :])
            ir_viz, ir_corners, ir_detected = detectGrid(
                ir_temp[ir_pts[0, 1]:ir_pts[1, 1], ir_pts[0, 0]:ir_pts[1, 0], :])
            rgb_temp[rgb_pts[0, 1]:rgb_pts[1, 1], rgb_pts[0, 0]:rgb_pts[1, 0], :] = rgb_viz
            ir_temp[ir_pts[0, 1]:ir_pts[1, 1], ir_pts[0, 0]:ir_pts[1, 0], :] = ir_viz

            rgb_place = cv2.resize(rgb_temp, (320, 240))

            ir_place[place_ir:place_ir + 206, :, :] = ir_temp
            disp = np.hstack((ir_place, rgb_place))
            disp = np.uint8(disp)
            cv2.imshow('vid', disp)
            if rgb_detected:
                for i in range(16):
                    pos1 = int(rgb_corners[i, 0, 0] / 2 + rgb_pts[0, 0] / 2)
                    pos2 = int(rgb_corners[i, 0, 1] / 2 + rgb_pts[0, 1] / 2)
                    cv2.circle(depth_frame, (pos1, pos2), 3, (0, 0, 255), -1)
            cv2.imshow('depth', depth_frame)

            rgb_thresh = cv2.getTrackbarPos('RGB Threshold', 'vid')
            rgb_blur = cv2.getTrackbarPos('RGB Blur', 'vid')
            ir_thresh = cv2.getTrackbarPos('IR Threshold', 'vid')
            ir_blur = cv2.getTrackbarPos('IR Blur', 'vid')

            if (rgb_blur % 2) != 1:
                rgb_blur += 1
            if (ir_blur % 2) != 1:
                ir_blur += 1

            if g == 27:  # esc
                leave = True
            if ((g == 115) & rgb_detected & ir_detected):
                # resize corners
                rgb_pts = rgb_pts / 2
                rgb_corners = rgb_corners / 2
                rgb_corners[:, 0, 0] = rgb_corners[:, 0, 0] + rgb_pts[0, 0]
                rgb_corners[:, 0, 1] = rgb_corners[:, 0, 1] + rgb_pts[0, 1]
                ir_corners[:, 0, 0] = ir_corners[:, 0, 0] + ir_pts[0, 0]
                ir_corners[:, 0, 1] = ir_corners[:, 0, 1] + ir_pts[0, 1]
                # get homography
                H, mask = cv2.findHomography(rgb_corners.reshape(-1, 2),
                                             ir_corners.reshape(-1, 2), cv2.RANSAC, 44)
                Hinv, mask2 = cv2.findHomography(
                    ir_corners.reshape(-1, 2), rgb_corners.reshape(-1, 2), cv2.RANSAC, 44)
                H = np.hstack((H, [[0], [0], [0]]))
                Hinv = np.hstack((Hinv, [[0], [0], [0]]))
                distance = np.zeros((16, 1), dtype='uint16')
                for i in range(16):
                    pos1 = int(rgb_corners[i, 0, 1])
                    pos2 = int(rgb_corners[i, 0, 0])
                    distance[i] = np.average(full_depth[pos1 - 3:pos1 + 3, pos2 - 3:pos2 + 3])
                distance = np.reshape(distance, (4, 4))
                H = np.vstack((H, distance))
                Hinv = np.vstack((Hinv, distance))
                num += 1
                np.savetxt("/home/julian/Documents/Hmatrix_rgb_to_ir_" + str(num) + ".out", H)
                np.savetxt("/home/julian/Documents/Hinvmatrix_ir_to_rgb_" + str(num) + ".out", Hinv)
                leave = True

    # display and write video
    disp = np.hstack((depth_place, ir_place, rgb_frame))
    cv2.imshow("live", disp)
    rgb_vid.write(rgb_frame)
    ir_full_vid.write(get_8bit(full_ir))
    ir_vid.write(ir_frame)
    depth_full_vid.write(get_8bit(full_depth))
    depth_vid.write(depth_frame)

    f += 1
    print ("frame No.", f)
    if k == 27:  # esc key
        done = True

# release resources and destoy windows
rgb_stream.stop()
depth_stream.stop()
openni2.unload()
rgb_vid.release()
ir_vid.release()
ir_full_vid.release()
depth_vid.release()
depth_full_vid.release()
cv2.destroyWindow("vid")
cv2.destroyWindow("ir")
cv2.destroyWindow("rgb")
cv2.destroyWindow("live")
print ("Completed video generation using {} codec". format(fourcc))

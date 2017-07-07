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
    output = np.zeros((h,w,3))
    temp1 = frame >> 8
    temp2 = (frame << 8) >> 8 
    output[:,:,1] = temp1.astype('uint8', casting = 'unsafe')
    output[:,:,0] = temp2.astype('uint8', casting = 'unsafe')
    output[:,:,2] = output[:,:,0]
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
depth_h, depth_w, depth_channels = depth_frame.shape
ir_place = np.zeros((rgb_h, ir_w, channels), dtype='uint8')
ir_w, ir_h = ir_frame.shape
depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
place_ir = rgb_h / 2 - ir_h / 2
place_depth = rgb_h / 2 - depth_h / 2
fps = 8.0

# ==============================================================================
# THE CODECS
# ==============================================================================
fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
video_location = '/home/julian/Videos/pos_1_'
rgb_vid = cv2.VideoWriter(video_location+'rgb_vid.avi', fourcc, fps, (rgb_w, rgb_h), 1)
ir_vid = cv2.VideoWriter(video_location+'ir_vid.avi', fourcc, fps, (ir_w, ir_h), 1)
ir_full_vid = cv2.VideoWriter(video_location+'ir_full_vid.avi', fourcc, fps, (ir_w, ir_h), 1)
depth_vid = cv2.VideoWriter(video_location+'depth_vid.avi', fourcc, fps, (depth_w, depth_h), 1)
depth_full_vid = cv2.VideoWriter(video_location+'depth_full_vid.avi', fourcc, fps, (depth_w, depth_h), 1)

print ("Press 'esc' to terminate")
f = 0   # frame counter
done = False
while not done:
    k = cv2.waitKey(1) & 255
    # capture frames
    rgb_frame = get_rgb()
    full_ir = therm.get_frame()
    full_depth, depth_frame = get_depth()
    
    #make visible
    ir_frame = therm.get_8bit_frame(full_ir)
    ir_place[place_ir:place_ir + ir_h, :, :] = ir_frame
    depth_place[place_depth:place_depth + depth_h, :, :] = depth_frame
    
    #display and write video
    disp = np.hstack((depth_place, ir_place, rgb_frame))
    disp = cv2.flip(disp, 1)
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
cv2.destroyWindow("live")
print ("Completed video generation using {} codec". format(fourcc))

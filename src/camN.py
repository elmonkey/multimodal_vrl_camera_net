"""
Created on Thur Jul  6 09:54:53 2017

@author: Julian

Use to record with the primesense camera RGB and depth cameras and the seek thermal camera confirming using server
"""
import numpy as np
import cv2
import time
import os
import shutil
import pandas as pd
import client
from primesense import openni2  # , nite2
from primesense import _openni2 as c_api
from seek_camera import thermal_camera

# Device number
devN = 2

#############################################################################
# set-up primesense camera
dist = '/home/julian/Install/OpenNI2/OpenNI2-x64/Redist'
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

# ==============================================================================
# Server Communication setup
# ==============================================================================
#############################################################################


def talk2server(cmd='connect', devN=1):
    """
    Communicate with server 'if active'
    inputs:
        cmd = str 'connect' ,'check' , 'sync' or 'close'
        devN = int 1, 2, ... n, (must be declared in server_threads.py)
    outputs:
        server_reponse = str, server response
        server_time = str, server timestamp
    usage:
    server_response, server_time = talk2server(cmd='connect',devN=1)
    """
    try:
        server_response, server_time = client.check_tcp_server(cmd=cmd, dev=devN).split("_")
        server_response, server_time = clientConnectThread.get_command()
    except:  # noserver response
        server_time = "na"
        server_response = "none"
    # print "server reponse: {} and timestamp: {}".format(server_response, server_time)
    return server_response, server_time

# TCP communication
# Start the client thread:
clientConnectThread = client.ClientConnect("connect", "{}".format(devN))
clientConnectThread.setDaemon(True)
clientConnectThread.start()  # launching thread
# time.sleep(1)
server_time = 0.0
server_response = "none"
response = clientConnectThread.get_command()
if "_" in response:
    server_response, server_time = response.split("_")
else:
    server_reponse = response
# Create a pandas dataframe to hold the information (index starts at 1)
cols = ["frameN", "localtime", "servertime"]
df = pd.DataFrame(columns=cols)
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
ir_h, ir_w = ir_frame.shape
ir_place = np.zeros((rgb_h, ir_w, channels), dtype='uint8')
depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
place_ir = rgb_h / 2 - ir_h / 2
place_depth = rgb_h / 2 - depth_h / 2
fps = 8.0

# ==============================================================================
# Video Recording set-up
# ==============================================================================
fourcc = cv2.cv.CV_FOURCC('M', 'P', 'E', 'G')
vid_num = 1
video_location = '/home/julian/Videos/'
rgb_vid = cv2.VideoWriter(video_location + 'rgb_vid_' + str(vid_num) + '.avi', fourcc, fps, (rgb_w, rgb_h), 1)
ir_vid = cv2.VideoWriter(video_location + 'ir_vid_' + str(vid_num) + '.avi', fourcc, fps, (ir_w, ir_h), 1)
depth_vid = cv2.VideoWriter(video_location + 'depth_vid_' + str(vid_num) + '.avi', fourcc, fps, (depth_w, depth_h), 1)

if os.path.exists(video_location + 'ir_full_vid_' + str(vid_num) + '/'):
    shutil.rmtree(video_location + 'ir_full_vid_' + str(vid_num) + '/')
os.makedirs(video_location + 'ir_full_vid_' + str(vid_num) + '/')
if os.path.exists(video_location + 'depth_full_vid_' + str(vid_num) + '/'):
    shutil.rmtree(video_location + 'depth_full_vid_' + str(vid_num) + '/')
os.makedirs(video_location + 'depth_full_vid_' + str(vid_num) + '/')
ir_name = video_location + 'ir_full_vid_' + str(vid_num) + '/ir_frame_'
depth_name = video_location + 'depth_full_vid_' + str(vid_num) + '/depth_frame_'

# 'warm-up' cameras
for i in range(80):
    rgb_frame = get_rgb()
    full_ir = therm.get_frame()
    full_depth, depth_frame = get_depth()

f = 0   # frame counter
tic = time.time()
rec = False
ready = True
new = True
rec_time = []

print ("Press 'esc' to terminate")
done = False
while not done:
    k = cv2.waitKey(1) & 255
    # capture frames
    rgb_frame = get_rgb()
    full_ir = therm.get_frame()
    full_depth, depth_frame = get_depth()

    # make visible
    ir_frame = therm.get_8bit_frame(full_ir)
    ir_place[place_ir:place_ir + ir_h, :, :] = ir_frame
    depth_place[place_depth:place_depth + depth_h, :, :] = depth_frame

    # display and write video
    disp = np.hstack((depth_place, ir_place, rgb_frame))
    disp = cv2.flip(disp, 1)
    cv2.imshow("live", disp)

    # Poll the server:
    if ready:
        clientConnectThread.update_command("ready")
    else:
        clientConnectThread.update_command("info")
    response = clientConnectThread.get_command()
    if "_" in response:
        server_response, server_time = response.split("_")
    else:
        server_response = response

    if server_response == "record":
        rec = True
        ready = True
        new = False
        
    elif server_response == "stop":
        rec = False
        ready = False
        new = False
        # release the previous videos recorded
        rgb_vid.release()
        ir_vid.release()
        depth_vid.release()
        timefile = open(video_location + 'frame_times_' + str(vid_num) + '.txt', 'w')
        for value in rec_time:
            timefile.write(str(value) + "/n")
        timefile.close()
        
    elif server_response == "restart":
        # release the videos to be rerecorded
        if not new:
            rgb_vid.release()
            ir_vid.release()
            depth_vid.release()
            rec_time = []
            rec = False
            ready = True
            new = True
            # set-up videos to be recorded
            f = 0
            rgb_vid = cv2.VideoWriter(video_location + 'rgb_vid_' + str(vid_num) + '.avi', fourcc, fps, (rgb_w, rgb_h), 1)
            ir_vid = cv2.VideoWriter(video_location + 'ir_vid_' + str(vid_num) + '.avi', fourcc, fps, (ir_w, ir_h), 1)
            depth_vid = cv2.VideoWriter(video_location + 'depth_vid_' + str(vid_num) + '.avi', fourcc, fps, (depth_w, depth_h), 1)
            
            if os.path.exists(video_location + 'ir_full_vid_' + str(vid_num) + '/'):
                shutil.rmtree(video_location + 'ir_full_vid_' + str(vid_num) + '/')
            os.makedirs(video_location + 'ir_full_vid_' + str(vid_num) + '/')
            if os.path.exists(video_location + 'depth_full_vid_' + str(vid_num) + '/'):
                shutil.rmtree(video_location + 'depth_full_vid_' + str(vid_num) + '/')
            os.makedirs(video_location + 'depth_full_vid_' + str(vid_num) + '/')
            ir_name = video_location + 'ir_full_vid_' + str(vid_num) + '/ir_frame_'
            depth_name = video_location + 'depth_full_vid_' + str(vid_num) + '/depth_frame_'
        
    elif server_response == "new":
        # release the previous videos recorded
        if not new:
            rgb_vid.release()
            ir_vid.release()
            depth_vid.release()
            timefile = open(video_location + 'frame_times_' + str(vid_num) + '.txt', 'w')
            for value in rec_time:
                timefile.write(str(value) + "/n")
            timefile.close()
            rec = False
            ready = True
            new = True
            # set-up new videos 
            vid_num += 1
            f = 0
            rgb_vid = cv2.VideoWriter(video_location + 'rgb_vid_' + str(vid_num) + '.avi', fourcc, fps, (rgb_w, rgb_h), 1)
            ir_vid = cv2.VideoWriter(video_location + 'ir_vid_' + str(vid_num) + '.avi', fourcc, fps, (ir_w, ir_h), 1)
            depth_vid = cv2.VideoWriter(video_location + 'depth_vid_' + str(vid_num) + '.avi', fourcc, fps, (depth_w, depth_h), 1)
            
            if os.path.exists(video_location + 'ir_full_vid_' + str(vid_num) + '/'):
                shutil.rmtree(video_location + 'ir_full_vid_' + str(vid_num) + '/')
            os.makedirs(video_location + 'ir_full_vid_' + str(vid_num) + '/')
            if os.path.exists(video_location + 'depth_full_vid_' + str(vid_num) + '/'):
                shutil.rmtree(video_location + 'depth_full_vid_' + str(vid_num) + '/')
            os.makedirs(video_location + 'depth_full_vid_' + str(vid_num) + '/')
            ir_name = video_location + 'ir_full_vid_' + str(vid_num) + '/ir_frame_'
            depth_name = video_location + 'depth_full_vid_' + str(vid_num) + '/depth_frame_'
        
    elif server_response == "close":
        done = True
        rec = False
        ready = False
        
    if rec:
        f += 1
        rgb_vid.write(rgb_frame)
        ir_vid.write(ir_frame)
        depth_vid.write(depth_frame)
        np.save(ir_name + str(f), full_ir)
        np.save(depth_name + str(f), full_depth)
        rec_time.append(server_time)
        print ("frame No. recorded ", f)

    if k == 27:  # esc key
        done = True

# release resources and destoy windows
rgb_stream.stop()
depth_stream.stop()
openni2.unload()
rgb_vid.release()
ir_vid.release()
depth_vid.release()
timefile = open(video_location + 'frame_times_' + str(vid_num) + '.txt', 'w')
for value in rec_time:
    timefile.write(str(value) + "/n")
timefile.close()
clientConnectThread.update_command("close_")
cv2.destroyWindow("live")
time.sleep(2)
print ("Completed video generation using {} codec". format(fourcc))
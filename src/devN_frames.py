#!/usr/bin/python
'''
Created on 17May2016
NOTE: OpenCV 2.4.9 VideoWriter has a huge memory leak and consumes all available
      when recoring multiple clips or a large number of frames in a single clip. 
      The memory is only released after the code is stopped completely. 

      workaround - save the frames as images.

Raspberry Pi3 TCP/IP communication network each controlling a primesense.
 - each pi records one .avi file: 
		RGB, 
		Depth, and 
		Distance map (Dmap - see details how this is manipulated)

Current features:
    1) Saves .avi files: rgb, depth, mask, skel, all
		individual frames can be saved by setting the flag       
		save_frames_flag=True
    
    2) Save frames/screenshots rgb and dmap (raw depth, not depth for display)
        * s-key to "screen" capture to screens folder
        * f-key to save nf-frames to frames folder
    
    3) Displays the images: normal, medium, or small
        rgb and depth

    4) Create a csv file via pandas
        name: 'ICU_dev<#>.csv'
    
    5) Threaded tcp client and server
        devid: dev1, dev2, ..devN
        Add/Remove access by removing/adding devs from server_threads.py
        
            client  commands: connect, check, close
            servers response: save, wait

Execution:
NETWORKING
#Open a terminal, set server IP, and run the server
ctrl+alt+t
sudo ifconfig eth0 192.168.0.11 netmask 255.255.255.0 # landline
sudo ifconfig wlan0 192.168.0.11 netmask 255.255.255.0 # wireless
# Activate server
python server_threads.py # first terminal
python dev1_videos.py # second terminal

@author: Carlos Torres <carlitos408@gmail.com>
'''
from primesense import openni2
from primesense import _openni2 as c_api
import cv2
import cv
import sys
import time
import os
import csv
import numpy as np
import pandas as pd
from time import localtime, strftime, gmtime
import pickle
import client

import serial

# Drawing
radius = 10
green = (0, 255, 0)
blue = (255, 0, 0)
red = (0, 0, 255)
colors = [green, blue, red]
#confs  = [1.0, 0.5, 0.0]

# Device resolution
w = 640
h = 480
# Center of the image
x = h / 2
y = w / 2

# Device number
devN = 1

# Array to store the image modalities+overlayed_skeleton (4images)
#rgb   = np.zeros((480,640,3), np.uint8)
#rgbdm = np.zeros((480,640*4, 3), np.uint8)

"""
Initialize and configure the primesense device
"""
# pi3
dist = "/home/carlos/Install/kinect/OpenNI-Linux-Arm-2.2/Redist/"

# initialize openni and check
openni2.initialize(dist)
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"
# if

# Register the device
dev = openni2.Device.open_any()


# create the streams stream
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()

# configure the depth_stream
depth_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=w, resolutionY=h, fps=30))
# Check and configure the depth_stream -- set automatically based on bus speed
# print 'The rgb video mode is', rgb_stream.get_video_mode() # Checks rgb video configuration
rgb_stream.set_video_mode(c_api.OniVideoMode(
    pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=w, resolutionY=h, fps=30))

# Configure the mirroring, which by default is enabled (True)
depth_stream.set_mirroring_enabled(False)
rgb_stream.set_mirroring_enabled(False)

# start the stream
rgb_stream.start()
depth_stream.start()

# synchronize the streams
dev.set_depth_color_sync_enabled(True)  # synchronize the streams

# IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),
                        dtype=np.uint8).reshape(h, w, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb
# get_rgb


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
    depth_frame = depth_stream.read_frame()
    dmap = np.fromstring(depth_frame.get_buffer_as_uint16(),
                         dtype=np.uint16).reshape(h, w)  # Works & It's FAST
    d4d = dmap.astype(float) * 255 / 2**12 - 1  # Correct the range. Depth images are 12bits
    d4d = cv2.cvtColor(np.uint8(d4d), cv2.COLOR_GRAY2RGB)

    temp1 = np.zeros(dmap.shape, dtype=np.uint8)
    temp1 = dmap - 2**8  # most significant: 2**8-2**15

    temp2 = dmap.copy()  # least significant: 2**0-2**7
    temp1[temp1 < 0] = 0
    temp2[temp2 > 255] = 0
    dmap = np.uint8(np.dstack((temp2, temp1, temp2)))
    # print dmap.shape, type(dmap), dmap.dtype
    return dmap, d4d
# get_depth


def createSessionFolders(folder, session=0):
    """
    Creates folders for session frames and csv files and returns paths.
        folder4frames='root/icudata/patient_<number+1>/session_<number>/<frames>/'
        folder4csv   ='root/icudata/patient_<number+1>/session_<number>/csv/'
    """
    print "Data Directories In {} Do'nt Exist. Creating Them".format(folder)
    if "session" in folder:  # remove session from path containing session_{}
        folder = folder.replace("/session_{}".format(session - 1), "")
    folder4frames = folder + "/session_{}".format(session)

    folder4csv = folder4frames + '/csv/'
    os.makedirs(folder4frames + '/rgb/')
    os.makedirs(folder4frames + '/depth/')
    os.makedirs(folder4frames + '/dmap/')
    os.makedirs(folder4csv)

    return folder4frames, folder4csv
# createSessionFolders()


def createActorFolder(actor='patient_0', save2sdcard=False):
    """
    Checks if a sdcard is connected and if a folder exists under 
        root/icudata/patient_<number>
    If it exists a new name is created
        newname = root/icudata/patient_<number+1>
        foldername = newfoldername
    Ultimately, creates folders for frames and csv files and returns paths.
        folder4frames='root/icudata/patient_<number+1>/frames/'
        folder4csv   ='root/icudata/patient_<number+1>/csv/'
    """
    # memory card name
    #sdname = "6563-3566"
    sdname = "pi{}mem/".format(devN)
    local = "/media/carlos/" + sdname
    # Two storage locations: media or local
    # if save2sdcard:
    #    media = '/media/'+ sdname +'/'
    #    root = media+'icudata{}/'.format(devN)
    #    storage = 'External'
    # else: #if not, then save on local hdd
    # usr = "carlos" #os.listdir('/home/')[0]
    #local = '/home/'+usr+'/Documents/Python/MICU/micu_openni2_v3/'
    local = "/home/carlos/Documents/Python/MICU/micu_openni2_v3/"
    root = local + 'icudata{}/'.format(devN)
    storage = 'Local'
    # end ifsdcards

    # check existence of directory.
    folder = root + actor
    name, idx = actor.split('_')
    done = False
    while not done:
        if os.path.isdir(folder):
            'Directory exists {}... renaming!'.format(folder)
            name, idx = actor.split('_')
            idx = str(int(idx) + 1)
            print idx
            actor = ('_').join([name, idx])
            folder = root + actor

        else:
            print 'Directory {} Doesnt Exist. Creating It'.format(folder)
            folder4frames = folder  # + '/session{}/'.format(session)
            #folder4vids = folder+'/videos/'
            folder4csv = folder4frames  # +'/csv/'
            os.makedirs(folder4frames)
            # os.makedirs(folder4csv)
            done = True
            print '{} storage location {}'.format(storage, root)
            print 'Directory for videos : {}'.format(folder4frames)
            print 'Directory for csv files: {}'.format(folder4csv)
    return folder4frames, folder4csv
# createActorFolder()


def save_frames(frame, rgb, depth, dmap, p="../data/frames/"):
    """
    Saves the images to as lossless pngs and appends the frame number n
    """
    # save te images to the path
    print "Saving image {} to {}".format(frame, p)
    cv2.imwrite(p + '/rgb/' + "rgb_" + str(frame) + ".png", rgb)
    cv2.imwrite(p + '/depth/' + "depth_" + str(frame) + ".png", depth)
    cv2.imwrite(p + '/dmap/' + "dmap_" + str(frame) + ".png", dmap)
    return
# save_frames


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


# ======== MAIN =========
def run_old_loop():
    """
    Creates session videos. However, opencv video release is not working. HUGE memory
    leak that.
#if __name__ == "__main__":
    #time.sleep(20) # secs pause! for startup
    #devN = 1

    """
    synctype = "relaxed"
    actorname = "patient_0"

    # Flags
    vis_frames = True  # True   # display frames
    save_frames_flag = False  # save all frames
    test_flag = True

    test_frames = 5000000
    fps = 30
    c = 0
    ## Runtime and Controls
    nf = 500  # 172800# 60*60*24*2 # Number of video frames in each clip and video
    f = 1  # frame counter
    tic = 0
    run_time = 0
    total_t = 0
    fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')

    done = False

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
    # print(server_response, server_time)

    # Create a pandas dataframe to hold the information (index starts at 1)
    cols = ["frameN", "localtime", "servertime"]
    df = pd.DataFrame(columns=cols)
    df.loc[c] = [0, server_time, time.time()]

    # The folders for all data
    folder4frames, folder4csv = createFolders(actorname)
    print "Creating Video Headers"
    # Initialize the videowriter
    vid_num = 0
    video_rgb = cv2.VideoWriter(folder4frames + "/rgb/dev" + str(devN) +
                                "rgb" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
    video_depth = cv2.VideoWriter(folder4frames + "/depth/dev" + str(devN) +
                                  "depth" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
    video_dmap = cv2.VideoWriter(folder4frames + "/dmap/dev" + str(devN) +
                                 "dmap" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))

    # Get the first timestamp
    tic = time.time()
    start_t = tic

    # --- main loop ---
    done = False
    while not done:  # view <= nviews
        # RGB-D Streams
        rgb = get_rgb()
        dmap, d4d = get_depth()

        if vis_frames:  # Display the streams
            rgbdm = np.hstack((rgb, d4d, dmap))
            # rgbdm_small = rgbdm # orginal size
            # rgbdm_small = cv2.resize(rgbdm,(1280,240)) # medium
            # rgbdm_small = cv2.resize(rgbdm,(640,240)) # smallest
            rgbdm_small = cv2.resize(rgbdm, (960, 240))  # smallest
            cv2.imshow("1:4 scale", rgbdm_small)
            # === Keyboard Commands ===
            key = cv2.waitKey(1) & 255
            if key == 27:
                print "Terminating code!"
                done = True
        # Poll the server:
        clientConnectThread.update_command("check")
        response = clientConnectThread.get_command()
        if "_" in response:
            server_response, server_time = response.split("_")
        else:
            server_reponse = response

        run_time = time.time() - tic
        print "Processing {} session and frame number {}".format(vid_num, f)

        # === check synchronization type
        if synctype == 'strict':
            if server_response == 'save':
                video_rgb.write(rgb)    # --> rgb vid file
                video_depth.write(d4d)  # --> depth vid file
                video_dmap.write(dmap)  # --> dmap vid file
                # Write Datarows
                df.loc[c] = [f, strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()), server_time]
                f += 1
                c += 1
        elif synctype == 'relaxed':
            video_rgb.write(rgb)    # --> rgb vid file
            video_depth.write(d4d)  # --> depth vid file
            video_dmap.write(dmap)  # --> dmap vid file

            # Write Datarows
            df.loc[c] = [f, run_time, server_time]
            f += 1
            c += 1
        else:
            print "synchronization type unknown"

        if test_flag and (f == test_frames):
            print "Terminating code!"
            done = True

        # if np.mod(f,nf) == 0: # close and create new csv and video
        if f == nf:
            df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(vid_num) + '.csv')
            print "session {} saved".format(vid_num)
            # release video writers
            video_rgb.release()
            video_depth.release()
            video_dmap.release()
            del df
            del video_rgb
            del video_depth
            del video_dmap
            print "\t Cleared the vars and memory and prepping next session"
            vid_num += 1
            # Create new video writers
            video_rgb = cv2.VideoWriter(folder4frames + "/rgb/dev" + str(devN) +
                                        "rgb" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
            video_depth = cv2.VideoWriter(folder4frames + "/depth/dev" + str(devN) +
                                          "depth" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
            video_dmap = cv2.VideoWriter(folder4frames + "/dmap/dev" + str(devN) +
                                         "dmap" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
            # reset pandas dataframe
            df = pd.DataFrame(columns=cols)
            c = 0
            f = 0
            # done = True #stop after the first recording.

        # elif chr(key) =='s':  #s-key to save current screen
        #    save_frames(f,rgb,dmap,p=folder4screens)

        # if
        # --- keyboard commands ---
    # while

    # TERMINATE
    print "=== Terminating code! ==="
    # Close carmine context and stop device
    print "==== Closing carmine context"
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    # write last datapoints
    print "==== Writing last portions of data."
    vid_num = +1
    df.loc[c] = [f, run_time, server_time]
    video_rgb.write(rgb)    # write to vid file
    video_depth.write(d4d)  # write to vid file
    video_dmap.write(dmap)
    # Write data to csv
    df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(vid_num) + '.csv')
    # release video writers
    print "==== Releasing the video writers"
    video_rgb.release()
    video_depth.release()
    video_dmap.release()
    # Disconnect the client from the server
    print "==== Disconecting client and closing the server"
    clientConnectThread.update_command("close")
    # Release video/image resources
    cv2.destroyAllWindows()
    # print some timing information:
    fps = f / run_time
    print "\nTime metrics for {} frames:" .format(f)
    print ("\ttotal run time is %.2f secs over" % run_time)
    print ("\tfps: %.2f" % fps)
    sys.exit(1)
# if __main__


def save_videos(folder4frames, c, vid_num, tic, nf=360):
    """
    save videos
    """
    fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
    fps = 30

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
    # print(server_response, server_time)

    # Create a pandas dataframe to hold the information (index starts at 1)
    cols = ["frameN", "localtime", "servertime"]
    df = pd.DataFrame(columns=cols)
    df.loc[c] = [0, server_time, time.time()]

    print "Creating Video Headers"
    # Initialize the videowriter
    #vid_num     = 0
    video_rgb = cv2.VideoWriter(folder4frames + "/rgb/dev" + str(devN) +
                                "rgb" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
    video_depth = cv2.VideoWriter(folder4frames + "/depth/dev" + str(devN) +
                                  "depth" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
    video_dmap = cv2.VideoWriter(folder4frames + "/dmap/dev" + str(devN) +
                                 "dmap" + str(vid_num) + ".avi", fourcc, fps=fps, frameSize=(w, h))
    rgb_stream.start()
    depth_stream.start()
    for f in range(nf):
        openni2.wait_for_any_stream()
        # RGB-D Streams
        rgb = get_rgb()
        dmap, d4d = get_depth()
        # if vis_frames: # Display the streams
        rgbdm = np.hstack((rgb, d4d, dmap))
        rgbdm_small = cv2.resize(rgbdm, (960, 240))  # smallest
        cv2.imshow("1:4 scale", rgbdm_small)
        # === Keyboard Commands ===
        key = cv2.waitKey(1) & 255
        if key == 27:
            print "Terminating code!"
            df.loc[c] = [f, run_time, server_time]
            video_rgb.write(rgb)    # write to vid file
            video_depth.write(d4d)  # write to vid file
            video_dmap.write(dmap)
            # Write data to csv
            df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(vid_num) + '.csv')
            # release video writers
            print "==== Releasing the video writers"
            video_rgb.release()
            video_depth.release()
            video_dmap.release()
            #done = True
            sys.exit(0)

        # Poll the server:
        clientConnectThread.update_command("check")
        sresponse = clientConnectThread.get_command()
        if "_" in response:
            server_response, server_time = response.split("_")
        else:
            server_reponse = response

        run_time = time.time() - tic
        print "\t Processing {} session and frame number {} -- {}".format(vid_num, f, (vid_num * nf) + f)

        video_rgb.write(rgb)    # --> rgb vid file
        video_depth.write(d4d)  # --> depth vid file
        video_dmap.write(dmap)  # --> dmap vid file

        # Write Datarows
        df.loc[c] = [f, run_time, server_time]

        c += 1
    # endfor f in range nf
    df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(vid_num) + '.csv')
    # release video writers
    video_rgb.release()
    video_depth.release()
    video_dmap.release()
    done = True
    del df
    del video_rgb
    del video_depth
    del video_dmap
    print "\n === Cleared the video and pandas variables".format(vid_num)
    # endfor vid_num in range(0, t_num_vids):
    rgb_stream.stop()
    depth_stream.stop()
    return c, run_time



# ======== MAIN =========
if __name__ == "__main__":
    # time.sleep(20) # secs pause! for startup
    #devN = 1
    synctype = "relaxed"
    actorname = "patient_0"

    # Flags
    vis_frames = True  # True   # display frames
    save_frames_flag = False  # save all frames
    test_flag = True

    test_frames = 50000000
    num_sessions = 10000  # 0000 #1000000
    nf = 3600  # 3600 at ~1fps := 1 hr clips |172800# 60*60*24*2 # Number of video frames in each videoclip
    c = 0      # global frame counter

    # The folders for all data
    folder4frames, folder4csv = createActorFolder(actorname)

    # print "Folder for frames: ", folder4frames
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
    # print(server_response, server_time)

    tic = time.time()
    #start_t = tic
    for session in range(0, num_sessions):
        folder4frames, folder4csv = createSessionFolders(folder4frames, session=session)
        # Create a pandas dataframe to hold the information (index starts at 1)
        cols = ["frameN", "localtime", "servertime"]
        df = pd.DataFrame(columns=cols)
        df.loc[c] = [0, server_time, time.time()]

        for f in range(nf):
            run_time = time.time() - tic
            # RGB-D Streams
            rgb = get_rgb()
            dmap, d4d = get_depth()
            # if vis_frames: # Display the streams
            rgbdm = np.hstack((rgb, d4d, dmap))
            rgbdm_small = cv2.resize(rgbdm, (960, 240))  # smallest
            cv2.imshow("1:4 scale", rgbdm_small)
            # === Keyboard Commands ===
            key = cv2.waitKey(1) & 255
            if key == 27:
                print "\n\t ====> ESC detected. Terminating code!"
                df.loc[c] = [f, run_time, server_time]
                # saveframes
                save_frames(f, rgb, d4d, dmap, p=folder4frames)
                # Write data to csv
                df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(session) + '.csv')
                #done = True
                sys.exit(0)

            # Poll the server:
            clientConnectThread.update_command("check")
            sresponse = clientConnectThread.get_command()
            if "_" in response:
                server_response, server_time = response.split("_")
            else:
                server_reponse = response
            df.loc[c] = [f, run_time, server_time]
            c += 1
            # saveframes
            save_frames(f, rgb, d4d, dmap, p=folder4frames)
        df.to_csv(folder4csv + "dev" + str(devN) + '_data' + str(session) + '.csv')

        # print "Collecting and Saving Video Number: {}".format(vid_num)
        #c, run_time = save_videos(folder4frames, c, session, tic, nf=nf)

        # --- main loop ---

    # TERMINATE
    print "=== Terminating code. After recording {} Total Video Clips! ===".format(num_sessions)
    # Close carmine context and stop device
    print "==== Closing carmine context"
    rgb_stream.stop()
    depth_stream.stop()
    openni2.unload()
    print "==== Disconecting client and closing the server"
    clientConnectThread.update_command("close")
    # Release video/image resources
    cv2.destroyAllWindows()
    # print some timing information:
    fps = f * num_sessions / run_time
    print "\nTime metrics for {} frames:" .format(f)
    print ("\ttotal run time is %.2f secs over" % run_time)
    print ("\tfps: %.2f" % fps)
    sys.exit(1)
# if __main

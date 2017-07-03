# You will need to have python 2.7 (3+ may work)
# and PyUSB 1.0
# and PIL 1.1.6 or better
# and numpy
# and scipy
# and ImageMagick

# Many thanks to the folks at eevblog, especially (in no particular order) 
#   miguelvp, marshallh, mikeselectricstuff, sgstair and many others
#     for the inspiration to figure this out
# This is not a finished product and you can use it if you like. Don't be
# surprised if there are bugs as I am NOT a programmer..... ;>))


## https://github.com/sgstair/winusbdotnet/blob/master/UsbDevices/SeekThermal.cs

## Incase I ever need it
#output = open('data.pkl', 'wb')
#pickle.dump(im2arrF,output)

import usb.core
import usb.util
from PIL import Image
import numpy as np
import sys, time
import cv2
from primesense import openni2#, nite2
from primesense import _openni2 as c_api
import pickle
#from heatmap import thermal_map

###################################################################################################################
# set-up primesense camera
dist ='/home/julian/Downloads/temp/OpenNI-Linux-x64-2.2/Redist'

## Initialize openni and check
openni2.initialize(dist) #
if (openni2.is_initialized()):
    print "openNI2 initialized"
else:
    print "openNI2 not initialized"

## Register the device
prime = openni2.Device.open_any()

## Create the streams 
rgb_stream = prime.create_color_stream()
depth_stream = prime.create_depth_stream()

## Configure the depth_stream -- changes automatically based on bus speed
#print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=320, resolutionY=240, fps=30))

## Check and configure the mirroring -- default is True
## Note: I disable mirroring
# print 'Mirroring info1', depth_stream.get_mirroring_enabled()
depth_stream.set_mirroring_enabled(False)
rgb_stream.set_mirroring_enabled(False)

## More infor on streams depth_ and rgb_
#help(depth_stream)

## Start the streams
rgb_stream.start()
depth_stream.start()

## Synchronize the streams
prime.set_depth_color_sync_enabled(True) 

## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
prime.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)


def get_rgb():
    """
    Returns numpy 3L ndarray to represent the rgb image.
    """
    bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
    rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb
#get_rgb


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
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(240,320)  # Works & It's FAST
    d4d = np.uint8(dmap.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d,cv2.COLOR_GRAY2RGB)
#    output = open('data.pkl', 'wb')
#    pickle.dump(d4d, output)
    return dmap, d4d
#get_depth
###################################################################################################################
# set-up thermal stream
# find our Seek Thermal device  289d:0010
therm = usb.core.find(idVendor=0x289d, idProduct=0x0010)
if not therm: raise ValueError('Device not found')

def send_msg(bmRequestType, bRequest, wValue=0, wIndex=0, data_or_wLength=None, timeout=None):
    assert (therm.ctrl_transfer(bmRequestType, bRequest, wValue, wIndex, data_or_wLength, timeout) == len(data_or_wLength))

# alias method to make code easier to read
receive_msg = therm.ctrl_transfer

def deinit():
    '''Deinit the device'''
    msg = '\x00\x00'
    for i in range(3):
        send_msg(0x41, 0x3C, 0, 0, msg)


# set the active configuration. With no arguments, the first configuration will be the active one
therm.set_configuration()

# get an endpoint instance
cfg = therm.get_active_configuration()
intf = cfg[(0,0)]

custom_match = lambda e: usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT
ep = usb.util.find_descriptor(intf, custom_match=custom_match)   # match the first OUT endpoint
assert ep is not None


# Setup device
try:
    msg = '\x01'
    send_msg(0x41, 0x54, 0, 0, msg)
except Exception as e:
    deinit()
    msg = '\x01'
    send_msg(0x41, 0x54, 0, 0, msg)

#  Some day we will figure out what all this init stuff is and
#  what the returned values mean.

send_msg(0x41, 0x3C, 0, 0, '\x00\x00')
ret1 = receive_msg(0xC1, 0x4E, 0, 0, 4)
#print ret1
ret2 = receive_msg(0xC1, 0x36, 0, 0, 12)
#print ret2

send_msg(0x41, 0x56, 0, 0, '\x20\x00\x30\x00\x00\x00')
ret3 = receive_msg(0xC1, 0x58, 0, 0, 0x40)
#print ret3

send_msg(0x41, 0x56, 0, 0, '\x20\x00\x50\x00\x00\x00')
ret4 = receive_msg(0xC1, 0x58, 0, 0, 0x40)
#print ret4

send_msg(0x41, 0x56, 0, 0, '\x0C\x00\x70\x00\x00\x00')
ret5 = receive_msg(0xC1, 0x58, 0, 0, 0x18)
#print ret5

send_msg(0x41, 0x56, 0, 0, '\x06\x00\x08\x00\x00\x00')
ret6 = receive_msg(0xC1, 0x58, 0, 0, 0x0C)
#print ret6

send_msg(0x41, 0x3E, 0, 0, '\x08\x00')
ret7 = receive_msg(0xC1, 0x3D, 0, 0, 2)
#print ret7

send_msg(0x41, 0x3E, 0, 0, '\x08\x00')
send_msg(0x41, 0x3C, 0, 0, '\x01\x00')
ret8 = receive_msg(0xC1, 0x3D, 0, 0, 2)
#print ret8

im2arrF = None
# filter array
kernel = np.ones((3,3),np.float32)/9
def get_thermal_image():
    global im2arrF
    while True:
        # Send read frame request
        send_msg(0x41, 0x53, 0, 0, '\xC0\x7E\x00\x00')

        try:
            ret9  = therm.read(0x81, 0x3F60, 1000)
            ret9 += therm.read(0x81, 0x3F60, 1000)
            ret9 += therm.read(0x81, 0x3F60, 1000)
            ret9 += therm.read(0x81, 0x3F60, 1000)
        except usb.USBError as e:
            sys.exit()

        #  Let's see what type of frame it is
        #  1 is a Normal frame, 3 is a Calibration frame
        #  6 may be a pre-calibration frame
        #  5, 10 other... who knows.
        status = ret9[20]
        #print ('%5d'*21 ) % tuple([ret9[x] for x in range(21)])

        if status == 1:
            #  Convert the raw calibration data to a string array
            calimg = Image.frombytes("I", (208,156), ret9, "raw", "I;16")

            #  Convert the string array to an unsigned numpy int16 array
            im2arr = np.asarray(calimg)
            im2arrF = im2arr.astype('uint16')

        if status == 3 and im2arrF is not None:
            #  Convert the raw calibration data to a string array
            img = Image.frombytes("I", (208,156), ret9, "raw", "I;16")

            #  Convert the string array to an unsigned numpy int16 array
            im1arr = np.asarray(img)
            im1arrF = im1arr.astype('uint16')

            #  Subtract the calibration array from the image array and add an offset
            additionF = abs((im1arrF-im2arrF) + 880)
            thermal = cv2.filter2D(additionF,-1,kernel)
            return thermal

fps_t = 0
fps_f = 0

def get_thermal_frame():
    global fps_t, fps_f
    
    frame = get_thermal_image()
    
    bit_correction = frame
#    d4d = thermal_map(dd)
    # Change bit-depth
    bit_correction >>= 6   
    bit_correction <<= 4
    # How Carlos limits bit depth
#    d4d = np.uint8(dd.astype(float) *255/ 2**12-1) # Correct the range. Depth images are 12bits
    bit_correction = cv2.cvtColor(bit_correction,cv2.COLOR_GRAY2BGR)

    bit_correction = bit_correction.astype('uint8',casting = 'unsafe')
    # Shown unknowns in black
    
    bit_correction = 255 - bit_correction
    bit_correction = np.rot90(bit_correction, 1)
    
    now = int(time.time())
    fps_f += 1
    if fps_t == 0:
        fps_t = now
    elif fps_t < now:
        print '\rFPS: %.2f' % (1.0 * fps_f / (now-fps_t)),
        sys.stdout.flush()
        fps_t = now
        fps_f = 0
        
    return bit_correction
# End of set-up - Start doing stuff
##==============================================================================
## Video .avi output setup
##==============================================================================
#vidname= "../video/test.avi" # make sure video directory exists
###################################################################################################################
# detect cameras and find homography
rgb_frame = get_rgb()
dmap, depth_frame = get_depth()
rgb_h, rgb_w, channels = rgb_frame.shape
depth_h, depth_w, depth_channels = depth_frame.shape
ir_w = 156
ir_h = 208
ir_place = np.zeros((rgb_h, ir_w, channels),dtype='uint8')
depth_place = np.zeros((depth_h, depth_w, channels), dtype='uint8')
place_ir = rgb_h/2 - ir_h/2
place_depth = rgb_h/2 - depth_h/2
fps = 8.0
nf  = 60  # total number of video frames 60 frames at 30fps == 2 second video
f   = 0   # frame counter

##==============================================================================
## THE CODECS
##==============================================================================
#fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
fourcc = cv2.cv.CV_FOURCC('M','J','P','G') 

rgb_vid = cv2.VideoWriter('Videos/rgb_vid.avi', fourcc, fps, (rgb_w,rgb_h),1)
ir_vid = cv2.VideoWriter('Videos/ir_vid.avi', fourcc, fps, (ir_w,ir_h),1)
depth_vid = cv2.VideoWriter('Videos/depth_vid.avi', fourcc, fps, (depth_w,depth_h),1)

print ("Press 'spacebar' to start recording avi or 'esc' to terminate")

done = False
max_area = 0

while not done:
    k = cv2.waitKey(1)
    #capture frames
    rgb_frame = get_rgb()
    ir_frame = get_thermal_frame()
    dmap, depth_frame = get_depth()
    ir_place[place_ir:place_ir+ir_h,:,:] = ir_frame
    depth_place[place_depth:place_depth+depth_h,:,:] = depth_frame
    
    disp = np.hstack((depth_place, ir_place, rgb_frame))
    disp = cv2.flip(disp,1)
    cv2.imshow("live", disp)
    rgb_vid.write(rgb_frame)
    ir_vid.write(ir_frame)
    depth_vid.write(depth_frame)
    
    f+=1
    print ("frame No.",f)
#        if f == nf:
#            done = True
    if k == 27: #esc key
        done = True
    elif k == ord(' '): #spacebar
        print ("Saving frame {}".format(f))
        cv2.imwrite('Images/'+str(f)+'.png', rgb_frame)
#   if detected: cv2.waitKey(0)
            

# release resources and destoy windows
rgb_stream.stop()
depth_stream.stop()
openni2.unload()
rgb_vid.release()
ir_vid.release()
depth_vid.release()
cv2.destroyWindow("live")
print ("Completed video generation using {} codec". format(fourcc) )

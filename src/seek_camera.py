# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:17:35 2017

@author: julian
"""
import numpy as np
import cv2
import usb.core
import usb.util
import sys


class thermal_camera():
    offset = 0x4000
    ffc_filename = 'seek_config.png'
    width = 156
    height = 208

    def send_msg(self, bmRequestType, bRequest, wValue=0, wIndex=0, data_or_wLength=None, timeout=None):
        assert (self.therm.ctrl_transfer(bmRequestType, bRequest, wValue,
                                         wIndex, data_or_wLength, timeout) == len(data_or_wLength))

    def deinit(self):
        '''Deinit the device'''
        msg = '\x00\x00'
        for i in range(3):
            self.send_msg(0x41, 0x3C, 0, 0, msg)

    def __init__(self):  # intial set-up of cameras
        self.therm = usb.core.find(idVendor=0x289d, idProduct=0x0010)
        if not self.therm:
            raise ValueError('Device not found')

        self.receive_msg = self.therm.ctrl_transfer
        # set the active configuration. With no arguments, the first configuration will be the active one
        self.therm.set_configuration()
        # get an endpoint instance
        cfg = self.therm.get_active_configuration()

        def custom_match(e): return usb.util.endpoint_direction(
            e.bEndpointAddress) == usb.util.ENDPOINT_OUT
        # match the first OUT endpoint
        ep = usb.util.find_descriptor(cfg[(0, 0)], custom_match=custom_match)
        assert ep is not None

        # Setup device
        try:
            msg = '\x01'
            self.send_msg(0x41, 0x54, 0, 0, msg)
        except Exception as e:
            self.deinit()
            msg = '\x01'
            self.send_msg(0x41, 0x54, 0, 0, msg)
        self.send_msg(0x41, 0x3C, 0, 0, '\x00\x00')
        ret1 = self.receive_msg(0xC1, 0x4E, 0, 0, 4)
        ret2 = self.receive_msg(0xC1, 0x36, 0, 0, 12)
        self.send_msg(0x41, 0x56, 0, 0, '\x20\x00\x30\x00\x00\x00')
        ret3 = self.receive_msg(0xC1, 0x58, 0, 0, 0x40)
        self.send_msg(0x41, 0x56, 0, 0, '\x20\x00\x50\x00\x00\x00')
        ret4 = self.receive_msg(0xC1, 0x58, 0, 0, 0x40)
        self.send_msg(0x41, 0x56, 0, 0, '\x0C\x00\x70\x00\x00\x00')
        ret5 = self.receive_msg(0xC1, 0x58, 0, 0, 0x18)
        self.send_msg(0x41, 0x56, 0, 0, '\x06\x00\x08\x00\x00\x00')
        ret6 = self.receive_msg(0xC1, 0x58, 0, 0, 0x0C)
        self.send_msg(0x41, 0x3E, 0, 0, '\x08\x00')
        ret7 = self.receive_msg(0xC1, 0x3D, 0, 0, 2)
        self.send_msg(0x41, 0x3E, 0, 0, '\x08\x00')
        self.send_msg(0x41, 0x3C, 0, 0, '\x01\x00')
        ret8 = self.receive_msg(0xC1, 0x3D, 0, 0, 2)

        self.dead_pixels = np.loadtxt('dead_pixels.txt', dtype='uint8')
        self.constant_ffc = cv2.imread('seek_ffc.png')

    def remove_dead_pixels(self, frame):
        corrected = frame.copy()

        for i in range(self.dead_pixels.size / 2):
            total = 0
            amount = 0
            x = self.dead_pixels[i, 0]
            y = 207 - self.dead_pixels[i, 1]
            if x != 0 and x != (self.width - 1):
                total += frame[x - 1, y]
                total += frame[x + 1, y]
                amount += 2
            elif x == 0:
                total += frame[x + 1, y]
                amount += 1
            elif x == self.width - 1:
                total += frame[x - 1, y]
                amount += 1
            if y != 0 and y != self.height - 1:
                total += frame[x, y - 1]
                total += frame[x, y + 1]
                amount += 2
            elif y == 0:
                total += frame[x, y + 1]
                amount += 1
            elif y == self.height - 1:
                total += frame[x, y - 1]
                amount += 1
            corrected[x, y] = total / amount
        return corrected

    def get_frame(self):
        while True:
            # Send read frame request
            self.send_msg(0x41, 0x53, 0, 0, '\xC0\x7E\x00\x00')

            try:
                from_usb = self.therm.read(0x81, 0x3F60, 1000)
                from_usb += self.therm.read(0x81, 0x3F60, 1000)
                from_usb += self.therm.read(0x81, 0x3F60, 1000)
                from_usb += self.therm.read(0x81, 0x3F60, 1000)
            except usb.USBError as e:
                sys.exit()

            #  Let's see what type of frame it is
            #  1 is a Normal frame, 3 is a Calibration frame
            #  6 may be a pre-calibration frame
            #  5, 10 other... who knows.
            status = from_usb[20]

            if status == 1:
                #  Convert the raw calibration data to a string array
                self.ffc_frame = np.fromstring(from_usb, 'uint16')
                self.ffc_frame = np.reshape(self.ffc_frame, (156, 208))

            if status == 3 and self.ffc_frame is not None:
                #  Convert the raw calibration data to a string array
                recieved = np.fromstring(from_usb, 'uint16')
                recieved = np.reshape(recieved, (156, 208))

                #  Subtract the calibration array from the image array and add an offset
                frame = abs((recieved - self.ffc_frame) + self.offset)
                frame = self.remove_dead_pixels(frame)
                self.full_frame = frame[:, 0:-2]
                self.full_frame = cv2.flip(self.full_frame, 0)
                return self.full_frame

    def get_8bit_frame(self, frame):
        output = frame >> 2
        output = cv2.cvtColor(np.rot90(output.astype('uint8'), 1), cv2.COLOR_GRAY2BGR)
        return output

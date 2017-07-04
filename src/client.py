# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:40:55 2014

tcp_client.py

ref: 
    https://docs.python.org/2/library/socketserver.html

Version 2: 03 April 2016
    Using threads to read server-client communications
     - Thx to C wheat.

@author: carlos
"""
import socket
import threading
import time
import errno
from socket import error as socket_error
#import sys

# ===========================================================================
# Check response from tcp server
# ----------------------------------------------------------------------------

# HOST = "localhost" # Local network
HOST = "192.168.0.100"  # Local network


def update_command(request='check'):
    return request


class ClientConnect(threading.Thread):
    """
        spawing a thread to listen for connections
    """

    # list of avalable ports
    dev_dict = {
        'dev1': {'PORT': 50007},
        'dev2': {'PORT': 50008},
        'dev3': {'PORT': 50009},
        'dev4': {'PORT': 50010},
        #'dev5':{'PORT':50011},
        #'dev6':{'PORT':50012}
    }

    def __init__(self, cmd='connect', dev=1):
        """
            Starts thread 
        """
        threading.Thread.__init__(self)
        self.command = (None, None)
        self.connected = False
        self.cb = None
        self.done = False
        self.cmd = cmd
        self.dev = dev
        devid = "dev{}".format(dev)
        self.PORT = self.dev_dict[devid]["PORT"]  # The same port as used by the server
        print "Connecting to PORT: ", self.PORT
        # Create a socket (SOCK_STREAM means a TCP socket)

        #self.sock.connect((HOST, self.PORT))
    def run(self):
        """
            called after start()
            Connects to server and polls for commands
        """
        while not self.done:
            data = str(self.dev) + " " + self.cmd
            try:
                # Connect to server and send data
                # self.sock.close()
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10)
                self.sock.connect((HOST, self.PORT))
                self.sock.sendall(data + "\n")

                # Receive data from the server and shut down
                received = self.sock.recv(1024)
                # self.sock.close()
                self.sock.shutdown(1)
                # print "Sent:     {}".format(data)
                # print "Received: {}".format(received)
                self.command = received
                if self.cb:
                    self.cb()
                if self.cmd == "close":
                    self.done == True
                # if self.connected == False:# and self.command == "connect":
                #    print "Connection successful"
                #    self.connected = True
                #    self.update_command("check")
                    #self.cmd = "check"
                # else:
                #    self.command = update_command()
            # except socket_error as serr:
            #	if serr.errno != errno.ECONNREFUSED:
            #		# Not the error we are looking for, re-raise
            #		raise serr
            #	print "Connection refused"
            except socket.timeout:
                print "Connection timed out"
                print "Attempting to Reconnect"
            # finally:
            #	self.sock.close()
            time.sleep(1)

    def callback(self, cb):
        """
        Set a callback on recieving
        """
        self.cb()

    def check_tcp_server():
        """

        """
        return self.command

    def get_command(self):
        return self.command

    def update_command(self, cmd):
        self.cmd = cmd
        return self.cmd

    def __del__(self):
        """
        On destruction closes thread and conneciton
        """
        self.done == True
        print "Closing Thread"
        # self.sock.close()
        print "Closing Socket"


def check_tcp_server(cmd='check', dev=1):
    """
    Check the server (the status of other devices).
    (str, str) -> (str)
    cmd   = str that can take one of three values: check, connect, disconnect
    devid = str that can take various values that must be included in the 
    server. E.g., dev1, dev2, or dev3. Each computer should have a 
    unique dev<#>.
    received = a list of connected devices
        """
    # ====== Client Variables:
    #HOST = "localhost"
    HOST = "192.168.0.100"
    received = ""
    devid = "dev{}".format(dev)
    dev_dict = {'dev1': {'PORT': 50007},
                'dev2': {'PORT': 50008},
                'dev3': {'PORT': 50009},
                'dev4': {'PORT': 50010},
                # 'dev5':{'PORT':50011},
                # 'dev6':{'PORT':50012}
                }
    PORT = dev_dict[devid]["PORT"]  # The same port as used by the server
    # Create a socket (SOCK_STREAM means a TCP socket)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data = str(dev) + " " + cmd
    try:
        # Connect to server and send data
        sock.connect((HOST, PORT))
        sock.sendall(data + "\n")
        # Receive data from the server and shut down
        received = sock.recv(1024)
        # print "Sent:     {}".format(data)
        # print "Received: {}".format(received)
    except:
        pass

#    finally:
#        sock.close()

    return received
# check_tcp_server()

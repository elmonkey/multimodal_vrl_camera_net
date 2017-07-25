#!/usr/bin/env python
"""
Created on Wed Dec  3 14:34:13 2014

TCP Server Example

ref: 
    https://docs.python.org/2/library/socketserver.html
    
Open a terminal, set server IP, and run the server
ctrl+alt+t
sudo ifconfig eth0 192.168.0.12 netmask 255.255.255.0
python server_threads.py

NOTE:
    Try changin the list structures for sets - to speed up the process
    
@author: carlos
"""
import SocketServer
import threading
import time
import sys
from time import strftime, localtime

devs = []
roll = []
conc = True  # flag for connecting
disc = False  # flag to disconnect
done = False  # flag to terminate
save = False  # flag to save
close = False
action = ""
ready = []


# Add/Remove devices to/from dev_list
dev_list = ['dev1', 'dev2']  # Allowed devices , 'dev3', 'dev4'

terminate_list = dev_list[:]
terminate = False  # termiantion flag


dev_dict = {'dev1': {'PORT': 50007},
            'dev2': {'PORT': 50008},
            #            'dev3': {'PORT': 50009},
            }

roll = {}
for d in dev_list:
    roll[d] = 'n'
    ready.append(False)


class MyTCPHandler(SocketServer.BaseRequestHandler):
    """
    The RequestHandler class for our server.

    It is instantiated once per connection to the server, and must
    override the handle() method to implement communication to the
    client.

    Add new devices to dev_list

    """

    def handle(self):
        global done, roll, conc, disc, terminate, devs, save, close, action, ready

        # self.request is the TCP socket connected to the client
        self.data = self.request.recv(1024).strip().split(" ")
        self.devid = self.data[0]
        self.cmd = self.data[1].split("_")
        self.msg = ""
        self.tic = time.time()  # server time tic
        self.time = strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())

        dev = "dev{}".format(self.devid)

#        print " ====== SERVER -- RUNNING ====== "
        print "{} wrote:".format(self.client_address[0])
        print "\t", self.data
        print "\t at local time {}".format(self.tic)

        # Check device is on the list of allowed devices:
        if dev in dev_list:
            print "Device {} recognized." .format(dev)
            if not terminate:

                # --- Connect the device <dev1#> "connect" command
                if self.cmd[0].lower() == "connect":
                    print "\tAttempting to {} {}".format(self.cmd, dev)
                    if dev in devs:
                        self.msg = "dev{} ready_{}".format(self.devid, self.tic)
                    else:
                        devs.append(dev)
                        self.msg = "dev{} connected_{}".format(self.devid, self.tic)

#                # --- Get server time stamp: laxed synchronization
#                elif self.cmd.lower() == "sync":  # get server time stamp
#                    self.msg = "sync_{}".format(self.tic)
#
#                # --- Strict synchronization: "check" server has registered all dev
#                elif self.cmd.lower() == "check":
#                    if (save) and (dev in devs):
#                        # print "Here is where we remove the dev!"
#                        self.msg = "save_{}".format(self.tic)
#                        devs.remove(dev)
#                    else:  # either save = False or device is not in the list
#                        # print "Device {} saved the img and is no longer enabled".format(self.devid)
#                        # print "\tstill connected: ", str(devs).strip('[]')
#                        self.msg = "wait_{}".format(self.tic)
#
                # --- Allow the clients to request termination using "close"
                elif self.cmd[0].lower() == "close":
                    print "Terminating all threads"
                    self.msg = "close_{}".format(self.tic)
                    close = True
                    ready[int(self.devid) - 1] = False
                
                # --- Tell the node what to do
                elif self.cmd[0].lower() == "info":
                    ready[int(self.devid) - 1] = False
                    if action == "new":
                        self.msg = "new_{}".format(self.tic)
                    elif action == "restart":
                        self.msg = "restart_{}".format(self.tic)
                    elif close == True:
                        self.msg = "close_{}".format(self.tic)
                    else:
                        self.msg = "wait_{}".format(self.tic)
                
                # --- Tell the node what to do
                elif self.cmd[0].lower() == "ready":
                    ready[int(self.devid) - 1] = True
                    if action == "record":
                        if all(device == True for device in ready):
                            self.msg = "record_{}".format(self.tic)
                        else:
                            self.msg = "wait_{}".format(self.tic)
                    elif action == "new":
                        self.msg = "new_{}".format(self.tic)
                    elif action == "restart":
                        self.msg = "restart_{}".format(self.tic)
                    elif action == "stop":
                        self.msg = "stop_{}".format(self.tic)
                    elif close == True:
                        self.msg = "close_{}".format(self.tic)
                
                # --- Manage commands from controling device  
                if (int(self.devid) == 1):
                    if self.cmd[1].lower() == "record":
                        action = "record"
                        self.msg = "record_{}".format(self.tic)
                    elif self.cmd[1].lower() == "stop":
                        action = "stop"
                        self.msg = "stop_{}".format(self.tic)
                    elif self.cmd[1].lower() == "restart":
                        action = "restart"
                        self.msg = "restart_{}".format(self.tic)
                    elif self.cmd[1].lower() == "new":
                        action = "new"
                        self.msg = "new_{}".format(self.tic)

                else:  # unknown command
                    print "Unknown command {}. Use connect, check, or close".format(self.cmd)

            else:  # terminate
                if dev in devs:
                    terminate_list.remove(dev)
                    self.msg = "terminate"
                if len(terminate_list) == 0:
                    done = True

        else:  # dev not in list
            print "Unknown device {}. Please check devid try again!".format(dev)
            self.msg = "dev{} -Not recognized by the server!!".format(self.devid)

        print "The registered devices are {}".format(devs)

        if len(devs) == len(dev_list):  # roll.values().count('y') == len(dev_list):
            conc = False  # done connecting all devs
            disc = True  # allows to disconnect
            save = True
            print "All CONNECTED"
        elif len(devs) == 0:  # roll.values().count('y') == 0:
            conc = True   # can begin connecting devices
            disc = False  # done disconnecting all devices
            save = False
            print "All DISCONNECTED"

        # print 'msg: ', self.msg
        self.request.sendall(self.msg.lower() + '=' +
                             strftime("%a, %d %b %Y %H:%M:%S +0000", localtime()))
        # print'Devices ready: ', devs
# MyTCPHandler()


class ServerThread(threading.Thread):
    #HOST = "localhost"
    HOST = "192.168.0.12"  # Local net

    def __init__(self, serverid='dev1', HOST=HOST, PORT=50007):
        print 'serving %s' % serverid
        threading.Thread.__init__(self)
        self.server = SocketServer.TCPServer((HOST, PORT), MyTCPHandler)

    def run(self):
        self.server.serve_forever()
        print "CLOSED!"
# ServerThread()


if __name__ == "__main__":

    time.sleep(5)  # secs pause!
    print " ====== SERVER -- RUNNING ====== "

    #HOST = "localhost"
    HOST = "192.168.0.12"  # Local net

#    dev_list1 = ['dev1']
    server_thread_list = []
    for d in dev_dict:
        server_thread = ServerThread(d, HOST=HOST, PORT=dev_dict[d]['PORT'])
        #server_thread.daemon = True
        server_thread.start()
        server_thread_list.append(server_thread)

    try:
        while not done:
            pass
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        for s in server_thread_list:
            s.server.shutdown()
        sys.exit(0)

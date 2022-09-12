# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:45:19 2022

@author: AKN
"""

from threading import Thread
from time import sleep
import labrad

class TemperaturePoll(Thread):
    
    def __init__(self,ls):
        Thread.__init__(self)
        self.daemon = True
        self.delay = 0.01
        self.ls = ls
        self.tempD4 = 0
        self.tempD5 = 0
        self.setp1 = 0
        self.setp2 = 0
        self.hrange1 = 0
        self.hrange2 = 0
        self.start()
        
    def run(self):
        while True:
            try:
                self.tempD4 = float(self.ls.read_temp('D4'))
                sleep(self.delay)
                self.tempD5 = float(self.ls.read_temp('D5'))
                sleep(self.delay)
                self.setp1 = float(self.ls.read_p(1))
                sleep(self.delay)
                self.hrange1 = int(self.ls.read_heater_range(1))
                sleep(self.delay)
                self.setp2 = float(self.ls.read_p(2))
                sleep(self.delay)
                self.hrange2 = int(self.ls.read_heater_range(2))
                sleep(self.delay)
            except:
                print("Temperature read error.")

if __name__ == "__main__":
    
    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # LakeShore 350
    ls350 = cxn.lakeshore_350()
    ls350.select_device() 
    
    temp_server = TemperaturePoll(ls350)
    
    while True:
        print("temp D4 = {}".format(temp_server.tempD4))
        print("temp D5 = {}".format(temp_server.tempD5))
        print("Setp 1 = {}".format(temp_server.setp1))
        print("Setp 2 = {}".format(temp_server.setp2))
        print("Heater range 1 = {}".format(temp_server.hrange1))
        print("Heater range 2 = {}".format(temp_server.hrange2))
        sleep(1)
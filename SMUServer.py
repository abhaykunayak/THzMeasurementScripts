# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:20:39 2022

@author: AKN
"""

from threading import Thread
from time import sleep
import labrad

class CurrentPoll(Thread):
    
    def __init__(self,smu):
        Thread.__init__(self)
        self.daemon = True
        self.delay = 0.01
        self.smu = smu
        self.I = 0
        self.start()
        
    def run(self):
        while True:
            try:
                self.I = self.smu.read_i()
                sleep(self.delay)
            except:
                print("Current read error.")

if __name__ == "__main__":
    
    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # LakeShore 350
    smu = cxn.K2400()
    smu.select_device() 
    
    smu_server = CurrentPoll(smu)
    
    while True:
        print("Current = {:3.3f} nA".format(smu_server.I*1e9))
        sleep(1)
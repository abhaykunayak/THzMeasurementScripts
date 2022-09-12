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
        self.ls = ls
        self.tempA = 0
        self.tempB = 0
        self.start()
        
    def run(self):
        while True:
            try:
                self.tempA = float(self.ls.read_temp('D4'))
                self.tempB = float(self.ls.read_temp('D5'))
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
        print("temp A = {}".format(temp_server.tempA))
        sleep(1)
        print("temp B = {}".format(temp_server.tempB))
        sleep(1)
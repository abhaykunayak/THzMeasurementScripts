# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:01:38 2022

@author: AKN
"""

import time
import yaml
import os
import labrad
import numpy as np
import scipy.io as sio
from datetime import datetime
from threading import Thread
from scipy.ndimage import gaussian_filter

class Scan(Thread):
    
    def __init__(self,
                 params,
                 spot,
                 dac_d,
                 dac_m,
                 save_data=False):
        
        Thread.__init__(self)
        self.daemon = True
        self.params = params
        self.rootdir = params['ROOTDIR'][2:-1]
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.save_data = save_data
        self.spot = spot
        self.dac_d = dac_d
        self.dac_m = dac_m
        self.dac_d_dummy_ch = params['DAC_OUTPUT_CH_DUMMY']
        if self.spot=='E':
            self.dac_m_ch = params['DAC_M_CH'][0:2]
        else:
            self.dac_m_ch = params['DAC_M_CH'][2:4]
        self.dac_d_ch = params[spot]['DAC_D_IN_CH']

        self.delay = params['DELAY']
        self.max_I = 0
        self.max_x = 0
        self.max_y = 0
            
    def voltage_ramp(self,dac_ch,v_initial,v_final):
        if v_initial == v_final:
            print('[{}] No voltage ramp needed.'.format(self.current_time()))
            return
        step = 10
        v_steps = np.linspace(v_initial, v_final, step)
        print('[{}] Ramping voltage...'.format(self.current_time()))
        for i in range(step):
            self.dac_d.set_voltage(dac_ch,v_steps[i])
            time.sleep(self.delay)
        print('[{}] Voltage ramp completed.'.format(self.current_time()))
        time.sleep(self.delay*5)
        
    def current_time(self):
        now = datetime.now()
        val = now.strftime("%H:%M:%S")
        return val
    
    def set_scan_range(self,x_center,y_center,scan_range,scan_step):
        self.x_center = x_center
        self.y_center = y_center
        self.scan_range = scan_range
        self.scan_step = scan_step
        
    def set_datafile(self,dv):
        self.dv = dv
        
    def save_to_datavault(self,s,x,y,I):
        Ns = np.arange(np.size(s))
        Nx = np.arange(np.size(x,1))
        Ny = np.arange(np.size(y,2))
        
        if self.save_data:
                print('[{}] Saving to datavault...'.format(self.current_time()))
                mesh = np.meshgrid(Ns,Nx,Ny,indexing='ij')
                data = np.column_stack((np.ndarray.flatten(mesh[0]),
                                        np.ndarray.flatten(mesh[1]),
                                        np.ndarray.flatten(mesh[2]),
                                        np.ndarray.flatten(x),
                                        np.ndarray.flatten(y),
                                        np.ndarray.flatten(I)))
                self.dv.add(data)

    def save_to_mat(self,s,x,y,I):
        Ns = np.arange(np.size(s))
        Nx = np.arange(np.size(x,1))
        Ny = np.arange(np.size(y,2))
        if self.save_data:
                print('[{}] Saving to *.mat...'.format(self.current_time()))
                mesh = np.meshgrid(Ns,Nx,Ny,indexing='ij')
                data = np.column_stack((
                        np.ndarray.flatten(mesh[0]),
                        np.ndarray.flatten(mesh[1]),
                        np.ndarray.flatten(mesh[2]),
                        np.ndarray.flatten(x),
                        np.ndarray.flatten(y),
                        np.ndarray.flatten(I)))
                sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                            {'data':data})
        
    def run(self):
        # Sweeps
        N = self.params['MSWEEPS']
        swp = np.arange(N)

        # Scan range
        x_rng = np.linspace((self.x_center-self.scan_range/2),(self.x_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        y_rng = np.linspace((self.y_center-self.scan_range/2),(self.y_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        
        currentRead = np.zeros((N,len(x_rng),len(y_rng)),dtype=float)
        xv = np.zeros((N,len(x_rng),len(y_rng)),dtype=float)
        yv = np.zeros((N,len(x_rng),len(y_rng)),dtype=float)

        for k in swp:
            # Sweep
            print("[{}] Sweep # {} out of {}.".format(
                self.current_time(),k+1,N))
            # def sweep action

            # scan in X
            for i in range(len(x_rng)):
                self.dac_m.set_voltage(self.dac_m_ch[0], x_rng[i])
                
                percent = (x_rng[i] - np.min(x_rng))/(np.max(x_rng) - np.min(x_rng))
                print("[{}] Scanning {} {:.2f} % complete".format(
                    self.current_time(),self.spot,100*percent))

                # scan in Y    
                for j in range(len(y_rng)):
                    self.dac_m.set_voltage(self.dac_m_ch[1], y_rng[j])
                    
                    time.sleep(self.params['LIA']['TIME_CONST'])

                    xv[k][i][j] = x_rng[i]
                    yv[k][i][j] = y_rng[j]
                    currentRead[k][i][j] = np.mean(self.dac_d.buffer_ramp(
                        self.dac_d_dummy_ch,
                        [self.dac_d_ch],
                        [0.0],
                        [0.0],
                        1,
                        0.1*self.params['LIA']['TIME_CONST']*1e6,
                        self.params['AVGS']))
                    # currentRead[k][i][j] = self.dac_d.read_voltage(self.dac_d_ch)
        
        # Save to datavault
        self.save_to_datavault(swp, xv, yv, currentRead)
        # Save to *.mat format
        self.save_to_mat(swp, xv, yv, currentRead)

        currentRead = np.abs(currentRead)
        currentRead = gaussian_filter(currentRead[-1], sigma=1)
        self.max_I = np.max(currentRead)
        [max_x_idx, max_y_idx] = np.unravel_index(np.argmax(currentRead),(len(x_rng),len(y_rng)))
        self.max_x = x_rng[max_x_idx]
        self.max_y = y_rng[max_y_idx]
        print("[{}] Max signal {} {:.4f} V at X: {:.4f} V, Y: {:.4f} V".format(
            self.current_time(),self.spot,self.max_I,self.max_x,self.max_y))
        
        self.dac_m.set_voltage(self.dac_m_ch[0], self.max_x)
        self.dac_m.set_voltage(self.dac_m_ch[1], self.max_y)
        return
    
def scan_mirror(params,spot):
    # Initialize labrad and servers
    cxn_m = labrad.connect()
    cxn_d = labrad.connect()
        
    # DAC-ADC for mirrors
    dac_m = cxn_m.dac_adc
    dac_m.select_device(params['DAC_MIRROR'])

    # DAC-ADC for switch
    dac_d = cxn_d.dac_adc
    dac_d.select_device(params['DAC_DATA'])
    dac_d.initialize()
    
    # Delay stage 
    ds = cxn_d.esp300()
    ds.select_device()
    ds.move_absolute(1,params['STAGE_POS'])
    time.sleep(5)

    # SR 830
    if params['SPOT'] == 'A':
        sr = cxn_d.sr860()
    else:
        sr = cxn_d.sr830()
    sr.select_device()
    sr.time_constant(params['LIA']['TIME_CONST'])
    sr.sensitivity(params['LIA']['SENS'])

    # Data vault
    dv = cxn_m.data_vault()
    
    # Change to data directory
    dv.cd(params['DATADIR'])
    
    # Create new data file
    dv.new('{}_'.format(spot)+params['FILENAME'], 
           ['IDX 0 [AU]', 'IDX 1 [AU]', 'IDX 2 [AU]', 'X Pos [V]', 'Y Pos [V]'], 
           ['I Measure [A]'])
    
    # Instatiate mirror scan object
    scan = Scan(params,
                  spot,
                  dac_d,
                  dac_m,
                  True)
    
    # DataVault File
    scan.set_datafile(dv)
    
    # Coarse
    scan.set_scan_range(params[spot]['X_CENTER'],params[spot]['Y_CENTER'], 
                        params[spot]['RANGE'],params[spot]['STEP'])
    
    # Measurement
    start = time.time()
    print("[{}] Estimated total time: {:.0f} s"
          .format(scan.current_time(),
                  0.04*(params[spot]['RANGE']**2)/(params[spot]['STEP']**2)))
    
    # Voltage ramp
    scan.voltage_ramp(params[spot]['DAC_D_OUT_CH'], 0, params['BIAS'])

    # Scan the Mirror
    scan.start()
    
    # Wait to finish scan
    scan.join()
    
    # Voltage ramp doen
    scan.voltage_ramp(params[spot]['DAC_D_OUT_CH'], params['BIAS'], 0)
    
    # Measurements ended
    end = time.time()
    print("[{}] DONE! Time Consumption: {:.1f} s".format(scan.current_time(),end-start) )

def main():
    # Load config file
    CONFIG_FILENAME = 'MirrorScanDACConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    scan_mirror(params,params['SPOT'])

if __name__ == '__main__':
    main()
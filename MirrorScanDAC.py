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
        self.dac_d_dummy_ch = params['DAC_D_DUMMY_CH']
        if self.spot=='E':
            self.dac_m_ch = params['DAC_M_CH'][0:2]
            self.dac_d_ch = params['DAC_D_E_IN_CH']
        else:
            self.dac_m_ch = params['DAC_M_CH'][2:4]
            self.dac_d_ch = params['DAC_D_A_IN_CH']

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
        time.sleep(self.delay)
        
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
        
    def save_to_datavault(self,x,y,I):
        if self.save_data:
                data = np.column_stack((np.ndarray.flatten(x),
                                        np.ndarray.flatten(y),
                                         np.ndarray.flatten(I)))
                self.dv.add(data)

    def save_to_mat(self,x_idx,y_idx,x,y,I):
        if self.save_data:
                print('[{}] Saving to *.mat...'.format(self.current_time()))
                mesh = np.meshgrid(x_idx,y_idx,indexing='ij')
                data = np.column_stack((np.ndarray.flatten(mesh[0]),
                                        np.ndarray.flatten(mesh[1]),
                                        np.ndarray.flatten(x),
                                        np.ndarray.flatten(y),
                                         np.ndarray.flatten(I)))
                sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                            {'data':data})
        
    def run(self):
        # Scan range
        x_rng = np.linspace((self.x_center-self.scan_range/2),(self.x_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        y_rng = np.linspace((self.y_center-self.scan_range/2),(self.y_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        x_rng_idx = np.arange(np.size(x_rng))
        y_rng_idx = np.arange(np.size(y_rng))
        
        currentRead = np.zeros((len(x_rng),len(y_rng)),dtype=float)
        xv = np.zeros((len(x_rng),len(y_rng)),dtype=float)
        yv = np.zeros((len(x_rng),len(y_rng)),dtype=float)
        
        # scan in X
        for i in range(len(x_rng)):
            self.dac_m.set_voltage(self.dac_m_ch[0], x_rng[i])
            
            percent = (x_rng[i] - np.min(x_rng))/(np.max(x_rng) - np.min(x_rng))
            print("[{}] Scanning {} {:.2f} % complete".format(
                self.current_time(),self.spot,100*percent))
        
            # scan in Y    
            for j in range(len(y_rng)):
                self.dac_m.set_voltage(self.dac_m_ch[1], y_rng[j])
                xv[i][j] = x_rng[i]
                yv[i][j] = y_rng[j]
                currentRead[i][j] = np.mean(self.dac_d.buffer_ramp(
                    [self.dac_d_dummy_ch],
                    [self.dac_d_ch],
                    [0.0],
                    [0.0],
                    self.params['AVGS'],
                    1e-3*1e6,
                    self.params['READINGS']))
        
        # Save to datavault
        self.save_to_datavault(xv, yv, currentRead)
        # Save to *.mat format
        self.save_to_mat(x_rng_idx,y_rng_idx,xv,yv,currentRead)

        currentRead = -gaussian_filter(currentRead, sigma=1)
        self.max_I = np.max(currentRead)
        [max_x_idx, max_y_idx] = np.unravel_index(np.argmax(currentRead),(len(x_rng),len(y_rng)))
        self.max_x = x_rng[max_x_idx]
        self.max_y = y_rng[max_y_idx]
        print("[{}] Max signal {} {:.4f} V at X: {:.4f} V, Y: {:.4f} V".format(
            self.current_time(),self.spot,self.max_I,self.max_x,self.max_y))
        
        self.dac_m.set_voltage(self.dac_m_ch[0], self.max_x)
        self.dac_m.set_voltage(self.dac_m_ch[1], self.max_y)
        
        return

def main():
    # Load config file
    CONFIG_FILENAME = 'MirrorScanDACConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn_e = labrad.connect()
    cxn_a = labrad.connect()
    cxn_d = labrad.connect()
        
    # DAC-ADC for mirrors
    dac_e = cxn_e.dac_adc
    dac_e.select_device(params['DAC_MIRROR'])
    
    dac_a = cxn_a.dac_adc
    dac_a.select_device(params['DAC_MIRROR'])
    
    # DAC-ADC for switch
    dac_d = cxn_d.dac_adc
    dac_d.select_device(params['DAC_DATA'])
   
    # Data vault
    dv_e = cxn_e.data_vault()
    dv_a = cxn_a.data_vault()
    
    # Change to data directory
    dv_e.cd(params['DATADIR'])
    dv_a.cd(params['DATADIR'])
    
    # Create new data file
    dv_e.new('E_'+params['FILENAME'], ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
    dv_a.new('A_'+params['FILENAME'], ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
    
    # Instatiate mirror scan object
    scan_e = Scan(params,
                  'E',
                  dac_d,
                  dac_e,
                  True)
    scan_a = Scan(params,
                'A',
                dac_d,
                dac_e,
                True)
    
    # DataVault File
    scan_e.set_datafile(dv_e)
    scan_a.set_datafile(dv_a)
    
    # Coarse
    scan_e.set_scan_range(params['EX_CENTER'],params['EY_CENTER'], params['RANGE'],params['STEP'])
    scan_a.set_scan_range(params['AX_CENTER'],params['AY_CENTER'], params['RANGE'],params['STEP'])
    
    # Measurement
    start = time.time()
    print("[{}] Estimated total time: {:.0f} s"
          .format(scan_e.current_time(),
                  4.5*params['DELAY']*(params['RANGE']*params['RANGE'])/(params['STEP']*params['STEP'])))
    
    # Voltage ramp
    #scan_e.voltage_ramp(params['DAC_D_E_OUT_CH'], 0, params['BIAS'])
    scan_a.voltage_ramp(params['DAC_D_A_OUT_CH'], 0, params['BIAS'])

    # Scan the Mirror
    #scan_e.start()
    scan_a.start()
    
    # Wait to finish scan
    #scan_e.join()
    scan_a.join()
    
    # Voltage ramp doen
    #scan_e.voltage_ramp(params['DAC_D_E_OUT_CH'], params['BIAS'], 0)
    scan_a.voltage_ramp(params['DAC_D_A_OUT_CH'], params['BIAS'], 0)
    
    # Measurements ended
    end = time.time()
    print("[{}] DONE! Time Consumption: {:.1f} s".format(scan_e.current_time(),end-start) )

if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:01:38 2022

@author: AKN
"""

import time
import labrad
import numpy as np
import yaml
from datetime import datetime
from threading import Thread
from scipy.ndimage import gaussian_filter

class Scan(Thread):
    
    def __init__(self,spot,smu,dac,dac_ch_x,dac_ch_y,x_center,y_center,scan_range,scan_step):
        Thread.__init__(self)
        self.spot = spot
        self.smu = smu
        self.dac = dac
        self.dac_ch_x = dac_ch_x
        self.dac_ch_y = dac_ch_y
        self.x_center = x_center
        self.y_center = y_center
        self.scan_range = scan_range
        self.scan_step = scan_step
        self.max_I = 0
        self.max_x = 0
        self.max_y = 0
            
    def voltage_ramp(self,v_initial,v_final):
        if v_initial == v_final:
            print('[{}] No voltage ramp needed.'.format(self.current_time()))
            return
        step = 10
        v_steps = np.linspace(v_initial, v_final, step)
        print('[{}] Ramping voltage...'.format(self.current_time()))
        self.smu.output_on()
        for i in range(step):
            self.smu.set_volts(v_steps[i], 5e-6)
            time.sleep(0.1)
        print('[{}] Voltage ramp completed.'.format(self.current_time()))
        time.sleep(1)
        
    def current_time(self):
        now = datetime.now()
        val = now.strftime("%H:%M:%S")
        return val
    
    def run(self):

        # Scan range
        x_rng = np.linspace((self.x_center-self.scan_range/2),(self.x_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        y_rng = np.linspace((self.y_center-self.scan_range/2),(self.y_center+self.scan_range/2),
                            int(self.scan_range/self.scan_step)+1)
        
        currentRead = np.zeros((len(x_rng),len(y_rng)),dtype=float)
        
        # scan in X
        for i in range(len(x_rng)):
            self.dac.set_voltage(self.dac_ch_x, x_rng[i])
            
            percent = (x_rng[i] - np.min(x_rng))/(np.max(x_rng) - np.min(x_rng))
            print("[{}] Scanning {} {:.2f} % complete".format(
                self.current_time(),self.spot,100*percent))
        
            # scan in Y    
            for j in range(len(y_rng)):
                self.dac.set_voltage(self.dac_ch_y, y_rng[j])
                time.sleep(0.01)
                currentRead[i][j] = self.smu.read_i()
                
        currentRead = gaussian_filter(currentRead, sigma=1)
        self.max_I = np.max(currentRead)
        [max_x_idx, max_y_idx] = np.unravel_index(np.argmax(currentRead),(len(x_rng),len(y_rng)))
        self.max_x = x_rng[max_x_idx]
        self.max_y = y_rng[max_y_idx]
        print("Max current {:.4f} nA at X: {:.4f} V, Y: {:.4f} V".format(self.max_I*1e9,self.max_x,self.max_y))
        
        self.dac.set_voltage(self.dac_ch_x, self.max_x)
        self.dac.set_voltage(self.dac_ch_y, self.max_y)
        
        return

    
def main():
    
    # Define parameters
    params = dict()

    # params['SCAN_BEAM'] = 'A'           # E or A

    # params['ROOTDIR'] = r"C:\Users\Marconi\Young Lab Dropbox\Young Group\THz\Raw Data"                               
    # params['DATADIR'] = "2022_09_07_TL2715_AKNDB010_5E"                  
    # params['FILENAME'] = "a_T_303K"
        
    params['EY_CENTER'] = -0.3534       #DAC1 refl: -0.3; trans: -0.35
    params['EX_CENTER'] = 4.8381        #DAC0 refl: 5.16; trans: 6.25
    params['AY_CENTER'] = 0.5494        #DAC3
    params['AX_CENTER'] = 1.5460        #DAC2 
    params['RANGE'] = 0.2
    params['STEP'] = 0.01

    params['DAC_DATA'] = "DAC-ADC_AD7734-AD5791 (COM5)"         # DAC for signal
    params['DAC_MIRROR'] = "DAC-ADC_AD7734-AD5791_4x4 (COM3)"   # DAC for mirrors
    params['DAC_CH_Y'] = 1
    params['DAC_CH_X'] = 0

    params['BIAS'] = 10
    params['DELAY'] = 0.05
    params['SMU_RANGE'] = 10e-6         # current range on K2450

    params['COMPL'] = 5e-6
    params['NPLC'] = 1                  # 0.1 -- 10
    params['FILT'] = "OFF"              # 'ON' or 'OFF'
    params['FILT_TYPE'] = "REPeat"      # 'REPeat' or 'MOVing'
    params['FILT_COUNT'] = 1            # 1 -- 100

    
    # Initialize labrad and servers
    cxn_m = labrad.connect()
        
    # DAC-ADC
    dac_m = cxn_m.dac_adc
    dac_m.select_device(params['DAC_MIRROR'])

    smu2400 = cxn_m.k2400()
    smu2400.select_device()
    smu2400.gpib_write(":ROUTe:TERMinals FRONt") # FRONt or REAR
    
    smu2400.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2400.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2400.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2400.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    
    smu2450 = cxn_m.k2450()
    smu2450.select_device()
    smu2450.gpib_write(":ROUTe:TERMinals FRONt") # FRONt or REAR
    
    smu2450.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2450.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2450.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2450.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    
    # Instatiate mirror scan object
    scan_e = Scan('E',smu2400,dac_m,0,1,params['EX_CENTER'],params['EY_CENTER'],
                         params['RANGE'],params['STEP'])
    scan_a = Scan('A',smu2450,dac_m,2,3,params['AX_CENTER'],params['AY_CENTER'],
                         params['RANGE'],params['STEP'])
    
    # Measurement
    start = time.time()
    print("[{}] Estimated total time: {:.0f} s"
          .format(scan_e.current_time(),
                  4.5*params['DELAY']*(params['RANGE']*params['RANGE'])/(params['STEP']*params['STEP'])))
    
    # Voltage ramp
    scan_e.voltage_ramp(0, params['BIAS'])
    scan_a.voltage_ramp(0, params['BIAS'])

    # Scan the Mirror
    # scan_e.start()
    scan_a.start()
    
    # Wait to finish scan
    # scan_e.join()
    scan_a.join()
    
    # Voltage ramp doen
    scan_e.voltage_ramp(params['BIAS'], 0)
    scan_a.voltage_ramp(params['BIAS'], 0)
    
    # Reset the SMU
    smu2400.gpib_write('*RST')
    smu2450.gpib_write('*RST')
    
    # Measurements ended
    end = time.time()
    print("[{}] DONE! Time Consumption: {:.1f} s".format(scan_e.current_time(),end-start) )

if __name__ == '__main__':
    main()
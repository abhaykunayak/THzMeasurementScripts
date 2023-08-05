# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:01:38 2022

@author: AKN
"""

import time
import labrad
import numpy as np
import scipy.io as sio
from datetime import datetime
from threading import Thread
from scipy.ndimage import gaussian_filter

class Scan(Thread):
    
    def __init__(self,params,spot,smu,dac,dac_ch_x,dac_ch_y,save_data=False):
        Thread.__init__(self)
        self.daemon = True
        self.rootdir = params['ROOTDIR']
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.save_data = save_data
        self.spot = spot
        self.smu = smu
        self.dac = dac
        self.dac_ch_x = dac_ch_x
        self.dac_ch_y = dac_ch_y
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
            self.smu.set_volts(v_steps[i], 10e-6)
            time.sleep(0.1)
        print('[{}] Voltage ramp completed.'.format(self.current_time()))
        time.sleep(1)
        
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
                data = np.array([x, y, I])
                self.dv.add(data)
    
    def save_to_mat(self,x,y,I):
        Nx = np.arange(np.size(x))
        Ny = np.arange(np.size(y))
        if self.save_data:
                print('[{}] Saving to *.mat...'.format(self.current_time()))
                mesh = np.meshgrid(Nx,Ny,indexing='ij')
                mesh_data = np.meshgrid(x,y,indexing='ij')
                data = np.column_stack((
                        np.ndarray.flatten(mesh[0]),
                        np.ndarray.flatten(mesh[1]),
                        np.ndarray.flatten(mesh_data[0]),
                        np.ndarray.flatten(mesh_data[1]),
                        np.ndarray.flatten(I)))
                sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                            {'data':data})
                
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
                # time.sleep(0.01)
                currentRead[i][j] = self.smu.read_i()
                self.save_to_datavault(x_rng[i], y_rng[j], currentRead[i][j])
                
        self.save_to_mat(x_rng,y_rng,currentRead)
        currentRead = gaussian_filter(currentRead, sigma=1)
        self.max_I = np.max(currentRead)
        [max_x_idx, max_y_idx] = np.unravel_index(np.argmax(currentRead),(len(x_rng),len(y_rng)))
        self.max_x = x_rng[max_x_idx]
        self.max_y = y_rng[max_y_idx]
        print("[{}] Max current {} {:.4f} nA at X: {:.4f} V, Y: {:.4f} V".format(
            self.current_time(),self.spot,self.max_I*1e9,self.max_x,self.max_y))
        
        self.dac.set_voltage(self.dac_ch_x, self.max_x)
        self.dac.set_voltage(self.dac_ch_y, self.max_y)
        
        return

def main():
    
    # Define parameters
    params = dict()

    params['ROOTDIR'] = r"C:\Users\Marconi\Young Lab Dropbox\Young Group\THz\Raw Data"
    params['DATADIR'] = "2023_07_28_AKN_DMLG_08"                  
    params['FILENAME'] = "MirrorScanSMU"
        
    params['EY_CENTER'] = -0.4140         #DAC1 refl: -0.4510; trans: -0.35
    params['EX_CENTER'] = 5.3355         #DAC0 refl: 5.8450; trans: 6.25 # 5.9750 
    params['AY_CENTER'] = 0.7145         #DAC3 (old: 0.5210)
    params['AX_CENTER'] = 1.1005         #DAC2 (old: 1.3215)
    params['RANGE'] = 0.01               # usual: 0.2
    params['STEP'] = 0.0005               # usual: 0.01

    params['DAC_DATA'] = "DAC-ADC_AD7734-AD5791 (COM5)"         # DAC for signal
    params['DAC_MIRROR'] = "DAC-ADC_AD7734-AD5791_4x4 (COM3)"   # DAC for mirrors
    params['DAC_CH_Y'] = 1
    params['DAC_CH_X'] = 0

    params['BIAS'] = 10
    params['DELAY'] = 0.05
    params['SMU_RANGE'] = 100e-6         # current range on K2450

    params['COMPL'] = 20e-6
    params['NPLC'] = 1                  # 0.1 -- 10
    params['FILT'] = "OFF"              # 'ON' or 'OFF'
    params['FILT_TYPE'] = "REPeat"      # 'REPeat' or 'MOVing'
    params['FILT_COUNT'] = 1            # 1 -- 100

    params["AUTOZOOM"] = False
    
    # Initialize labrad and servers
    cxn_e = labrad.connect()
    cxn_a = labrad.connect()
        
    # DAC-ADC
    dac_e = cxn_e.dac_adc
    dac_e.select_device(params['DAC_MIRROR'])
    
    dac_a = cxn_a.dac_adc
    dac_a.select_device(params['DAC_MIRROR'])

    smu2400 = cxn_e.k2400()
    smu2400.select_device()
    smu2400.gpib_write(":ROUTe:TERMinals FRONt") # FRONt or REAR
    
    smu2400.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2400.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2400.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2400.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    
    smu2450 = cxn_a.k2450()
    smu2450.select_device()
    smu2450.gpib_write(":ROUTe:TERMinals FRONt") # FRONt or REAR
    
    smu2450.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2450.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2450.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2450.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    
    # Data vault
    dv_e = cxn_e.data_vault()
    dv_a = cxn_e.data_vault()
    
    # Change to data directory
    dv_e.cd(params['DATADIR'])
    dv_a.cd(params['DATADIR'])
    
    # Create new data file
    dv_e.new('E_'+params['FILENAME'], ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
    dv_a.new('A_'+params['FILENAME'], ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
    
    # Instatiate mirror scan object
    scan_e = Scan(params,'E',smu2400,dac_e,0,1,True)
    scan_a = Scan(params,'A',smu2450,dac_a,2,3,True)
    
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
    scan_e.voltage_ramp(0, params['BIAS'])
    scan_a.voltage_ramp(0, params['BIAS'])

    # Scan the Mirror
    scan_e.start()
    scan_a.start()
    
    # Wait to finish scan
    scan_e.join()
    scan_a.join()
    
    e_max_x = scan_e.max_x
    e_max_y = scan_e.max_y
    
    a_max_x = scan_a.max_x
    a_max_y = scan_a.max_y
    
    if params["AUTOZOOM"]:
        print("[{}] Auto-zoom...".format(scan_e.current_time()) )
        # Create new data file
        dv_e.new('E_'+params['FILENAME']+'_fine', ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
        dv_a.new('A_'+params['FILENAME']+'_fine', ['X Pos [V]', 'Y Pos [V]'], ['I Measure [A]'])
        
        # Instatiate mirror scan object
        scan_e = Scan('E',smu2400,dac_e,0,1,True)
        scan_a = Scan('A',smu2450,dac_a,2,3,True)
        
        scan_e.set_datafile(dv_e)
        scan_a.set_datafile(dv_a)
        
        # Fine
        scan_e.set_scan_range(e_max_x,e_max_y,params['RANGE']/20,params['STEP']/20)
        scan_a.set_scan_range(a_max_x,a_max_y, params['RANGE']/20,params['STEP']/20)
        
        # Scan the Mirror
        scan_e.start()
        scan_a.start()
        
        # Wait to finish scan
        scan_e.join()
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
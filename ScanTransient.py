# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:50:53 2022

@author: AKN
"""

import time
import numpy as np
import labrad
import yaml
from datetime import datetime
import os
import logging
import TemperatureServer
import SMUServer
import MirrorScan

class Transient:
    
    def __init__(self,dv,ds,dac,dac_e,dac_a,smu_e,smu_a,tempServer):
        self.idx = 0
        self.dv = dv
        self.ds = ds
        self.dac = dac
        self.dac_e = dac_e
        self.dac_a = dac_a
        self.smu_e = smu_e
        self.smu_a = smu_a
        self.tempServer = tempServer
        self.I_e = 0
        self.I_a = 0
        
        
    def set_scan_params(self,params):
        self.e_max_x = params['EX_CENTER']
        self.e_max_y = params['EY_CENTER']
        self.a_max_x = params['AX_CENTER']
        self.a_max_y = params['AY_CENTER']
        self.range = params['RANGE']
        self.step = params['STEP']
        
    def current_time(self):

        now = datetime.now()
        val = now.strftime("%H:%M:%S")
        return val
    
    def log_message(self,msg):

        msg_timestamp = '[{}] {}'.format(self.current_time(),msg)
        logging.info(msg_timestamp)
        print(msg_timestamp)
        
    def setup_logfile(self,params):

        print('[{}] Setting up datafile and logging...'.format(self.current_time())) 
        
        # Change the working directory
        fullpath = os.path.join(params['ROOTDIR'],params['DATADIR']+'.dir')
        os.chdir(fullpath)
        
        # Setup logging
        filename_log = "{}_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"),params['FILENAME'])
        logging.basicConfig(filename="{}".format(filename_log), level=logging.INFO, force=True)
        self.log_message('Started logging...')
    
    def setup_datavault(self,params):

        self.dv.new(params['FILENAME']+'_sweep_{:03d}'.format(self.idx), params['DEPENDENTS'],
                ['Lockin X [A]', 'Lockin Y [A]', 'Input 2 [V]', 'Input 3 [V]', 'Temp A [K]', 'Temp B [K]', 
                'Emitter Current [A]', 'AB Current [A]'])
        
        self.dv.add_parameter('delay_mm_rng', (params['DELAY_RANGE_MM']))
        self.dv.add_parameter('delay_mm_pnts', params['DELAY_POINTS'])
        self.dv.add_parameter('delay_ps_rng', (params['DELAY_RANGE_PS']))
        self.dv.add_parameter('delay_ps_pnts',  params['DELAY_POINTS'])
        self.dv.add_parameter('live_plots', (('delay_ps', 'Lockin X'), 
                                        ('delay_ps', 'Lockin Y'), 
                                        ('delay_ps', 'Emitter Current'), 
                                        ('delay_ps', 'AB Current')))
        
        # Save all parameters
        config_filename = params['ROOTDIR']+"\\"+params['DATADIR']+".dir\\"+self.dv.get_name()+".yml"
        with open(config_filename, 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
            self.log_message("Config file written.")

    def voltage_ramp_smu(self,smu,v_initial,v_final):
        
        self.log_message("Starting SMU voltage ramp...")
        if v_initial == v_final:
            return
        step = 10
        v_steps = np.linspace(v_initial, v_final, step)
        for i in range(step):
            smu.set_v_meas_i(v_steps[i])
            time.sleep(0.1)
        time.sleep(1)
        self.log_message("Voltage SMU ramp ended.")

    def init_stage_position(self,params):
        # Move stage to starting position
        self.log_message("Moving stage to starting position...")
        self.ds.gpib_write("1VA20.0")
        self.ds.move_absolute(1,params['DELAY_RANGE_MM'][0])
        time.sleep(5)
        
        for i in range(10):
            try:
                current_pos = self.ds.tell_position(1)
                if abs(current_pos-params['DELAY_RANGE_MM'][0])<0.001:
                    self.log_message("Stage in position.")
                    return
            except:
                self.log_message("Error in reading stage position.")
            
            # wait and try reading the position again
            time.sleep(1)
        
        self.log_message("Moving stage timeout.")        
        
    def save_to_datavault(self,dmm,dps,in1,in2):
        # Read temperature                
        self.tempD4  = self.tempServer.tempD4
        self.tempD5  = self.tempServer.tempD5
        
        data = np.array([dmm,dps,in1,in2,0,0,self.tempD4,self.tempD5,self.I_e,self.I_a])
        self.dv.add(data)
        
    def scan_mirror(self,k=1):
        # Instatiate mirror scan object
        scan_e = MirrorScan.Scan('E',self.smu_e,self.dac_e,0,1,False)
        scan_a = MirrorScan.Scan('A',self.smu_a,self.dac_a,2,3,False)
        
        # Scan Ramge
        scan_e.set_scan_range(self.e_max_x,self.e_max_y, self.range/k,self.step/k)
        scan_a.set_scan_range(self.a_max_x,self.a_max_y, self.range/k,self.step/k)
        
        # Scan the Mirror
        scan_e.start()
        scan_a.start()
        
        # Wait to finish scan
        scan_e.join()
        scan_a.join()
        
        self.e_max_x = scan_e.max_x
        self.e_max_y = scan_e.max_y
        
        self.a_max_x = scan_a.max_x
        self.a_max_y = scan_a.max_y
    
    def scan_transient(self,params,delay_mm,delay_ps):
        
        SMUServer_e = SMUServer.CurrentPoll(self.smu_e)
        SMUServer_a = SMUServer.CurrentPoll(self.smu_a)
        
        _ = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                           params['DAC_INPUT_CH'],
                                           [0.0],
                                           [0.0],
                                           10,
                                           params['SAMPLING']*params['TIME_CONST']*1e6,
                                           params['AVGS']))
                
        for i in range(params['DELAY_POINTS']):
                percent = (delay_mm[i]-np.min(delay_mm))/(np.max(delay_mm)-np.min(delay_mm))
                if i%10==0:
                    self.log_message("Stepping stage to {:.2f} mm; {:.3f} ps; {:.2f} %"
                          .format(delay_mm[i], delay_ps[i], 100*percent))

                # Move stage
                try:
                    self.ds.move_absolute(1,delay_mm[i])
                except:
                    self.log_message("Stage movement error.")
                
                # Buffer ramp DAC
                br_data = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                                   params['DAC_INPUT_CH'],
                                                   [0.0],
                                                   [0.0],
                                                   1,
                                                   params['SAMPLING']*params['TIME_CONST']*1e6,
                                                   params['AVGS']))

                in1 = br_data[0][0]*params['SENS']/10.0/params['GAIN']
                in2 = br_data[1][0]*params['SENS']/10.0/params['GAIN']
                self.I_e = SMUServer_e.I
                self.I_a = SMUServer_a.I
                self.save_to_datavault(delay_mm[i], delay_ps[i], in1, in2)
        
        SMUServer_e.stop_thread = True
        SMUServer_a.stop_thread = True
            

def main():
    
    # Define parameters
    params = dict()

    params['ROOTDIR'] = r"C:\Users\Marconi\Young Lab Dropbox\Young Group\THz\Raw Data"
    params['DATADIR'] = "2022_09_07_TL2715_AKNDB010_5E"  
    params['FILENAME'] = "refl_transient_T_303K_ms"

    params['DEPENDENTS'] = ['delay_mm', 'delay_ps']
    params['INDEPENDENTS'] = ['Lockin X [A]', 'Lockin Y [A]', 'Input 2 [V]', 
                              'Input 3 [V]', 'Temp A [K]', 'Temp B [K]', 
                              'Emitter Current [A]', 'AB Current [A]']

    params['DELAY_RANGE_MM'] = [29,38]  # mm refl full: [29,37] | sample: [32,36] trans: [32,36]
    params['DELAY_POINTS'] = 100*(params['DELAY_RANGE_MM'][1]-params['DELAY_RANGE_MM'][0])+1        # normal = 100 pts/mm

    params['TIME_CONST'] = 0.01         # s; Lockin
    params['SENS'] = 0.05               # s; Lockin | full: 0.05 | sample: 0.005

    params['SWEEPS'] = 3                # Sweeps
    params['AVGS'] = 200                # Averages
    params['SAMPLING'] = 0.1            # DAC buffered ramp oversample

    params['GAIN'] = 1e8                # TIA gain
    params['BIAS_E'] = 10               # V on emitter
    params['BIAS_AB'] = 0.0             # V on AB line
    params['SMU_RANGE'] = 1e-6         # current range on K2450
    params['MAX_I_E'] = 0               # current on E
    params['MAX_I_AB'] = 0              # current on AB

    params['NPLC'] = 1                  # 0.1 -- 10
    params['FILT'] = "OFF"              # 'ON' or 'OFF'
    params['FILT_TYPE'] = "REPeat"      # 'REPeat' or 'MOVing'
    params['FILT_COUNT'] = 1            # 1 -- 100

    # Mirror
    params['EY_CENTER'] = -0.3534       #DAC1
    params['EX_CENTER'] = 4.8381         #DAC0
    params['AY_CENTER'] = 0.5542        #DAC3
    params['AX_CENTER'] = 1.5308        #DAC2 
    params['RANGE'] = 0.2
    params['STEP'] = 0.01

    # DAC
    params['DAC_DATA'] = "DAC-ADC_AD7734-AD5791 (COM5)"         # DAC for signal
    params['DAC_MIRROR'] = "DAC-ADC_AD7734-AD5791_4x4 (COM3)"   # DAC for mirrors
    params['DAC_CONV_TIME'] = 500       # ADC conversion time in us (82 -- 2686)
    params['DAC_OUTPUT_CH'] = 2         # Output channel number
    params['DAC_OUTPUT_CH_DUMMY'] = [0] # Dummy output channel number
    params['DAC_INPUT_CH'] = [0,1]  # Input channel number
    params['DAC_CH_Y'] = 1
    params['DAC_CH_X'] = 0

    params['DELAY_MEAS'] = 0.1            # Some delay
    
    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # Initialize labrad and servers
    cxn_m = labrad.connect()
    
    # DAC - lockin
    dac = cxn.dac_adc
    dac.select_device(params['DAC_DATA'])
    dac.initialize()
    for i in range(4):
        dac.set_conversiontime(i,params['DAC_CONV_TIME'])
    
    
    # DAC - Mirror
    dac_m = cxn_m.dac_adc
    dac_m.select_device(params['DAC_MIRROR'])
    for i in range(4):
        dac_m.set_conversiontime(i,params['DAC_CONV_TIME'])
    
    # K2400
    smu2400 = cxn.k2400()
    smu2400.select_device()
    smu2400.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2400.gpib_write(":ROUTe:TERMinals FRONt")
    
    # smu2400_current = SMUServer.CurrentPoll(smu2400)
    
    # K2450
    smu2450 = cxn.k2450()
    smu2450.select_device()
    smu2450.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2450.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2450.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2450.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    smu2450.set_i_read_range(params['SMU_RANGE'])
    
    # smu2450_current = SMUServer.CurrentPoll(smu2450)
    
    # LakeShore 350
    ls350 = cxn.lakeshore_350()
    ls350.select_device()
    
    tempServer = TemperatureServer.TemperaturePoll(ls350)
    
    # Delay stage
    ds = cxn.esp300()
    ds.select_device()
    ds.gpib_write("1VA20.0")
    
    # Lockin
    lck = cxn.sr860()
    lck.select_device()
    
    lck.time_constant(params['TIME_CONST'])
    lck.sensitivity(params['SENS'])
    
    # Calculate delay
    delay_mm = np.linspace(params['DELAY_RANGE_MM'][0], params['DELAY_RANGE_MM'][1], params['DELAY_POINTS'])
    conv_range = [(params['DELAY_RANGE_MM'][0]/0.299792458)*2, (params['DELAY_RANGE_MM'][1]/0.299792458)*2]
    delay_ps = np.linspace(conv_range[0], conv_range[1], params['DELAY_POINTS'])
    params['DELAY_RANGE_PS'] = [delay_ps.tolist()[0], delay_ps.tolist()[-1]]
    
    # DataVault
    dv = cxn.data_vault
    dv.cd(params['DATADIR']) # Change to data directory
    
    scanTransient = Transient(dv, ds, dac, dac_m, dac_m, smu2400, smu2450, tempServer)
    
    # Set mirror scan parameters
    scanTransient.set_scan_params(params)
    
    # Setup log file
    scanTransient.setup_logfile(params)
    
    # Measurement start
    start = time.time()
    scanTransient.log_message('Measurement Started')
    
    # Voltage ramp up on SMUs
    scanTransient.voltage_ramp_smu(smu2400, smu2400.read_v(), params['BIAS_E'])
    scanTransient.voltage_ramp_smu(smu2450, smu2450.read_v(), params['BIAS_E'])
    
    # Rough Scans
    scanTransient.log_message('Coarse mirror Scan E and A...')
    scanTransient.scan_mirror()
    
    try:
        for i in range(params['SWEEPS']):
            # Sweep
            scanTransient.log_message("Starting sweep #{:03d}...".format(i))
            
            # Initialize the stage to starting position
            scanTransient.init_stage_position(params)
            
            # Check current on E and A
            scanTransient.log_message("Checking for max current on E switch...")
            scanTransient.voltage_ramp_smu(smu2450, smu2450.read_v(), params['BIAS_E'])
            scanTransient.scan_mirror(k=20)
            scanTransient.voltage_ramp_smu(smu2450, smu2450.read_v(), 0)
            
            # Setup DataVault file and Config file
            scanTransient.log_message("Setting up datavault and config file...")
            scanTransient.idx = i
            scanTransient.setup_datavault(params)
            
            # Transient
            scanTransient.log_message("Starting transient measurement...")
            scanTransient.scan_transient(params, delay_mm, delay_ps)
            
    except KeyboardInterrupt:
        scanTransient.log_message("Measurement interrupted.")
        
    # Kill Temperature server
    tempServer.stop_thread = True
    
    # Kill SMU server
    # SMUServer_e.stop_thread = True
    # SMUServer_a.stop_thread = True
    
    # Initialize the stage to starting position
    scanTransient.init_stage_position(params)
        
    # Voltage ramp down
    scanTransient.voltage_ramp_smu(smu2400, smu2400.read_v(), 0)
    scanTransient.voltage_ramp_smu(smu2450, smu2450.read_v(), 0)
    
    # Measurements end
    scanTransient.log_message("Measurement Ended.")
    end = time.time()
    scanTransient.log_message("Total time: {:2f} s".format(end-start))
    
    return

if __name__ == '__main__':
    main()

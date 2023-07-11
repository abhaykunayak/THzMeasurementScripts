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
    
    def __init__(self,dv,ds,dac,dac_e,dac_a,smu_e,smu_a,ls350,tempServer):
        self.idx = 0
        self.dv = dv
        self.ds = ds
        self.dac = dac
        self.dac_e = dac_e
        self.dac_a = dac_a
        self.smu_e = smu_e
        self.smu_a = smu_a
        self.ls = ls350
        self.tempServer = tempServer
        self.SMUServer_e = 0
        self.I_e = 0
        self.I_a = 0
        self.v_gate = 0
        
        
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
        
    def wait_sec(self,t):
        for i in range(t):
            time.sleep(1)
            
    def setup_logfile(self,params):

        print('[{}] Setting up datafile and logging...'.format(self.current_time())) 
        
        # Change the working directory
        fullpath = os.path.join(params['ROOTDIR'],params['DATADIR']+'.dir')
        os.chdir(fullpath)
        
        # Setup logging
        filename_log = "{}_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"),params['FILENAME'])
        logging.basicConfig(filename="{}".format(filename_log), level=logging.INFO, force=True)
        self.log_message('Started logging...')
        params['LOGFILE'] = filename_log
    
    def save_config(self,params):
        # Save all parameters
        config_filename = params['ROOTDIR']+"\\"+params['DATADIR']+".dir\\"+self.dv.get_name()+".yml"
        with open(config_filename, 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
            self.log_message("Config file written.")
            
    def get_DAC_time(self,params):
        
        br_start = time.time()
        # Buffer ramp DAC
        _ = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                            params['DAC_INPUT_CH'],
                                            [0.0],
                                            [0.0],
                                            params['FPOINTS'],
                                            params['SAMPLING']*params['TIME_CONST']*1e6,
                                            params["READINGS"]),dtype=np.float64)   
        
        br_end = time.time()
        total_time = br_end-br_start
        self.log_message("Finished reading buffer in {:.6f}s.".format(total_time))
        
        
    def setup_datavault(self,params):
        
        self.tempD4  = self.tempServer.tempD4
        
        self.dv.new(params['FILENAME']+'_T_{:.2f}'.format(self.tempD4)
                    +'_Vg_{:03d}mV'.format(int(self.v_gate*1e3))+'_sw_{:02d}'.format(self.idx), 
                    params['DEPENDENTS'], params['INDEPENDENTS'])
        
        self.dv.add_parameter('delay_mm_rng', (params['DELAY_RANGE_MM']))
        self.dv.add_parameter('delay_mm_pnts', params['DELAY_POINTS'])
        self.dv.add_parameter('delay_ps_rng', (params['DELAY_RANGE_MM']))
        self.dv.add_parameter('delay_ps_pnts',  params['DELAY_POINTS'])
        self.dv.add_parameter('live_plots', (('delay_ps', 'Lockin X'), 
                                        ('delay_ps', 'Lockin Y'), 
                                        ('delay_ps', 'Input 2'), 
                                        ('delay_ps', 'Input 3')))
    
    def setup_delay(self,delay_start,delay_end,pts):
        # Calculate delay
        self.delay_mm = np.linspace(delay_start,delay_end,pts)
        self.delay_ps = np.linspace((delay_start/0.299792485)*2,
                                    (delay_end/0.299792458)*2,pts)
        
    def voltage_ramp_smu(self,smu,v_initial,v_final):
        
        self.log_message("Starting SMU voltage ramp...")
        if v_initial == v_final:
            return
        step = 10
        smu.output_on()
        v_steps = np.linspace(v_initial, v_final, step)
        for i in range(step):
            smu.set_volts(v_steps[i],1e-6)
            time.sleep(0.1)
        time.sleep(1)
        self.log_message("Voltage SMU ramp ended.")
        
    def voltage_ramp_dac(self,dac,ch,v_initial,v_final):
        
        self.log_message("Starting DAC voltage ramp on CH {}...".format(ch))
        if v_initial == v_final:
            return
        step = 10
        v_steps = np.linspace(v_initial, v_final, step)
        for i in range(step):
            dac.set_voltage(ch,v_steps[i])
            time.sleep(0.1)
        time.sleep(1)
        self.log_message("DAC CH {} Voltage ramp ended.".format(ch))
    
    def init_delay_stage(self,params):
        # Delay stage settings
        self.log_message("Setting up the delay stage...")
        self.ds.gpib_write("1MO1")
        self.ds.gpib_write("1VA1.0")
        self.ds.gpib_write("1AC0.1")
        self.ds.gpib_write("1AG0.1")
        self.ds.gpib_write("1BM9,1")
        self.ds.gpib_write("1BN1")
        self.ds.gpib_write("BO 2H")
        
    def init_stage_position(self,params):
        # Move stage to starting position
        self.log_message("Moving stage to starting position...")
        self.ds.gpib_write("1VA1.0")
        self.ds.gpib_write("1PA{:.6f}".format(params['DELAY_RANGE_MM'][0]))
        self.ds.gpib_write("1WS1000")
        
        self.wait_sec(30)
        
        self.log_message("Moving stage timeout.")        
        
    def save_to_datavault(self,dmm,dps,in1,in2,in3,in4,delay_pos):
        # Read temperature                
        self.tempD4  = self.tempServer.tempD4
        self.tempD5  = self.tempServer.tempD5
        
        data = np.array([dmm,dps,in1,in2,in3,in4,self.tempD4,self.tempD5,
                         self.I_e,self.I_a,delay_pos])
        self.dv.add(data)
        
    def scan_mirror(self,params,k=1):
        # Ramp bias voltage
        self.voltage_ramp_smu(self.smu_e, self.smu_a.read_v(), params['BIAS_E'])
        self.voltage_ramp_smu(self.smu_a, self.smu_a.read_v(), params['BIAS_E'])
        
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
        
        # Ramp down bias on A
        self.voltage_ramp_smu(self.smu_a, self.smu_a.read_v(), 0)
        
        # Save max data
        self.e_max_I = scan_e.max_I
        self.I_e = self.e_max_I
        self.e_max_x = scan_e.max_x
        self.e_max_y = scan_e.max_y
        
        self.a_max_I = scan_a.max_I
        self.I_a = self.a_max_I
        self.a_max_x = scan_a.max_x
        self.a_max_y = scan_a.max_y
    
    def scan_transient(self,params):
        
        self.SMUServer_e = SMUServer.CurrentPoll(self.smu_e)
        # self.SMUServer_a = SMUServer.CurrentPoll(self.smu_a)
        
        for i in range(params['DELAY_POINTS']):
                percent = (self.delay_mm[i]-np.min(self.delay_mm))/(np.max(self.delay_mm)-np.min(self.delay_mm))
                if i%10==0:
                    self.log_message("Stepping stage to {:.2f} mm; {:.3f} ps; {:.2f} %"
                          .format(self.delay_mm[i], self.delay_ps[i], 100*percent))

                # Move stage
                try:
                    self.ds.gpib_write("1PA{:.6f}".format(self.delay_mm[i]))
                    self.ds.gpib_write("1WS")
                    time.sleep(params['TIME_CONST']*5)
                    # delay_pos = float(self.ds.gpib_query("1TP?"))
                except:
                    self.log_message("Stage movement error.")
                
                self.I_e = self.SMUServer_e.I
                # self.I_a = self.SMUServer_a.I    
                
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
                in3 = br_data[2][0]*params['SENS']/10.0/params['GAIN']
                in4 = br_data[3][0]*params['SENS']/10.0/params['GAIN']
                
                self.save_to_datavault(self.delay_mm[i], self.delay_ps[i], 
                                       in1, in2, in3, in4, delay_pos=0)
        
        self.SMUServer_e.stop_thread = True
        # SMUServer_a.stop_thread = True
    
    def scan_transient_fast(self,params):
        
        # Start moving stage
        self.log_message("Moving to final position...")
        self.ds.gpib_write("1VA{:.6f}".format(params['STAGE_VEL']))
        self.ds.gpib_write("WT1000")
        self.ds.gpib_write("1PA{:.6f}".format(params['DELAY_RANGE_MM'][1]))
        self.ds.gpib_write("1WS1000")
        
        br_start = time.time()
        # Buffer ramp DAC
        br_data = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                            params['DAC_INPUT_CH'],
                                            [0.0],
                                            [0.0],
                                            params['FPOINTS'],
                                            params['SAMPLING']*params['TIME_CONST']*1e6,
                                            params["READINGS"]),dtype=np.float64)   
        
        br_end = time.time()
        self.log_message("Finished reading buffer in {:.6f}s.".format(br_end-br_start))
        
        # Save data tp Data Vault
        br_data[0:2] = br_data[0:2]*params['SENS']/10.0/params['GAIN']
        
        dv_data = np.concatenate(([self.delay_mm], [self.delay_ps],
                                  br_data,
                                  np.ones((1,params['FPOINTS']))*self.tempServer.tempD4,
                                  np.ones((1,params['FPOINTS']))*self.tempServer.tempD5,
                                  np.ones((1,params['FPOINTS']))*self.I_e,
                                  np.ones((1,params['FPOINTS']))*self.I_a),axis=0).T
        
        self.dv.add(dv_data)
            
        # Start moving stage
        self.log_message("Moving to start position...")
        self.ds.gpib_write("1VA{:.6f}".format(1))
        self.ds.move_absolute(1,params['DELAY_RANGE_MM'][0])
        
        # Sleep
        time.sleep(10)

    def scan_transient_sweep(self,params):
        try:
            for i in range(params['SWEEPS']):

                # Sweep
                self.log_message("Starting sweep #{:03d}...".format(i))
                
                # Initialize the delay stage
                self.init_delay_stage(params)
                
                # Initialize the stage to starting position
                self.init_stage_position(params)
                
                # Check current on E and A
                #self.log_message("Checking for max current on E switch...")
                #self.scan_mirror(params,k=15)
                
                # Setup DataVault file and Config file
                self.log_message("Setting up datavault and config file...")
                self.idx = i
                self.setup_datavault(params)
                self.save_config(params)
                
                # Transient
                self.log_message("Starting transient measurement...")
                _ = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                                   params['DAC_INPUT_CH'],
                                                   [0.0],
                                                   [0.0],
                                                   6,
                                                   params['SAMPLING']*params['TIME_CONST']*1e6,
                                                   params['AVGS']))
                # Clear DAC buffer
                self.dac.stop_ramp()

                if params['MEASURE_MODE'] == 'FAST':
                    self.scan_transient_fast(params)
                else:
                    self.scan_transient(params)
                
        except KeyboardInterrupt:
            self.log_message("Sweep measurement interrupted.")
        
    def scan_transient_temp_sweep(self,params):
        temp_range = np.linspace(params['T_INITIAL'],params['T_FINAL'],params['T_STEPS'])
        try:
            for t in range(params['T_STEPS']):
                self.log_message('Starting temperature sweep #: {} of {}'.format(t,params['T_STEPS']))
                self.tempD4 = self.tempServer.tempD4
                self.tempD5 = self.tempServer.tempD5
                
                # Set temperature
                self.log_message("Setting temperature T = {:.2f} K...".format(temp_range[t]))
                self.ls.set_p(1,temp_range[t])
                #self.log_message("Setting output #2 set point: {:.3f}.".format(temp_range[t]-3.0))
                #self.ls.set_p(2,temp_range[t]-3.0)

                time.sleep(60*2)
                
                # Rough Scans
                self.log_message('Mid mirror Scan E and A...')
                self.scan_mirror(params,k=5)
                
                # Transient
                self.scan_transient_sweep(params)
                
        except KeyboardInterrupt:
            self.log_message("Temp sweep measurement interrupted.")

    def scan_transient_gate_sweep(self,params):
        gate_range = np.linspace(params['V_GATE_I'],params['V_GATE_F'],params['V_GATE_STEPS'])
        try:
            for v in range(params['V_GATE_STEPS']):
                self.log_message('Starting Gate sweep #: {} of {}'.format(v,gate_range[v]))
                
                # Set Gate Voltage
                self.log_message("Setting Gate Voltage V = {:.2f} V...".format(gate_range[v]))
                self.v_gate = gate_range[v]
                self.voltage_ramp_dac(self.dac,params['DAC_OUTPUT_CH'],
                                      self.dac.read_voltage(2),
                                      gate_range[v])
                
                # Transient
                self.scan_transient_sweep(params)
                
        except KeyboardInterrupt:
            self.log_message("Gate sweep measurement interrupted.")                
            
def main():
    
    # Load config file
    CONFIG_FILENAME = 'ScanTransientDACConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # Initialize labrad and servers
    cxn_m = labrad.connect()
    
    # DAC - Mirror
    dac_m = cxn_m.dac_adc
    dac_m.select_device(params['DAC_MIRROR'])
    for i in range(4):
        dac_m.set_conversiontime(i,params['DAC_CONV_TIME'])
    
    # DAC - lockin
    dac = cxn.dac_adc
    dac.select_device(params['DAC_DATA'])
    dac.initialize()
    for i in range(4):
        dac.set_conversiontime(i,params['DAC_CONV_TIME'])
        
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
    
    # Start temperature server
    tempServer = TemperatureServer.TemperaturePoll(ls350)
    
    # Delay stage
    ds = cxn.esp300()
    ds.select_device()
    
    # Lockin
    lck = cxn.sr860()
    lck.select_device()
    
    lck.time_constant(params['TIME_CONST'])
    lck.sensitivity(params['SENS'])
      
    # DataVault
    dv = cxn.data_vault
    dv.cd(params['DATADIR']) # Change to data directory
    
    # Instantiate scan transient object
    scanTransient = Transient(dv, ds, dac, dac_m, dac_m, smu2400, smu2450, 
                              ls350, tempServer)
    
    # Set up delay
    if params['MEASURE_MODE'] == 'FAST':
        scanTransient.setup_delay(params['DELAY_RANGE_MM'][0], params['DELAY_RANGE_MM'][1], params['FPOINTS'])
    else:
        scanTransient.setup_delay(params['DELAY_RANGE_MM'][0], params['DELAY_RANGE_MM'][1], params['DELAY_POINTS'])
    
    # Initialize the delay stage
    scanTransient.init_delay_stage(params)
    
    # Set mirror scan parameters
    scanTransient.set_scan_params(params)
    
    # Setup log file
    scanTransient.setup_logfile(params)
    
    # Measurement start
    start = time.time()
    scanTransient.log_message('Measurement Started')
    
    # Voltage ramp up on E SMU
    scanTransient.voltage_ramp_smu(smu2400, smu2400.read_v(), params['BIAS_E'])
    scanTransient.voltage_ramp_dac(dac,0,0,10)
    try:
        # Rough Scans
        #scanTransient.log_message('Coarse mirror Scan E and A...')
        #scanTransient.scan_mirror(params)
        # Fine Scans
        #scanTransient.log_message('Fine mirror Scan E and A...')
        #scanTransient.scan_mirror(params,k=10)
        
        # Sweeps
        scanTransient.scan_transient_sweep(params)
        
        # Temperature sweeps - comment below to remove variable temperature scans
        #scanTransient.scan_transient_temp_sweep(params)
        
        # Gate Sweeps
        #scanTransient.scan_transient_gate_sweep(params)
            
        # Kill Temperature server
        tempServer.stop_thread = True
        
        try:
            # Kill SMU server
            scanTransient.SMUServer_e.stop_thread = True
            scanTransient.SMUServer_a.stop_thread = True
        except:
            scanTransient.log_message("SMU servers are dead already!.")
            
    except:
        scanTransient.log_message("Safe exit in process.")
    
    # Initialize the stage to starting position
    scanTransient.init_stage_position(params)
    
    # Gate Voltage ramp down
    scanTransient.voltage_ramp_dac(dac,params['DAC_OUTPUT_CH'],dac.read_voltage(2),0)
    
    # Voltage ramp down
    scanTransient.voltage_ramp_smu(smu2400, smu2400.read_v(), 0)
    scanTransient.voltage_ramp_smu(smu2450, smu2450.read_v(), 0)
    scanTransient.voltage_ramp_dac(dac,0,10,0)
    
    # Measurements end
    scanTransient.log_message("Measurement Ended.")
    end = time.time()
    scanTransient.log_message("Total time: {:2f} s".format(end-start))
    
    return

if __name__ == '__main__':
    main()

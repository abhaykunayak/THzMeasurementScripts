# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 18:50:53 2022

@author: AKN
"""

import time
import numpy as np
import scipy.io as sio
import labrad
import yaml
from datetime import datetime
import os
import glob
import logging
import TemperatureServer
import MirrorScanDAC

class Transient:
    '''
    description
    '''
    def __init__(self, params, dv, ds, dac, dac_m, ls350, tempServer):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        self.idx = 0
        self.rootdir = params['ROOTDIR']
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.dv = dv
        self.ds = ds
        self.dac = dac
        self.dac_m = dac_m
        self.ls = ls350
        self.tempServer = tempServer
        self.I_e = 0
        self.I_a = 0
        self.v_gate = 0
        
    def current_time(self):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        now = datetime.now()
        val = now.strftime("%H:%M:%S")
        return val
    
    def log_message(self, msg):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        msg_timestamp = '[{}] {}'.format(self.current_time(),msg)
        logging.info(msg_timestamp)
        print(msg_timestamp)
        
    def wait_sec(self,t):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        for i in range(t):
            time.sleep(1)
            
    def setup_logfile(self,params):
        '''
        Setup the logfile

        Parameters
        ----------
        params : params

        Returns
        -------
        None
        '''

        print('[{}] Setting up datafile and logging...'.format(self.current_time())) 
        
        # Change the working directory
        fullpath = os.path.join(params['ROOTDIR'],params['DATADIR']+'.dir')
        os.chdir(fullpath)
        files = glob.glob(fullpath+'\*.hdf5')
        if files == []:
            latest_file_num = 1
        latest_file = max(files, key=os.path.getctime)
        latest_file_num = int(os.path.basename(latest_file)[0:5]) + 1
        filename_log = '{:05d} - logfile.log'.format(latest_file_num)
        
        # Setup logging
        logging.basicConfig(filename="{}".format(filename_log), level=logging.INFO, force=True)
        self.log_message('Started logging into {}'.format(filename_log))
    
    def save_config(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        # Save all parameters
        config_filename = params['ROOTDIR']+"\\"+params['DATADIR']+".dir\\"+self.dv.get_name()+".yml"
        with open(config_filename, 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
            self.log_message("Config file written.")
            
    def get_DAC_time(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        br_start = time.time()
        # Buffer ramp DAC
        _ = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                            params['DAC_INPUT_CH'],
                                            [0.0],
                                            [0.0],
                                            params['FPOINTS'],
                                            params['SAMPLING']*params['LIA']['TIME_CONST']*1e6,
                                            params["READINGS"]),dtype=np.float64)   
        
        br_end = time.time()
        total_time = br_end-br_start
        self.log_message("Finished reading buffer in {:.6f}s.".format(total_time))
        
        
    def setup_datavault(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        self.tempD4  = self.tempServer.tempD4
        
        self.dv.new(params['FILENAME']+'_T_{:.2f}'.format(self.tempD4)
                    +'_Vg_{:03d}mV'.format(int(self.v_gate*1e3))+'_sw_{:02d}'.format(self.idx), 
                    params['DEPENDENTS'], params['INDEPENDENTS'])
        
        self.dv.add_parameter('delay_mm_rng', (params['DELAY_RANGE_MM']))
        self.dv.add_parameter('delay_mm_pnts', params['DELAY_POINTS'])
        self.dv.add_parameter('delay_ps_rng', (params['DELAY_RANGE_MM']))
        self.dv.add_parameter('delay_ps_pnts',  params['DELAY_POINTS'])
        self.dv.add_parameter('live_plots', (('delay_ps', 'Lockin X'), 
                                        ('delay_ps', 'Lockin Y')))
    
    def setup_delay(self,delay_start,delay_end,pts):
        '''
        description

        Parameters
        ----------

        Returns
        -------

        '''
        # Calculate delay
        self.delay_mm = np.linspace(delay_start,delay_end,pts)
        self.delay_ps = np.linspace((delay_start/0.299792485)*2,
                                    (delay_end/0.299792458)*2,pts)
        
    def voltage_ramp_dac(self, dac, ch: list, v_initial: list, v_final: list):
        '''
        Ramp the voltage of a DAC channel.

        Parameters
        ----------
        dac : larbad connection object

        ch : list
            channel number between 0 and 3

        v_initial : list
            initial voltage of the DAC channel

        v_final : list
            final voltage of the DAC channel

        Returns
        -------
        None

        '''
        self.log_message("Starting DAC voltage ramp on CH {}...".format(ch))
        if v_initial == v_final:
            return
        self.log_message("Ramping gate voltage from {} V to {} V ...".format(v_initial,v_final))
        _ = dac.buffer_ramp(ch, [0],
                        v_initial, v_final, 1000, 2000, 1)
        
        self.log_message("DAC CH {} Voltage ramp ended.".format(ch))
    
    def init_delay_stage(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
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
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        # Move stage to starting position
        self.log_message("Moving stage to starting position...")
        self.ds.gpib_write("1VA1.0")
        self.ds.gpib_write("1PA{:.6f}".format(params['DELAY_RANGE_MM'][0]))
        self.ds.gpib_write("1WS1000")
        time.sleep(10)

        # Check if stage in position
        stage_in_pos = False
        for i in range(10):
            time.sleep(1)
            stage_motion = self.dac.read_voltage(7)
            if stage_motion>3.0:
                self.log_message("Stage in position.")
                stage_in_pos = True
                break
            else:
                self.log_message("Waiting for stage position...")
                if i == 9:
                    self.log_message("Moving stage timeout.")

        if not stage_in_pos:
            self.log_message("Stage not in position.")      
        

    def save_to_datavault(self,dmm,dps,in1,in2,in3,in4,delay_pos):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        # Read temperature                
        self.tempD4  = self.tempServer.tempD4
        self.tempD5  = self.tempServer.tempD5
        
        data = np.array([dmm,dps,in1,in2,in3,in4,self.tempD4,self.tempD5,
                         self.I_e,self.I_a,delay_pos])
        self.dv.add(data)
    
    def save_to_mat(self,data,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        print('[{}] Saving to *.mat...'.format(self.current_time()))
        sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                    {'data': data, 'config': params})
                
    def scan_mirror(self,params,spot):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        # Instatiate mirror scan object
        scan = MirrorScanDAC.Scan(params, spot, self.dac, self.dac_m, False)
        print('setup complete')        
        # Coarse Scan Range
        scan.set_scan_range(params[spot]['X_CENTER'],params[spot]['Y_CENTER'], 
                        params[spot]['RANGE'],params[spot]['STEP'])

        # Scan the Mirror
        scan.start()
    
        # Wait to finish scan
        scan.join()

    def scan_transient(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        for i in range(params['DELAY_POINTS']):
                percent = (self.delay_mm[i]-np.min(self.delay_mm))/(np.max(self.delay_mm)-np.min(self.delay_mm))
                if i%10==0:
                    self.log_message("Stepping stage to {:.2f} mm; {:.3f} ps; {:.2f} %"
                          .format(self.delay_mm[i], self.delay_ps[i], 100*percent))

                # Move stage
                try:
                    self.ds.gpib_write("1PA{:.6f}".format(self.delay_mm[i]))
                    self.ds.gpib_write("1WS")
                    time.sleep(params['LIA']['TIME_CONST']*5)
                    # delay_pos = float(self.ds.gpib_query("1TP?"))
                except:
                    self.log_message("Stage movement error.")
                
                # Buffer ramp DAC
                br_data = np.array(self.dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                                   params['DAC_INPUT_CH'],
                                                   [0.0],
                                                   [0.0],
                                                   1,
                                                   params['SAMPLING']*params['LIA']['TIME_CONST']*1e6,
                                                   params['AVGS']))

                in1 = br_data[0][0]*params['LIA']['SENS']/10.0/params['GAIN']
                in2 = br_data[1][0]*params['LIA']['SENS']/10.0/params['GAIN']
                in3 = br_data[2][0]*params['LIA']['SENS']/10.0/params['GAIN']
                in4 = br_data[3][0]*params['LIA']['SENS']/10.0/params['GAIN']
                
                self.save_to_datavault(self.delay_mm[i], self.delay_ps[i], 
                                       in1, in2, in3, in4, delay_pos=0)
    
    def scan_transient_fast(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        stage_vel = (params['DELAY_RANGE_MM'][1]-params['DELAY_RANGE_MM'][0])/params['DAC_TIME']
        # Start moving stage
        self.log_message("Moving to final position...")
        self.ds.gpib_write("1VA{:.6f}".format(stage_vel))
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
                                            params['SAMPLING']*params['LIA']['TIME_CONST']*1e6,
                                            params["READINGS"]),dtype=np.float64)   
        
        br_end = time.time()
        self.log_message("Finished reading buffer in {:.6f}s.".format(br_end-br_start))
        
        # Save data tp Data Vault
        br_data[0:2] = br_data[0:2]*params['LIA']['SENS']/10.0/params['GAIN']
        br_data[2:4] = br_data[5:7]*params['LIA_2']['SENS']/10.0/params['IAC']
        
        dv_data = np.concatenate(([self.delay_mm], [self.delay_ps],
                                  br_data,
                                  np.ones((1,params['FPOINTS']))*self.tempServer.tempD4,
                                  np.ones((1,params['FPOINTS']))*self.tempServer.tempD5
                                  ),axis=0).T
        
        self.dv.add(dv_data)
        self.save_to_mat(dv_data,params)
            
        # Start moving stage
        self.log_message("Moving to start position...")
        self.ds.gpib_write("1VA{:.6f}".format(1))
        self.ds.move_absolute(1,params['DELAY_RANGE_MM'][0])
        
        # Sleep
        time.sleep(10)

    def scan_transient_sweep(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        try:
            for i in range(params['SWEEPS']):

                # Sweep
                self.log_message("Starting sweep #{:03d}...".format(i))
                
                if params['MIRROR_SCAN']:
                    # Check current on E and A
                    self.log_message("Checking for max current on E switch...")
                    self.log_message("Moving stage to {} mm.".format(params['STAGE_POS']))
                    self.ds.move_absolute(1,params['STAGE_POS'])
                    time.sleep(2)
                    for j in range(20):
                        curr_pos = self.dac.read_voltage(3)
                        if curr_pos>3.0:
                            self.log_message("Stage in position.")
                            break
                        else:
                            self.log_message("Waiting for stage position...")
                        time.sleep(2)

                    self.scan_mirror(params,'E')
                    self.scan_mirror(params,'A')

                # Initialize the delay stage
                self.init_delay_stage(params)
                
                # Initialize the stage to starting position
                self.init_stage_position(params)
                
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
                                                   params['SAMPLING']*params['LIA']['TIME_CONST']*1e6,
                                                   params['AVGS']))
                # Clear DAC buffer
                self.dac.stop_ramp()

                if params['MEASURE_MODE'] == 'FAST':
                    self.scan_transient_fast(params)
                else:
                    self.scan_transient(params)
                
        except KeyboardInterrupt:
            self.log_message("Sweep measurement interrupted.")

    def scan_transient_gate2D(self,params):
        '''
        description

        Parameters
        ----------

        Returns
        -------
        
        '''
        v_gate_bot_ch = params['V_GATE_BOT_CH']
        v_gate_bot_pnts = params['V_GATE_BOT_PNTS']
        v_gate_bot_rng = np.linspace(params['V_GATE_BOT_RNG'][0], params['V_GATE_BOT_RNG'][1], v_gate_bot_pnts)
        v_gate_top_ch = params['V_GATE_TOP_CH']
        v_gate_top_pnts = params['V_GATE_TOP_PNTS']
        v_gate_top_rng = np.linspace(params['V_GATE_TOP_RNG'][0], params['V_GATE_TOP_RNG'][1], v_gate_top_pnts)
        v_gate_total_pnts = v_gate_top_pnts * v_gate_bot_pnts
        # Gate voltage ramp up to starting values
        v_gate_bot_next = 0.0
        v_gate_bot_last = v_gate_bot_rng[0]
        v_gate_top_next = 0.0
        v_gate_top_last = v_gate_top_rng[0]
        
        self.log_message("Ramping up gate voltage to starting voltage: {} V ...".format([v_gate_bot_last, v_gate_top_last]))
        self.voltage_ramp_dac(self.dac, [v_gate_bot_ch, v_gate_top_ch], [v_gate_bot_next, v_gate_top_next], [v_gate_bot_last, v_gate_top_last])
        
        # Gate voltage ramp for measurements
        for i in np.arange(v_gate_top_pnts):
            # Top Gate Voltage
            v_gate_top_next = v_gate_top_rng[i]
            self.log_message("Gate sweep # {}/{}".format((i+i*v_gate_bot_pnts),v_gate_total_pnts))
            self.log_message("Ramping up top gate voltage to: {} V ...".format([v_gate_top_next]))
            self.voltage_ramp_dac(self.dac, [v_gate_top_ch], [v_gate_top_last], [v_gate_top_next])

            for j in np.arange(v_gate_bot_pnts):
                # Bottom Gate Voltage
                v_gate_bot_next = v_gate_bot_rng[j]
                self.log_message("Ramping up bottom gate voltage to: {} V ...".format([v_gate_bot_next]))
                self.voltage_ramp_dac(self.dac, [v_gate_bot_ch], [v_gate_bot_last], [v_gate_bot_next])
                
                self.v_gate = 0.0
                # self.scan_transient_sweep(params)

                v_gate_bot_last = v_gate_bot_next

            v_gate_top_last = v_gate_top_next
            
        # Gate voltage ramp down
        self.log_message("Ramping down gate voltages from {} V to 0 V...".format([v_gate_bot_last, v_gate_top_last]))
        self.voltage_ramp_dac(self.dac, [v_gate_bot_ch, v_gate_top_ch], [v_gate_bot_last, v_gate_top_last], [0.0, 0.0])
             
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
    
    # LakeShore 350
    ls350 = cxn.lakeshore_350()
    ls350.select_device()
    
    # Start temperature server
    tempServer = TemperatureServer.TemperaturePoll(ls350)
    
    # Delay stage
    ds = cxn.esp300()
    ds.select_device()
    
    # Lockin - THz
    lck1 = cxn.sr860()
    lck1.select_device(1)
    lck1.time_constant(params['LIA']['TIME_CONST'])
    lck1.sensitivity(params['LIA']['SENS'])
    
    # Lockin - Transport
    # lck2 = cxn.sr860()
    # lck2.select_device(0)    
    # lck2.time_constant(params['LIA_2']['TIME_CONST'])
    # lck2.sensitivity(params['LIA_2']['SENS'])
    # lck2.sine_out_amplitude(params['LIA_2']['AMPL'])
    # lck2.frequency(params['LIA_2']['FREQ'])

    # DataVault
    dv = cxn.data_vault
    dv.cd(params['DATADIR']) # Change to data directory
    
    # Instantiate scan transient object
    scanTransient = Transient(params,
                              dv, 
                              ds, 
                              dac,
                              dac_m, 
                              ls350, 
                              tempServer)
    
    # Set up delay
    if params['MEASURE_MODE'] == 'FAST':
        scanTransient.setup_delay(params['DELAY_RANGE_MM'][0], params['DELAY_RANGE_MM'][1], params['FPOINTS'])
    else:
        scanTransient.setup_delay(params['DELAY_RANGE_MM'][0], params['DELAY_RANGE_MM'][1], params['DELAY_POINTS'])
    
    # Initialize the delay stage
    scanTransient.init_delay_stage(params)
    
    # Setup log file
    scanTransient.setup_logfile(params)
    
    # Measurement start
    start = time.time()
    scanTransient.log_message('Measurement Started')
    
    # Voltage ramp up on E SMU
    scanTransient.voltage_ramp_dac(dac,[params['DAC_OUTPUT_CH']],[0],[params['BIAS_E']])

    try:
        # Sweeps
        # scanTransient.scan_transient_sweep(params)

        # Gate
        scanTransient.scan_transient_gate2D(params)

        # Kill Temperature serverp
        tempServer.stop_thread = True
            
    except Exception as error:
        scanTransient.log_message("Safe exit in process. {}".format(error))
    
    # Initialize the stage to starting position
    scanTransient.init_stage_position(params)
    
    # Voltage ramp down
    scanTransient.voltage_ramp_dac(dac,[params['DAC_OUTPUT_CH']],[params['BIAS_E']],[0])

    # Measurements end
    scanTransient.log_message("Measurement Ended.")
    end = time.time()
    scanTransient.log_message("Total time: {:2f} s".format(end-start))
    
    return

if __name__ == '__main__':
    main()

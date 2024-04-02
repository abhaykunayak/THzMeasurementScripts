import labrad
import numpy as np
import yaml
import time
import os
from datetime import datetime
import glob
import logging
import scipy.io as sio
from threading import Thread


class Transport(Thread):
    
    def __init__(self,
                 params,
                 dac,
                 ls):
        
        Thread.__init__(self)
        self.daemon = True
        self.params = params
        self.dac = dac
        self.ls = ls
        self.rootdir = params['ROOTDIR']
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.tempD4 = 0
        self.tempD5 = 0

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

    def setup_datavault(self, params, dv):
        
        dv.new(params['FILENAME'], 
               params['DEPENDENTS'],
               params['INDEPENDENTS'])
        
        dv.add_parameter('n0_rng', params['MEAS_PARAMS']['n0_RANGE'])
        dv.add_parameter('p0_rng', params['MEAS_PARAMS']['p0_RANGE'])
        dv.add_parameter('n0_pnts', params['MEAS_PARAMS']['n0_PNTS'])
        dv.add_parameter('p0_pnts', params['MEAS_PARAMS']['p0_PNTS'])

        dv.add_parameter('live_plots', (('n0', 'Lockin Transport X'), 
                                        ('n0', 'Lockin THz X'), 
                                        ('n0', 'p0', 'Lockin Transport X'), 
                                        ('n0', 'p0', 'Lockin THz X')))
        
        self.log_message('Data vault file: {}'.format(dv.get_name()))

        self.dv = dv

    def save_to_mat(self, params):
        '''
        saves data in matlab format

        Parameters
        ----------
        params : YAML parameter file

        Returns
        -------
        None

        '''  
        # get data from data vault file
        data = self.dv.get()

        sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                    {'data':data, 'config': params})
    
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
    
    def find_vt_vb_n0p0(self, p0, n0, c_delta):
        '''
        Converts (n0,p0) to (vb,vt) 

        Parameters
        ----------
        p0 : float
        n0 : float
        c_delta : float
            asymmetry of the top and bottom gates.

        Returns
        -------
        vt : float
        vb : float

        '''
        return (0.5 * (n0 + p0) / (1.0 + c_delta)), 0.5 * (n0 - p0) / (1.0 - c_delta)
    
    def mesh_n0p0(self, p0_range, n0_range, delta, pxsize):
        """
        mesh function to convert n0,p0 to vb,vt

        Parameters
        ----------
        p0_range : (float, float)
        n0_range : (float, float)
        delta: float
        pxsize : (int, int)

        Returns
        -------
        (v_fast, v_slow), (p0, n0) : ([float],[float]), ([float],[float])

        """
        
        p0 = np.linspace(p0_range[0], p0_range[1], pxsize[1])
        n0 = np.linspace(n0_range[0], n0_range[1], pxsize[0])
        
        n0, p0 = np.meshgrid(n0, p0)  # p0 - slow; n0 - fast
        
        v_fast, v_slow = self.find_vt_vb_n0p0(p0, n0, delta)

        return np.dstack((v_fast, v_slow)), np.dstack((p0, n0))

    
    def scan_gate(self,params):
        meas_params = params['MEAS_PARAMS']

        pxsize = (meas_params['n0_PNTS'], meas_params['p0_PNTS'])
        extent = (meas_params['n0_RANGE'][0], meas_params['n0_RANGE'][1], 
                meas_params['p0_RANGE'][0], meas_params['p0_RANGE'][1])
        num_n0 = pxsize[0]
        num_p0 = pxsize[1]

        mesh_vtvb, mesh_p0n0 = self.mesh_n0p0(p0_range = (extent[2], extent[3]),
                n0_range = (extent[0], extent[1]), delta = meas_params['DELTA'],
                pxsize = pxsize)

        self.log_message("Starting transport measurement...")
        start_time = time.time()

        for i in np.arange(num_p0):
            # Measure temperature
            try:
                self.tempD4 = float(self.ls.read_temp('D4'))
                self.tempD5 = float(self.ls.read_temp('D5'))
            except:
                self.log_message("Error in reading temperature")

            # Empty variable for buffer ramp data
            br_data_full = np.zeros((np.size(params['DAC_IN_CH']),num_n0),dtype=float)
            
            # Extract real voltages for vt and vb
            vec_vt = mesh_vtvb[i, :][:, 0]
            vec_vb = mesh_vtvb[i, :][:, 1]

            mesh_p0 = mesh_p0n0[i, :][:, 0]
            mesh_n0 = mesh_p0n0[i, :][:, 1]

            # Check voltage limits
            mask = np.logical_and(np.logical_and(vec_vt <= meas_params['MAX_vt'], 
                                            vec_vt >= meas_params['MIN_vt']),
                              np.logical_and(vec_vb <= meas_params['MAX_vb'], 
                                            vec_vt >= meas_params['MIN_vb']))
        
            if np.any(mask == True):
                start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]
                num_points = stop - start + 1

                if i == 0:
                    self.log_message('Ramping up to starting gate voltages...')
                    self.voltage_ramp_dac(self.dac, params['DAC_OUT_CH'], 
                              [0.0, 0.0], [vec_vt[start], vec_vb[start]])
                else:
                    self.log_message('Ramping gate voltages for next line...')
                    self.voltage_ramp_dac(self.dac, params['DAC_OUT_CH'], 
                              [last_vt, last_vb], [vec_vt[start], vec_vb[start]])
                    
                # measurement
                time.sleep(params['LIA_R']['TIME_CONST']*3.0)
                br_data = np.array(self.dac.buffer_ramp(params['DAC_OUT_CH'],
                                                params['DAC_IN_CH'],
                                                [vec_vt[start], vec_vb[start]],
                                                [vec_vt[stop], vec_vb[stop]],
                                                num_points,
                                                params['SAMPLING']*params['LIA_R']['TIME_CONST']*1e6,
                                                params['AVGS']))

                # Lock-in data
                # THz data
                br_data[0:2] = br_data[0:2]*params['LIA_THZ']['SENS']/10.0/params['GAIN']
                # Transport data
                br_data[2:4] = br_data[2:4]*params['LIA_R']['SENS']/10.0/params['IAC']

                br_data_full[:,start:stop + 1] = br_data
                n0_idx = np.ones(num_n0) * i
                p0_idx = np.linspace(0, num_n0 - 1, num_n0)
                tempD4 = np.ones(num_n0) * self.tempD4
                tempD5 = np.ones(num_n0) * self.tempD5
                
                # Format data for data vault
                data = np.concatenate((
                            [n0_idx], [p0_idx],
                            [mesh_n0], [mesh_p0],
                            [vec_vb], [vec_vt],
                            br_data_full,
                            [tempD4], [tempD5]
                            ),axis=0).T

                self.dv.add(data)

                last_vt = vec_vt[stop]
                last_vb = vec_vb[stop]

        # Ramp down all gate voltages
        self.log_message('Ramping down gate voltages...')
        self.voltage_ramp_dac(self.dac, params['DAC_OUT_CH'], 
                              [last_vt, last_vb], [0.0, 0.0])
        
        end_time = time.time()
        self.log_message("Finished all sweeps in {:.2f}s.".format(end_time-start_time))

        return

    def IV(self,params):
        '''
        Description of the function
        '''
        self.log_message('MCBC')

def main():
    # Load config file
    CONFIG_FILENAME = 'TransportConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # SR 860 Lockin THz
    lck_thz = cxn.sr860()
    lck_thz.select_device(1)
    lck_thz.time_constant(params['LIA_THZ']['TIME_CONST'])
    lck_thz.sensitivity(params['LIA_THZ']['SENS'])

    # SR 860 Lockin Transport 
    lck_r = cxn.sr860()
    lck_r.select_device(0)
    lck_r.sensitivity(params['LIA_R']['SENS'])
    lck_r.time_constant(params['LIA_R']['TIME_CONST'])
    lck_r.sine_out_amplitude(params['LIA_R']['AMPL'])
    lck_r.frequency(params['LIA_R']['FREQ'])

    # DAC-ADC for data
    dac = cxn.dac_adc
    dac.select_device(params['DAC_DATA'])
    dac.initialize()
    for i in range(4):
        dac.set_conversiontime(i,params['DAC_CONV_TIME'])

    # LS 330
    ls350 = cxn.lakeshore_350()
    ls350.select_device()

    # Data vault
    dv = cxn.data_vault()
    
    # Change to data directory
    dv.cd(params['DATADIR'])
    
    # Transport class instance
    rt = Transport(params, dac, ls350)
    
    # Setup log file
    rt.setup_logfile(params)        

    # DataVault File
    rt.setup_datavault(params, dv)

    # Measure
    rt.scan_gate(params)
    
    # Save config file
    rt.save_config(params)

    # Save data to matlab
    rt.save_to_mat(params)

if __name__ == '__main__':
    main()
    
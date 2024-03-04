import labrad
import numpy as np
import yaml
import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
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
        self.rootdir = params['ROOTDIR'][2:-1]
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.tempD4 = 0
        self.tempD5 = 0
            
    def setup_datavault(self,params, dv):
        
        dv.new(params['FILENAME'], 
               params['DEPENDENTS'],
               params['INDEPENDENTS'])
        self.dv = dv
        
    def save_to_datavault(self,data):
        self.dv.add(data)

    def save_to_mat(self,data):
        sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                    {'data':data})
    
    def voltage_ramp_dac(self,ch,v_initial,v_final):
        print("Starting DAC voltage ramp on CH {}...".format(ch))
        if v_initial == v_final:
            return
        step = 50
        v_steps = np.linspace(v_initial, v_final, step)
        for i in range(step):
            self.dac.set_voltage(ch,v_steps[i])
            time.sleep(0.2)
        time.sleep(1)
        print("DAC CH {} Voltage ramp ended.".format(ch))
    
    def scan_gate(self,params):
        v_rng = np.linspace(params['V_GATE_I'],params['V_GATE_F'],
                    params['V_GATE_STEPS']+1)
        swp = np.arange(params['SWEEPS'])

        print("Starting transport measurement...")
        start_time = time.time()
        
        for i in swp:
            print("Starting sweep: {} out of {}...".format(i+1,params['SWEEPS']))
            
            # Measure temperature
            try:
                self.tempD4 = float(self.ls.read_temp('D4'))
                self.tempD5 = float(self.ls.read_temp('D5'))
            except:
                print("Error in reading temperature")
            
            # Gate voltage ramp
            if i==0:
                self.voltage_ramp_dac(params['V_GATE_CH'],0,params['VIMD'])
                self.voltage_ramp_dac(params['V_GATE_CH'],params['VIMD'],v_rng[0])
            else:
                self.voltage_ramp_dac(params['V_GATE_CH'],v_rng[-1],params['VIMD'])
                self.voltage_ramp_dac(params['V_GATE_CH'],params['VIMD'],v_rng[0])

            # Buffer Ramp
            br_data = np.array(self.dac.buffer_ramp([params['V_GATE_CH']],
                                            params['DAC_IN_CH'],
                                            [params['V_GATE_I']],
                                            [params['V_GATE_F']],
                                            params['V_GATE_STEPS']+1,
                                            params['SAMPLING']*params['LIA']['TIME_CONST']*1e6,
                                            params['AVGS']))

            # Lock-in data
            br_data[1:3] = br_data[1:3]*params['LIA']['SENS']/10.0/params['IAC']

            # Format data
            data = np.concatenate(([np.ones_like(v_rng)*i],[v_rng],
                                br_data,
                                [np.ones_like(v_rng)*self.tempD4],
                                [np.ones_like(v_rng)*self.tempD5]),axis=0).T
            if i>0:
                data_swp = np.vstack((data_swp,data))
            else:
                data_swp = data

        # Gate voltage ramp
        self.voltage_ramp_dac(params['V_GATE_CH'],v_rng[-1],0)
        
        end_time = time.time()
        print("Finished all sweeps in {:.2f}s.".format(end_time-start_time))

        return data_swp

    def IV(self,params):
        '''
        Description of the function
        '''
        print('MCBC')

def main():
    # Load config file
    CONFIG_FILENAME = 'TransportConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # SR 830 Lockin 
    sr830 = cxn.sr830()
    sr830.select_device()
    sr830.sensitivity(params['LIA']['SENS'])
    sr830.time_constant(params['LIA']['TIME_CONST'])
    sr830.sine_out_amplitude(params['LIA']['AMPL'])
    sr830.frequency(params['LIA']['FREQ'])

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
        
    rt = Transport(params, dac, ls350)
    
    # DataVault File
    rt.setup_datavault(params, dv)

    # Measure 
    data = rt.scan_gate(params)

    # Save data
    rt.save_to_datavault(data)
    rt.save_to_mat(data)

if __name__ == '__main__':
    main()
    
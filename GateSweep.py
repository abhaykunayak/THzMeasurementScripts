import labrad
import numpy as np
import yaml
import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from threading import Thread


class GateSweep(Thread):
    
    def __init__(self,
                 params,
                 dac,
                 ls,
                 ds):
        
        Thread.__init__(self)
        self.daemon = True
        self.params = params
        self.dac = dac
        self.ls = ls
        self.ds = ds
        self.rootdir = params['ROOTDIR'][2:-1]
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"
        self.delay_pos = 0
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
    
    def save_config(self,params):
        # Save all parameters
        config_filename = self.datapath+"\\"+self.dv.get_name()+".yml"
        with open(config_filename, 'w') as f:
            yaml.dump(params, f, sort_keys=False, default_flow_style=False)
            print("Config file written.")

    def voltage_ramp(self,ch,v_initial,v_final):
        print("Starting DAC voltage ramp on CH {}...".format(ch))
        if v_initial == v_final:
            return
        step = 20
        v_steps = np.linspace(v_initial, v_final, step)
        for i in range(step):
            self.dac.set_voltage(ch,v_steps[i])
            time.sleep(0.2)
        time.sleep(1)
        print("DAC CH {} Voltage ramp ended.".format(ch))
    
    def move_stage(self,pos):
        self.delay_pos = pos
        # move stage to position
        print("Moving stage to position: {} mm".format(self.delay_pos))
        self.ds.move_absolute(1,self.delay_pos)
        time.sleep(3)

    def sweep_gate(self,params):
        # delay range
        delay_rng = np.linspace(params['DELAY_I'],params['DELAY_F'],
                            params['DELAY_STEPS'])
        
        # gate range
        v_rng = np.linspace(params['V_GATE_I'],params['V_GATE_F'],
                            params['V_GATE_STEPS']+1)
        
        # Sweeps
        swp = np.arange(params['SWEEPS'])

        print("Starting measurement...")
        start_time = time.time()
        
        for j in range(params['DELAY_STEPS']):
            
            self.move_stage(delay_rng[j])

            for i in swp:
                print("Starting sweep: {} out of {}...".format(i+1,params['SWEEPS']))
                
                # Measure temperature
                self.tempD4 = float(self.ls.read_temp('D4'))
                self.tempD5 = float(self.ls.read_temp('D5'))
                
                # Ramp gate voltage
                self.voltage_ramp(params['V_GATE_CH'],0,v_rng[0])

                # Buffer Ramp
                br_data = np.array(self.dac.buffer_ramp([params['V_GATE_CH']],
                                                params['DAC_IN_CH'],
                                                [params['V_GATE_I']],
                                                [params['V_GATE_F']],
                                                params['V_GATE_STEPS']+1,
                                                params['SAMPLING']*params['LIA_THZ']['TIME_CONST']*1e6,
                                                params['AVGS']))
            
                # Ramp gate voltage
                self.voltage_ramp(params['V_GATE_CH'],v_rng[-1],0)

                # Format data
                data = np.concatenate(([np.ones_like(v_rng)*j],
                                    [np.ones_like(v_rng)*self.delay_pos],
                                    [np.ones_like(v_rng)*i],[v_rng],
                                    br_data[0:2][:]*params['LIA_THZ']['SENS']/10.0/params['GAIN'],
                                    br_data[2:4][:]*params['LIA_TRANSPORT']['SENS']/10.0,
                                    [np.ones_like(v_rng)*self.tempD4],
                                    [np.ones_like(v_rng)*self.tempD5]),axis=0).T
                if i>0 or j>0:
                    data_swp = np.vstack((data_swp,data))
                else:
                    data_swp = data

        end_time = time.time()
        print("Finished all sweeps in {:.2f}s.".format(end_time-start_time))

        return data_swp

def main():
    # Load config file
    CONFIG_FILENAME = 'GateSweepConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # SR 860
    sr860 = cxn.sr860()
    sr860.select_device()
    sr860.sensitivity(params['LIA_THZ']['SENS'])

    # SR 830
    sr830 = cxn.sr830()
    sr830.select_device()
    sr830.sensitivity(params['LIA_TRANSPORT']['SENS'])

    # DAC-ADC for data
    dac = cxn.dac_adc
    dac.select_device(params['DAC_DATA'])
    dac.initialize()
    for i in range(4):
        dac.set_conversiontime(i,params['DAC_CONV_TIME'])

    # LS 350
    ls350 = cxn.lakeshore_350()
    ls350.select_device()

    # Delay stage
    ds = cxn.esp300()
    ds.select_device()
    ds.move_absolute(1,params['DELAY_I'])
    
    # Data vault
    dv = cxn.data_vault()
    
    # Change to data directory
    dv.cd(params['DATADIR'])
    
    gs = GateSweep(params, dac, ls350, ds)
    
    # DataVault File
    gs.setup_datavault(params, dv)
    gs.save_config(params)

    # Switch bias 
    gs.voltage_ramp(0,0,10)

    # Measure 
    data = gs.sweep_gate(params)

    # Switch bias 
    gs.voltage_ramp(0,10,0)

    # Save data
    gs.save_to_datavault(data)
    gs.save_to_mat(data)

if __name__ == '__main__':
    main()
    
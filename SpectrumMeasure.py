import labrad
import numpy as np
import yaml
import time
import os
import scipy.io as sio
import matplotlib.pyplot as plt
from threading import Thread


class SpectrumMeasure(Thread):
    
    def __init__(self,
                 params,
                 sr770,
                 save_data=False):
        
        Thread.__init__(self)
        self.daemon = True
        self.params = params
        self.sr770 = sr770
        self.rootdir = params['ROOTDIR'][2:-1]
        self.datadir = params['DATADIR']
        self.datapath = self.rootdir+"\\"+self.datadir+".dir"

        self.save_data = save_data
            
    def set_datafile(self,dv):
        self.dv = dv
        
    def save_to_datavault(self,x,y):
        data = np.column_stack((x,y))
        self.dv.add(data)

    def save_to_mat(self,x,y):
        data = np.column_stack((x,y))
        sio.savemat(self.datapath+"\\"+self.dv.get_name()+".mat",
                    {'data':data})
                    
    def set_span(self,s):
        self.sr770.gpib_write('SPAN{:d}'.format(s))

    def set_center_freq(self,f):
        self.sr770.gpib_write('CRTF{:d}'.format(f))

    def set_meas(self,m):
        self.sr770.gpib_write('MEAS0,{:d}'.format(m))

    def set_averaging(self,v,navg=2):
        self.sr770.gpib_write('AVGO{:d}'.format(v))
        if v:
            self.sr770.gpib_write('NAVG{:d}'.format(navg))

    def get_data(self):
        self.sr770.start()
        time.sleep(0.1)
        y = self.sr770.gpib_query('SPEC?0').split()
        y = y[0].split(',')
        y = np.array([np.float64(y[i]) for i in range(400)])

        x0 = np.float64(self.sr770.gpib_query('BVAL?0,0').split()[0])
        x1 = np.float64(self.sr770.gpib_query('BVAL?0,399').split()[0])
        x = np.linspace(x0,x1,400)

        return x,y



def main():
    # Load config file
    CONFIG_FILENAME = 'SpectrumMeasureConfig.yml'

    with open(os.path.realpath(CONFIG_FILENAME),'r') as f:
        params = yaml.safe_load(f)

    # Initialize labrad and servers
    cxn = labrad.connect()
    
    # SR 770 Spectrum Analyzer
    sr770 = cxn.signal_analyzer_sr770()
    sr770.select_device()

    # Data vault
    dv = cxn.data_vault()
    
    # Change to data directory
    dv.cd(params['DATADIR'])
    
    # Create new data file
    dv.new(params['FILENAME'], ['Freq [Hz]'], ['Y [dbV]'])
        
    spec = SpectrumMeasure(params, 
                           sr770,
                           save_data=True)
    
    # DataVault File
    spec.set_datafile(dv)

    # Averaging
    #spec.sr770.gpib_write('*RST')
    #spec.set_meas(1)
    spec.set_span(10)
    spec.set_averaging(1,10)
    spec.set_center_freq(3150)

    # Measure 
    x,y = spec.get_data()

    # Save data
    spec.save_to_datavault(x,y)
    spec.save_to_mat(x,y)

if __name__ == '__main__':
    main()
    
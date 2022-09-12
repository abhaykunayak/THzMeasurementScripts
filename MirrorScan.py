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

class ScanE(Thread):
    
# Define parameters
params = dict()

params['SCAN_BEAM'] = 'A'           # E or A

params['ROOTDIR'] = r"C:\Users\Marconi\Young Lab Dropbox\Young Group\THz\Raw Data"                               
params['DATADIR'] = "2022_09_07_TL2715_AKNDB010_5E"                  
params['FILENAME'] = "a_T_303K"
    
params['Y_CENTER'] = -0.3534          #DAC1 refl: -0.3; trans: -0.35
params['Y_RANGE'] = 0.2
params['Y_STEP'] = 0.01

params['X_CENTER'] = 4.8381         #DAC0 refl: 5.16; trans: 6.25
params['X_RANGE'] = 0.2
params['X_STEP'] = 0.01

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

if params['SCAN_BEAM']=='A':
    params['Y_CENTER'] = 0.5494   #DAC3 refl: 0.900 | trans:-0.4163
    params['X_CENTER'] = 1.5460      #DAC2 refl: 1.506 | trans: 1.7368
    params['DAC_CH_Y'] = 3
    params['DAC_CH_X'] = 2
    params['SMU'] = "K2450"

params['AUTOZOOM'] = True          # or False
    
# Functions
def current_time():
    '''
    Gets the current time in the right format

    Returns
    -------
    val : TYPE
        DESCRIPTION.

    '''
    now = datetime.now()
    val = now.strftime("%H:%M:%S")
    return val

def voltage_ramp(smu, v_initial, v_final):
    '''
    Voltage ramp of the Keithley source meters

    Parameters
    ----------
    smu : LABRAD Keithley server object
        DESCRIPTION.
    v_initial : FLOAT
        initial voltage.
    v_final : FLOAT
        DESCRIPTION.

    Returns
    -------
    None.

    '''    
          
    if v_initial == v_final:
        print('[%s] No voltage ramp needed.' %current_time())
        return
    step = 20
    v_steps = np.linspace(v_initial, v_final, step)
    print('[%s] Ramping voltage...' %current_time())
    smu.output_on()
    for i in range(step):
        smu.set_volts(v_steps[i], params['COMPL'])
        time.sleep(params['DELAY'])
    print('[%s] Voltage ramp completed.' %current_time())
    time.sleep(params['DELAY'])
    
    return

def scan_mirror(dac,dac_readout,smu,ls,dv):
    '''
    Scan the galvo mirrors

    Parameters
    ----------
    x_rng : TYPE
        DESCRIPTION.
    y_rng : TYPE
        DESCRIPTION.
    dac : TYPE
        DESCRIPTION.
    smu : TYPE
        DESCRIPTION.
    lsci : TYPE
        DESCRIPTION.
    dv : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    max_x = 0
    max_y = 0
    max_I = 0
    tempA = 0
    tempB = 0
    
    # Scan range
    x_rng = np.linspace((params['X_CENTER']-params['X_RANGE']/2),(params['X_CENTER']+params['X_RANGE']/2),
                        int(params['X_RANGE']/params['X_STEP'])+1)
    y_rng = np.linspace((params['Y_CENTER']-params['Y_RANGE']/2),(params['Y_CENTER']+params['Y_RANGE']/2),
                        int(params['Y_RANGE']/params['Y_STEP'])+1)
    
    # scan in X
    for i in range(len(x_rng)):
    
        dac.set_voltage(params['DAC_CH_X'], x_rng[i])
        
        percent = (x_rng[i] - np.min(x_rng))/(np.max(x_rng) - np.min(x_rng))
        print("[{}] Scanning {:.2f} % complete".format(current_time(),100*percent))
    
        # scan in Y    
        for j in range(len(y_rng)):
            
            dac.set_voltage(params['DAC_CH_Y'], y_rng[j])
            
            x_pos_read = dac.read_voltage(0)
            y_pos_read = dac.read_voltage(1)
            currentRead = smu.read_i()
            # tempB = float(ls.read_temp('A'))
            # tempA = float(ls.read_temp('B'))
            dac_output = dac_readout.read_voltage(3)
            
            data = np.array([x_rng[i], y_rng[j], params['BIAS'],
                                currentRead, dac_output, x_pos_read, y_pos_read, tempA, tempB])
            dv.add(data)
            
            # store max current
            if currentRead>max_I:
                max_I = currentRead
                max_x = x_rng[i]
                max_y = y_rng[j]
                
            time.sleep(params['DELAY'])

    print("[{}] Max current {:.4f} nA at X: {:.4f} V, Y: {:.4f} V".format(current_time(),max_I*1e9,max_x,max_y))            
    dac.set_voltage(params['DAC_CH_X'], max_x)
    dac.set_voltage(params['DAC_CH_Y'], max_y)
    params['X_CENTER'] = max_x
    params['Y_CENTER'] = max_y
    
    return
        
def main():
    
    # Initialize labrad and servers
    cxn_m = labrad.connect()
        
    # DAC-ADC
    dac_m = cxn_m.dac_adc
    dac_m.select_device(params['DAC_MIRROR'])
    
    
    # DAC-ADC for readout
    cxn = labrad.connect()
    dac_data = cxn.dac_adc()
    dac_data.select_device(params['DAC_DATA'])
    
    
    # K2400
    if params['SCAN_BEAM']=='A':
        smu = cxn.k2450()
        smu.select_device()
        smu.gpib_write(":ROUTe:TERMinals FRONt")
    else:
        smu = cxn.k2400()
        smu.select_device()
        smu.gpib_write(":ROUTe:TERMinals FRONt") # FRONt or REAR
    
    smu.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    
    # LS 350
    ls = cxn.lscimodel331s()
    ls.select_device()
    
    # Data vault
    dv = cxn.data_vault()
    
    # Change to data directory
    dv.cd(params['DATADIR'])
    
    # Create new data file
    dv.new(params['FILENAME'], ['X Pos [V]', 'Y Pos [V]', 'Bias [V]'],
           ['I Measure [A]', 'DAC Read [V]', 'X Pos Read [V]', 'Y Pos Read [V]', 'Temp A [K]', 'Temp B [K]'])
    
    
    # Save parameters
    # dv.add_parameter('X points', len(x_rng))
    # dv.add_parameter('Y points', len(y_rng))
    
    # Save all parameters
    # params['X_POINTS'] = len(x_rng)
    # params['Y_POINTS'] = len(y_rng)
    config_filename = params['ROOTDIR']+"\\"+params['DATADIR']+".dir\\"+dv.get_name()+".yml"
    with open(config_filename, 'w') as f:
        yaml.dump(params, f)
    
    # Measurement
    start = time.time()
    print("[{}] Estimated total time: {:.0f} s"
          .format(current_time(),
                  4.5*params['DELAY']*(params['X_RANGE']*params['Y_RANGE'])/(params['X_STEP']*params['Y_STEP'])))
    
    # Voltage ramp
    voltage_ramp(smu, 0, params['BIAS'])
    if params['SCAN_BEAM'] == 'A':
        smu.set_i_read_range( params['SMU_RANGE'] )

    # Scan the Mirror
    scan_mirror(dac_m,dac_data,smu,ls,dv)
    
    if params['AUTOZOOM']:
        print("[{}] Auto-zooming...".format(current_time()))
        params['X_RANGE'] = 0.0125
        params['X_STEP'] = 0.0005
        params['Y_RANGE'] = 0.0125
        params['Y_STEP'] = 0.0005
        # Create new data file
        dv.new(params['FILENAME']+"_fine", ['X Pos [V]', 'Y Pos [V]', 'Bias [V]'],
               ['I Measure [A]', 'DAC Read [V]', 'X Pos Read [V]', 'Y Pos Read [V]', 'Temp A [K]', 'Temp B [K]'])
        scan_mirror(dac_m,dac_data,smu,ls,dv)
        
    # Voltage ramp
    voltage_ramp(smu, params['BIAS'], 0)
    
    # Reset the SMU
    smu.gpib_write('*RST')
    
    # Measurements ended
    end = time.time()
    print("[{}] DONE! Time Consumption: {:.1f} s".format(current_time(),end-start) )

if __name__ == '__main__':
    main()
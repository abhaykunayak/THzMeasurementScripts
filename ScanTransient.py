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
from scipy.ndimage import gaussian_filter
import TemperatureServer
import SMUServer

# Define parameters
params = dict()

params['ROOTDIR'] = r"C:\Users\Marconi\Young Lab Dropbox\Young Group\THz\Raw Data"
params['DATADIR'] = "2022_09_07_TL2715_AKNDB010_5E"  
params['FILENAME'] = "refl_transient_T_303K_ms"

params['DEPENDENTS'] = ['delay_mm', 'delay_ps']
params['INDEPENDENTS'] = ['Lockin X [A]', 'Lockin Y [A]', 'Input 2 [V]', 
                          'Input 3 [V]', 'Temp A [K]', 'Temp B [K]', 
                          'Emitter Current [A]', 'AB Current [A]', 
                          'DAC_M0 [V]','DAC_M1 [V]', 'DAC_M2 [V]', 'DAC_M3 [V]']

params['DELAY_RANGE_MM'] = [29,38]  # mm refl full: [29,37] | sample: [32,36] trans: [32,36]
params['DELAY_POINTS'] = 100*(params['DELAY_RANGE_MM'][1]-params['DELAY_RANGE_MM'][0])+1        # normal = 100 pts/mm

params['TIME_CONST'] = 0.01         # s; Lockin
params['SENS'] = 0.05               # s; Lockin | full: 0.05 | sample: 0.005

params['SWEEPS'] = 5                # Sweeps
params['AVGS'] = 200                # Averages
params['SAMPLING'] = 0.1            # DAC buffered ramp oversample

params['GAIN'] = 1e8                # TIA gain
params['BIAS_E'] = 10               # V on emitter
params['BIAS_AB'] = 0.0             # V on AB line
params['SMU_RANGE'] = 10e-6         # current range on K2450
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

def log_message(msg):
    '''
    

    Parameters
    ----------
    msg : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    msg_timestamp = '[{}] {}'.format(current_time(),msg)
    logging.info(msg_timestamp)
    print(msg_timestamp)
    
    return
    
def setup_datavault(params,dv,sweep_idx):
    '''
    description goes here

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    dv : TYPE
        DESCRIPTION.

    Returns
    -------
    dv : TYPE
        DESCRIPTION.

    '''
    
    # cols = np.array([["Lockin {} {:04d} [A]".format(j,i) for j in ['X', 'Y', '2', '3']] 
    #                   for i in range(params['AVGS'])]).flatten()
    # dv.new(params['FILENAME']+'_sweep_{:03d}'.format(sweep_idx), params['DEPENDENTS'],
    #         ['Lockin X [A]', 'Lockin Y [A]', 'Input 2 [V]', 'Input 3 [V]', 'Temp A [K]', 'Temp B [K]', 
    #         'Emitter Current [A]', 'AB Current [A]',
    #         'DAC_M0 [V]','DAC_M1 [V]', 'DAC_M2 [V]', 'DAC_M3 [V]']+cols.tolist())

    dv.new(params['FILENAME']+'_sweep_{:03d}'.format(sweep_idx), params['DEPENDENTS'],
            ['Lockin X [A]', 'Lockin Y [A]', 'Input 2 [V]', 'Input 3 [V]', 'Temp A [K]', 'Temp B [K]', 
            'Emitter Current [A]', 'AB Current [A]',
            'DAC_M0 [V]','DAC_M1 [V]', 'DAC_M2 [V]', 'DAC_M3 [V]'])
    
    dv.add_parameter('delay_mm_rng', (params['DELAY_RANGE_MM']))
    dv.add_parameter('delay_mm_pnts', params['DELAY_POINTS'])
    dv.add_parameter('delay_ps_rng', (params['DELAY_RANGE_PS']))
    dv.add_parameter('delay_ps_pnts',  params['DELAY_POINTS'])
    dv.add_parameter('live_plots', (('delay_ps', 'Lockin X'), ('delay_ps', 'Lockin Y'), ('delay_ps', 'Emitter Current'), ('delay_ps', 'AB Current')))
    
    # Save all parameters
    config_filename = params['ROOTDIR']+"\\"+params['DATADIR']+".dir\\"+dv.get_name()+".yml"
    with open(config_filename, 'w') as f:
        yaml.dump(params, f, sort_keys=False, default_flow_style=False)
        log_message("Config file written.")
    return dv

def setup_logfile(params):
    '''
    Setup the log file.

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    print('[%s] Setting up datafile and logging...' %current_time()) 
    
    # Change the working directory
    fullpath = os.path.join(params['ROOTDIR'],params['DATADIR']+'.dir')
    os.chdir(fullpath)
    
    # Setup logging
    filename_log = "{}_{}.log".format(datetime.now().strftime("%Y%m%d%H%M%S"),params['FILENAME'])
    logging.basicConfig(filename="{}".format(filename_log), level=logging.INFO, force=True)
    log_message('Started logging...')
    return
    
def voltage_ramp(dac, v_initial, v_final):
    '''
    Voltage ramp of the DAC output

    Parameters
    ----------
    dac : LABRAD DAC server object
        DESCRIPTION.
    v_initial : FLOAT
        initial voltage.
    v_final : FLOAT
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    log_message("Starting DAC voltage ramp...")
    if v_initial == v_final:
        return
    step = 20
    v_steps = np.linspace(v_initial, v_final, step)
    for i in range(step):
        dac.set_voltage(params['DAC_OUTPUT_CH'], v_steps[i])
        time.sleep(params['TIME_CONST'])
    
    log_message("Voltage DAC ramp ended.")
    
    return

def voltage_ramp_smu(smu, v_initial, v_final):
    '''
    Voltage ramp of the DAC output

    Parameters
    ----------
    dac : LABRAD DAC server object
        DESCRIPTION.
    v_initial : FLOAT
        initial voltage.
    v_final : FLOAT
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    log_message("Starting SMU voltage ramp...")
    if v_initial == v_final:
        return
    step = 20
    v_steps = np.linspace(v_initial, v_final, step)
    for i in range(step):
        smu.set_v_meas_i(v_steps[i])
        time.sleep(params['TIME_CONST'])

    log_message("Voltage SMU ramp ended.")
         
    return

def init_stage_position(params,ds):
    '''
    

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    ds : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    # Move stage to starting position
    log_message("Moving stage to starting position...")
    ds.gpib_write("1VA20.0")
    ds.move_absolute(1,params['DELAY_RANGE_MM'][0])
    time.sleep(5)
    
    for i in range(10):
        try:
            current_pos = ds.tell_position(1)
            if abs(current_pos-params['DELAY_RANGE_MM'][0])<0.001:
                log_message("Stage in position.")
                return
        except:
            log_message("Error in reading stage position.")
        
        # wait and try reading the position again
        time.sleep(1)
    
    log_message("Moving stage timeout.")        
    
    return

def scan_mirror(params,dac,smu,spot,k):
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
    
    if spot == 'E':
        params['X_CENTER'] = params['EX_CENTER']         #DAC0
        params['Y_CENTER'] = params['EY_CENTER']         #DAC1
        params['DAC_CH_X'] = 0
        params['DAC_CH_Y'] = 1
        params['X_RANGE'] = 0.2/k
        params['X_STEP'] = 0.01/k
        params['Y_RANGE'] = 0.2/k
        params['Y_STEP'] = 0.01/k
        
    else:
        params['X_CENTER'] = params['AX_CENTER']         #DAC2  
        params['Y_CENTER'] = params['AY_CENTER']         #DAC3    
        params['DAC_CH_X'] = 2
        params['DAC_CH_Y'] = 3
        params['X_RANGE'] = 0.2/k
        params['X_STEP'] = 0.01/k
        params['Y_RANGE'] = 0.2/k
        params['Y_STEP'] = 0.01/k
    

    
    # smu.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    
    # Scan range
    x_rng = np.linspace((params['X_CENTER']-params['X_RANGE']/2),(params['X_CENTER']+params['X_RANGE']/2),
                        int(params['X_RANGE']/params['X_STEP'])+1)
    y_rng = np.linspace((params['Y_CENTER']-params['Y_RANGE']/2),(params['Y_CENTER']+params['Y_RANGE']/2),
                        int(params['Y_RANGE']/params['Y_STEP'])+1)
    
    currentRead = np.zeros((len(x_rng),len(y_rng)),dtype=float)
    
    # Scan in X
    for i in range(len(x_rng)):
    
        dac.set_voltage(params['DAC_CH_X'], x_rng[i])
        
        percent = (x_rng[i] - np.min(x_rng))/(np.max(x_rng) - np.min(x_rng))
        log_message("Scanning {:.2f} % complete".format(100*percent))
    
        # Scan in Y    
        for j in range(len(y_rng)):
            dac.set_voltage(params['DAC_CH_Y'], y_rng[j])
            # time.sleep(params['DELAY_MEAS'])`
            currentRead[i][j] = smu.read_i()
    
    currentRead = gaussian_filter(currentRead, sigma=1)
    max_I = np.max(currentRead)
    [max_x_idx, max_y_idx] = np.unravel_index(np.argmax(currentRead),(len(x_rng),len(y_rng)))
    max_x = x_rng[max_x_idx]
    max_y = y_rng[max_y_idx]
    log_message("Max current {:.4f} nA at X: {:.4f} V, Y: {:.4f} V".format(max_I*1e9,max_x,max_y))
    
    dac.set_voltage(params['DAC_CH_X'], max_x)
    dac.set_voltage(params['DAC_CH_Y'], max_y)
    
    if spot == 'E':
        params['EX_CENTER'] = max_x
        params['EY_CENTER'] = max_y
    else:
        params['AX_CENTER'] = max_x
        params['AY_CENTER'] = max_y
    
    return max_I, max_x, max_y

def transient(params,delay_mm,delay_ps,ds,temp_server,dac,dac_m,dv,smu2400_current,smu2450):
    
    current_e = 0
    current_ab = 0
    dac_m0 = 0
    dac_m1 = 0
    dac_m2 = 0
    dac_m3 = 0
    tempA = 0
    tempB = 0
    input_2_avg = 0
    input_3_avg = 0
    
    # Buffer ramp DAC
    # _ = np.array(dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
    #                                    params['DAC_INPUT_CH'],
    #                                    [0.0],
    #                                    [0.0],
    #                                    1,
    #                                    params['SAMPLING']*params['TIME_CONST']*1e6,
    #                                    params['AVGS']))
            
    for i in range(params['DELAY_POINTS']):

            percent = (delay_mm[i]-np.min(delay_mm))/(np.max(delay_mm)-np.min(delay_mm))
            if i%10==0:
                log_message("Stepping stage to {:.2f} mm; {:.3f} ps; {:.2f} %"
                      .format(delay_mm[i], delay_ps[i], 100*percent))
            
            # Read temperature                
            tempA  = temp_server.tempA
            tempB  = temp_server.tempB
            
            # Check current on E and AB
            current_e = smu2400_current.I
            current_ab = smu2400_current.I
            
            # Move stage
            try:
                ds.move_absolute(1,delay_mm[i])
            except:
                log_message("Stage movement error.")
            
            # if np.abs((current_e-params['MAX_I_E'])/params['MAX_I_E'])>0.05 or np.abs((current_ab-params['MAX_I_AB'])/params['MAX_I_AB'])>0.05:
            #     print("[{}] Low current on E/AB. Exiting sweep...".format(current_time()))
            #     break
            
            # Stop sweep when E current is low 
            # if np.abs((current_e-params['MAX_I_E'])/params['MAX_I_E'])>0.1:
                # print("[{}] Low current on E/AB. E: {:.3f}nA. Exiting sweep...".format(current_time(),current_e*1e9))
                # break
            
            # Buffer ramp DAC
            br_data = np.array(dac.buffer_ramp(params['DAC_OUTPUT_CH_DUMMY'],
                                               params['DAC_INPUT_CH'],
                                               [0.0],
                                               [0.0],
                                               1,
                                               params['SAMPLING']*params['TIME_CONST']*1e6,
                                               params['AVGS']))

            input_0_avg = br_data[0][0]*params['SENS']/10.0/params['GAIN']
            input_1_avg = br_data[1][0]*params['SENS']/10.0/params['GAIN']

            
            # mirror control positions
            # dac_m0 = dac_m.read_voltage(0)
            # dac_m1 = dac_m.read_voltage(1)
            # dac_m2 = dac_m.read_voltage(2)
            # dac_m3 = dac_m.read_voltage(3)
            
            # data = np.append([delay_mm[i], delay_ps[i], input_0_avg, input_1_avg, input_2_avg, input_3_avg,
            #                     tempA, tempB, current_e, current_ab, dac_m0, dac_m1, dac_m2, dac_m3],
            #         br_data.flatten(order='F'))
            
            data = np.array([delay_mm[i], delay_ps[i], input_0_avg, input_1_avg, input_2_avg, input_3_avg,
                                tempA, tempB, current_e, current_ab, dac_m0, dac_m1, dac_m2, dac_m3])
            
            # Save data to datavault
            dv.add(data)

    return

def main():
    
    # Setup log file
    setup_logfile(params)
    
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
    
    smu2400_current = SMUServer.CurrentPoll(smu2400)
    
    # K2450
    smu2450 = cxn.k2450()
    smu2450.select_device()
    smu2450.gpib_write(':SENSe:CURRent:NPLCycles {}'.format(params['NPLC']))
    smu2450.gpib_write(':SENSe:AVERage:TCONtrol {}'.format(params['FILT_TYPE']))
    smu2450.gpib_write(':SENSe:AVERage:COUNt {}'.format(params['FILT_COUNT']))
    smu2450.gpib_write(':SENSe:AVERage {}'.format(params['FILT']))
    smu2450.set_i_read_range(params['SMU_RANGE'])
    
    smu2450_current = SMUServer.CurrentPoll(smu2450)
    
    # LakeShore 350
    ls350 = cxn.lakeshore_350()
    ls350.select_device()
    
    temp_server = TemperatureServer.TemperaturePoll(ls350)
    
    # LakeShore 331
    # ls = cxn.lscimodel331s()
    # ls.select_device() 
    
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
    
    # Measurement start
    start = time.time()
    log_message('Measurement Started')
    
    # Voltage ramp up on DAC, also on SMUs
    # voltage_ramp(dac, 0, -params['BIAS_E'])
    voltage_ramp_smu(smu2400, smu2400.read_v(), params['BIAS_E'])
    voltage_ramp_smu(smu2450, smu2450.read_v(), params['BIAS_E'])
    
    # Rough Scans
    log_message('Coarse mirror Scan E...')
    params["MAX_I_E"], params['EX_CENTER'], params['EY_CENTER'] = scan_mirror(params,dac_m,smu2400,'E',1)
    
    log_message('Coarse mirror Scan A...')
    params["MAX_I_AB"], params['AX_CENTER'], params['AY_CENTER'] = scan_mirror(params,dac_m,smu2450,'A',1)
    
    try:
        for i in range(params['SWEEPS']):
            # Sweep
            log_message("Starting sweep #{:03d}...".format(i))
            
            # Initialize the stage to starting position
            init_stage_position(params, ds)
            
            # Check current on E
            log_message("Checking for max current on E switch...")
            params["MAX_I_E"], params['EX_CENTER'], params['EY_CENTER'] = scan_mirror(params,dac_m,smu2400,'E',20)
            
            # Check current on A
            log_message("Checking for max current on A switch...")
            voltage_ramp_smu(smu2450, smu2450.read_v(), params['BIAS_E'])
            params["MAX_I_AB"], params['AX_CENTER'], params['AY_CENTER'] = scan_mirror(params,dac_m,smu2450,'A',20)
            voltage_ramp_smu(smu2450, params['BIAS_E'], params['BIAS_AB'])

            # Setup DataVault file and Config file
            log_message("Setting up datavault and config file...")
            dv = setup_datavault(params,dv,i)
            
            # Transient
            log_message("Starting transient measurement...")
            transient(params, delay_mm, delay_ps, ds, temp_server, dac, dac_m, dv, smu2400_current, smu2450_current)
            
    except KeyboardInterrupt:
        log_message("Measurement interrupted.")

    # Initialize the stage to starting position
    init_stage_position(params, ds)
        
    # Voltage ramp down
    # voltage_ramp(dac, -params['BIAS_E'], 0)
    voltage_ramp_smu(smu2400, smu2400.read_v(), 0)
    voltage_ramp_smu(smu2450, smu2450.read_v(), 0)
    
    # Measurements end
    log_message("Measurement Ended.")
    end = time.time()
    log_message("Total time: {:2f} s".format(end-start))
    
    return

if __name__ == '__main__':
    main()

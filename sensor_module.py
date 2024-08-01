#import board
#import digitalio
#import adafruit_max31865
#from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219
import numpy as np
import pandas as pd
from datetime import datetime
"""
def initialize_sensors():
    spi = board.SPI()
    cs = digitalio.DigitalInOut(board.D6)
    max31855 = adafruit_max31865.MAX31865(spi, cs)
    i2c_bus = board.I2C()
    ina219 = INA219(i2c_bus, 0x41)
    return spi, cs, max31855, i2c_bus, ina219
"""
def get_sensor_data(df_combined):
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'])
    timestamps_combined = df_combined['Timestamp'].dt.strftime('%m/%d/%Y %H:%M:%S')
    data_combined = df_combined[['Probe_Temp', 'Board_Voltage', 'Compressor_Voltage']].values
    return timestamps_combined, data_combined

#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Python 3 module to read a Wind/Waves L2 60S averaged data file.
"""

# ________________ IMPORT _________________________
# (Include here the modules to import, e.g. import sys)
import sys
import os
import re
from datetime import datetime
import struct
import logging

# ________________ HEADER _________________________

# Mandatory
__version__ = '1.0.0'
__author__ = 'Xavier Bonnin'
__date__ = '02/03/2022'

# Optional
__license__ = 'CeCILL-C'
__credit__ = ['Wind/Waves']
__maintainer__ = ''
__email__ = 'xavier.bonnin@obspm.fr'
__project__ = 'Wind/Waves'
__institute__ = 'LESIA, Observatoire de Paris'
__changes__ = {'1.0.0': 'First release'}


# ________________ Global Variables _____________
# (define here the global variables)

# Define default behavior for logger
LOGGER_BASIC_FORMAT = '%(asctime)s - %(levelname)-8s - %(message)s'
LOGGER_BASIC_STRTFORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(level=logging.INFO,
                    format=LOGGER_BASIC_FORMAT,
                    datefmt=LOGGER_BASIC_STRTFORMAT)
logging.root.setLevel(logging.INFO)


# ________________ Class Definition __________
# (If required, define here classes)


# ________________ Global Functions __________
# (If required, define here global functions)
def read_l2_60s(filepath,
                to_array=True,
                logger=logging):
    """
    Method to read a Wind/Waves l2 60 seconds averaged data file

    Example:
        from read_wi_wa_l2_60s_data import read_l2_60s
        file = 'wi_wa_rad1_l2_60s_20010101_v01.dat'
        header, data = read_l2_60s(file)

    :param filepath: Path of the Wind/Waves L2 60SEC binary file
    :param to_array: If True, return data as a dictionaries containing
                     the following elements:
                        - FKHZ - List of frequencies in kHz
                        - TIME - List of sample UTC times (numpy array of datetime.datetime objects)
                        - SMIN - numpy 2D array containing intensity minimal values
                        - SMAX - numpy 2D array containing intensity maximal values
                        - SMEAN - numpy 2D array containing intensity mean values
    :param logger: logging.getLogger instance
    :return: header and data as two lists of dictionaries
    """
    header_fields = ('P_FIELD', 'JULIAN_DAY_B1', 'JULIAN_DAY_B2', 'JULIAN_DAY_B3', 'MSEC_OF_DAY',
                     'RECEIVER_CODE',
                     'JULIAN_SEC',
                     'YEAR', 'MONTH', 'DAY',
                     'HOUR', 'MINUTE', 'SECOND',
                     'MOYSEC', 'IUNIT', 'NFREQ',
                     'X_GSE', 'Y_GSE', 'Z_GSE')
    header_dtype = '>bbbbihLhhhhhhhhhfff'

    header = []
    data = []
    nsweep = 1
    nsample = 0
    with open(filepath, 'rb') as frb:
        while (True):
            try:
                logger.debug('Reading sweep #%i' % (nsweep))
                # Reading number of octets in the current sweep
                block = frb.read(4)
                if (len(block) == 0):
                    break
                loctets1 = struct.unpack('>i', block)[0]
                # Reading header parameters in the current sweep
                block = frb.read(44)
                header_i = dict(
                    zip(header_fields, struct.unpack(header_dtype, block)))
                nfreq = header_i['NFREQ']
                # Reading frequency list (kHz) in the current sweep
                block = frb.read(4 * nfreq)
                freq = struct.unpack('>' + 'f' * nfreq, block)
                # Reading mean, min, max intensity
                block = frb.read(4 * nfreq)
                Smoy = struct.unpack('>' + 'f' * nfreq, block)
                block = frb.read(4 * nfreq)
                Smin = struct.unpack('>' + 'f' * nfreq, block)
                block = frb.read(4 * nfreq)
                Smax = struct.unpack('>' + 'f' * nfreq, block)
                # Reading number of octets in the current sweep
                block = frb.read(4)
                loctets2 = struct.unpack('>i', block)[0]
                if (loctets2 != loctets1):
                    logger.info(f'Error reading file! ({loctets1} != {loctets2})')
                    return None
            except EOFError:
                logger.exception('End of file reached')
                break
            else:
                header.append(header_i)
                time = datetime(header_i['YEAR'],
                                header_i['MONTH'],
                                header_i['DAY'],
                                header_i['HOUR'],
                                header_i['MINUTE'],
                                header_i['SECOND'])
                data.append({'TIME': time,
                             'FKHZ': freq, 'SMEAN': Smoy,
                             'SMIN': Smin, 'SMAX': Smax})
                nsweep += 1
                nsample += len(freq)

    logger.info('{0} samples extracted from {1}'.format(nsample, filepath))

    if to_array:
        try:
            import numpy as np
        except:
            logger.exception('numpy Python package is required!')
        else:
            logger.debug('Returning data as 2D array')
            data_array = {}
            data_array['FKHZ'] = data[0]['FKHZ']
            ntime = nsweep - 1
            nfreq = len(data_array['FKHZ'])

            time = np.zeros(ntime, dtype=datetime)
            smin = np.zeros((ntime, nfreq), dtype=float)
            smax = np.zeros((ntime, nfreq), dtype=float)
            smean = np.zeros((ntime, nfreq), dtype=float)
            for i, sweep in enumerate(data):
                time[i] = sweep['TIME']
                smin[i, :] = sweep['SMIN'][:]
                smax[i, :] = sweep['SMAX'][:]
                smean[i, :] = sweep['SMEAN'][:]
            data_array['TIME'] = time
            data_array['SMIN'] = smin
            data_array['SMAX'] = smax
            data_array['SMEAN'] = smean

            data = data_array

    return header, data

# _________________ Main ____________________________
if (__name__ == "__main__"):
    print("Python module to read Wind/Waves L2 data file.")

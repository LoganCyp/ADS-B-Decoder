# Python ADS-B Demodulator 

from bitstring import BitArray
import pandas as pd
from rtlsdr import RtlSdr
from datetime import datetime
import numpy as np
import math

# Generates buffers of ~256k samples
BUFFER_SIZE = 256 * 1024 

# ADBS Preamble length in symbols
'''
We calculate this with the sample rate of the SDR in mind, in our case we sample at 2 MHz, to calculate the preamble in symbols, we
note that the preamble is 8 microseconds, then multiply this by the sample rate, 2*10^6 * 8*10^-6 = 16
'''
ADBS_PREAMBLE_SYM = 16

# The length of the ADBS Message in bits
ADBS_MSG_LEN_BITS = 112

# The length of the ADBS Message in symbols
'''
Each bit in the ADS-B message is represented by 2 symbols, because of Manchester encoding, to get the length in symbols we multiply 
by 2
'''
ADBS_MSG_LEN_SYM = ADBS_MSG_LEN_BITS*2

# The length of the entire ADBS packet in symbols, including the preamble and message
ADBS_PACKET_LEN_SYM = ADBS_PREAMBLE_SYM + ADBS_MSG_LEN_SYM

# Validate CRC (cyclic redundancy check)

# The aforementioned CRC code
CRC_GENERATOR_CODE = BitArray(bin='1111111111111010000001001')

def check_crc(message: BitArray):
    '''
    Cyclic Redundancy Check (CRC) is used to detect errors in transmitted data by appending a checksum code to the original information.
    The checksum is recalculated at the receiving end to verify the integrity of the data.

    The generator code for the ADS-B CRC is 1111111111111010000001001, derived from Gertz, J.L. 1984. 
    Fundamentals of mode s parity coding. Massachusetts Institute of Technology, Lincoln Laboratory.
    '''
    # We want to save a copy of our received CRC, as to compare to our computed CRC
    original_crc = message[-24:]
    message[-24:] = 0x000

    # We want the first 88 bits of the message, as the last 24 are the parity bits
    # Length of message - 24 is used for general robustness rather than practicality
    for i in range(len(message) - 24):
        if message[i] == 1:
            # XOR 25 bits of the message with CRC_GENERATOR_CODE
            message[i:i+25] = message[i:i+25] ^ CRC_GENERATOR_CODE

    # Our computed CRC
    crc = message[-24:]

    # If the two values are equal, the integrity of the data has been maintained.
    return crc == original_crc

# Initialize a pandas dataframe to hold all received data
data_df = pd.DataFrame(columns = ['Downlink Format', 'ICAO', 'Message', 'Timestamp'])

def get_lat_long_data(data_df, icao):
    # Group the data by the ICAO
    grouped_data = data_df[data_df['ICAO'] == icao]

    if grouped_data is not None:
        if len(grouped_data) >= 2:
            # Sort the grouped_data by Timestamp in descending order
            grouped_data_sorted = grouped_data.sort_values(by='Timestamp', ascending=False)

            # Initialize variables to hold the most recent even and odd entries
            most_recent_even = None
            most_recent_odd = None

            # Loop through each row in grouped_data_sorted
            for index, row in grouped_data_sorted.iterrows():
                msg = row['Message']

                # The message should start with a zero bit, otherwise it gives the wrong latitude / longitude, this was found through observation
                if int(msg[21]) == 0 and int(msg[0]) == 0 and most_recent_even is None:
                    # Most recent even entry
                    most_recent_even = row
                elif int(msg[21]) == 1 and  int(msg[0]) == 0 and most_recent_odd is None:
                    # Most recent odd entry
                    most_recent_odd = row

                # If both most recent even and odd entries are found, break the loop
                if most_recent_even is not None and most_recent_odd is not None:
                    break
            
            if most_recent_even is not None and most_recent_odd is not None:

                # Combine the most recent even and odd entries into a DataFrame
                result_df = pd.DataFrame([most_recent_even, most_recent_odd])

                # Reset index for the resulting DataFrame
                result_df.reset_index(drop=True, inplace=True)

                return result_df
            else:
                return None
        else:
            return None



def cpr_NL(lat, Nz):
    '''
    In the 1090 MHz Riddle, it is mentioned that you need to check the NL values with respect
    to the even and odd latitudes, it can be found in section 5.2.2 equation 5.3

    NL calculates the number of longitude zones used for even encoding
    '''

    # Some initial edge cases laid out in the 1090 MHz Riddle, there is a small amount of error allowed since the latitude
    # Will likely never be exactly 87 or 0.
    if abs(lat) <= 1e-08:
        return 59
    elif abs(abs(lat) - 87) <= 1e-08 + 1e-05 * 87:
        return 2
    elif lat > 87 or lat < -87:
        return 1
    else:
        # Simply breaking up the numerator and denomintaor to increase readability

        num = 1 - math.cos(math.pi / (2 * Nz))
        dem = math.cos(math.pi / 180 * abs(lat)) ** 2

        # Calculate NL
        NL = 2 * math.pi / (math.acos(1 - num / dem))
        NL = math.floor(NL)

        return NL

# Globally unambiguous position decoding
def lat_long_pos(msg0, t0, msg1, t1):
    '''
    The 21st bit of the message is the Compact Position Reporting (CPR) Format

    0: Even Frame
    1: Odd Frame

    We need an even frame and an odd frame to calculate the latitude and longitude 
    '''

    even_message = int(msg0[21])
    odd_message = int(msg1[21])


    if even_message == 0 and odd_message == 1:
        # Pass through the conditional as the even and odd messages are passed correctly 
        pass
    elif even_message == 1 and odd_message == 0:
        # We want to invert the assignments if a odd frame gets assigned to the even frame and vice versa
        msg0, msg1 = msg1, msg0
        t0, t1 = t1, t0
    elif (even_message == 0 and odd_message == 0) or (even_message == 1 and odd_message == 1):
        print('Either two even or two odd messages where passed')
        return 0, 0
    
    '''
    In the encoded message

    LAT-CPR is 17 bits, comprised of bits 23-39 in ME
    LON-CPR is 17 bits, comprised of bits 40-56 in ME

    We divide the integer value by 2^17 since there are 17 bits in the latitude and longitude values
    We divide by 131072 as 2^17 = 131072
    '''

    even_lat_cpr = msg0[22:39].int / 131072
    even_lon_cpr = msg0[39:56].int / 131072

    odd_lat_cpr = msg1[22:39].int / 131072
    odd_lon_cpr = msg1[39:56].int / 131072

    '''
    Latitude zone sizes are defined as 

    dLatEven = 360/4Nz
    dLatOdd = 360/(4Nz - 1)

    Where Nz is the number of latitude zones between the equator and a pole. In Mode S, Nz is defined as a constant 15
    '''
    Nz = 15

    dLatEven = 360 / (4 * Nz)
    dLatOdd = 360 / ((4 * Nz) - 1)

    '''
    The formula for latitude zone index is denoted a j
    '''

    j = math.floor(59 * even_lat_cpr - 60 * odd_lat_cpr + 0.5)


    lat_even = float(dLatEven * (j % 60 + even_lat_cpr))
    lat_odd = float(dLatOdd * (j % 59 + odd_lat_cpr))

    '''
    In the southern hemisphere, values returned range from 270 to 360, we want to make sure latitude is within the range
    of [-90, 90], which can be adjusted based on the conditional below
    '''

    if lat_even >= 270:
        lat_even = lat_even - 360
    
    if lat_odd >= 270:
        lat_odd = lat_odd - 360

    # If the pair of messages are from different longitude zones, it is not possible to compute the correct global positioning 
    if cpr_NL(lat_even, Nz) != cpr_NL(lat_odd, Nz):
        print('The messages are from different longitude zones')
        return 0, 0
    
    # If the even timestamp is greater than the odd timestamp, we use the even latitude and longitude, otherwise we use the odd latitude and longitude
    if t0 >= t1:
        lat = lat_even
        # Calculating the longitude index, m, can be calculated similary to the latitude index
        nl = cpr_NL(lat, Nz)
        m = math.floor(even_lon_cpr * (nl - 1) - odd_lon_cpr * nl + 0.5)
        # Calculating the longitude zone size is depedent on latitude, and can be calculated as follows
        n_even = max(nl, 1)
        # The longitude zone size can be defined as follows
        dLonEven = 360 / n_even
        # Calculate the longitude
        lon = dLonEven * (m % n_even + even_lon_cpr)
    else:
        lat = lat_odd
        nl = cpr_NL(lat, Nz)
        m = math.floor(even_lon_cpr * (nl - 1) - odd_lon_cpr * nl + 0.5)
        n_odd = max((nl - 1), 1)
        dLonOdd = 360 / n_odd
        lon = dLonOdd * (m % n_odd + odd_lon_cpr)

    if lon >= 180:
        lon = lon - 360

    return lat, lon

# Work in progress for returning altitude
def get_altitude(tc, alt):
    # Default the altitude to zero, sometimes the altitude is unreachable in specfic messages
    altitude = 0

    if tc is None or tc < 9 or tc == 19 or tc > 22:
        print(f"TypeCode: {tc} is not an airborne position message")
        return 0
 

    # If the Type Code (TC) is between 9 and 18, it uses barometric altitude. 
    if tc <= 18 & tc >= 9:
        '''
        The 8th bit of the 12 bit altitude field is the Q bit. It indicates whether the altitude is encoded with an increment of 
        25 feet or 100 feet. When Q = 1, the altitude is encoded in 25 feet increments.
        '''
        # Checking if Q = 1
        if alt[7] == 1:
            # Remove the 8th bit (Q bit) from the binary sequence 
            alt = alt[:7] + alt[8:]
            # Convert the altitude to decimal
            decimal_alt = alt.uint
            # The formula for altitude under these conditions is h = 25 * N - 1000
            altitude = 25 * decimal_alt - 1000

    return altitude


def adsb_decode_bits(bits: str, filter_df=False):

    data = BitArray(bin=''.join(bits))

    # The Datalink Format is composed of the first 5 bits
    df = data[0:5].uint
    # The 
    ca = data[5:8].uint
    icao = data[8:32].hex
    me = data[32:88]
    crc = data[88:112]

    crc_ok = check_crc(BitArray(data))

    if not crc_ok:
        return 0

    if filter_df and filter_df != df:
        return 0

    tc = me[0:5].uint

    alt = me[8:20]

    altitude = get_altitude(tc, alt)

    # Get the current datetime for latitude / longitude calculations
    timestamp = datetime.now()

    data_row = [df, icao, me, timestamp]

    # Append the new row using loc
    data_df.loc[len(data_df)] = data_row

    lat_lon_data = get_lat_long_data(data_df, icao)

    if lat_lon_data is not None:
        print(lat_lon_data)
        msg0 = lat_lon_data.iloc[0]['Message']
        t0 = lat_lon_data.iloc[0]['Timestamp']
        msg1 = lat_lon_data.iloc[1]['Message']
        t1 = lat_lon_data.iloc[1]['Timestamp']

        lat, lon = lat_long_pos(msg0, t0, msg1, t1)
        print(f'lat {lat} \n lon {lon}')

    info_message = f"Raw hex message: {data.hex} \n Altitude {altitude} \n ICAO {icao} \n {tc}"

    print(info_message)

    return 1


# In-Phase / Quadrature to Amplitude Demodulation

def iq_to_amp(iq_data: np.ndarray) -> np.ndarray:
    """
    Convert IQ data to amplitude.
    """
    # Shift the data to fit the frame
    iq_data = iq_data.view(np.int8) - 128
    # Break into in-phase and quadrature components
    signal_i, signal_q = iq_data[::2], iq_data[1::2]
    
    # Return the magnitude of the signal
    return np.abs(signal_i + 1j * signal_q)


'''
The 16-bit fixed preamble can be represented as 1010000101000000 in binary.

(Reference : https://mode-s.org/decode/book-the_1090mhz_riddle-junzi_sun.pdf Section 1.4.2)
'''

ADS_B_PREAMBLE = BitArray(bin='1010000101000000')


# Decode bits into information

def decode_symbols(demod_signal: np.ndarray):
    stats = 0
    adbs_message_b = np.zeros(ADBS_MSG_LEN_BITS , np.uint8)
    
    '''
    There are many ways to detect the 8us preamble, this one specfically stuck out to me as it finds 
    the highest correlation in signal between the preamble and the demodulated signal. Other algorithms
    exist that are probably more effective but this was a simple implementation
    '''

    # Find the signal correlations between the ADS-B Preamble and the demodulated signal
    correlation = np.correlate(demod_signal, ADS_B_PREAMBLE, mode='valid')
    # While this is a fairly strict and possibly inaccurate threshold, it almost always ensures that the preamble will be found, this can be adjusted.
    threshold = max(correlation)
    # Find the most recent peak (presuming that is the preamble) where the correlation is greater than or equal to the threshold
    peaks = np.where(correlation >= threshold)[0]
    # In the case that there are multiple peaks, chose the first since the preamble comes first.
    start_index = peaks[0]
    
    if peaks[0] <= demod_signal.size - ADBS_PACKET_LEN_SYM:
        adbs_message_s = demod_signal[start_index+ADBS_PREAMBLE_SYM:start_index+ADBS_PACKET_LEN_SYM]

        # Based on the magnitude of the signal, we can decode the signal into bits
        for (i, (first, second)) in enumerate(adbs_message_s.reshape(ADBS_MSG_LEN_BITS, 2)):
            if first > second:
                adbs_message_b[i] = 1
            elif first < second:
                adbs_message_b[i] = 0
            else:
                break

        stats += adsb_decode_bits(''.join(str(i) for i in adbs_message_b), filter_df=17)

    return stats

def fetch_data(total_samples_read):
    """
    Reads from an rtlsdr device.
    """
    iq_data = np.ctypeslib.as_array(sdr.read_bytes(BUFFER_SIZE))
    total_samples_read += BUFFER_SIZE
    read_size = BUFFER_SIZE

    return iq_data, read_size, total_samples_read

if __name__ == '__main__':
    demod_signal = np.zeros(BUFFER_SIZE//2 + ADBS_PACKET_LEN_SYM)
    read_new_samples = None

    total_samples_read = 0
    stats = 0

    # Initialize RTL SDR Object, ensure you plug in a compatible RTL-SDR device with proper drivers installed.
    sdr = RtlSdr()
    # Set the sample rate to 2 MHz
    sdr.sample_rate = 2e6
    # Set the center frequency to our desired target of 1090 MHz to receive ADS-B messages
    sdr.center_freq = 1090e6
    # Automatically determine the gain
    sdr.gain = 'auto'
        
    while True:

        iq_data, read_size, total_samples_read = fetch_data(total_samples_read)

        # Save first bits of the array for messages between 2 buffers
        demod_signal[ADBS_PACKET_LEN_SYM:ADBS_PACKET_LEN_SYM + read_size//2] = iq_to_amp(iq_data)

        stats += decode_symbols(demod_signal)

        # Save last un-iterated symbols at the begining of the array for the next iteration
        demod_signal[:ADBS_PACKET_LEN_SYM] = demod_signal[-ADBS_PACKET_LEN_SYM:]





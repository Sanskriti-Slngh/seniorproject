import RPi.GPIO as gpio
import time
from datetime import datetime

# Pin connection
dpin = 3  # Data pin
cpin = 5  # Clock pin

# Set up GPIO
gpio.setwarnings(False)
gpio.setmode(gpio.BOARD)
gpio.setup(cpin, gpio.OUT)
gpio.setup(dpin, gpio.IN)

""" 
Frequency Definitions:
1 Hz = 1s
1 KHz = 1ms
1 MHz = 1us
"""

# Frequency, Duty Cycle, and other parameters
f = 500  # Frequency in KHz
Duty = (50, 50)  # Duty cycle in percentage (on-time, off-time)
sample = 0  # 0 for positive edge, 1 for negative edge
prev_reading = None
n = 1000  # Number of samples

# Program setup
T = 1e3 / f
onT = T * Duty[0] / 100
offT = T * Duty[1] / 100
assert (Duty[0] + Duty[1] == 100)

data = 0
j = 0
ignore = True
skip_n_after_ignore = 0
num_bits_in_group = 10
num_readings = 1
all_readings = []
mreading = []


def sample_data():
    # Function to sample data from the sensor
    global data, ignore, j, skip_n_after_ignore, mreading

    x = gpio.input(dpin)
    time.sleep(1 / 1e6)

    if ignore and x:
        return
    ignore = False

    if skip_n_after_ignore:
        skip_n_after_ignore -= 1
        return
    data |= (x << j)

    if (j == (num_bits_in_group - 1)):
        x = (data >> 1) % 128

        if (x >= 32 and x <= 127):
            mreading.append(chr(x))
        data = 0
        j = 0
    else:
        j += 1


def get_reading():
    # Function to get the reading from the sampled data
    global mreading, n

    for i in range(n):
        # Positive cycle
        gpio.output(cpin, gpio.HIGH)
        if not sample:
            sample_data()
        time.sleep(onT / 1e6)
        # Negative cycle
        gpio.output(cpin, gpio.LOW)
        if sample:
            sample_data()
        time.sleep(offT / 1e6)

    mreading = "".join(mreading)
    reading = mreading[4:12]
    return reading


while True:
    # Clear all variables
    data = 0
    j = 0
    ignore = True
    skip_n_after_ignore = 0
    mreading = []

    # Get reading from the sensor
    reading = get_reading()

    # Check for changes in the reading
    if reading != prev_reading:
        # Record timestamp and reading
        ts = datetime.now()
        all_readings.append((ts.strftime("%m/%d/%Y %H:%M:%S"), reading))

        # Check if enough readings are collected
        if len(all_readings) == num_readings:
            # Write readings to local file
            file = open('meter_data.txt', 'a')
            for x in all_readings:
                file.write(str(x) + "\n")
            file.close()

            # Write readings to web file
            file = open('/var/www/html/meter_data.txt', 'a')
            for x in all_readings:
                file.write(str(x) + "\n")
            file.close()

            # Clear the list of readings
            all_readings.clear()

    # Update previous reading and wait
    prev_reading = reading
    time.sleep(10)

import random
import pickle
import numpy as np

# set the seed for repeatability
random.seed(100)

# Define data directories and file paths
data_dir_path = '/Users/tiyasingh/Desktop/seniorproject/data'
fname_meter_data = "/Users/tiyasingh/Desktop/seniorproject/meter_data.txt"
fname_water_spreadsheet = "/Users/tiyasingh/Desktop/seniorproject/water_spreadsheet.txt"

# Read meter data
with open(fname_meter_data, 'r') as fin:
    data = fin.read()
flow_data = data.split('\n')

# Read water spreadsheet data
with open(fname_water_spreadsheet, 'r') as f:
    data = f.read()
label_data = data.split('\n')

# Define label mappings
label2num = {'f': 1, 'w': 2, 's': 3, 'l': 4, 'h': 5, 'f+w': 6, 'dw': 7}
num2label = ['N', 'f', 'w', 's', 'l', 'h', 'f+w', 'dw']


def time_to_index(time):
    """
    Normal time to minutes
    :param time: 4:12
    :return: 252
    """
    n_time = time.split(":")
    index = (int(n_time[0])*60) + int(n_time[1])
    return index


def gallons_to_dec(g):
    """
    Meter text reading into float
    :param g: 0000263
    :return: 26.3
    """
    return float(int(g)/10)


def random_num():
    """
    Generate a random number between 1 to 100
    :return:
    """
    return random.randint(1, 100)


def w_appliance(i, l):
    """
    Get the label for current point based on start and end on previous time stamps
    :param i: index of current point
    :param l: all of the annotation data for current date
    :return: label for current point
    """
    answer = []
    if l[i] != 0 and (l[i][-1] != 's' or l[i][-1] != 'e'):
        answer.append(label2num[l[i]])
    for appliance in ['w', 'h', 's', 'l', 'dw', 'f+w']:
        index = i
        while True:
            label = l[index]
            if label != 0 and label[:-1] == appliance:
                if label[-1] == 's':
                    ans = label2num[label[:-1]]
                    answer.append(ans)
                    break
                if label[-1] == 'e':
                    break
            if index == 0:
                break
            index -= 1
    return answer


# Function to calculate the order sum
def order_sum(lst):
    s = 0
    ind = 0
    for i in lst:
        s += (i * 2**ind)
        ind += 1
    dnm = sum([2**i for i in range(16)]) * 16
    s = s/dnm
    return s


# Function to check if a string can be converted to a float
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


# Use the small older dataset for Java Final
current_date = "02/01/2023"
prev_gallon = 33.1
prev_date = None
textallindex = 0
plot = True
plot_all = True
twindow = 7
nosamples = 10000
nosamplesdev = 2000

x_all = []
y_all = []
text_all = []

x_s = [i for i in range(24 * 60)]
y_s = [0 for i in range(24 * 60)]
text_s = [[] for i in range(24 * 60)]
on_apps = []

for lno, l_data in enumerate(label_data):
    if lno == 0:
        # First line is header row
        continue

    day, start_time, end_time, label = l_data.split(",")
    label = label.strip()

    # TBD : Ingore unknowns for now
    if label == 'u':
        continue

    # All data collected from previous day
    if day != current_date:
        text_all.append(text_s)
        # Reset text_s
        text_s = [[] for i in range(24 * 60)]
        # Change date to new date
        current_date = day

    # convert time to index
    start_time = time_to_index(start_time)
    if end_time == "":
        end_time = start_time
    else:
        end_time = time_to_index(end_time)

    for _ in range(start_time, end_time+1):
        text_s[_].append(label2num[label])

text_all.append(text_s)

# reset current date
current_date = "02/01/2023"
for data in flow_data:
    new_data = data.replace("(", "")
    new_data = new_data.replace(")", "")
    new_data = new_data.replace("'", "")
    lst = new_data.split(" ")
    date = lst[0]
    time = lst[1]

    # some readings are not captured correctly, ignore them
    if len(lst[2]) != 8 or isfloat(lst[2]) is False:
        continue
    gallons = gallons_to_dec(lst[2])

    # Collected all the data for current date
    if date != current_date:
        x_all.append(x_s)
        y_all.append(y_s)

        # Change to new day
        current_date = date
        x_s = [i for i in range(24 * 60)]
        y_s = [0 for i in range(24 * 60)]

    # collect the data into current day
    t_index = time_to_index(time)
    delta_g = max(gallons - prev_gallon, 0)
    prev_gallon = gallons
    y_s[t_index] += delta_g

# add last day of data
x_all.append(x_s)
y_all.append(y_s)

maximum = max([max(i) for i in y_all])
y_ind = 0
for x in y_all:
    ind = 0
    for i in x:
        x[ind] = i/maximum
        ind += 1
    y_all[y_ind] = x
    y_ind += 1

# Create training data
x_train = []
y_train = []
x_dev = []
y_dev = []

P = np.zeros((16))
for i in range(16):
    P[i] = np.sin(i)

while True:
    # Pick a random day and minute
    day = random.randint(0, 17)
    minute = random.randint(8, 1440-10)
    xs = [y_all[day][minute - i] for i in range(-7, 8)]
    xs.append(minute/1440)
    labels = text_all[day][minute]

    # if no current label, then skip
    if len(labels) == 0 and random.randint(1,100) > 1:
        continue

    if len(x_train) < nosamples:
        x_train.append(xs)
        if len(labels) == 0:
            y_train.append(0)
        else:
            y_train.append(labels[0])
        continue

    if len(x_dev) < nosamplesdev:
        x_dev.append(xs)
        if len(labels) == 0:
            y_dev.append(0)
        else:
            y_dev.append(labels[0])
        continue

    break

x_train = np.array(x_train)
y_train = np.array(y_train)
x_dev = np.array(x_dev)
y_dev = np.array(y_dev)

# save data
with open(data_dir_path + f'/data_train.pickle', 'wb') as f:
    pickle.dump((x_train, y_train), f, pickle.HIGHEST_PROTOCOL)
with open(data_dir_path + f'/data_dev.pickle', 'wb') as f:
    pickle.dump((x_dev, y_dev), f, pickle.HIGHEST_PROTOCOL)

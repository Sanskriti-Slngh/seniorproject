from datagenerator import CSVDataset
from fnn import MLP
import pickle
from torch.utils.data import DataLoader
from base_model import BaseModel


# parameters
lr = 0.00001
batch_size = 32
epochs = 100
model_name = "fnn"
train = True
val = False
test = True
evaluate = True
accuracy = False

data_dir_path = '/Users/tiyasingh/Desktop/seniorproject/data'

# Load your pickled data
with open(data_dir_path + f'/data_train.pickle', 'rb') as f:
    x_train, y_train = pickle.load(f)

with open(data_dir_path + f'/data_dev.pickle', 'rb') as f:
    x_dev, y_dev = pickle.load(f)

# split dev into val/test
middle_index = len(x_dev) // 2
x_val = x_dev[:middle_index]
x_test = x_dev[middle_index:]

middle_index = len(y_dev) // 2
y_val = y_dev[:middle_index]
y_test = y_dev[middle_index:]


if train:
    train_dataset = CSVDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
if val:
    val_dataset = CSVDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
if test:
    test_dataset = CSVDataset(x_val, y_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

fnn = MLP()
model = BaseModel(False, fnn)

if train:
    for i in range(epochs):
        model.load_model()
        model.compile(lr)
        model.train(epochs, train_loader)
        model.save(i)
        model.save()

if evaluate:
    model.load_model()
    if val:
        if accuracy:
            model.predict_model(val_loader)
        model.evaluate(val_loader)
    if test:
        if accuracy:
            model.predict_model(test_loader)
        model.evaluate(test_loader)


# README

## DIRECTIONS - HOW TO RUN

1. Run `main.py` with parameter configuration:
    - `lr = 0.00001`: Learning rate for model training
    - `batch_size = 32`: Batch size for model training
    - `epochs = 100`: Number of epochs for model training
    - `model_name = "fnn"`: Type of model (currently only supports "fnn")
    - `train = True/False`: Set to True if the model needs to be trained
    - `val = True/False`: Set to True if evaluation should be done on the validation dataset
    - `test = True/False`: Set to True if evaluation should be done on the test dataset
    - `evaluate = True/False`: Set to True if visual analysis is desired
    - `accuracy = True/False`: Set to True if accuracy is wanted on the validation/test dataset

## OUTPUT INTERPRETATION:
- `TRAIN = True`: Displays Epoch number, Batch number, and Categorical Cross Entropy Loss during training.
- `EVALUATE = True`: Generates a plot of water meter changes over time, titled with real and predicted appliances turned on at that moment.
- `ACCURACY = True`: Shows the accuracy performance of the model on the dataset.

## EXAMPLES:
- **TRAIN MODEL `main.py` PARAMETERS:**
    ```
    lr = 0.000001
    batch_size = 32
    epochs = 100
    model_name = "fnn"
    train = True
    val = False
    test = False
    evaluate = False
    accuracy = False
    ```

- **FIND ACCURACY OF MODEL ON VALIDATION AND TEST SET `main.py` PARAMETERS:**
    ```
    lr = 0.000001
    batch_size = 32
    epochs = 100
    model_name = "fnn"
    train = False
    val = True
    test = True
    evaluate = False
    accuracy = True
    ```

- **VISUALIZE PERFORMANCE OF MODEL ON VALIDATION AND TEST SET `main.py` PARAMETERS:**
    ```
    lr = 0.000001
    batch_size = 32
    epochs = 100
    model_name = "fnn"
    train = False
    val = True
    test = True
    evaluate = True
    accuracy = False
    ```

## FILE OVERVIEW

- **`main.py`**: Main file combining all files/classes together and serving as the main control source.

- **`base_model.py`**: Contains all functions compatible with future models (senior project).
    - FUNCTIONS:
        - `compile(learning rate)`
        - `save(epoch number if needed)`
        - `load(epoch number if needed)`
        - `train(number of epochs)`
        - `evaluate(data loader)`
        - `predict_model(data loader)`
    - CLASSES:
        - `BaseModel`

- **`fnn.py`**: Model architecture used for model training.
    - FUNCTIONS:
        - `forward(layer)`
    - CLASSES:
        - `MLP()`

- **`meter_reading.py`**: Generates a text file that contains the meter readings (taken from Raspberry Pi).
    - Creates "data/meter_data.txt"
    - FUNCTIONS:
        - `sample_data()`
        - `get_reading()`

- **`data_process.py`**: Generates the datasets used for model training/testing and data generator.
    - Uses "data/meter_data.txt"
    - Uses "data/water_spreadsheet.txt"
    - Creates "data/data_train.pickle"
    - Creates "data/data_dev.pickle"
    - FUNCTIONS:
        - `time_to_index(time)`
        - `gallons_to_dec(gallons)`
        - `random_num()`
        - `w_appliance(current point, annotated data with appliances for current data)`
        - `order_sum(lst)`
        - `isfloat(num)`

- **`datagenerator.py`**: Builds a data loader for the PyTorch model(s).
    - CLASSES:
        - `CSVDataset()`

## DATA
Self-built data is attached in the zip file under the "data" folder:
- **`meter_data.txt`:**
  The date/time and the corresponding meter reading.
- **`water_spreadsheet.txt`:**
  The date/time (start/end) and the appliance that was turned on.
- **`data_dev.pickle`:**
  The dataset created using the .txt files for validation and testing.
- **`data_train.pickle`:**
  The dataset created using the .txt files for training.

## MODELS
Each trained model is saved here:
- **`epoch_.pt`:**
  The final model to train/validate/test.

## VISUALS
The output screenshots.


# import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import io
import matplotlib.pyplot as plt


"""
Sanskriti Singh
Model controls all basic functions of all models regardless of architecture
"""
class BaseModel:

    def __init__(self, aug, model):
        self.augment = aug
        self.name = "model"
        self.model = model
        self.criterion = None
        self.optimizer = None

    # save model
    def save(self, i=''):
        torch.save(self.model.state_dict(), f"models/epoch_{i}.pt")

    # load model
    def load_model(self, i=''):
        checkpoint_path = f"models/epoch_{i}.pt"

        with open(checkpoint_path, 'rb') as f:
            buffer = io.BytesIO(f.read())

        checkpoint = torch.load(buffer, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)

    # compile model
    def compile(self, lr):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    # train the model
    def train(self, num_epochs, trainloader, valloader):
        for epoch in range(num_epochs):
            # Set the model to training mode
            self.model.train()

            for batch_idx, (inputs, labels) in enumerate(trainloader):
                self.optimizer.zero_grad()  # Zero the gradients

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()  # Backward pass to compute gradients
                self.optimizer.step()  # Update the model's parameters

                # Print training progress
                if batch_idx % 100 == 0:  # Adjust the frequency of printing as needed
                    print(f'Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item()}')

            print("\nAccuracy for train set with current model: " + str(self.predict_model(trainloader)))
            print("Accuracy for validation set with current model: " + str(self.predict_model(valloader))+"\n")

    # evaluate the model
    def evaluate(self, data_loader):
        # Mapping from numerical labels to actual labels
        num2label = {
            0: 'nothing',
            1: 'flush',
            2: 'wash',
            3: 'shower',
            4: 'laundry',
            5: 'hose',
            6: 'flush + wash',
            7: 'dishwasher'
        }

        # Set the model to evaluation mode
        self.model.eval()

        # Lists to store input data and predicted labels
        x_s = []
        y_s = []
        y_pred = []

        # Disable gradient computation during evaluation
        with torch.no_grad():
            # Loop through the data loader
            for inputs, _ in data_loader:
                # Convert inputs to a list and append to x_s
                for i in inputs.tolist():
                    x_s.append(i)
                # Convert outputs to a list and append to y_s
                for j in _.tolist():
                    y_s.append(j.index(1.0))

                # Forward pass through the model
                outputs = self.model(inputs)

                # Calculate class probabilities using softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Extract predicted labels and append to y_pred
                for i in torch.argmax(probabilities, dim=1).tolist():
                    y_pred.append(i)

        # Plot each data point with the predicted label as the title
        for key, values in enumerate(x_s):
            plt.plot(values, marker='o')
            plt.title("Real: " + num2label[y_s[key]] + ", Predicted: " + num2label[y_pred[key]])
            plt.ylabel("Change in gallons")
            plt.show()

    # predict with the model
    def predict_model(self, data_loader):
        # Set the model to evaluation mode
        self.model.eval()

        # Lists to store input data and predicted labels
        y_true = []
        y_pred = []

        # Disable gradient computation during evaluation
        with torch.no_grad():
            # Loop through the data loader
            for inputs, _ in data_loader:
                # Convert outputs to a list and append to y_true
                for i in _.tolist():
                    y_true.append(i.index(1.0))

                # Forward pass through the model
                outputs = self.model(inputs)

                # Calculate class probabilities using softmax
                probabilities = torch.nn.functional.softmax(outputs, dim=1)

                # Extract predicted labels and append to y_pred
                for i in torch.argmax(probabilities, dim=1).tolist():
                    y_pred.append(i)

        # print accuracy
        def calculate_accuracy(y_pred, y_true):
            correct_predictions = sum(pred == true for pred, true in zip(y_pred, y_true))
            total_predictions = len(y_pred)
            accuracy = correct_predictions / total_predictions
            return accuracy

        return (calculate_accuracy(y_pred, y_true))

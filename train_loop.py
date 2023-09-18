import torch
import torch.nn as nn
import numpy as np
from move_detection import LSTMMovePredictor
from MovesDataset import RubiksCubeDataset, CubeCroppedDataset, BiCNNDataset, SpatialImageDataset, CombinedDataset
from torch.utils.data import Dataset, DataLoader
from conv_lstm import ConvLSTM
from BiCNN import SpatialCNN, TemporalCNN, SpatialCNN2
from combined_models import Combined


def train_LSTM():
    # load in the dataset for training
    root_dir = 'moves'
    max_tokens = 10  # Maximum number of tokens (frames) per move

    transform = None

    dataset = RubiksCubeDataset(root_dir,  max_tokens, transform=transform)
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    # load in the model to be trained
    input_dim = 3  # Dimensionality of each landmark
    hidden_dim = 128  # Number of hidden units in LSTM
    num_layers = 3  # Number of LSTM layers
    output_dim = 3  # Number of possible cube move classes
    num_landmarks = 42
    model = LSTMMovePredictor(input_dim, hidden_dim, num_layers, output_dim, num_landmarks)
    model = model.to('cuda')

    # define training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    label_to_index = {'U': 0, 'R': 1, 'F': 2}

    for epoch in range(num_epochs):
        for batch_data, batch_delta_data, batch_labels in train_dataloader:  # Assuming you have a DataLoader
            batch_data = batch_data.to('cuda')
            batch_delta_data = batch_delta_data.to('cuda')  # Move batch data to GPU
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')
            optimizer.zero_grad()


            # Forward pass
            outputs = model(batch_data)

            # Compute loss
            loss = criterion(outputs, batch_indices)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'LSTM_model/model.pt')

    # test model
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_data, batch_delta_data, batch_labels in test_dataloader:
            batch_data = batch_data.to('cuda')
            batch_delta_data = batch_delta_data.to('cuda')  # Move batch data to GPU

            # Convert string labels to integer labels
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')

            # Forward pass
            outputs = model(batch_data)

            # Get predicted labels (indices with maximum probability)
            predicted_indices = torch.argmax(outputs, dim=1)

            # Count correct predictions
            total_correct += torch.sum(predicted_indices == batch_indices).item()
            total_samples += batch_indices.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.2%}")


def train_ConvLSTM():
    dataset = CubeCroppedDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

    model = ConvLSTM(input_dim=3,
                     hidden_dim=32,
                     kernel_size=(3,3),
                     num_layers=1,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)
    model = model.to('cuda')

    # define training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 300

    label_to_index = {'U': 0, 'R': 1, 'F': 2}

    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_dataloader:  # Assuming you have a DataLoader
            batch_data = batch_data.to('cuda')
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')
            if epoch > 200:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            optimizer.zero_grad()

            # Forward pass
            # print(batch_data.shape)
            outputs, _ = model(batch_data)

            # Compute loss
            loss = criterion(outputs, batch_indices)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'ConvLSTM_model/model.pt')

    # test model
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_data, batch_labels in test_dataloader:
            batch_data = batch_data.to('cuda')

            # Convert string labels to integer labels
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')

            # Forward pass
            outputs, _ = model(batch_data)

            # Get predicted labels (indices with maximum probability)
            predicted_indices = torch.argmax(outputs, dim=1)

            # Count correct predictions
            total_correct += torch.sum(predicted_indices == batch_indices).item()
            total_samples += batch_indices.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.2%}")

def train_BiCNN():
    dataset = BiCNNDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=True)

    model_spatial = SpatialCNN()
    model_temporal = TemporalCNN()
    model_spatial = model_spatial.to('cuda')
    model_temporal = model_temporal.to('cuda')

    # define training loop
    criterion = nn.CrossEntropyLoss()
    optimizer_spatial = torch.optim.Adam(model_spatial.parameters(), lr=0.001)
    optimizer_temporal = torch.optim.Adam(model_temporal.parameters(), lr=0.001)

    num_epochs = 100

    label_to_index = {'U': 0, 'R': 1, 'F': 2}

    for epoch in range(num_epochs):
        for batch_data_spatial, batch_data_temporal, batch_labels in train_dataloader:  # Assuming you have a DataLoader
            batch_data_spatial = batch_data_spatial.to('cuda')
            batch_data_temporal = batch_data_temporal.to('cuda')
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')
            optimizer_spatial.zero_grad()
            optimizer_temporal.zero_grad()

            # Forward pass
            # print(batch_data.shape)
            outputs_spatial = model_spatial(batch_data_spatial)

            # Compute loss
            loss_spatial = criterion(outputs_spatial, batch_indices)

            # Backpropagation and optimization
            loss_spatial.backward()
            optimizer_spatial.step()

            outputs_temporal = model_temporal(batch_data_temporal)

            # Compute loss
            loss_temporal = criterion(outputs_temporal, batch_indices)

            # Backpropagation and optimization
            loss_temporal.backward()
            optimizer_temporal.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss Spatial: {loss_spatial.item():.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss Temporal: {loss_temporal.item():.4f}")

    # test model
    model_spatial.eval()  # Set the model to evaluation mode
    model_temporal.eval()
    total_correct_spatial = 0
    total_samples_spatial = 0
    total_correct_temporal = 0
    total_samples_temporal = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_data_spatial, batch_data_temporal, batch_labels in test_dataloader:
            batch_data_spatial = batch_data_spatial.to('cuda')
            batch_data_temporal = batch_data_temporal.to('cuda')
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')

            # Forward pass
            outputs_spatial = model_spatial(batch_data_spatial)
            outputs_temporal = model_temporal(batch_data_temporal)

            # Get predicted labels (indices with maximum probability)
            predicted_indices_spatial = torch.argmax(outputs_spatial, dim=1)
            predicted_indices_temporal = torch.argmax(outputs_temporal, dim=1)

            # Count correct predictions
            total_correct_spatial += torch.sum(predicted_indices_spatial == batch_indices).item()
            total_samples_spatial += batch_indices.size(0)
            total_correct_temporal += torch.sum(predicted_indices_temporal == batch_indices).item()
            total_samples_temporal += batch_indices.size(0)

    accuracy_spatial = total_correct_spatial / total_samples_spatial
    accuracy_temporal = total_correct_temporal / total_samples_temporal
    print(f"Test Accuracy Spatial: {accuracy_spatial:.2%}")
    print(f"Test Accuracy Temporal: {accuracy_temporal:.2%}")


def train_spatialCNN():
    dataset = SpatialImageDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    model = SpatialCNN2()
    model = model.to('cuda')

    # define training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\

    num_epochs = 50

    label_to_index = {'U': 0, 'R': 1, 'F': 2}

    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_dataloader:  # Assuming you have a DataLoader
            batch_data = batch_data.to('cuda')
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')
            optimizer.zero_grad()

            # Forward pass
            # print(batch_data.shape)
            outputs = model(batch_data)

            # Compute loss
            loss = criterion(outputs, batch_indices)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # test model
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_data, batch_labels in test_dataloader:
            batch_data = batch_data.to('cuda')

            # Convert string labels to integer labels
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')

            # Forward pass
            outputs = model(batch_data)

            # Get predicted labels (indices with maximum probability)
            predicted_indices = torch.argmax(outputs, dim=1)

            # Count correct predictions
            total_correct += torch.sum(predicted_indices == batch_indices).item()
            total_samples += batch_indices.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.2%}")

def train_combined():
    dataset = CombinedDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = Combined()
    model.load_state_dict(torch.load('Combined_model/model.pt'))
    model = model.to('cuda')

    # define training loop
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    label_to_index = {'U': 0, 'R': 1, 'F': 2}

    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_dataloader:  # Assuming you have a DataLoader

            batch_data[0] = batch_data[0].to('cuda')
            batch_data[1] = batch_data[1].to('cuda')
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')
            optimizer.zero_grad()

            # Forward pass
            # print(batch_data.shape)
            outputs = model(batch_data)
            torch.reshape(outputs, (1, 3))

            # Compute loss
            loss = criterion(outputs, batch_indices)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), 'Combined_model/model.pt')
    # test model
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        for batch_data, batch_labels in test_dataloader:

            batch_data[0] = batch_data[0].to('cuda')
            batch_data[1] = batch_data[1].to('cuda')

            # Convert string labels to integer labels
            batch_indices = [label_to_index[label] for label in batch_labels]
            batch_indices = torch.tensor(batch_indices, dtype=torch.long).to('cuda')

            # Forward pass
            outputs = model(batch_data)
            torch.reshape(outputs, (1, 3))

            # Get predicted labels (indices with maximum probability)
            predicted_indices = torch.argmax(outputs, dim=1)

            # Count correct predictions
            total_correct += torch.sum(predicted_indices == batch_indices).item()
            total_samples += batch_indices.size(0)

    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    train_combined()
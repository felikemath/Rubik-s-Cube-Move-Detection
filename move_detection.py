import torch
import torch.nn as nn

class LSTMMovePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, num_landmarks):
        super(LSTMMovePredictor, self).__init__()

        self.lstm = nn.LSTM(input_dim * num_landmarks, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape input data to fit LSTM input shape
        batch_size, sequence_length, num_landmarks, landmark_dim = x.size()
        x = x.view(batch_size, sequence_length, -1)  # Flatten the landmarks

        lstm_output, _ = self.lstm(x)
        last_hidden_state = lstm_output[:, -1, :]
        output = self.fc(last_hidden_state)
        # output = self.softmax(output)
        return output


input_dim = 3  # Dimensionality of each landmark
hidden_dim = 128  # Number of hidden units in LSTM
num_layers = 3  # Number of LSTM layers
output_dim = 3  # Number of possible cube move classes
num_landmarks = 42
# Print the model architecture

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_classes):
        super(ConvLSTM, self).__init__()

        self.lstm = nn.LSTMCell(input_channels, hidden_channels)
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(hidden_channels, num_classes, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        h_t = torch.zeros(batch_size, self.hidden_channels, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_channels, dtype=x.dtype, device=x.device)

        outputs = []
        for t in range(seq_len):
            lstm_input = x[:, t, :, :, :].view(batch_size, -1)
            h_t, c_t = self.lstm(lstm_input, (h_t, c_t))
            output = self.conv(h_t)
            outputs.append(output)

        return torch.stack(outputs, dim=1)

input_channels = 3  # Number of input channels (e.g., RGB images)
hidden_channels = 64  # Number of hidden channels in LSTM
kernel_size = 3  # Kernel size for convolutional layers
num_classes = 3  # Number of classes for classification

if __name__ == '__main__':
    model = LSTMMovePredictor(3, 128, 5, 3, 42)
    x = torch.rand((1,10,42,3))

    print(model(x))


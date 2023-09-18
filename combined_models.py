import torch.nn as nn
import torch
from move_detection import LSTMMovePredictor
from conv_lstm import ConvLSTM

class Combined(nn.Module):

    def __init__(self):
        super(Combined, self).__init__()

        self.lstm = LSTMMovePredictor(
            input_dim=3,
            hidden_dim=128,
            output_dim=3,
            num_landmarks=42,
            num_layers=3,
        )

        self.lstm.load_state_dict(torch.load('LSTM_model/model.pt'))
        for param in self.lstm.parameters():
            param.requires_grad = False  # Freeze LSTM parameters

        self.convlstm = ConvLSTM(input_dim=3,
                     hidden_dim=32,
                     kernel_size=(3,3),
                     num_layers=1,
                     batch_first=True,
                     bias=True,
                     return_all_layers=False)

        self.convlstm.load_state_dict(torch.load('ConvLSTM_model/model.pt'))
        for param in self.convlstm.parameters():
            param.requires_grad = False  # Freeze ConvLSTM parameters

        self.fc = nn.Linear(6, 3)

    def forward(self, x):

        x_0 = self.lstm(x[0])
        x_1, _ = self.convlstm(x[1])
        x_ = torch.concat((x_0, x_1), dim=1)

        output = self.fc(x_)

        return output

if __name__ == '__main__':
    x = (torch.rand((1,10,42,3)), torch.rand((1,5,3,224,224)))
    model = Combined()
    print(model)
    print(model(x))
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualBranch(nn.Module):
    def __init__(self):
        super(VisualBranch, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = 256 * 14 * 14

        self.fc1 = nn.Linear(self.flatten_size, 256)

    def forward(self, x):
        batch_size, channels, time_steps, height, width = x.shape
        x = x.view(batch_size * time_steps, channels, height, width)

        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool4(F.relu(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        x = x.view(batch_size, time_steps, -1)
        x = torch.mean(x, dim=1)

        return x

class SignalBranch(nn.Module):
    def __init__(self):
        super(SignalBranch, self).__init__()
        
        # Adjust input channels to match the input size (400 instead of 200)
        self.conv1 = nn.Conv1d(400, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Adjust the input to the linear layer (new flattened size)
        # After 2 maxpooling layers (stride=2), the 5 timestep dimension will be reduced
        self.fc1 = nn.Linear(128 * 1, 256)  # Flattened feature size

    def forward(self, x):
        # Apply convolution and pooling layers
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))

        # Flatten the tensor for fully connected layer
        x = x.view(x.size(0), -1)

        # Apply fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        return x

class DualCNNFusion(nn.Module):
    def __init__(self):
        super(DualCNNFusion, self).__init__()
        self.visual_branch = VisualBranch()
        self.signal_branch = SignalBranch()

        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)

        self.fc_fusion = nn.Linear(128, 3)

    def forward(self, visual_input, signal_input):
        visual_features = self.visual_branch(visual_input)
        signal_features = self.signal_branch(signal_input)

        combined_features = torch.cat((visual_features.unsqueeze(1), signal_features.unsqueeze(1)), dim=2)

        lstm_out, _ = self.lstm1(combined_features)
        lstm_out, _ = self.lstm2(lstm_out)

        lstm_out = lstm_out[:, -1, :]

        output = F.softmax(self.fc_fusion(lstm_out), dim=1)

        return output

# model = DualCNNFusion()

# visual_input = torch.randn(16, 3, 16, 224, 224)
# signal_input = torch.randn(16, 400, 5)

# output = model(visual_input, signal_input)
# print(output.shape)

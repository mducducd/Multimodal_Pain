import torch
import torch.nn as nn

class DualBranchBiLSTM(nn.Module):
    def __init__(self, signal_input_dim, lstm_hidden_dim, num_layers, alpha_init=0.5):
        super(DualBranchBiLSTM, self).__init__()

        # Define the CNN for visual input
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 128, stride=1),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512)  # Output feature size of 512
        )

        # Bi-LSTM for visual input
        self.visual_lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_dim, 
                                   num_layers=num_layers, bidirectional=True, batch_first=True)
        self.visual_attention = self.AttentionLayer(lstm_hidden_dim * 2, lstm_hidden_dim)

        # Bi-LSTM for signal input
        self.signal_lstm = nn.LSTM(input_size=signal_input_dim, hidden_size=lstm_hidden_dim, 
                                   num_layers=num_layers, bidirectional=True, batch_first=True)
        self.signal_attention = self.AttentionLayer(lstm_hidden_dim * 2, lstm_hidden_dim)

        # Learnable parameter alpha for fusion
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Final fully connected layer after fusion
        self.fc = nn.Linear(lstm_hidden_dim * 2, 3)  # Assuming 3 classes for classification

    def _make_layer(self, in_channels, out_channels, stride):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        return nn.Sequential(*layers)

    class AttentionLayer(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super().__init__()
            self.attention = nn.Linear(input_dim, hidden_dim)
            self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, lstm_output):
            attention_weights = torch.tanh(self.attention(lstm_output))
            attention_weights = self.context_vector(attention_weights).squeeze(-1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            weighted_output = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
            return weighted_output, attention_weights

    def forward(self, visual_input, signal_input):
        B, C, T, H, W = visual_input.shape
        
        # Extract features for each frame using the CNN
        cnn_features = []
        for t in range(T):
            frame = visual_input[:, :, t, :, :]  # (B, C, H, W)
            feature = self.cnn(frame)
            cnn_features.append(feature)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # (B, T, cnn_feature_dim)
        
        # Visual branch
        visual_lstm_out, _ = self.visual_lstm(cnn_features)
        visual_attention_out, _ = self.visual_attention(visual_lstm_out)

        # Signal branch
        signal_lstm_out, _ = self.signal_lstm(signal_input)
        signal_attention_out, _ = self.signal_attention(signal_lstm_out)

        # Fusion
        fusion_out = self.alpha * visual_attention_out + (1 - self.alpha) * signal_attention_out

        # Final classification
        output = self.fc(fusion_out)
        return output

# Example usage
B = 8  # Batch size
T = 16  # Number of frames
visual_input = torch.randn(B, 3, T, 224, 224)
signal_input = torch.randn(B, 400, 5)

# Instantiate the dual-branch Bi-LSTM model
model = DualBranchBiLSTM(signal_input_dim=5, lstm_hidden_dim=128, num_layers=4)
output = model(visual_input, signal_input)
print(output.shape)  # Expected output shape: (B, 3)

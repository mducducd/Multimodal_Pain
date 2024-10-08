import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBottleneck(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, num_classes=3, attention_dim=128):
        super(AttentionBottleneck, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.attention_dim = attention_dim

        # Attention layers to project the inputs to a common attention dimension
        self.attention_query1 = nn.Linear(input_dim1, attention_dim)
        self.attention_key1 = nn.Linear(input_dim1, attention_dim)
        self.attention_value1 = nn.Linear(input_dim1, attention_dim)

        self.attention_query2 = nn.Linear(input_dim2, attention_dim)
        self.attention_key2 = nn.Linear(input_dim2, attention_dim)
        self.attention_value2 = nn.Linear(input_dim2, attention_dim)

        # Linear layer to process the concatenated attention outputs
        self.fc1 = nn.Linear(attention_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def attention(self, query, key, value):
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_dim, dtype=torch.float32))
        weights = F.softmax(scores, dim=-1)

        # Apply attention weights
        attended = torch.matmul(weights, value)
        return attended

    def forward(self, output1, output2):
        # Assuming output1 shape: (B, D1) and output2 shape: (B, D2)
        B, D1 = output1.shape
        print('dddddddddddddddd', output2.shape)
        B, D2 = output2.shape

        # Compute attention for output1
        query1 = self.attention_query1(output1).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        key1 = self.attention_key1(output1).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        value1 = self.attention_value1(output1).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        attended_output1 = self.attention(query1, key1, value1).squeeze(1)  # Shape: (B, attention_dim)

        # Compute attention for output2
        query2 = self.attention_query2(output2).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        key2 = self.attention_key2(output2).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        value2 = self.attention_value2(output2).unsqueeze(1)  # Shape: (B, 1, attention_dim)
        attended_output2 = self.attention(query2, key2, value2).squeeze(1)  # Shape: (B, attention_dim)

        # Concatenate the attended features
        combined_features = torch.cat((attended_output1, attended_output2), dim=1)  # Shape: (B, 2*attention_dim)

        # Pass through a feedforward network
        x = F.relu(self.fc1(combined_features))
        logits = self.fc2(x)  # Shape: (B, num_classes)

        return logits

# Example usage:
B, D1, D2 = 32, 64, 128  # Example dimensions
hidden_dim = 256
attention_dim = 128
output1 = torch.rand(B, D1)  # Example output1
output2 = torch.rand(B, D2)  # Example output2

model = AttentionBottleneck(input_dim1=D1, input_dim2=D2, hidden_dim=hidden_dim, num_classes=3, attention_dim=attention_dim)
logits = model(output1, output2)

print("Logits shape:", logits.shape)  # Should be (B, 3)
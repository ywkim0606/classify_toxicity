import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, voc_size, emb_dim=10, hid_dim=10, conv_out=30, class_size=4):
        
        super(CNN_LSTM, self).__init__()
        
        self.emb_dim = emb_dim
        self.voc_size = voc_size
        self.conv_out = conv_out
        self.hid_dim = hid_dim
        self.class_size = class_size
        
        self.embedding = torch.nn.Embedding(num_embeddings=self.voc_size, embedding_dim=self.emb_dim)
        self.conv_layer = nn.Conv1d(self.emb_dim, self.conv_out, kernel_size=3)
        self.lstm = nn.LSTM(23, hid_dim, num_layers = 1, bias=True, batch_first = True)
        self.linear = nn.Linear(self.hid_dim, class_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3, stride=2)

    def forward(self, X):
        emb_x = self.embedding(X)
        emb_x = torch.permute(emb_x.view(emb_x.shape[0], emb_x.shape[1], emb_x.shape[2]), (0, 2, 1))
        conv_x = self.conv_layer(emb_x)
        maxpool_x = self.maxpool(conv_x)
        dropout_x = self.dropout(maxpool_x)
        relu_x = self.relu(dropout_x)
        lstm_x, (hn, cn) = self.lstm(relu_x)
        linear_x = self.linear(lstm_x)
        return linear_x

if __name__ == '__main__':
    X = torch.randint(10, (16, 50))
    cnn_lstm = CNN_LSTM(voc_size=300)
    out = cnn_lstm(X)
    assert out.shape == torch.Size([16,30,4])

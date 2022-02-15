import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialNet(nn.Module):
    def __init__(self, EEGchannels, EEGsamples) -> None:
        super().__init__()
        self.spatial_conv = nn.Conv2d(1, 1, kernel_size=(1,EEGchannels), padding= "same")
        self.spatial_maxpool = nn.MaxPool2d((1, EEGsamples))

        self.spatial_fc = nn.Linear(EEGchannels, 8)
        self.spatial_out = nn.Linear(8, EEGchannels)

        self.out_conv = nn.Conv2d(1, 1, kernel_size=(10,EEGchannels))
        self.out_maxpool = nn.MaxPool2d(4)
    
    def forward(self, x):

        #Getting a mask of the spatial features
        spa = F.elu(self.spatial_conv(x))
        spa = self.spatial_maxpool(spa)
        spa = spa.view(spa.size(0), -1)
        spa = F.elu(self.spatial_fc(spa))
        spa = self.spatial_out(spa)
        spa = spa.unsqueeze(2).expand(-1,-1,x.shape[-1]).unsqueeze(1)
        masked = x * spa
        #Block output
        out = torch.tanh(self.out_conv(masked))
        out = self.out_maxpool(out)
        return out, spa
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads = 1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output, scores

    def forward(self, q, k, v, mask=None, dropout=False):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores, attention_weights = self.attention(
        q, k, v, self.d_k, mask, self.dropout if dropout else None)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output, attention_weights    


class CompleteNet(nn.Module):
    def __init__(self, EEGchannels, EEGsamples) -> None:
        super().__init__()
        self.EEGSamples = EEGsamples
        self.EEGChannels = EEGchannels
        self.spatial_net = SpatialNet(EEGchannels, EEGsamples)
        spatial_shapes = self.compute_spatial_shapes()
        self.attention_net = MultiHeadAttention(d_model=spatial_shapes[-1])
        self.fc = nn.Linear(spatial_shapes[-2] * spatial_shapes[-1], 1)

    def compute_spatial_shapes(self):
        test = torch.rand(1,1,self.EEGChannels,self.EEGSamples)
        test = self.spatial_net(test)
        return test[0].shape

    def forward(self, x):
        spatial_features, spatial_weights = self.spatial_net(x)
        output, attention_weights = self.attention_net(spatial_features, spatial_features, spatial_features)
        output = self.fc(torch.flatten(output,1))
        return output, attention_weights, spatial_weights
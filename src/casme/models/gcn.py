import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(GraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_nodes = num_nodes
        
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.A = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.A, 1e-4)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (B, C, T, V)
        B, C, T, V = x.size()
        
        # apply adjacency: x_A = x * A -> (B, C, T, V)
        # x is (B, C, T, V), A is (V, V)
        x_A = torch.einsum('bctv,vw->bctw', x, self.A)
        
        # apply weight: x_W = W * x_A -> (B, out_C, T, V)
        # weight is (C, out_C)
        out = torch.einsum('bctv,co->botv', x_A, self.weight) + self.bias.view(1, -1, 1, 1)
        return out


class STGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, residual=True):
        super(STGCN_Block, self).__init__()
        
        self.gcn = GraphConvolution(in_channels, out_channels, num_nodes)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), stride=(stride, 1), padding=(4, 0)),
            nn.BatchNorm2d(out_channels),
        )
        
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, num_nodes=478, dropout=0.5):
        super(STGCN, self).__init__()
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        
        self.layers = nn.ModuleList([
            STGCN_Block(in_channels, 64, num_nodes, residual=False),
            STGCN_Block(64, 64, num_nodes),
            STGCN_Block(64, 128, num_nodes, stride=2),
            STGCN_Block(128, 128, num_nodes),
            STGCN_Block(128, 256, num_nodes, stride=2),
            STGCN_Block(256, 256, num_nodes)
        ])
        
        self.fc = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T, V) or maybe dict with pixel_values?
        # we will expect (B, C, T, V) where V is num_nodes
        if x.dim() == 5:
            # wait, if input is (B, C, T, H, W) where H*W = V?
            pass
            
        B, C, T, V = x.size()
        
        # BN
        x = x.permute(0, 1, 3, 2).contiguous() # (B, C, V, T)
        x = x.view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous() # (B, C, T, V)
        
        for layer in self.layers:
            x = layer(x)
            
        # Global pooling over T and V
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(B, -1)
        
        x = self.dropout(x)
        out = self.fc(x)
        
        class Output:
            def __init__(self, logits):
                self.logits = logits
        return Output(out)

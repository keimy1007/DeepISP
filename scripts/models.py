import torch
import torch.nn as nn
import torch.nn.init as init


class LinearWithSkip(nn.Module):
    def __init__(self, dim):
        super(LinearWithSkip, self).__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear(x) + x


class ISP2HFAModel(nn.Module):
    def __init__(self, hidden_dims=[64, 64], dropout_rate=0.1):
        super(ISP2HFAModel, self).__init__()
        self.hidden_dims = hidden_dims

        last_dim = 29
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            if hidden_dim == last_dim:
                self.layers.append(LinearWithSkip(hidden_dim))
            else:
                self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        # task specific layers
        self.fc_y_pattern = nn.Linear(last_dim, 52 * 5)
        self.fc_y_total = nn.Linear(last_dim, 52 * 5)
        self.fc_y_md = nn.Linear(last_dim, 1)
        self.fc_y_psd = nn.Linear(last_dim, 1)
        self.fc_y_vfi = nn.Linear(last_dim, 1)
        self.fc_y_ght = nn.Linear(last_dim, 1)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # common layers
        for layer in self.layers:
            x = layer(x)
        # task_specific_layers
        y_pattern = self.fc_y_pattern(x)
        y_total = self.fc_y_total(x)
        y_md = self.fc_y_md(x)
        y_psd = self.fc_y_psd(x)
        y_vfi = self.fc_y_vfi(x)
        y_ght = self.fc_y_ght(x)
        return y_pattern, y_total, y_md, y_psd, y_vfi, y_ght


class ISP2HFAprogModel(nn.Module):
    def __init__(self, hidden_dims=[16, 16], dropout_rate=0.1):
        super(ISP2HFAprogModel, self).__init__()
        self.hidden_dims = hidden_dims

        last_dim = 29
        self.layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            if hidden_dim == last_dim:
                self.layers.append(LinearWithSkip(hidden_dim))
            else:
                self.layers.append(nn.Linear(last_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        # task specific layers
        self.fc_y_mdslope = nn.Linear(last_dim, 1)
        self.fc_y_vfislope = nn.Linear(last_dim, 1)
        self.fc_y_mdprog = nn.Linear(last_dim, 1)
        self.fc_y_vfiprog = nn.Linear(last_dim, 1)

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # common layers
        for layer in self.layers:
            x = layer(x)
        # task_specific_layers
        y_mdslope = self.fc_y_mdslope(x)
        y_vfislope = self.fc_y_vfislope(x)
        y_mdprog = self.fc_y_mdprog(x)
        y_vfiprog = self.fc_y_vfiprog(x)
        return y_mdslope, y_vfislope, y_mdprog, y_vfiprog
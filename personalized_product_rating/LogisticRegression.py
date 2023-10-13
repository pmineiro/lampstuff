import torch
        
class MLP(torch.nn.Module):
    @staticmethod
    def new_gelu(x):
        import math
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def __init__(self, dim):
        super().__init__()
        self.c_fc    = torch.nn.Linear(dim, 4 * dim)
        self.c_proj  = torch.nn.Linear(4 * dim, dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = MLP(dim)

    def forward(self, x):
        return x + self.layer(x)
        
class ResidualLogisticRegressor(torch.nn.Module):
    def __init__(self, in_features, out_features, depth, *, model_id = None):
        import parameterfree
        super().__init__()
        self._in_features = in_features
        self._depth = depth
        self.blocks = torch.nn.Sequential(*[ Block(in_features) for _ in range(depth) ])
        self.linear  = torch.nn.Linear(in_features=in_features, out_features=out_features)
        with torch.no_grad():
            if model_id:
                state_dict = torch.load(f'{model_id}/logistic.pth')
                self.load_state_dict(state_dict)
        self.optim = parameterfree.COCOB(self.parameters())

    def forward(self, X):
        import torch.nn.functional as F
        return F.log_softmax(self.linear(self.blocks(X)), dim=-1)

    def predict(self, X):
        self.eval()
        return self(X)

    def learn(self, X, Y):
        import torch.nn.functional as F
        self.train()
        self.optim.zero_grad()
        output = self(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        self.optim.step()
        return loss.item()

    def save_pretrained(self, model_id):
        torch.save(self.state_dict(), f'{model_id}/logistic.pth')

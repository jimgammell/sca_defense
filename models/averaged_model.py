import torch
from torch import nn, optim

def get_averaged_model(model_class, device, dist_fn=lambda x1, x2: (x1-x2).norm(p=2), avg_fn=lambda x1, x2: 0.9999*x1+0.0001*x2):
    class AveragedModel(model_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.dist_fn = dist_fn
            self.avg_fn = avg_fn
            self.averaged_parameters = [param.detach().clone().to(device) for param in self.parameters() if param.requires_grad]
            for param in self.averaged_parameters:
                param.requires_grad = False

        def get_avg_departure_penalty(self):
            penalty = torch.tensor(0.0, requires_grad=True, device=self.averaged_parameters[0].device, dtype=torch.float)
            for p1, p2 in zip([p for p in self.parameters() if p.requires_grad], self.averaged_parameters):
                penalty = penalty + self.dist_fn(p1, p2)
            return penalty

        @torch.no_grad()
        def update_avg(self):
            for (idx, p1), p2 in zip(enumerate(self.averaged_parameters), [p for p in self.parameters() if p.requires_grad]):
                self.averaged_parameters[idx] = self.avg_fn(p1, p2)
        
        @torch.no_grad()
        def reset_avg(self):
            for (idx, p1), p2 in zip(enumerate(self.averaged_parameters), [p for p in self.parameters() if p.requires_grad]):
                self.averaged_parameters[idx] = p2
    
    return AveragedModel
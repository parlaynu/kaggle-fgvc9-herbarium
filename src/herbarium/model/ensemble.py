import torch
import torch.nn as nn


reducers = {
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.amax,
    "none": None
}


class Ensemble(nn.Module):
    def __init__(self, num_categories, models, reducer):
        super().__init__()

        self._models = models
        self._reducer = reducers[reducer]
        
        # freeze the model layers
        for model in self._models:
            model.eval()
            model.requires_grad_(False)
        
        # a dummy classifier for the module
        self.classifier = nn.Linear(in_features=10, out_features=2)

    def to(self, device):
        for idx, model in enumerate(self._models):
            self._models[idx] = model.to(device)
        return super().to(device)
    
    def forward(self, inputs):
        # collect outputs from all the models
        outputs = []
        for model in self._models:
            outputs.append(model(inputs))

        # stack for dimensions: BATCH_SIZE, NMODELS, NCATS
        outputs = torch.stack(outputs, dim=1)
        if self._reducer:
            outputs = self._reducer(outputs, dim=1)
        
        return outputs


def ensemble(num_categories, models, *, reducer="mean", use_gpu=False):
    model = Ensemble(num_categories, models, reducer=reducer)
    
    device = torch.device('cuda') if (use_gpu and torch.cuda.is_available()) else torch.device('cpu')
    model = model.to(device)

    # set extra attributes on the model
    model.fullname = "herbarium.model.ensemble"
    model.device = device
    model.num_outputs = num_categories

    return model


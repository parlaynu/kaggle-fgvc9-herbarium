import torch
import torch.optim

def AdamW(model, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False, *, debug=False):
    
    if hasattr(model, "named_param_groups"):
        groups = []
        for n, p in model.named_param_groups():
            if debug:
                print(f"- {n}: lr={lr:.2e}")
            
            groups.append({
                "params": p,
            })
    
        return torch.optim.AdamW(groups, lr, betas, eps, weight_decay, amsgrad)

    else:
        return torch.optim.AdamW(model.parameters(), lr, betas, eps, weight_decay, amsgrad)


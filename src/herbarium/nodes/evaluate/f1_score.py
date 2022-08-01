from collections import defaultdict

import torch

from ..node import Node


class F1Score(Node):
    
    def __init__(self, inode, num_categories):
        super().__init__(inode)
        self._num_categories = num_categories
        self.reset()
    
    def reset(self):
        # TP, FN+FP
        self._data = None
        
    def __len__(self):
        return len(self.inode)
    
    def __iter__(self):
        self.reset()
        
        for idx, item in enumerate(self.inode):
            targets = item['target']
            outputs = torch.argmax(item['output'], dim=1)

            correct = (targets == outputs).long()
            
            if self._data is None:
                self._data = torch.zeros(2, self._num_categories, device=targets.device)
            
            self._data[0, targets] += correct
            self._data[1, targets] += 1.0 - correct
            self._data[1, outputs] += 1.0 - correct

            yield item
        
        f1_scores = 2*self._data[0] / (2*self._data[0] + self._data[1] + 1e-9)
        item['metrics']['f1_data'] = self._data
        item['metrics']['f1_scores'] = f1_scores
        item['metrics']['f1_score'] = f1_scores.mean().item()
        

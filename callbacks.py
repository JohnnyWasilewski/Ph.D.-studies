from pl_bolts.callbacks.printing import dicts_to_table
import pytorch_lightning as pl
  
import copy
from itertools import zip_longest
from typing import Any, Callable, Dict, List, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_info


class PrintTableMetricsCallback2(Callback):
    def __init__(self, display_every_n_epochs) -> None:
        self.metrics: List = []
        self.iter = 0
        self.display_every_n_epochs = display_every_n_epochs
            
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:        
        if self.iter % (2  * self.display_every_n_epochs) == 0:
            metrics_dict = copy.copy(trainer.callback_metrics)
            rank_zero_info(dicts_to_table([metrics_dict]))            
        self.iter += 1
        
class CheckBatchGradient(pl.Callback):
    
    def on_train_start(self, trainer, model):
        n = 0

        example_input = model.example_input_array.to(model.device)
        example_input.requires_grad = True

        model.zero_grad()
        output = model(example_input)
        output[n].abs().sum().backward()
        
        zero_grad_inds = list(range(example_input.size(0)))
        zero_grad_inds.pop(n)
        
        if example_input.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")
"""
@file earlystopper.py
@brief Training Callback for Overfitting Prevention.

This module implements the Early Stopping mechanism used during the "Fine-tune Parameters"
and "Retrain Model Parameters" stages. It monitors a specific metric (e.g., validation loss)
and terminates the training loop if improvement stalls, saving computational resources
and preventing the model from overfitting to the training set.
"""

from typing import Dict, List

from .sequential import Sequential

class EarlyStopper:
    """
    Callback to stop training when a monitored metric has stopped improving.
    """
    def __init__(
        self, 
        monitor_metric: str,
        delta:float = 0.0,
        patience:int = 0,
        mode:str = "min",
        restore_best_state_dict:bool = False
    ):
        """
        Initialize the EarlyStopper.

        Args:
            monitor: Name of the metric to monitor (e.g., 'validation_loss', 'validation_accuracy').
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            patience: Number of epochs with no improvement after which training will be stopped.
            mode: One of {'min', 'max'}. 
                  - 'min': Training stops when the quantity monitored has stopped decreasing.
                  - 'max': Training stops when the quantity monitored has stopped increasing.
            restore_best_weights: Whether to restore model weights from the epoch with the best value.
        """
        self.monitor_metric = monitor_metric
        self.delta = delta
        self.patience = patience

        assert mode == "min" or mode == "max", "Mode must be either \"min\" or \"max\""
        self.mode = mode
        self.restore_best_state_dict = restore_best_state_dict

        self.best_metric_val = None
        self.best_epoch = 0
        self.best_state_dict = None


    def __call__(
            self,
            model:Sequential,
            history:Dict[str, List[float]],
            epoch:int
    ):
        """
        Checks whether training should stop.
        
        Called at the end of every epoch by the Sequential.fit() loop.

        Args:
            model: The model instance being trained.
            history: Dictionary of training metrics history.
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        assert self.monitor_metric in history.keys(), f"Metric {self.monitor_metric} is not the model training dictionary, {history.keys()}!"

        if self.best_metric_val is None or \
           (self.mode == "min" and (self.best_metric_val - history[self.monitor_metric][-1]) > self.delta) or\
           (self.mode == "max" and (history[self.monitor_metric][-1] - self.best_metric_val) > self.delta):
                self.best_metric_val = history[self.monitor_metric][-1]
                self.best_epoch = epoch

                if self.restore_best_state_dict: 
                    self.best_state_dict = model.state_dict()
                return False
        
        if epoch - self.best_epoch >= self.patience:
            print(f"Stopping Training of {model.__class__.__name__} with at {self.best_epoch} epoch with best {self.monitor_metric} = {self.best_metric_val}")
            
            # Restore best weights if configured
            if self.restore_best_state_dict and self.best_state_dict is not None:
                model.load_state_dict(self.best_state_dict, strict=True)
            return True
            
        return False
            
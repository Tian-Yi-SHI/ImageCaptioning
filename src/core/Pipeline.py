from abc import ABC, abstractmethod
from typing import Optional, Any

class Pipeline:
    '''
    Base class for image captioning task pipelines.
    Defines the skeleton of training and testing workflows, with concrete implementations deferred to subclasses.
    '''

    def __init__(self):
        self.model = None

        self.training_params = None

        self.is_model_ready = False
        self.is_params_ready = False

    @abstractmethod
    def build_model(self, model: Optional[Any] = None) -> Any:
        """
        model construction or load
        Subclasses must call self._mark_model_ready() after successful initialization.

        Args:
            model: trained model(optional)
        Return: model constructed
        """
        raise NotImplementedError("function build_model not implemented")
    
    def _mark_model_ready(self):
        """Mark model as ready (called by subclasses to avoid external modification)"""
        if self.model is None:
            raise RuntimeError("Cannot mark model as ready - model has not been initialized")
        self.is_model_ready = True

    def set_training_params(self, batch_size: int, n_epoch: int, lr: float, wd: float = 0.0):
        """
        set and check training params
        Args:
            batch_size: 
            n_epoch: number of training epoch
            lr: learning rate
            wd: weight decay
        """
        # validation
        param_checks = [
            (isinstance(batch_size, int) and batch_size > 0, 
             f"Invalid batch_size {batch_size}: must be a positive integer"),
            (isinstance(n_epoch, int) and n_epoch > 0, 
             f"Invalid n_epoch {n_epoch}: must be a positive integer"),
            (isinstance(lr, (int, float)) and lr > 0, 
             f"Invalid learning rate {lr}: must be a positive number"),
            (isinstance(wd, (int, float)) and wd >= 0, 
             f"Invalid weight decay {wd}: cannot be negative")
        ]
        for is_valid, err_msg in param_checks:
            if not is_valid:
                raise ValueError(err_msg)

        # save params
        self.training_params = {
            "batch_size": batch_size,
            "n_epoch": n_epoch,
            "lr": lr,
            "wd": wd
        }
        self.is_params_ready = True

    def train(self, train_loader: Any, val_loader: Optional[Any] = None) -> dict:
        """
        Defines training workflow skeleton.
        Subclasses implement specific training logic in _train_core.
        
        Args:
            train_loader: Data loader for training data
            val_loader: Optional data loader for validation data
        Return: Dictionary of training results
        """
        # validation
        self._check_train_ready()

        # core training logic
        return self._train_core(train_loader, val_loader)

    @abstractmethod
    def _train_core(self, train_loader: Any, val_loader: Optional[Any] = None) -> dict:
        """
        Subclasses implement specific training logic here.
        
        Args:
            train_loader: Data loader for training data
            val_loader: Optional data loader for validation data
        Return: Dictionary of training results
        """
        raise NotImplementedError("Subclasses must implement the _train_core method")

    def test(self, test_loader: Any) -> dict:
        """
        Defines testing workflow skeleton.
        Subclasses implement specific testing logic in _test_core.
        
        Args:
            test_loader: Data loader for test data
        Return: Dictionary of test results (e.g., evaluation metrics)
        """
        # validation
        self._check_test_ready()

        # core testing logic
        return self._test_core(test_loader)

    @abstractmethod
    def _test_core(self, test_loader: Any) -> dict:
        """
        Subclasses implement specific testing logic here.
        
        Args:
            test_loader: Data loader for test data
        Return: Dictionary of test results
        """
        raise NotImplementedError("Subclasses must implement the _test_core method")

    def _check_train_ready(self):
        """Validate all prerequisites for training"""
        if not self.is_model_ready:
            raise RuntimeError("Cannot start training - model not ready. Call build_model first and ensure proper initialization.")
        if not self.is_params_ready:
            raise RuntimeError("Cannot start training - parameters not set. Call set_training_params first with valid values.")

    def _check_test_ready(self):
        """Validate all prerequisites for testing"""
        if not self.is_model_ready:
            raise RuntimeError("Cannot start testing - model not ready. Call build_model first and ensure proper initialization.")
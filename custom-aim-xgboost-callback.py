# modeling/custom_xgboost_aim_callback.py

from aim.sdk.run import Run
from aim.ext.resource.configs import DEFAULT_SYSTEM_TRACKING_INT
from aim.xgboost import AimCallback
from typing import Optional, Dict, Any, Tuple, List
import xgboost as xgb


class CustomAimCallback(AimCallback):
    """
    Custom AimCallback for XGBoost to track and log all evaluation metrics,
    including custom metrics, to AimStack during training.

    Attributes:
        repo_path (Optional[str]): Path to the Aim repository.
        experiment (Optional[str]): Name of the Aim experiment.
        system_tracking_interval (Optional[int]): Interval for system metrics tracking.
        log_system_params (Optional[bool]): Whether to log system parameters.
        capture_terminal_logs (Optional[bool]): Whether to capture terminal logs.
        run (Run): Aim Run instance for logging metrics.
        parameters (Optional[Dict[str, Any]]): Hyperparameters and other run parameters.
        tags (Optional[List[str]]): Tags for categorizing the run.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment: Optional[str] = None,
        system_tracking_interval: Optional[int] = DEFAULT_SYSTEM_TRACKING_INT,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ):
        """
        Initializes the CustomAimCallback.

        Args:
            repo (Optional[str]): Path to the Aim repository. Defaults to None.
            experiment (Optional[str]): Name of the Aim experiment. Defaults to None.
            system_tracking_interval (Optional[int]): Interval for system metrics tracking. Defaults to DEFAULT_SYSTEM_TRACKING_INT.
            log_system_params (Optional[bool]): Whether to log system parameters. Defaults to True.
            capture_terminal_logs (Optional[bool]): Whether to capture terminal logs. Defaults to True.
            parameters (Optional[Dict[str, Any]]): Hyperparameters and other run parameters. Defaults to None.
            tags (Optional[List[str]]): Tags for categorizing the run. Defaults to None.

        Raises:
            ValueError: If the Aim Run initialization fails.
        """
        super().__init__()
        self.repo_path = repo
        self._experiment = experiment
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params
        self.capture_terminal_logs = capture_terminal_logs
        self.parameters = parameters or {}
        self.tags = tags or []

        # Initialize Aim Run
        try:
            self.run = Run(
                repo=self.repo_path,
                experiment=self._experiment,
                system_tracking_interval=self.system_tracking_interval,
                log_system_params=self.log_system_params,
                capture_terminal_logs=self.capture_terminal_logs,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Aim Run: {e}")

        # Log run parameters
        if self.parameters:
            print(f"Trying to log these params: {self.parameters}")
            for param, value in self.parameters.items():
                self.run[param] = value

        # Add tags to the run
        if self.tags:
            for tag in self.tags:
                self.run.add_tag(tag)

    def after_iteration(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Dict[str, Any]]) -> bool:
        """
        Callback function executed after each training iteration.

        Args:
            model (xgb.Booster): The XGBoost model.
            epoch (int): Current iteration number.
            evals_log (Dict[str, Dict[str, Any]]): Evaluation metrics logged by XGBoost.

        Returns:
            bool: Whether to stop training. Always returns False to continue training.
        """
        if not evals_log:
            return False  # Continue training if no evaluation metrics are present

        for dataset_name, metrics in evals_log.items():
            for metric_name, metric_values in metrics.items():
                if not metric_values:
                    continue  # Skip if no metric values are present

                latest_value = metric_values[-1]

                if isinstance(latest_value, tuple):
                    # Metric with standard deviation (e.g., 'metric', ('value', 'std'))
                    value, stdv = latest_value
                    # Log the main metric
                    self.run.track(
                        value,
                        name=metric_name,
                        step=epoch,
                        context={"dataset": dataset_name, "stdv": False},
                    )
                    # Log the standard deviation as a separate metric
                    stdv_metric_name = f"{metric_name}_stdv"
                    self.run.track(
                        stdv,
                        name=stdv_metric_name,
                        step=epoch,
                        context={"dataset": dataset_name, "stdv": True},
                    )
                else:
                    # Standard metric without standard deviation
                    self.run.track(
                        latest_value,
                        name=metric_name,
                        step=epoch,
                        context={"dataset": dataset_name},
                    )

        return False  # Continue training

    def end_training(self, model: xgb.Booster, evals_log: Dict[str, Dict[str, Any]]) -> None:
        """
        Callback function executed at the end of training.

        Args:
            model (xgb.Booster): The XGBoost model.
            evals_log (Dict[str, Dict[str, Any]]): Evaluation metrics logged by XGBoost.
        """
        # Optionally, perform any final logging or cleanup here
        self.run.close()

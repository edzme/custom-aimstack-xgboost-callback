# custom-aimstack-xgboost-callback

### Custom Aim / AimStack callback for xgboost.

- Sets the hyperparams as run params
- Logs all custom eval metrics
- Captures terminal logs
- Logs system params (cpu, gpu, etc)
- Fixes the epoch = 0 issue in the default aimxgboost callback


#### Example implementation using xgboost's native api:

```python
from custom_xgboost_aim_callback import CustomAimCallback

aim_server_host = os.environ.get("AIM_SERVER_HOST")
aim_server_port = os.environ.get("AIM_SERVER_PORT")
aim_host_path = f"aim://{aim_server_host}:{aim_server_port}"

aim_experiment_name = 'test'

callbacks = []

# **Add AimCallback for experiment tracking**
aim_callback = CustomAimCallback(
    repo=aim_host_path, experiment=aim_experiment_name, parameters=hyperparams
)
callbacks.append(aim_callback)
log.debug(f"Added AimCallback for experiment '{aim_experiment_name}'.")

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=max_boost_round,
    evals=evals,
    custom_metric=custom_evals,
    callbacks=callbacks,
    verbose_eval=True,
)
```
import os
import pandas as pd
from multiprocessing import Pool
from scripts.pipeline import Pipeline


def process_method(method_type_method):
    method_type, method = method_type_method
    config_path = os.path.join(config_dir, method_type, method, f"{data}.yaml")
    pipeline = Pipeline(
        config_path=config_path, device="cpu", workspace=f"workspaces/{method}", seed=1
    )
    train_loss, test_loss = pipeline.train(metric="MAPE")
    print(f"{method_type}/{method}: {train_loss}, {test_loss}")
    return [method, train_loss, test_loss]


config_dir = "configs/baselines"
data = "mix"

method_type_methods = [
    (method_type, method)
    for method_type in os.listdir(config_dir)
    for method in os.listdir(os.path.join(config_dir, method_type))
]

if __name__ == "__main__":
    with Pool(4) as p:
        result = p.map(process_method, method_type_methods)

    res = pd.DataFrame(data=result, columns=["method", "train_MAPE", "test_MAPE"])
    res.to_csv(f"data/outputs/{data}.csv", index=False)

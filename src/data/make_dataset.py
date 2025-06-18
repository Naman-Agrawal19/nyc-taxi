
from yaml import safe_load
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

current_path = Path(__file__).parent.parent.parent.resolve()
config_path = current_path / "config.yml"
params_path = current_path / "params.yaml"

def load_config():
    with open(config_path, "r") as f:
        config = safe_load(f)
    return config

def load_params():
    with open(params_path, "r") as f:
        params = safe_load(f)
    return params

def raw_train_path():
    return Path(load_config()['raw_data']['train_data'])

def get_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def save_data(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)

def main():
    raw_path = raw_train_path()
    raw_train_df = get_data(raw_path)
    test_size = load_params()['make_dataset']['test_size']
    random_state = load_params()['make_dataset']['random_state']

    train_data, val_data = train_test_split(raw_train_df, test_size=test_size, random_state=random_state)

    save_data(train_data, Path(load_config()['interim_data']['train_data']))
    save_data(val_data, Path(load_config()['interim_data']['val_data']))

if __name__ == "__main__": 
    main()
    if Path(load_config()['interim_data']['train_data']).exists() and Path(load_config()['interim_data']['val_data']).exists():
        print("make dataset done")
    else:
        print("make dataset failed")
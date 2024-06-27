import os
import pandas as pd


def main():
    
    # this folder contains stateless cleaned data + stateful + feat eng
    dir_path = os.path.join("data", "dev", "06_stateful")
    
    df = pd.DataFrame()
    for file in os.listdir(dir_path):
        if file.endswith(".csv"):
            df = pd.concat([df, pd.read_csv(os.path.join(dir_path, file))])
    
    df.to_csv(os.path.join("data", "reference", "reference_data.csv"), index=False)
    print("Reference data saved.")
    print(f"Shape of reference data: {df.shape}")


if __name__ == '__main__':
    main()
import os
import pandas as pd
import numpy as np


def main():
    # loading original data
    original_data = pd.read_csv(
        os.path.join("data", "dev", "01_raw", "diabetic_data.csv")
    )

    # sampling data
    sampled_data = original_data.sample(
        frac=0.1,  # sampling 10%
        random_state=42)
    
    # removing the target column
    # as prod data should not have it
    sampled_data = sampled_data.drop(columns=["readmitted"])

    # additional transformations to show drift:
    # age column
    age_values = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
    sampled_data["age"] = np.random.choice(age_values, size=sampled_data.shape)

    # saving data
    sampled_data.to_csv(
        os.path.join("data", "prod", "01_raw", "example_prod_data.csv"),
        index=False
    )
    
    print("Original data shape:", original_data.shape)
    print("Sampled data shape:", sampled_data.shape)
    print("Data saved to data/prod/01_raw/example_prod_data.csv")
    

if __name__ == '__main__':
    main()
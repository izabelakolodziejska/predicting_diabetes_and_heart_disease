import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_data, fit_preprocess, split_data, transform, save_metadata_pickle

import pandas as pd


if __name__ == "__main__":
    print('Loading data..')
    df = load_data()
    
    # print(f"Liczba rekordów: {df.shape[0]}")
    # print(f"Liczba cech: {df.shape[1]}")


    print('Splitting data..')
    X_train_raw, X_val_raw, X_test_raw, y_diabetes_train, y_diabetes_val, y_diabetes_test, y_heart_train, y_heart_val, y_heart_test = split_data(df)

    print("Fitting preprocessor on training data..")
    metadata, cat_idxs, cat_dims = fit_preprocess(X_train_raw)
    save_metadata_pickle(cat_idxs, cat_dims)

    print("Transforming data..")
    X_train = transform(X_train_raw, metadata)
    X_val = transform(X_val_raw, metadata)
    X_test = transform(X_test_raw, metadata)


    print(f"Liczba rekordów: {X_train.shape[0]}")
    print(f"Liczba rekordów: {X_val.shape[0]}")
    print(f"Liczba rekordów: {X_test.shape[0]}")
    print(f"Liczba cech: {X_train.shape[1]}")
    print(f"Liczba cech: {X_val.shape[1]}")
    print(f"Liczba cech: {X_test.shape[1]}")
    

    print("Saving processed data..")
    dir = f'data/processed'
    os.makedirs(f'{dir}', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_val.to_csv('data/processed/X_val.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)

    y_diabetes_train.to_csv('data/processed/y_diabetes_train.csv', index=False)
    y_diabetes_val.to_csv('data/processed/y_diabetes_val.csv', index=False)
    y_diabetes_test.to_csv('data/processed/y_diabetes_test.csv', index=False)

    y_heart_train.to_csv('data/processed/y_heart_train.csv', index=False)
    y_heart_val.to_csv('data/processed/y_heart_val.csv', index=False)
    y_heart_test.to_csv('data/processed/y_heart_test.csv', index=False)

    print(f"prepare_data.py finished, train size: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}") 


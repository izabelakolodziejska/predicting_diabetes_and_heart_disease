import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

def load_data():
    #zdecydowałam się pobrać dodatkowe dane z innego roku, znacząco zwiekszyło to wyniki modelu
    raw_2023 = pd.read_sas("data/raw/LLCP2023.XPT", format="xport")

    raw_2024 = pd.read_sas("data/raw/LLCP2024.XPT", format="xport")

    filtered_24 = raw_2024[(raw_2024['_MICHD'] == 1) | (raw_2024['DIABETE4'] == 1)]

    common_columns = raw_2023.columns.intersection(filtered_24.columns).tolist()

    raw23 = raw_2023[common_columns]
    raw24 = filtered_24[common_columns]

    raw = pd.concat([raw23, raw24], ignore_index=True)   
    

    return raw

def fit_preprocess(X_train, target_columns=("DIABETE4", "_MICHD"), 
                   threshold=12, drop_null_threshold=0.50):

    features_all = [c for c in X_train.columns if c not in target_columns]
    X_raw = X_train[features_all].copy()


    #usunięcie kolumn gdzie ponad połowa rekordów to braki
    cols_to_drop = [c for c in X_raw.columns if X_raw[c].isnull().mean() > drop_null_threshold]
    X_raw = X_raw.drop(columns=cols_to_drop)
    features = list(X_raw.columns)

    #jeśli ilość odpowiedzi jest mniejsza niż 12 to klasyfikuję te cechy jako kategoryczne
    cat_cols = [c for c in features if X_raw[c].nunique(dropna=False) <= threshold]
    num_cols = [c for c in features if c not in cat_cols]

    cat_fill_values = {}
    for c in cat_cols:
        mode_val = X_raw[c].mode(dropna=True) #najczęstsza wartość
        fill_val = mode_val.iloc[0] if len(mode_val) else 0
        cat_fill_values[c] = fill_val

    num_fill_values = {}
    for c in num_cols:
        num_fill_values[c] = X_raw[c].median() #mediana dla cech numerycznych

    encoders = {}
    categorical_dims = {}
    for c in cat_cols:
        oe = OrdinalEncoder() #niestety w tym projekcie liczba cech jest za duza by je analizowac ręcznie, zdecydowałam się na razie na LabelEncoder dla cech kategorycznych ponieważ większość cech ma odpowiedzi w określonym porżadku (np kategorie BMI)
        filled_col = X_raw[c].fillna(cat_fill_values[c]).values.reshape(-1, 1)
        oe.fit(filled_col)
        encoders[c] = oe
        categorical_dims[c] = len(oe.categories_[0])

    scaler = None
    if len(num_cols) > 0:
        scaler = MinMaxScaler()
        filled_nums = X_raw[num_cols].fillna(pd.Series(num_fill_values))
        scaler.fit(filled_nums)

    cat_idxs = [i for i, f in enumerate(features) if f in cat_cols] #indeksy cech kategorycznych
    cat_dims = [categorical_dims[f] for f in features if f in cat_cols] #liczba unikalnych cech kategorycznych

    metadata = {
        "features": features,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "encoders": encoders,
        "categorical_dims": categorical_dims,
        "scaler": scaler,
        "cat_fill_values": cat_fill_values,
        "num_fill_values": num_fill_values,
        "target_columns": target_columns,
        "cols_dropped": cols_to_drop,
    }

    return metadata, cat_idxs, cat_dims


def transform(X, metadata):
    #faktyczne transformacje cech
    features = metadata["features"]
    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]
    encoders = metadata["encoders"]
    cat_fill_values = metadata["cat_fill_values"]
    num_fill_values = metadata["num_fill_values"]
    scaler = metadata["scaler"]

    X_raw = X[features].copy()

    for c in cat_cols:
        X_raw[c] = X_raw[c].fillna(cat_fill_values[c])
    for c in num_cols:
        X_raw[c] = X_raw[c].fillna(num_fill_values[c])

    for c in cat_cols:
        le = encoders[c]
        valid_categories = set(encoders[c].categories_[0])
        X_raw[c] = X_raw[c].apply(lambda x: x if x in valid_categories else cat_fill_values[c])
        X_raw[c] = le.transform(X_raw[c].values.reshape(-1, 1)).flatten()

    if scaler is not None and len(num_cols) > 0:
        X_raw[num_cols] = scaler.transform(X_raw[num_cols])

    return X_raw



def split_data(df, target_columns=['DIABETE4', '_MICHD'], test_size=0.15, val_size=0.1, random_state=0):
    df = df.drop(df[(df["DIABETE4"] == 2) | (df["DIABETE4"] == 4) | (df["DIABETE4"] == 5)].index)  #zostawiam tylko przypadki choroba lub nie w cukrzycy

    #kolumny która zawierają powtarzające się dane 
    columns_to_drop = [
    'CVDCRHD4', 'CVDINFR4', 'QSTVER', 'WTKG3', '_AGE65YR', 
    '_AGEG5YR', '_AGE_G', '_BMI5', 'CHILDREN', '_DUALUSE', 
    'EDUCA', '_RAWRAKE', '_RFCHOL3', '_SEX', 'FMONTH', 'DISPCODE', 'SAFETIME'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    y_diabetes = (df["DIABETE4"] == 1).astype(int) 
    y_heart = (df["_MICHD"] == 1).astype(int)

    y_combined = y_diabetes.astype(str) + "_" + y_heart.astype(str)

    X_temp, X_test, y_diabetes_temp, y_diabetes_test, y_heart_temp, y_heart_test = train_test_split(
            df, y_diabetes, y_heart,
            test_size=test_size,
            stratify=y_combined,
            random_state=random_state
        )
    
    y_combined_temp = y_diabetes_temp.astype(str) + "_" + y_heart_temp.astype(str)
    val_ratio = val_size / (1 - test_size)

    X_train, X_val, y_d_train, y_d_val, y_h_train, y_h_val = train_test_split(
            X_temp, y_diabetes_temp, y_heart_temp,
            test_size=val_ratio,
            stratify=y_combined_temp,
            random_state=random_state
        )
    
    return (X_train.reset_index(drop=True),
                X_val.reset_index(drop=True),
                X_test.reset_index(drop=True),
                y_d_train.reset_index(drop=True),
                y_d_val.reset_index(drop=True),
                y_diabetes_test.reset_index(drop=True),
                y_h_train.reset_index(drop=True),
                y_h_val.reset_index(drop=True),
                y_heart_test.reset_index(drop=True))
    




def save_metadata_pickle( cat_idxs, cat_dims, filepath='data/metadata'):
    #potrzebne do wydobycia cat_idxs, cat_dims później
    os.makedirs(filepath, exist_ok=True) 
     
    data = ( cat_idxs, cat_dims)
    with open(f'{filepath}/metadata.pkl', 'wb') as f:
        pickle.dump(data, f)






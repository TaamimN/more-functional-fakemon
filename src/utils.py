import pandas as pd
from config import ALL_TYPES

# convert types into multihot vector
def types_to_multihot(t1, t2):
    vec = [0] * len(ALL_TYPES)
    if pd.notna(t1):
        vec[ALL_TYPES.index(t1)] = 1
    if pd.notna(t2):
        vec[ALL_TYPES.index(t2)] = 1
    else:
        vec[ALL_TYPES.index("None")] = 1
    return vec

# add columns for dataframe in reference to csv format
def prepare_dataframe(df):
    df["type_vector"] = df.apply(
        lambda r: types_to_multihot(r["Type 1"], r["Type 2"]), 
        axis=1
    )
    df["stats"] = df.apply(
        lambda r: [r["HP"], r["Attack"], r["Defense"], 
                   r["Sp. Atk"], r["Sp. Def"], r["Speed"]], 
        axis=1
    )
    return df[["Identifier", "type_vector", "stats"]]
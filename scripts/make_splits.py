from sklearn.model_selection import train_test_split
import pandas as pd

RANDOM_SEED = 42

infrequent_cats = ['Sennacherib', 'Tiglath-Pileser III', 'Shalmaneser III']

df_ab = df[
    df["Tier"].isin(["A", "B"]) & ~df["Relief_ID"].isin(Disqualified_relief_id)
].copy()
df_ab['Authority'] = df_ab['Authority'].replace(infrequent_cats, 'OTHER')
df_ab['Authority'] = df_ab['Authority'].astype('category')
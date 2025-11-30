import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_dataset(path="data/food_nutrition_dataset.csv"):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.dropna()

    nutrients = ["calories", "protein", "carbs", "fat", "iron", "vitamin_c"]
    
    scaler = MinMaxScaler()
    df[nutrients] = scaler.fit_transform(df[nutrients])

    return df
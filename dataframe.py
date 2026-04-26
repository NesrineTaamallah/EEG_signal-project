from dataset import load_all_cleaned_with_features
import variables as v
import os

df = load_all_cleaned_with_features(
    cleaned_dir=v.DIR_CLEANED,
    sfreq=v.SFREQ,
    window_sec=1,
    overlap=0.5
)

print(df.shape)
print(df.columns)
print(df.head())


output_path = r"C:\Users\nesri\OneDrive\Desktop\signal\data\Data\dataframe.csv"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df.to_csv(output_path, index=False)
print(f"DataFrame sauvegardé dans {output_path}")

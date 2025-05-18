import pandas as pd

# Read the parquet file
df = pd.read_parquet('../datasets/argoversev2/val_metadata.parquet')

# Replace 'train' with 'val' in the camera columns
camera_cols = ['ring_front_left', 'ring_front_center', 'ring_front_right']
for col in camera_cols:
    df[col] = df[col].str.replace('train', 'val')

# Save back to parquet
df.to_parquet('../datasets/argoversev2/val_metadata.parquet', index=False)
print("Updated val_metadata.parquet camera paths from 'train' to 'val'")

import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
train, test = train_test_split(df, test_size=TEST_SIZE, shuffle=True, random_state=42, stratify=df['DEATH_EVENT'])

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
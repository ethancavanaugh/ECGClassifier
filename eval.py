import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

from ECGPyDataset import ECGPyDataset

if __name__ == "__main__":
    model = load_model('logs/backup_model_last.keras')
    test_df = pd.read_csv('./data/test.csv')
    test_dataset = ECGPyDataset(test_df, 1)


    yhat = model.predict(test_dataset)
    print(yhat)
    yhat = yhat.round()
    ytrue = test_df['A']


    report = classification_report(ytrue, yhat)
    print(report)
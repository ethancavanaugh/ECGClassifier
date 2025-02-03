import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn import metrics

from ECGPyDataset import ECGPyDataset

if __name__ == "__main__":
    model = load_model('logs/backup_model_best.keras')
    test_df = pd.read_csv('./data/test.csv')
    test_dataset = ECGPyDataset(test_df, 1)

    y_true = test_df['A'].map({0: 1, 1: 0})  # Swap values so that 0 = normal and 1 = abnormal
    y_pred = model.predict(test_dataset)
    test_predictions = pd.DataFrame({'true_label': y_true, 'model_output': y_pred.squeeze()})
    test_predictions.to_csv('eval/test_predictions.csv', index=False)
    #y_pred = pd.read_csv('eval/test_predictions.csv')['model_output']


    #Reciever operating curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    df_roc = pd.DataFrame({'Threshold': thresholds, 'FPR': fpr, 'TPR': tpr})
    df_roc.to_csv('eval/roc.csv', index=False)
    metrics.RocCurveDisplay.from_predictions(y_true, y_pred)
    plt.show()

    #Precision recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    df_prc = pd.DataFrame({'Threshold': np.append(thresholds, [1]), 'Precision': precision, 'Recall': recall})
    df_prc.to_csv('eval/precision-recall.csv', index=False)
    metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred)
    plt.show()
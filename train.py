import math
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard, ReduceLROnPlateau,
                                        CSVLogger, EarlyStopping)
from model import get_model
from ECGPyDataset import ECGPyDataset

if __name__ == "__main__":
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.0005
    batch_size = 16
    validation_split = 0.2
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]

    # Split into training and validation datasets
    df_train = pd.read_csv('data/train.csv')
    val_index = math.ceil(len(df_train) * (1 - validation_split))
    train_dataset = ECGPyDataset(df_train.iloc[:val_index], batch_size)
    val_dataset = ECGPyDataset(df_train.iloc[val_index:], batch_size)

    model = get_model(1)
    model.compile(loss=loss, optimizer=opt)

    # Create log
    callbacks += [TensorBoard(log_dir='logs/tb_logs', write_graph=False),
                  CSVLogger('logs/training.log', append=False)]
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('logs/backup_model_last.keras'),
                  ModelCheckpoint('logs/backup_model_best.keras', save_best_only=True)]

    history = model.fit(train_dataset, validation_data=val_dataset,
                        epochs=70, callbacks=callbacks, verbose=1)
    model.save("./final_model.keras")
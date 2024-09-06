
import torch
import pandas as pd
import numpy as np
import torchaudio
from tqdm import tqdm

import os
from glob import glob
import numpy as np
import librosa
import librosa.display
import time
import datetime
import random
from pathlib import Path
from torchaudio.models import wav2vec2_model
import json
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import (
    LightningDataModule,
    Trainer,
    seed_everything,
)

from pytorch_lightning.loggers import WandbLogger
from utils import sliding_window_cuting

# Load dataset
from torch.utils.data import Dataset, DataLoader
from model_pl_module import AvesModule

# -------------------------------------------------
    # DATASET
# -------------------------------------------------
class DCASEDataset(Dataset):
    def __init__(self, dataset_dataframe, is_train, audio_sr, duration_sec, classes):

        self.is_train = is_train
        self.audio_sr = audio_sr
        self.duration_sec = duration_sec
        self.data = dataset_dataframe     
        self.classes = classes   

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        out = {'Audio': row['Audio'], 'Label': row['Label']}
        return out
    


# Create dataset in a module
class DCASEDatamodule(LightningDataModule):

    def __init__(self, train_dir, valid_dir, batch_size, num_workers, sr, wind_dur, classes, on_the_fly=False, df_train_path=None, df_valid_path=None):
        super().__init__()
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.wind_dur = wind_dur
        self.classes = classes
        self.on_the_fly = on_the_fly

    def prepare_data(self):
        # download data
        pass

    def setup(self, stage):
        if stage=='fit':
            if self.on_the_fly:
                print("Creating segments from sliding window")
                df_trainset = self.create_df_dataset(self.train_dir, self.classes, is_training=True)
                df_validset = self.create_df_dataset(self.valid_dir, self.classes, is_training=False)
                # df_trainset.to_pickle('/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_training_PB24_seglen100ms.pkl')
                # df_validset.to_pickle('/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_validation_PB_seglen100ms.pkl')
            else:
                print("Loading segments from pickle")
                df_trainset = pd.read_pickle(df_train_path)
                df_validset = pd.read_pickle(df_valid_path)

        print(f"Train dataset have {len(df_trainset)} segments")
        print(f"Valid dataset have {len(df_validset)} segments")    
        self.train_dataset = DCASEDataset(df_trainset, True, self.sr, self.wind_dur, self.classes)
        self.valid_dataset = DCASEDataset(df_validset, False, self.sr, self.wind_dur, self.classes)
        # Test set to do later

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=self.num_workers)

    # Could add test and predict dataloaders
    def create_df_dataset(self, file_dir, class_map, is_training=False):

        extension = "*.csv"
        all_csv_files = [file for path_dir, _, _ in os.walk(file_dir) for file in glob(os.path.join(path_dir, extension))]
        all_csv_files = sorted(all_csv_files)
        all_wav_files = [csv_file.replace('.csv', '.wav') for csv_file in all_csv_files]
        
        df_set = pd.DataFrame()

        for wav_file, annot_file in tqdm(zip(all_wav_files, all_csv_files), total=len(all_csv_files)):

            # print('\n ---------------------------------------------------------------')
            # print(f" File {os.path.basename(wav_file)}")
            # print(' ---------------------------------------------------------------')

            #Load file and annotations
            waveform, sr = torchaudio.load(wav_file)
            if sr != SR:
                waveform = torchaudio.functional.resample(waveform, sr, SR)
            
            # Normalize the waveform
            waveform = (waveform - waveform.mean())/waveform.std()
            waveform = waveform[0].numpy()

            df = pd.read_csv(annot_file, sep=',')
            if ('Q' not in df.columns):
                # convert to binary annotation for training
                df["Q"] = df.apply(self.determine_q, axis=1)
                df = df[['Audiofilename','Starttime', 'Endtime', 'Q']]

            # Adaptative sliding window length
            # df_pos_annot = df[df.Q == 'POS']
            # mean_pos_duration = (df_pos_annot['Endtime'].head(5) - df_pos_annot['Starttime'].head(5)).mean()
            # wind_dur = mean_pos_duration
            # if wind_dur > 0.5:
            #     wind_dur = 0.5
            # elif(wind_dur < 0.025):
            #     wind_dur = 0.025
            # self.wind_dur = wind_dur
            # --------------------------------------
            
            # Create segments
            df_segments = sliding_window_cuting(waveform, df_annot=df, sr=self.sr, wind_dur=self.wind_dur, win_coverage_threshold=0.9, annot_coverage_threshold=0.5, overlap_ratio=0.)

            # Remove all rows where Label = UNK
            df_segments = df_segments[df_segments.Label != 'UNK']

            # Balance the dataset
            if is_training:
                # Find if there is more positive or negative annotation
                smallest_class = min(len(df_segments[df_segments.Label == 'Q']), len(df_segments[df_segments.Label == 'NEG']))

                # Sample same number of negative and positive annotations
                df_segments = df_segments.groupby('Label').apply(lambda x: x.sample(smallest_class)).reset_index(drop=True)

            # Add a column with filename
            df_segments['Audiofilename'] = os.path.basename(wav_file)
            # Add a column with dataset directory
            df_segments['Dataset_dir'] = os.path.basename(os.path.dirname(wav_file))

            df_set = pd.concat([df_set, df_segments])
        # Convert annotation to numeric for training
        df_set['Label'] = df_set['Label'].map(class_map)

        return df_set
    
    def determine_q(self, row):
        if "POS" in row.values:
            return "POS"
        elif "UNK" in row.values:
            return "UNK"
        else:
            return "NEG"




if __name__ == "__main__":
    # # Params and model
    SR = 16000
    BATCH_SIZE = 32
    FAST_DEV_RUN = False # Activate to test on one file only
    epochs = 20
    classes = {'NEG':0.0, 'Q':1.0}

    wandb_logger = WandbLogger(project='Test DCASEt5 Aves')
    run_name = wandb_logger.experiment.name

    # model
    lr = 1e-5
    weight_decay = 0.01
    print("Setting up model")
    model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
    # aves_model = AvesClassifier(model_path=model_path, model_config_path=model_config_path, trainable=True)
    # lightning_model = AvesModule(aves_model, lr, weight_decay)
    lightning_model = AvesModule(lr, weight_decay, trainable=False)

    # Datamodule
    data_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/'
    # train_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Training_Set/'
    train_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/PB24/'
    valid_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/PB/'
    # df_train_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_train_seglen200ms.pkl'
    # df_valid_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_valid_seglen200ms.pkl'
    df_train_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_training_PB24_seglen100ms.pkl'
    df_valid_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_validation_PB_seglen100ms.pkl'
    NUM_WORKERS = 20
    wind_dur = 0.025
    dm = DCASEDatamodule(train_dir, valid_dir,
                         BATCH_SIZE, NUM_WORKERS, SR,
                         wind_dur=wind_dur, classes=classes, 
                         on_the_fly=True,                                           # On the fly create the dataframe with sliding window function
                         df_train_path=df_train_path, df_valid_path=df_valid_path)
    
    # train model
    print("Setting up trainer")
    checkpoints_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/checkpoints/'
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path,filename=run_name+'_{epoch}-{val_loss:.2f}-{val_macro_f1:.2f}', save_top_k=1, monitor='val_loss', mode='min')
    trainer = Trainer(accelerator='gpu', devices=1, max_epochs=epochs, fast_dev_run=FAST_DEV_RUN, logger=wandb_logger,
                      limit_val_batches= 0.01, limit_train_batches=1.0,
                      callbacks=[checkpoint_callback])

    print("Starting Trainer Fit")
    # trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.fit(model=lightning_model, datamodule=dm)
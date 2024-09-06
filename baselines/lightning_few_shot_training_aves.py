
import pandas as pd
import torchaudio
import os
from glob import glob
import numpy as np

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from utils import sliding_window_cuting, get_pos_neg_segments, compute_protos

from torch.utils.data import Dataset, DataLoader
from model_few_shots_pl_module import AvesModule
from tqdm import tqdm
from datetime import datetime
import gc

# -------------------------------------------------
    # DATASET
# -------------------------------------------------
class DCASEDataset(Dataset):
    def __init__(self, dataset_dataframe, is_train, audio_sr, duration_sec, classes, is_training=True):

        self.is_train = is_train
        self.audio_sr = audio_sr
        self.duration_sec = duration_sec
        self.data = dataset_dataframe     
        self.classes = classes   
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.is_training:
            return {'Audio': row['Audio'], 'Label': row['Label']}
        else:
            return {'Audio': row['Audio']}
    


# Create dataset in a module
class DCASEDatamodule(LightningDataModule):

    def __init__(self, file_path, batch_size, num_workers, sr, classes):
        super().__init__()
        self.file_path = file_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sr = sr
        self.wind_dur = 0.
        self.classes = classes

    def prepare_data(self):
        # download data
        pass

    def setup(self, stage):

        if stage=='fit':
            print("Creating training set from few shot and validation from query time")
            df_trainset, df_validset = self.create_df_few_shot_dataset(self.file_path, self.classes)
            # df_trainset.to_pickle('/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_training_PB24_seglen100ms.pkl')
            # df_validset.to_pickle('/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/data pickle/df_validation_PB_seglen100ms.pkl')

            print(f"Train dataset have {len(df_trainset)} segments")
            print(f"Valid dataset have {len(df_validset)} segments")    
            self.train_dataset = DCASEDataset(df_trainset, True, self.sr, self.wind_dur, self.classes)
            # self.valid_dataset = DCASEDataset(df_validset, False, self.sr, self.wind_dur, self.classes)

        if stage=='test':
            _, df_testset = self.create_df_few_shot_dataset(self.file_path, self.classes, is_training=False)
            self.test_dataset = DCASEDataset(df_testset, False, self.sr, self.wind_dur, self.classes, is_training=False)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=self.num_workers)

    # def val_dataloader(self):
    #     return DataLoader(self.valid_dataset,
    #                         batch_size=self.batch_size,
    #                         shuffle=False,
    #                         drop_last=False,
    #                         num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=self.num_workers)
    
    def create_df_few_shot_dataset(self, wavfile_name, class_map, is_training=True):
        
        annot_file = wavfile_name.replace('.wav', '.csv')

        #Load file and annotations
        waveform, sr = torchaudio.load(wavfile_name)
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
            self.sr = SR
        
        # Normalize the waveform
        waveform = (waveform - waveform.mean())/waveform.std()
        waveform = waveform[0].numpy()

        df_annot = pd.read_csv(annot_file, sep=',')


        if ('Q' not in df_annot.columns):
            # convert to binary annotation for training
            df_annot["Q"] = df_annot.apply(self.determine_q, axis=1)
            df_annot = df_annot[['Audiofilename','Starttime', 'Endtime', 'Q']]

        # Get annot until the first N_SHOTS positive annotations
        pos_indices = df_annot[df_annot['Q'] == 'POS'].index[:N_SHOTS]
        last_annot_index = max(pos_indices)
        annot_few_shot_df = df_annot.loc[:last_annot_index]
        annot_query_df = df_annot.loc[last_annot_index+1:]

        # Get the mean duration of positive annotations
        df_pos_annot = df_annot[df_annot.Q == 'POS']
        mean_pos_duration = (df_pos_annot['Endtime'].head(N_SHOTS) - df_pos_annot['Starttime'].head(N_SHOTS)).mean()

        # Adaptative sliding window length
        wind_dur = mean_pos_duration
        if wind_dur > 1:
            wind_dur = 1
        elif(wind_dur < 0.2):
            wind_dur = 0.2
        self.wind_dur = wind_dur
        
        # print(f"Average pos annot {mean_pos_duration} --> Sliding window duration: ", wind_dur)

        win_coverage_threshold = 1 # Segment is positive if annotations cover more than 30% of the window
        annot_coverage_threshold = 0.5 # Segment is positive if more than 80% of the annotation is is the window
        overlap_ratio = 0.

        # Get Pos, Neg and Query segments
        start_query_sample = int(annot_few_shot_df['Endtime'].iloc[-1] * self.sr) #Add one window to avoid missing last annot
        waveform_few_shots, waveform_query = waveform[:start_query_sample],  waveform[start_query_sample:]
        # Shift query annotation to query start time
        annot_query_df['Starttime'].loc[:] -= annot_few_shot_df['Endtime'].iloc[-1]
        annot_query_df['Endtime'].loc[:] -= annot_few_shot_df['Endtime'].iloc[-1]

        # df_few_shots = sliding_window_cuting(waveform_few_shots, few_shot_df, SR, wind_dur, win_coverage_threshold, annot_coverage_threshold, overlap_ratio=0.)
        df_few_shots = get_pos_neg_segments(waveform_few_shots, annot_few_shot_df, self.sr, self.wind_dur)

        if is_training:
            df_query = sliding_window_cuting(waveform_query, annot_query_df, self.sr, self.wind_dur, win_coverage_threshold, annot_coverage_threshold, overlap_ratio=overlap_ratio)
            df_query = df_query[df_query.Label != 'UNK']
            
            # Balance the validation set
            smallest_class = min(len(df_query[df_query.Label == 'Q']), len(df_query[df_query.Label == 'NEG']))
            df_query = df_query.groupby('Label').apply(lambda x: x.sample(smallest_class)).reset_index(drop=True)
            df_query['Label'] = df_query['Label'].map(class_map)

        else:
            df_query = sliding_window_cuting(waveform_query, df_annot=None, sr=self.sr, wind_dur=self.wind_dur)

        df_few_shots = df_few_shots[df_few_shots.Label != 'UNK']

        # Balance the training dataset
        smallest_class = min(len(df_few_shots[df_few_shots.Label == 'Q']), len(df_few_shots[df_few_shots.Label == 'NEG']))
        df_few_shots = df_few_shots.groupby('Label').apply(lambda x: x.sample(smallest_class)).reset_index(drop=True)

        # Convert annotation to numeric for training
        df_few_shots['Label'] = df_few_shots['Label'].map(class_map)

        return df_few_shots, df_query
    

    
    def determine_q(self, row):
        if "POS" in row.values:
            return "POS"
        elif "UNK" in row.values:
            return "UNK"
        else:
            return "NEG"

    def get_query_time(self):
        annot_file = self.file_path.replace('.wav', '.csv')
        df_annot = pd.read_csv(annot_file, sep=',')
        if ('Q' not in df_annot.columns):
            # convert to binary annotation for training
            df_annot["Q"] = df_annot.apply(self.determine_q, axis=1)
            df_annot = df_annot[['Audiofilename','Starttime', 'Endtime', 'Q']]
        pos_indices = df_annot[df_annot['Q'] == 'POS'].index[:N_SHOTS]
        last_annot_index = max(pos_indices)
        annot_few_shot_df = df_annot.loc[:last_annot_index]

        # Get Pos, Neg and Query segments
        start_query_sample = int(annot_few_shot_df['Endtime'].iloc[-1] * self.sr)
        
        return start_query_sample/self.sr

if __name__ == "__main__":
    # # Params and model
    SR = 16000
    BATCH_SIZE = 16
    FAST_DEV_RUN = False # Activate to test on one file only
    N_SHOTS = 5
    epochs = 20
    classes = {'NEG':0.0, 'Q':1.0}
    NUM_WORKERS = 20
    lr = 1e-5
    weight_decay = 0.0001

    seed_everything(42)

    test_name = 'full_test_set_training'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prediction_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/{test_name}_{timestamp}.csv'
    prediction_path = "/home/reindert/Valentin_REVO/FSL_revo/Prediction/testGR_Lightning04.csv"

    # Datamodule
    data_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/'
    
    # FETCHING FILES
    # file_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Test_Set/'
    file_dir = '/home/reindert/Valentin_REVO/DCASE_2022/GR_Set/GR/'
    extension = '*.wav'

    # Load all files paths
    all_wav_files = [file for path_dir, _, _ in os.walk(file_dir) for file in glob(os.path.join(path_dir, extension))]
    all_wav_files = sorted(all_wav_files)

    # Only run for a few files
    # subset = [0, 10, 12, 22, 38]
    # all_wav_files = [all_wav_files[i] for i in subset]


    for wavfile_path in tqdm(all_wav_files):

        wandb_logger = WandbLogger(project='Test DCASEt5 Aves')
        run_name = wandb_logger.experiment.name

        print("Prediction on file: ", wavfile_path)

        dm = DCASEDatamodule(wavfile_path,
                            BATCH_SIZE, NUM_WORKERS, SR,
                            classes=classes)
        
        # Setup
        query_time = dm.get_query_time()

        # model
        print("Setting up model")
        lightning_model = AvesModule(lr, weight_decay, trainable=True, query_time=query_time, file_path=dm.file_path)

        # Initialize prototypes
        annot_file = wavfile_path.replace('.wav', '.csv') 
        lightning_model.protos = compute_protos(wavfile_path, annot_file, lightning_model.model, sr=SR)

        # Set up checkpoints to save model
        checkpoints_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/checkpoints/'
        # checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_path,filename=run_name+'_{epoch}-{val_loss:.2f}-{val_macro_f1:.2f}', save_top_k=1, monitor='val_macro_f1', mode='max')
        early_stopping_callback = EarlyStopping(monitor='train_loss', patience=3, verbose=True, mode='min', min_delta=1)

        # Trainer
        print("Setting up trainer")
        trainer = Trainer(accelerator='gpu', devices=1, max_epochs=epochs, fast_dev_run=FAST_DEV_RUN, logger=wandb_logger,
                        # limit_val_batches= 1.0, limit_train_batches=1.0,
                        callbacks=[early_stopping_callback])

        # Train the model on first few shots
        # trainer.fit(model=lightning_model, datamodule=dm)
        
        # Evaluate the file
        lightning_model.predict_csv_filepath = prediction_path
        trainer.test(model=lightning_model, datamodule=dm)

        # Clear the objects to free up memory and avoid state carryover
        del lightning_model
        del dm
        del trainer
        del wandb_logger
        gc.collect()


    threshold_range = [0.5]
    for threshold in threshold_range:
        print('Threshold = ', threshold)        
        post_process = True
        if post_process:
            from src.utils.post_proc import post_processing
            
            post_proc_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/{test_name}_{timestamp}_postprocessed.csv'
            post_proc_path = "/home/reindert/Valentin_REVO/FSL_revo/Prediction/testGR_Lightning01_postproc.csv"
            file_dir = "/home/reindert/Valentin_REVO/DCASE_2022/GR_Set"
            post_processing(file_dir, prediction_path, post_proc_path, n_shots=5, threshold=threshold)
            prediction_path = post_proc_path

        # Evaluate result
        eval_predictions = False
        if eval_predictions:
            from src.utils.evaluation import evaluate
    
            ref_files_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
            team_name = f'Bordoux'
            dataset = f'fish_sound_GR_{threshold}'
            savepath = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/'

            print('Evaluating file ', prediction_path)
            evaluate(prediction_path, ref_files_path, team_name, dataset, savepath)
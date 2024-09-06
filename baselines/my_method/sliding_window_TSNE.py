import torch
from transformers import ClapModel, ClapProcessor
import pandas as pd
import numpy as np
import librosa
import torchaudio
from tqdm import tqdm

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
# from src.models.components.protonet import AvesClassifier, ResNet
from glob import glob
import numpy as np
import librosa
import librosa.display
import time
import datetime
import random
from Z_evaluate_proto import evaluate_prototypes_AVES
from pathlib import Path
from utils import sliding_window_cuting
from model_few_shots_pl_module import AvesModule, AvesClassifier



def tsne_embedding(X, n_components, perplexity = 10, n_iter = 1000, learning_rate='auto'):
  # Fit and transform X into a reduced space of n_components dimensions using T-SNE
  # https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600
  tsne = TSNE(n_components,perplexity=perplexity,n_iter=n_iter, method='exact', learning_rate=learning_rate, verbose=0, init='pca')
  return tsne.fit_transform(X)



def generate_pos_neg_query_segments(self, audio_filepath=None, annot_filepath=None, seg_len=0.5, n_shot=5, model_sr=11025, call_type='Q', adaptative_seg_len=True):
    '''
    Create pos, neg and query array, containing respectively positive, negative and query segments.
        - audio_filepath: path to wav file
        - annot_filepath: path to txt file (annotation in Raven format)
    Output:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this 
    '''
    # Load annotation and wav file
    df_annot = pd.read_csv(annot_filepath, sep='\t')

    # Load audio
    waveform, sr = torchaudio.load(audio_filepath)

    if sr != model_sr:
        transform = torchaudio.transforms.Resample(sr, model_sr)
        waveform = transform(waveform)
        sr = model_sr

    df_pos = df_annot[df_annot[call_type] == 'POS']
    # If the given file do not contain the minimum number of annotation, skip the file
    if len(df_pos) < n_shot:
        print('Skipping file, not enough POS annotations: ', audio_filepath)
        return [], [], [], 0, 0, 0

    # Normalize the waveform
    waveform = (waveform - waveform.mean())/waveform.std()

    # Create features for positive proto
    n_shot_df = df_pos.sort_values('Starttime').head(n_shot)

    X_pos = []
    pos_annot_bounds = []

    for i, row in n_shot_df.iterrows():
        start_wav = int(row['Starttime']*sr)
        end_wav = int(row['Endtime']*sr)

        # AVES minimal input is 25 ms, if smaller, make the segment 20 ms long
        if (end_wav - start_wav)/sr < 0.025:
            end_wav = int(start_wav + 0.025*sr)
        
        pos_annot_bounds.append((start_wav, end_wav))
        X_pos.append(waveform[0][start_wav:end_wav])

    # # Compute pos proto in frame instead of windows - NOT USED ANYMORE
    # seg_len_in_sample = int(seg_len * sr)
    # X_pos_concat = cat(X_pos, 0)
    # num_segments = int(len(X_pos_concat) // seg_len_in_sample)
    # X_pos = np.array_split(X_pos_concat[:num_segments * seg_len_in_sample], num_segments) # Last part of the file (size<segment) is discarded

    # Compute the mean length of X_pos samples to use to generate seg_len
    if adaptative_seg_len:
        mean_length = sum(len(sample) for sample in X_pos) / len(X_pos)
        seg_len_in_sample = int(mean_length)
        seg_len = mean_length / sr
        print('Average length of positive samples: ', seg_len)
    else:
        seg_len_in_sample = int(seg_len * sr)

    assert seg_len_in_sample != 0, 'Error: seg_len is 0'
    
    # Save the ending time of the last annotation (where to start query set)
    last_annot_endtime = int(n_shot_df.iloc[-1]['End Time (s)']*sr)
    start_query = last_annot_endtime

    # Compute the proto by averaging all the space between the pos_call
    X_neg = neg_proto_all_between_pos(pos_annot_bounds, waveform, seg_len_in_sample)
    if len(X_neg) < n_shot:
        print("WARNING: Not enough negative samples between pos call, go for whole file method unexpected behavior might happen")
        num_segments = int(len(waveform[0]) // seg_len_in_sample)
        X_neg = np.array_split(waveform[0][:num_segments * seg_len_in_sample], num_segments) # Last part of the file (size<segment) is discarded

    # Find the shortest element in X_pos list
    min_length = min(len(sample) for sample in X_pos)
    min_length_sec = min_length / sr


    # Create features for query set
    query_waveform = waveform[0][last_annot_endtime:]
    num_segments_query = len(query_waveform) // seg_len_in_sample
    X_query = np.array_split(query_waveform[:num_segments_query * seg_len_in_sample], num_segments_query)


    return X_pos, X_neg, X_query, start_query, min_length_sec, seg_len
    


def neg_proto_all_between_pos(pos_annot_bounds, full_waveform, seg_len_in_sample):
    '''
    Function to convert all the space between positive annotations to negative samples

    Input:
    - waveform: the whole waveform
    - pos_annot_bounds: list of the start and end of the positive annotations

    Return:
    - X_neg: list of the negative samples
    '''
    # Select only the section of the waveform before the last positive sample (discard last annotation + query set)
    waveform = full_waveform.squeeze()[:pos_annot_bounds[-1][0]]

    # Create a new waveform with only the sections between the postitive annotations
    for bound in reversed(pos_annot_bounds[:-1]):
        waveform = torch.cat((waveform[:bound[0]], waveform[bound[1]:]), 0)

    # Compute the negative sample of segment_length and discard the rest
    num_segments_query = len(waveform) // seg_len_in_sample
    X_neg = np.array_split(waveform[:num_segments_query * seg_len_in_sample], num_segments_query)

    return X_neg

if __name__ == "__main__":
    
    # Params and model
    SR = 16000
    BATCH_SIZE = 8
    FAST_DEV_RUN = True # Activate to test on one file only
    feature_extractor = 'AVES'  # 'BioLingual' or 'AVES'
    device = torch.device('cuda')

    # BioLingual model
    if feature_extractor == 'BioLingual':
        SR = 22500
        model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", sampling_rate=SR)
        model.to(device)
        model.eval()
    
    # AVES model
    if feature_extractor == 'AVES': 
        # device = 'cuda'
        model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
        model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
        model = AvesClassifier(model_path=model_path, model_config_path=model_config_path, trainable=False)
        SR = 16000
        
        # Load the model from lightning checkpoint
        checkpoint_file = 'kind-dragon-139_epoch=3-val_loss=0.00-val_macro_f1=0.87.ckpt'
        checkpoint_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/checkpoints/{checkpoint_file}'
        lightning_module = AvesModule.load_from_checkpoint(checkpoint_path)
        model = lightning_module.model
    
        model.to(device)
        model.eval()

    # FETCHING FILES
    file_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
    extension = '*.csv'

    # Load all files paths
    all_csv_files = [file for path_dir, _, _ in os.walk(file_dir) for file in glob(os.path.join(path_dir, extension))]
    all_csv_files = sorted(all_csv_files)

    all_wav_files = [csv_file.replace('.csv', '.wav') for csv_file in all_csv_files]


    if FAST_DEV_RUN:
        # Select random file to run a test of the code on
        rand_idx = random.randint(0, len(all_csv_files) - 1)
        rand_idx = 12
        all_csv_files = [all_csv_files[rand_idx]]
        all_wav_files = [all_wav_files[rand_idx]]
        # all_csv_files = [all_csv_files[12], all_csv_files[20]]
        # all_wav_files = [all_wav_files[12], all_wav_files[20]]
    
    for wav_file, annot_file in tqdm(zip(all_wav_files, all_csv_files), total=len(all_csv_files)):

        #Load file and annotations
        waveform, sr = torchaudio.load(wav_file)
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
            sr = SR
        
        # Normalize the waveform
        waveform = (waveform - waveform.mean())/waveform.std()
        waveform = waveform[0].numpy()
        df_annot = pd.read_csv(annot_file, sep=',')

        # Adaptative sliding window length
        df_pos_annot = df_annot[df_annot.Q == 'POS']
        mean_pos_duration = (df_pos_annot['Endtime'].head(5) - df_pos_annot['Starttime'].head(5)).mean()
        wind_dur = mean_pos_duration
        if wind_dur > 1:
            wind_dur = 1
        elif(wind_dur < 0.025):
            wind_dur = 0.025

        # wind_dur = 0.025
        win_coverage_threshold = 0.9
        annot_coverage_threshold = 0.5

        # CUTTING AND ASSIGNING LABELS
        df_chunks = sliding_window_cuting(waveform, df_annot, SR, wind_dur, win_coverage_threshold, annot_coverage_threshold)
        
        # Subsample by ration to reduce inference time
        ratio = 1   
        print(f"Subsampling POS and UNK segments by {ratio} and negative segments to two time more than the other classes sum")

        subsample_df = df_chunks.groupby('Label').apply(lambda x: x.sample(n=int(len(x)/ratio))).reset_index(drop = True)

        # Reduce the negative class further as it is overrepresented
        neg_chunks = subsample_df.loc[subsample_df['Label'] == 'NEG']
        unk_chunks = subsample_df.loc[subsample_df['Label'] == 'UNK']
        pos_chunks = subsample_df.loc[subsample_df['Label'] == 'Q']

        # other_classes_df = subsample_df.loc[(subsample_df['Label'] != 'NEG') & (subsample_df['Label'] != 'UNK')]

        # Reduce each class to a maximum of 500 samples to speed up TSNE
        sampled_neg_chunks = neg_chunks.sample(n=np.min([len(neg_chunks), 500]), random_state=1)
        sampled_unk_chunks = unk_chunks.sample(n=np.min([len(unk_chunks), 500]), random_state=1)
        sampled_pos_chunks = pos_chunks.sample(n=np.min([len(pos_chunks), 500]), random_state=1)

        balanced_df = pd.concat([sampled_neg_chunks, sampled_unk_chunks, sampled_pos_chunks])

        print('Number of samples after subsampling in each of the class: ')
        print(balanced_df['Label'].value_counts())
        
        chunks = [row['Audio'] for _idx, row in balanced_df.iterrows()]
        labels = [row['Label'] for _idx, row in balanced_df.iterrows()]

        feat_array = torch.Tensor().to(device)

        # BATCHING AND EXTRACTING FEATURES
        num_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in range(num_batches):
            # Get the current batch of chunks
            batch_chunks = chunks[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            
            if feature_extractor == 'BioLingual':
                # Convert batch_chunks to the format expected by the processor
                start = time.time()
                inputs = processor(audios=batch_chunks, return_tensors="pt", sampling_rate=SR).to(device)
                proc_time = time.time() - start
                with torch.no_grad():
                    features = model.get_audio_features(**inputs)
                    print(f"Processor time: {proc_time} \n Model time: {time.time() - proc_time}")
            elif feature_extractor == 'AVES':
                # Convert batch_chunks to a tensor and add a channel dimension
                x = torch.stack([torch.Tensor(sample) for sample in batch_chunks]).to(device)
                with torch.no_grad():
                    features = model(x)
            else:
                raise ValueError("Feature extractor not supported")
            
            # Concatenate the features from this batch with the accumulated features
            feat_array = torch.cat((feat_array, features), dim=0)

        # TSNE
        generate_TSNE_and_save = True
        if generate_TSNE_and_save:
            tsne_X_2d = tsne_embedding(feat_array.cpu(), n_components=2, perplexity=10, n_iter=1000)

            result_df = pd.DataFrame({'component_1': tsne_X_2d[:,0], 'component_2': tsne_X_2d[:,1], 'label': labels})
            
            # Plot TSNE
            fig, ax = plt.subplots(1)
            sns.scatterplot(x='component_1', y='component_2', hue='label', data=result_df, ax=ax,s=120)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            ax.set_title(f'TSNE of {os.path.basename(wav_file)}')

            # Saving TSNE plots
            path, file = os.path.split(wav_file)
            dest_dir = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/TSNE/{datetime.datetime.now().strftime("%Y%m%d")}/{os.path.basename(os.path.normpath(path))}'
            os.makedirs(dest_dir, exist_ok=True)
            filename = f'prepretrained_TSNE_{feature_extractor}_{wind_dur}_{file}.png'
            plt.savefig(os.path.join(dest_dir, filename), bbox_inches='tight')
            print(f"TSNE plot for file {os.path.basename(wav_file)} saved to {dest_dir}")

            plt.show()

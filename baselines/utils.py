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
from glob import glob
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import time


# def old_sliding_window_cuting(waveform, df_annot=None, sr=16000, wind_dur=1., win_coverage_threshold=0.2, annot_coverage_threshold=0.8, overlap_ratio=0., verbose=0):
#     '''
#     Generate chunks of audio with label associated based on annotation file and audio file
#     '''
#     # TODO - Overlapping windows not working to fix

#     # Calculate frame length in samples
#     frame_length_sample = int(wind_dur * sr)
#     step_size = int(frame_length_sample * (1-overlap_ratio))

#     # Create a list to store chunks and start times
#     chunks = []
#     start_time = []

#     indices = np.arange(0, len(waveform) - frame_length_sample + 1, step_size)
#     chunks = [waveform[i:i+frame_length_sample] for i in indices]
#     start_time = indices / sr

#     df_chunks = pd.DataFrame({'Audio': chunks, 'Starttime': start_time})

#     # Create label if the frame contains annotation
#     if df_annot is not None:
#         labels = []
#         df_chunks['Endtime'] = df_chunks['Starttime'] + wind_dur

#         for i, chunk in df_chunks.iterrows():
#             # Find annotations in the current chunk
#             annotations_in_window = df_annot[
#                                 ((chunk['Starttime'] < df_annot['Starttime']) & (df_annot['Starttime'] < chunk['Endtime'])) |
#                                 ((chunk['Starttime'] < df_annot['Endtime']) & (df_annot['Endtime'] < chunk['Endtime'])) |
#                                 ((df_annot['Starttime'] < chunk['Starttime']) & (chunk['Endtime'] < df_annot['Endtime']))
#                                             ]
#             # Determine the label for the current chunk
#             label = determine_label(annotations_in_window, chunk['Starttime'], chunk['Endtime'], wind_dur, annot_coverage_threshold, win_coverage_threshold)
#             labels.append(label)

#         # Assign labels to the DataFrame
#         df_chunks.insert(2, 'Label', labels)
#         if verbose == 1:
#             print('Number of samples in each of the class: ')
#             print(df_chunks['Label'].value_counts())
#     return df_chunks


def determine_label(annotations_in_window, start_window, end_window, wind_dur, annot_coverage_threshold=0.8, win_coverage_threshold=0.2):
    label = 'NEG'
    
    if not annotations_in_window.empty:
        # Calculate durations using vectorized operations
        annotations_in_window['Duration'] = np.minimum(end_window, annotations_in_window['Endtime']) - np.maximum(start_window, annotations_in_window['Starttime'])
        
        total_annotation_duration = annotations_in_window['Duration'].sum()

        if total_annotation_duration + 1e-5 > win_coverage_threshold * wind_dur:
            # Find the longest annotation
            longest_annotation = annotations_in_window.loc[annotations_in_window['Duration'].idxmax()]

            if 'POS' in longest_annotation.values:
                pos_column = longest_annotation.index[longest_annotation == 'POS'].tolist()
                return pos_column[0]
            elif 'UNK' in longest_annotation.values:
                label = 'UNK'
        
        # Filter significant annotations
        significant_annotations = annotations_in_window[annotations_in_window['Endtime'] - annotations_in_window['Starttime'] >= 1e-5]

        for _, annotation in significant_annotations.iterrows():
            if annotation['Duration'] / (annotation['Endtime'] - annotation['Starttime']) >= annot_coverage_threshold - 1e-5:
                if 'POS' in annotation.values:
                    pos_column = annotation.index[annotation == 'POS'].tolist()
                    return pos_column[0]
                elif 'UNK' in annotation.values:
                    label = 'UNK'

    return label

def sliding_window_cuting(waveform, df_annot=None, sr=16000, wind_dur=1.0, win_coverage_threshold=0.2, annot_coverage_threshold=0.8, overlap_ratio=0.0, verbose=0):
    # Calculate frame length in samples
    frame_length_sample = int(wind_dur * sr)
    step_size = int(frame_length_sample * (1 - overlap_ratio))

    # Calculate start times
    indices = np.arange(0, len(waveform) - frame_length_sample + 1, step_size)
    start_times = indices / sr
    end_times = start_times + wind_dur

    # Extract chunks
    chunks = [waveform[i:i+frame_length_sample] for i in indices]
    df_chunks = pd.DataFrame({'Audio': chunks, 'Starttime': start_times, 'Endtime': end_times})

    if df_annot is not None:
        labels = []
        df_annot_start = df_annot['Starttime'].values
        df_annot_end = df_annot['Endtime'].values

        for start, end in zip(start_times, end_times):
            # Find annotations in the current chunk
            mask = ((start < df_annot_start) & (df_annot_start < end)) | \
                   ((start < df_annot_end) & (df_annot_end < end)) | \
                   ((df_annot_start < start) & (end < df_annot_end))
            annotations_in_window = df_annot[mask]

            # Determine the label for the current chunk
            label = determine_label(annotations_in_window, start, end, wind_dur, annot_coverage_threshold, win_coverage_threshold)
            labels.append(label)

        # Assign labels to the DataFrame
        df_chunks['Label'] = labels

        if verbose == 1:
            print('Number of samples in each of the class:')
            print(df_chunks['Label'].value_counts())

    return df_chunks


# def calculate_duration(row, start_window, end_window):
#     return min(end_window, row['Endtime']) - max(start_window, row['Starttime'])

# def old_determine_label(annotations_in_window, start_window, end_window, wind_dur, annot_coverage_threshold=0.8, win_coverage_threshold=0.2):

#     label = 'NEG'
    
#     if len(annotations_in_window) > 0:
#         annotations_in_window.loc[:,'Duration'] = annotations_in_window.apply(
#             lambda row: calculate_duration(row, start_window, end_window), axis=1)
        
#         total_annotation_duration = annotations_in_window['Duration'].sum()

#         if total_annotation_duration + 1e-5 > win_coverage_threshold * wind_dur:
#             longest_annotation = annotations_in_window.sort_values(by='Duration', ascending=False).iloc[0]
#             if (longest_annotation == 'POS').any():
#                 pos_column = longest_annotation.index[longest_annotation.isin(['POS'])].tolist()
#                 return pos_column[0]
#             elif (longest_annotation == 'UNK').any():
#                 label = 'UNK'
        
#         significant_annotations = annotations_in_window[(annotations_in_window['Endtime'] - annotations_in_window['Starttime']) >= 1e-5]
#         for _, annotation in significant_annotations.iterrows():

#             if annotation['Duration'] / (annotation['Endtime'] - annotation['Starttime']) >= annot_coverage_threshold - 1e-5:
#                 if (annotation == 'POS').any():
#                     pos_column = annotation.index[annotation.isin(['POS'])].tolist()
#                     return pos_column[0]
#                 elif (annotation == 'UNK').any():
#                     label = 'UNK'

#     return label



def get_pos_neg_segments(waveform_few_shots, few_shot_df, sr, wind_dur, min_window=0.025):

        # Create features for positive proto
        pos_wav_array = np.array([], dtype=np.float32)
        annot_bounds = []

        # Get positive annotations boundaries and positive segments
        for _, row in few_shot_df.iterrows():
            start_wav = int(row['Starttime']*sr)
            end_wav = int(row['Endtime']*sr)

            # AVES minimal input is 25 ms, if smaller, make the segment 20 ms long
            if (end_wav - start_wav)/sr < min_window:
                end_wav = int(start_wav + min_window*sr)

            annot_bounds.append((start_wav, end_wav))
            # Do not save waveform if label is UNK
            if row['Q'] == 'POS':
                pos_wav_array = np.append(pos_wav_array, waveform_few_shots[start_wav:end_wav])

        
        seg_len_in_sample = int(wind_dur * sr)
        
        # Create a new waveform with only the sections between the postitive annotations
        annot_start = 0
        neg_wav_array = np.array([], dtype=np.float32)
        for bounds in annot_bounds:
            neg_wav_array = np.append(neg_wav_array, waveform_few_shots[annot_start:bounds[0]])
            annot_start = bounds[1]

        # Slice the positive and negative segments into windows
        pos_num_segments = len(pos_wav_array) // seg_len_in_sample
        neg_num_segments = len(neg_wav_array) // seg_len_in_sample

        X_pos = np.split(pos_wav_array[:seg_len_in_sample*pos_num_segments], pos_num_segments)
        pos_label = ['Q'] * len(X_pos)
        X_neg = np.split(neg_wav_array[:seg_len_in_sample*neg_num_segments], neg_num_segments)
        neg_label = ['NEG'] * len(X_neg)

        df = pd.DataFrame({'Audio': X_pos+X_neg, 'Label': pos_label+neg_label})

        return df



def compute_protos(wav_file, annot_file, model, sr):

    waveform, native_sr = torchaudio.load(wav_file)

    if native_sr != sr:
        transform = torchaudio.transforms.Resample(native_sr, sr)
        waveform = transform(waveform)

    # Normalize the waveform
    waveform = (waveform - waveform.mean())/waveform.std()
    waveform = waveform[0].numpy()

    df_annot = pd.read_csv(annot_file, sep=',').sort_values(by=['Starttime'], ignore_index=True)

    # Get annot until the first 5 positive annotations
    pos_indices = df_annot[df_annot['Q'] == 'POS'].index[:5]
    last_annot_index = max(pos_indices)
    few_shot_df = df_annot.loc[:last_annot_index]

    # Get the mean duration of positive annotations
    df_pos_annot = df_annot[df_annot.Q == 'POS']
    mean_pos_duration = (df_pos_annot['Endtime'].head(5) - df_pos_annot['Starttime'].head(5)).mean()

    # Adaptative sliding window length
    wind_dur = mean_pos_duration
    if wind_dur > 1.:
        wind_dur = 1.
    elif(wind_dur < 0.025):
        wind_dur = 0.025
    
    print(f"Average pos annot {mean_pos_duration} --> Sliding window duration: ", wind_dur)

    # Get Pos, Neg and Query segments
    start_query_sample = int(few_shot_df['Endtime'].iloc[-1] * sr) + int(wind_dur * sr) #Add one window to avoid missing last annot
    waveform_few_shots = waveform[:start_query_sample]
    df_few_shots = get_pos_neg_segments(waveform_few_shots, few_shot_df, sr, wind_dur)

    # Balance the training dataset
    smallest_class = min(len(df_few_shots[df_few_shots.Label == 'Q']), len(df_few_shots[df_few_shots.Label == 'NEG']))
    df_few_shots = df_few_shots.groupby('Label').apply(lambda x: x.sample(smallest_class)).reset_index(drop=True)

    df_pos = df_few_shots[df_few_shots['Label'] == 'Q']
    df_neg = df_few_shots[df_few_shots['Label'] == 'NEG']

    X_pos = [torch.tensor(row['Audio']) for _, row in df_pos.iterrows()]
    X_neg = [torch.tensor(row['Audio']) for _, row in df_neg.iterrows()]

    batch_size = 16

    X_pos = torch.tensor(torch.stack(X_pos, dim=0))
    pos_dataset = torch.utils.data.TensorDataset(X_pos)
    pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, num_workers=20, batch_size=batch_size,shuffle=False)

    X_neg = torch.tensor(torch.stack(X_neg, dim=0))
    neg_dataset = torch.utils.data.TensorDataset(X_neg)
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, num_workers=20, batch_size=batch_size,shuffle=False)

    # pos_iterator = iter(pos_loader)
    device = 'cuda'
    model = model.to(device)
    feat_array_pos = torch.Tensor().to(device)


    with torch.no_grad():
        torch.cuda.empty_cache()

        for batch in pos_loader:
            x_pos = batch[0]
            x_pos = x_pos.to(device)
            feat_pos = model(x_pos)
            feat_array_pos = torch.cat((feat_array_pos, feat_pos), dim=0)                     
        # Compute positive prototype as the mean of all positive embeddings
        pos_proto = feat_array_pos.mean(dim=0).to(device)
        
        neg_iterator = iter(neg_loader)
        feat_array_neg = torch.Tensor().to(device)

        for batch in neg_iterator:
            x_neg = batch[0]
            x_neg = x_neg.to(device)
            feat_neg = model(x_neg)
            feat_array_neg = torch.cat((feat_array_neg, feat_neg), dim=0)
                                
        # Compute negative prototype as the mean of all negative embeddings
        neg_proto = feat_array_neg.mean(dim=0).to(device)

    protos = torch.cat([neg_proto.unsqueeze(0), pos_proto.unsqueeze(0)], axis=0)

    return protos


def get_probability(neg_proto, pos_proto ,query_set_out):
    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """
    
    prototypes = torch.stack([neg_proto, pos_proto]).squeeze(1)
    dists = euclidean_dist(query_set_out,prototypes)

    '''  Taking inverse distance for converting distance to probabilities'''
    logits = -dists

    #Testing prototypes similarity
    # dist_proto = euclidean_dist(proto_pos.unsqueeze(0), neg_proto.unsqueeze(0))
    # print("Similarity between pos and neg proto is ", 1/(1+dist_proto))
    # print("Where 0 is very different and 1 very similar")

    prob = torch.softmax(logits,dim=1)
    
    #prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,1]
    # return prob_pos.detach().cpu().tolist()   
    return prob_pos



def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    
    n, m = x.size(0), y.size(0)
    d = x.size(1)

    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import torchaudio
from torch import Tensor, from_numpy, cat



        

def generate_pos_neg_query_segments(waveform=None, df_annot=None, seg_len=0.5, n_shot=5, sr=11025, call_type='POS'):
    '''
    Create pos, neg and query array, containing respectively positive, negative and query segments.
        - audio_filepath: path to wav file
        - annot_filepath: path to txt file (annotation in Raven format)
    Output:
        - X_pos: Positive set features. Positive class prototypes will be calculated from this
        - X_query: Query set. Onset-offset prediction will be made on this    #     mean,std = norm_params(x[train_array])
    '''

    # Load annotation and wav file
    if 'Q' in df_annot.columns:
        df_pos = df_annot[df_annot['Q'] == 'POS']
    else:
        pos_column = df_annot.columns[df_annot.isin(['POS']).any()].tolist()
        print('Calls with positive events: ', pos_column)
        # Ask user to select the correct call
        if len(pos_column) > 1:
            call_type = input('Enter the correct call type: ')
        else:
            call_type = pos_column[0]
        df_pos = df_annot[df_annot[call_type] == 'POS']

    # If the given file do not contain the minimum number of annotation, skip the file
    if len(df_pos) < n_shot:
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
    seg_len_in_sample = int(seg_len * sr)
    # X_pos_concat = cat(X_pos, 0)
    # num_segments = int(len(X_pos_concat) // seg_len_in_sample)
    # X_pos = np.array_split(X_pos_concat[:num_segments * seg_len_in_sample], num_segments) # Last part of the file (size<segment) is discarded

    # Compute the mean length of X_pos samples to use to generate seg_len
    # mean_length = sum(len(sample) for sample in X_pos) / len(X_pos)
    # seg_len_in_sample = int(mean_length)
    # seg_len = mean_length / sr
    # print(f" Average positive annotation length is {seg_len} seconds")

    assert seg_len_in_sample != 0, "Segment length is 0"
    
    # Save the ending time of the last annotation (where to start query set)
    last_annot_endtime = int(n_shot_df.iloc[-1]['Endtime']*sr)
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


    return X_pos, X_neg, X_query, start_query 
    


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
        waveform = cat((waveform[:bound[0]], waveform[bound[1]:]), 0)

    # Compute the negative sample of segment_length and discard the rest
    num_segments_query = len(waveform) // seg_len_in_sample

    if num_segments_query == 0:
        breakpoint()

    X_neg = np.array_split(waveform[:num_segments_query * seg_len_in_sample], num_segments_query)

    return X_neg



def init_seed():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)



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


# TODO
# Implement a cosine distance to infer from protypical network


def get_probability(proto_pos,neg_proto,query_set_out):
    """Calculates the  probability of each query point belonging to either the positive or negative class
     Args:
     - x_pos : Model output for the positive class
     - neg_proto : Negative class prototype calculated from randomly chosed 100 segments across the audio file
     - query_set_out:  Model output for the first 8 samples of the query set

     Out:
     - Probabiility array for the positive class
     """
    
    prototypes = torch.stack([proto_pos,neg_proto]).squeeze(1)
    dists = euclidean_dist(query_set_out,prototypes)

    '''  Taking inverse distance for converting distance to probabilities'''
    logits = -dists

    #Testing prototypes similarity
    # dist_proto = euclidean_dist(proto_pos.unsqueeze(0), neg_proto.unsqueeze(0))
    # print("Similarity between pos and neg proto is ", 1/(1+dist_proto))
    # print("Where 0 is very different and 1 very similar")

    prob = torch.softmax(logits,dim=1)
    inverse_dist = torch.div(1.0, dists)
    
    #prob = torch.softmax(inverse_dist,dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]

    return prob_pos.detach().cpu().tolist()     




def evaluate_prototypes_AVES(encoder = None, device= None, waveform = None, df_annot = None, seg_len = 0.5, n_shot = 5, batch_size = 64, sr=11025):
    '''
    This function is ugly for now but manage to generate positive, negative and query prototypes with AVES
    It returns two arrays onset and offset, respectively the start and the end of predicted events
    '''
    seg_len = seg_len

    # If too many OOM error
    # device = 'cpu'
    # encoder = encoder.to(device)

    # Create waveform Dataloader if AVES
    X_pos, X_neg, X_query, start_query = generate_pos_neg_query_segments(waveform, df_annot, seg_len, sr=sr)

    'List for storing the combined probability across all iterations'
    prob_comb = []

    # if len(X_pos) < n_shot:
    #     print("Not enough positive samples, file is skipped")
    #     return [], [],
   
    # If the file have enough annotation, process to evaluation
    # X_pos = torch.tensor(torch.stack(X_pos, dim=0))
    # pos_dataset = torch.utils.data.TensorDataset(X_pos)
    # pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, batch_sampler=None,batch_size=conf.eval.pos_batch_size,shuffle=False)

    # If the file have enough annotation, process to evaluation
    X_neg = torch.tensor(torch.stack(X_neg, dim=0))
    neg_dataset = torch.utils.data.TensorDataset(X_neg)
    neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, batch_sampler=None,batch_size=batch_size,shuffle=False)

    X_query = torch.tensor(torch.stack(X_query, dim=0))
    query_dataset = torch.utils.data.TensorDataset(X_query)
    q_loader = torch.utils.data.DataLoader(dataset=query_dataset, batch_sampler=None,batch_size=batch_size,shuffle=False)
    

    # print("\n Adaptative segment length: Average length of positive samples: ", seg_len)
    print(f"Creating positive prototype from {len(X_pos)} samples")

    # pos_iterator = iter(pos_loader)
    feat_array_pos = torch.Tensor().to(device)

    # for batch in tqdm(pos_iterator):
    #     x_pos = batch[0]
    #     x_pos = x_pos.to(device)
    #     feat_pos = encoder(x_pos)
    #     feat_array_pos = torch.cat((feat_array_pos, feat_pos), dim=0)
                            
    # # Compute positive prototype as the mean of all positive embeddings
    # pos_proto = feat_array_pos.mean(dim=0).to(device)


    with torch.no_grad():
        torch.cuda.empty_cache()
        for pos_sample in tqdm(X_pos):
            try:
                feat = encoder(pos_sample.unsqueeze(0).to(device))
            except:
                breakpoint()
            # feat = feat.cpu()
            feat_mean = feat.mean(dim=0).unsqueeze(0)
            feat_array_pos = torch.cat((feat_array_pos, feat_mean), dim=0)
        pos_proto = feat_array_pos.mean(dim=0)


        prob_pos_iter = []
        
        print(f"Creating negative prototype from {len(X_neg)} samples")

        neg_iterator = iter(neg_loader)
        feat_array_neg = torch.Tensor().to(device)

        for batch in tqdm(neg_iterator):
            x_neg = batch[0]
            x_neg = x_neg.to(device)
            feat_neg = encoder(x_neg)
            feat_array_neg = torch.cat((feat_array_neg, feat_neg), dim=0)
                                
        # Compute negative prototype as the mean of all negative embeddings
        proto_neg = feat_array_neg.mean(dim=0).to(device)

        # Create query set
        print("Evaluating query set with prototypes")
        q_iterator = iter(q_loader)
        for batch in tqdm(q_iterator):
            x_q = batch[0]
            x_q = x_q.to(device)
            x_query = encoder(x_q)

            pos_proto = pos_proto.detach().cpu()
            proto_neg = proto_neg.detach().cpu()
            x_query = x_query.detach().cpu()
            
            probability_pos = get_probability(pos_proto, proto_neg, x_query)
            prob_pos_iter.extend(probability_pos)

    prob_comb.append(prob_pos_iter)
    prob_final = np.mean(np.array(prob_comb),axis=0)
    
    thresh = 0.5
    
    krn = np.array([1, -1])
    prob_thresh = np.where(prob_final > thresh, 1, 0)

    # prob_pos_final = prob_final * prob_thresh
    
    changes = np.convolve(krn, prob_thresh)

    # onset = start of events, offset = end of events
    onset_frames = np.where(changes == 1)[0]
    offset_frames = np.where(changes == -1)[0]

    onset = onset_frames * seg_len
    onset = onset + start_query / sr

    offset = offset_frames * seg_len
    offset = offset + start_query / sr
    assert len(onset) == len(offset)
    return onset, offset



if __name__ == "__main__":
    # Get one file paths
    filepath = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/temp_data_move_to_test_fast/Train/HT/a1.wav'
    annot_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/temp_data_move_to_test_fast/Train/HT/a1.csv'

    

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
from src.models.components.protonet import AvesClassifier, ResNet
from glob import glob
import numpy as np
import librosa
import librosa.display
import time
import datetime
import random
from Z_evaluate_proto import evaluate_prototypes_AVES, get_probability
from pathlib import Path
from utils import sliding_window_cuting


def tsne_embedding(X, n_components, perplexity = 10, n_iter = 1000, learning_rate='auto'):
  # Fit and transform X into a reduced space of n_components dimensions using T-SNE
  # https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600
  tsne = TSNE(n_components,perplexity=perplexity,n_iter=n_iter, method='exact', learning_rate=learning_rate, verbose=0, init='pca')
  return tsne.fit_transform(X)


if __name__ == "__main__":
    
    file_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
    extension = '*.csv'
    
    training = True
    if training:
        # Params and model
        SR = 16000
        BATCH_SIZE = 8
        FAST_DEV_RUN = True # Activate to test on one file only
        feature_extractor = 'AVES'  # 'BioLingual' or 'AVES'
        device = torch.device('cuda')

        # BioLingual model
        if feature_extractor == 'BioLingual':
            model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
            processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", sampling_rate=SR)
        
        # AVES model
        if feature_extractor == 'AVES': 
            device = 'cuda'
            model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
            model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
            model = AvesClassifier()
        
        model.to(device)
        model.eval()

        # FETCHING FILES
        all_csv_files = [file for path_dir, _, _ in os.walk(file_dir) for file in glob(os.path.join(path_dir, extension))]
        all_csv_files = sorted(all_csv_files)

        all_wav_files = [csv_file.replace('.csv', '.wav') for csv_file in all_csv_files]

        if FAST_DEV_RUN:
            # Select random file to run a test of the code on
            # idx = random.randint(0, len(all_csv_files) - 1)
            # idx = 12
            # all_csv_files = [all_csv_files[idx]]
            # all_wav_files = [all_wav_files[idx]]

            subset = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
            all_csv_files = [all_csv_files[i] for i in subset]
            all_wav_files = [all_wav_files[i] for i in subset]
        
        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])

        print(f"Number of files: {len(all_csv_files)}")
        for wav_file, annot_file in tqdm(zip(all_wav_files, all_csv_files), total=len(all_csv_files)):

            print('\n ---------------------------------------------------------------')
            print(f" File {os.path.basename(wav_file)}")
            print(' ---------------------------------------------------------------')

            #Load file and annotations
            waveform, sr = torchaudio.load(wav_file)

            if sr != SR:
                transform = torchaudio.transforms.Resample(sr, SR)
                waveform = transform(waveform)
                sr = SR

            # Normalize the waveform
            waveform = (waveform - waveform.mean())/waveform.std()
            waveform = waveform[0].numpy()

            df_annot = pd.read_csv(annot_file, sep=',').sort_values(by=['Starttime'])

            # Get annot until the first 5 positive annotations
            pos_indices = df_annot[df_annot['Q'] == 'POS'].index[:5]
            last_annot_index = max(pos_indices)
            few_shot_df = df_annot.loc[:last_annot_index]

            # Get the mean duration of positive annotations
            df_pos_annot = df_annot[df_annot.Q == 'POS']
            mean_pos_duration = (df_pos_annot['Endtime'].head(5) - df_pos_annot['Starttime'].head(5)).mean()

            # Adaptative sliding window length
            # wind_dur = mean_pos_duration
            if wind_dur > 0.5:
                wind_dur = 0.5
            elif(wind_dur < 0.025):
                wind_dur = 0.025
            
            # print(f"Average pos annot {mean_pos_duration} --> Sliding window duration: ", wind_dur)

            win_coverage_threshold = 1 # Segment is positive if annotations cover more than 30% of the window
            annot_coverage_threshold = 0.5 # Segment is positive if more than 50% of the annotation is is the window
            # wind_dur = 0.2
            overlap_ratio = 0.5

            # Get Pos, Neg and Query segments
            start_query_sample = int(few_shot_df['Endtime'].iloc[-1] * sr)
            waveform_few_shots, waveform_query = waveform[:start_query_sample],  waveform[start_query_sample:]

            # Get the positive segments
            X_pos = []
            pos_annot_bounds = []

            for i, row in df_pos_annot.iterrows():
                start_wav = int(row['Starttime']*sr)
                end_wav = int(row['Endtime']*sr)
                # AVES minimal input is 25 ms, if smaller, make the segment 20 ms long
                if (end_wav - start_wav)/sr < 0.025:
                    end_wav = int(start_wav + 0.025*sr)
                pos_annot_bounds.append((start_wav, end_wav))
                X_pos.append(waveform[0][start_wav:end_wav])

            # Create a new waveform with only the sections between the postitive annotations
            for bound in reversed(pos_annot_bounds[:-1]):
                waveform_few_shots = torch.cat((waveform_few_shots[:bound[0]], waveform_few_shots[bound[1]:]), 0)
            
            # Compute the negative sample of segment_length and discard the rest
            num_segments_query = len(waveform_few_shots) // wind_dur
            assert num_segments_query != 0

            X_neg = np.array_split(waveform_few_shots[:num_segments_query * wind_dur], num_segments_query)

            # df_few_shots = sliding_window_cuting(waveform_few_shots, few_shot_df, SR, wind_dur, win_coverage_threshold, annot_coverage_threshold, overlap_ratio=overlap_ratio)
            df_query = sliding_window_cuting(waveform_query, None, SR, wind_dur, win_coverage_threshold, annot_coverage_threshold, overlap_ratio=overlap_ratio)

            # Convert to list
            X_query = [torch.tensor(row['Audio']) for _, row in df_query.iterrows()]

            'List for storing the combined probability across all iterations'
            prob_comb = []
            batch_size = 16

            X_pos = torch.tensor(torch.stack(X_pos, dim=0))
            pos_dataset = torch.utils.data.TensorDataset(X_pos)
            pos_loader = torch.utils.data.DataLoader(dataset=pos_dataset, num_workers=20, batch_size=batch_size,shuffle=False)

            X_neg = torch.tensor(torch.stack(X_neg, dim=0))
            neg_dataset = torch.utils.data.TensorDataset(X_neg)
            neg_loader = torch.utils.data.DataLoader(dataset=neg_dataset, num_workers=20, batch_size=batch_size,shuffle=False)

            X_query = torch.tensor(torch.stack(X_query, dim=0))
            query_dataset = torch.utils.data.TensorDataset(X_query)
            q_loader = torch.utils.data.DataLoader(dataset=query_dataset, num_workers=20, batch_size=64,shuffle=False)
            
            # print("\n Adaptative segment length: Average length of positive samples: ", seg_len)

            # pos_iterator = iter(pos_loader)
            feat_array_pos = torch.Tensor().to(device)

            with torch.no_grad():
                torch.cuda.empty_cache()

                print(f"Creating positive prototype from {len(X_pos)} samples")
                for batch in tqdm(pos_loader):
                    x_pos = batch[0]
                    x_pos = x_pos.to(device)
                    feat_pos = model(x_pos)
                    feat_array_pos = torch.cat((feat_array_pos, feat_pos), dim=0)                     
                # Compute positive prototype as the mean of all positive embeddings
                pos_proto = feat_array_pos.mean(dim=0).to(device)

                prob_pos_iter = []
                
                print(f"Creating negative prototype from {len(X_neg)} samples")

                neg_iterator = iter(neg_loader)
                feat_array_neg = torch.Tensor().to(device)

                for batch in tqdm(neg_iterator):
                    x_neg = batch[0]
                    x_neg = x_neg.to(device)
                    feat_neg = model(x_neg)
                    feat_array_neg = torch.cat((feat_array_neg, feat_neg), dim=0)
                                        
                # Compute negative prototype as the mean of all negative embeddings
                proto_neg = feat_array_neg.mean(dim=0).to(device)

                # Create query set
                print("Evaluating query set with prototypes")
                q_iterator = iter(q_loader)
                for batch in tqdm(q_iterator):
                    x_q = batch[0]
                    x_q = x_q.to(device)
                    x_query = model(x_q)

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

            onset = onset_frames * wind_dur
            onset = onset + start_query_sample / sr

            offset = offset_frames * wind_dur
            offset = offset + start_query_sample / sr
            assert len(onset) == len(offset)

            # Save predictions
            name = np.repeat(os.path.basename(wav_file),len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        test_name = f'TEST002'
        prediction_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/{test_name}.csv'
        df_out.to_csv(prediction_path,index=False)
        print("File saved at ", prediction_path)

    # threshold sweep test
    threshold_range = np.arange(0.1, 0.9, 0.1)
    threshold_range = [0.5]
    for threshold in threshold_range:
        test_name = 'TEST002'
        print('Threshold = ', threshold)
        prediction_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/{test_name}.csv'
        
        post_process = True
        if post_process:
            from src.utils.post_proc import post_processing
            post_proc_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/{test_name}_postprocessed.csv'
            post_processing(file_dir, prediction_path, post_proc_path, n_shots=5, threshold=threshold)
            prediction_path = post_proc_path

        # Evaluate result
        eval_predictions = True
        if eval_predictions:
            from src.utils.evaluation import evaluate
    
            ref_files_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
            team_name = 'TEST_seglen_0.2'
            dataset = 'VAL'
            savepath = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/'

            print('Evaluating file ', prediction_path)
            evaluate(prediction_path, ref_files_path, team_name, dataset, savepath)

import torch
import pandas as pd
import numpy as np
import torchaudio
from tqdm import tqdm

import os
from sklearn.manifold import TSNE
# from src.models.components.protonet import AvesClassifier, ResNet
from model_few_shots_pl_module import AvesClassifier, AvesModule, BioLingualClassifier

from glob import glob
import numpy as np
from Z_evaluate_proto import get_probability
from utils import sliding_window_cuting, get_pos_neg_segments

# Allocated max memory to cuda
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10500"


def tsne_embedding(X, n_components, perplexity = 10, n_iter = 1000, learning_rate='auto'):
  # Fit and transform X into a reduced space of n_components dimensions using T-SNE
  # https://distill.pub/2016/misread-tsne/?_ga=2.135835192.888864733.1531353600-1779571267.1531353600
  tsne = TSNE(n_components,perplexity=perplexity,n_iter=n_iter, method='exact', learning_rate=learning_rate, verbose=0, init='pca')
  return tsne.fit_transform(X)


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    file_dir = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
    extension = '*.csv'
    
    create_eval = True
    if create_eval:
        # Params and model
        SR = 44100
        BATCH_SIZE = 1
        FAST_DEV_RUN = True # Activate to test on one file only
        feature_extractor = 'BioLingual'  # 'BioLingual' or 'AVES'
        device = torch.device('cuda')

        # BioLingual model
        if feature_extractor == 'BioLingual':
            # SR = 16000
            model = BioLingualClassifier(SR)
            model.to(device)
            model.eval()
            min_window = 0.01

        # AVES model
        if feature_extractor == 'AVES': 
            device = 'cuda'
            model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
            model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
            model = AvesClassifier(model_path=model_path, model_config_path=model_config_path)

            checkpoint_file = 'olive-bush-222_epoch=9-val_loss=0.00-val_macro_f1=1.00.ckpt'
            checkpoint_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/checkpoints/{checkpoint_file}'
            lightning_module = AvesModule.load_from_checkpoint(checkpoint_path)
            model = lightning_module.model
            min_window = 0.025
        
            model.to(device)
            model.eval()

        # FETCHING FILES
        all_csv_files = [file for path_dir, _, _ in os.walk(file_dir) for file in glob(os.path.join(path_dir, extension))]
        all_csv_files = sorted(all_csv_files)

        all_wav_files = [csv_file.replace('.csv', '.wav') for csv_file in all_csv_files]

        if FAST_DEV_RUN:
            # Select random file to run a test of the code on
            # idx = random.randint(0, len(all_csv_files) - 1)
            idx = 12
            all_csv_files = [all_csv_files[idx]]
            all_wav_files = [all_wav_files[idx]]

            # subset = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
            # all_csv_files = [all_csv_files[i] for i in subset]
            # all_wav_files = [all_wav_files[i] for i in subset]
        
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
            wind_dur = mean_pos_duration
            if wind_dur > 1.:
                wind_dur = 1.
            elif(wind_dur < 0.025):
                wind_dur = 0.025
            
            print(f"Average pos annot {mean_pos_duration} --> Sliding window duration: ", wind_dur)

            # Get Pos, Neg and Query segments
            start_query_sample = int(few_shot_df['Endtime'].iloc[-1] * sr) 
            waveform_few_shots, waveform_query = waveform[:start_query_sample],  waveform[start_query_sample:]

            # df_few_shots = sliding_window_cuting(waveform_few_shots, few_shot_df, SR, wind_dur, win_coverage_threshold, annot_coverage_threshold, overlap_ratio=0.)
            df_few_shots = get_pos_neg_segments(waveform_few_shots, few_shot_df, SR, wind_dur, min_window=min_window)
            df_query = sliding_window_cuting(waveform_query, None, SR, wind_dur)
            
            df_pos = df_few_shots[df_few_shots['Label'] == 'Q']
            df_neg = df_few_shots[df_few_shots['Label'] == 'NEG']

            X_pos = [torch.tensor(row['Audio']) for _, row in df_pos.iterrows()]
            X_neg = [torch.tensor(row['Audio']) for _, row in df_neg.iterrows()]
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
            

            # pos_iterator = iter(pos_loader)
            feat_array_pos = torch.Tensor().to(device)

            with torch.no_grad():
                torch.cuda.empty_cache()

                # print(f"Creating positive prototype from {len(X_pos)} samples")
                for batch in pos_loader:
                    x_pos = batch[0]
                    x_pos = x_pos.to(device)
                    feat_pos = model(x_pos)
                    feat_array_pos = torch.cat((feat_array_pos, feat_pos), dim=0)                     
                # Compute positive prototype as the mean of all positive embeddings
                pos_proto = feat_array_pos.mean(dim=0).to(device)

                prob_pos_iter = []
                
                # print(f"Creating negative prototype from {len(X_neg)} samples")

                neg_iterator = neg_loader
                feat_array_neg = torch.Tensor().to(device)

                for batch in neg_iterator:
                    x_neg = batch[0]
                    x_neg = x_neg.to(device)
                    feat_neg = model(x_neg)
                    feat_array_neg = torch.cat((feat_array_neg, feat_neg), dim=0)
                                        
                # Compute negative prototype as the mean of all negative embeddings
                proto_neg = feat_array_neg.mean(dim=0).to(device)


                # Create query set
                # print("Evaluating query set with prototypes")
                q_iterator = iter(q_loader)
                for batch in q_iterator:
                    x_q = batch[0]
                    x_q = x_q.to(device)
                    x_query = model(x_q)

                    pos_proto = pos_proto.detach().cpu()
                    proto_neg = proto_neg.detach().cpu()
                    x_query = x_query.detach().cpu()
                    
                    probability_pos = get_probability(pos_proto, proto_neg, x_query)
                    prob_pos_iter.extend(probability_pos)
                

            prob_comb.append(prob_pos_iter)
            prob_final = np.array(prob_comb)[0]
            
            thresh = 0.5
            
            krn = np.array([1, -1])
            prob_thresh = np.where(prob_final > thresh, 1, 0)            
            changes = np.convolve(krn, prob_thresh)

            # onset = start of events, offset = end of events
            onset_frames = np.where(changes == 1)[0]
            offset_frames = np.where(changes == -1)[0]

            # Get onset and offset time in seconds (query time + frames*stepsize)
            overlap_ratio = 0
            onset = start_query_sample / sr + onset_frames * (wind_dur*(1-overlap_ratio)) 
            offset = start_query_sample / sr + offset_frames * (wind_dur*(1-overlap_ratio))

            assert len(onset) == len(offset)

            # Save predictions
            name = np.repeat(os.path.basename(wav_file),len(onset))
            name_arr = np.append(name_arr,name)
            onset_arr = np.append(onset_arr,onset)
            offset_arr = np.append(offset_arr,offset)

        df_out = pd.DataFrame({'Audiofilename':name_arr,'Starttime':onset_arr,'Endtime':offset_arr})
        test_num = '01'
        test_name = f'BioLingual_one_file{test_num}'
        prediction_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/{test_name}.csv'
        df_out.to_csv(prediction_path,index=False)
        print("File saved at ", prediction_path)


    # threshold sweep test
    threshold_range = np.arange(0.5, 0.9, 0.1)
    # test_name = f'TEST005'

    threshold_range = [0.7]
    for threshold in threshold_range:
        print('Threshold = ', threshold)
        prediction_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/{test_name}.csv'
        
        post_process = True
        if post_process:
            from src.utils.post_proc import post_processing
            post_proc_path = f'/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/{test_name}_postprocessed.csv'
            post_processing(file_dir, prediction_path, post_proc_path, n_shots=5, threshold=threshold)
            prediction_path = post_proc_path

        # Evaluate result
        eval_predictions = True
        if eval_predictions:
            from src.utils.evaluation import evaluate
    
            ref_files_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/dcase2024_task5/Development_Set/Validation_Set/'
            team_name = f'one_file12_{test_num}_BioLingual'
            dataset = f'VAL_threshold_{threshold}'
            savepath = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/'

            print('Evaluating file ', prediction_path)
            evaluate(prediction_path, ref_files_path, team_name, dataset, savepath)


import torch
import pandas as pd
import numpy as np

import numpy as np
from torchaudio.models import wav2vec2_model
import json
import torch.nn as nn
import torch.nn.functional as F
import os

from pytorch_lightning import LightningModule, seed_everything

import torchmetrics
from transformers import ClapModel, ClapProcessor
from utils import get_probability, euclidean_dist
        
class FusionLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(FusionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, loss1, loss2):        
        # Compute the weighted sum of the losses
        total_loss = self.alpha * loss1 + self.beta * loss2
        return total_loss

from torch_audiomentations import (
    Compose,
    AddColoredNoise,
    PitchShift,
    Shift,
    Gain,
)


apply_augmentations = Compose(
    transforms=[
        AddColoredNoise(min_snr_in_db=5, max_snr_in_db=25, min_f_decay=0, max_f_decay=0, p=0.7, mode="per_channel"),
        Gain(min_gain_in_db=-10, max_gain_in_db=10, p=0.7, mode='per_channel'),
        PitchShift(min_transpose_semitones=-1, max_transpose_semitones=1, p=0.7, sample_rate=16000, mode='per_channel'),
        Shift(min_shift=-0.25, max_shift=0.25, p=0.7, mode='per_channel'),
    ]
)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        mode="elucidean",
        base_temperature=0.07,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.mode = mode
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def get_cdist(self, mat1, mat2):
        ret = torch.zeros((mat1.size(0), mat2.size(0))).to(mat1.device)
        for i in range(mat1.size(0)):
            for j in range(mat2.size(0)):
                ret[i, j] = torch.sqrt(torch.sum((mat1[i] - mat2[j]) ** 2))
        return ret

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        # import ipdb; ipdb.set_trace()
        # anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        anchor_dot_contrast = torch.div(
            torch.cdist(anchor_feature, contrast_feature, p=2.0), self.temperature
        )
        # anchor_dot_contrast = torch.div(self.get_cdist(anchor_feature, contrast_feature), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# --------------------------------
#    MODEL
# --------------------------------
class BioLingualClassifier(nn.Module):
    def __init__(self, sr, n_classes=2):
        super().__init__()
        self.sr = sr
        self.n_classes = n_classes
        self.model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        self.processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", sampling_rate=self.sr)
        # self.dropout = nn.Dropout(0.5)
        # self.classifier_head = nn.Linear(in_features=512, out_features=self.n_classes)

    def forward(self, x):
        np_x = x.cpu().squeeze().numpy()
        inputs = self.processor(audios=np_x, return_tensors="pt", sampling_rate=self.sr)
        inputs = inputs.to(x.device)
        mean_embedding = self.model.get_audio_features(**inputs)
        # mean_embedding_after_drop = self.dropout(mean_embedding)
        # logits = self.classifier_head(mean_embedding_after_drop)
        # return mean_embedding, logits
        return mean_embedding



class AvesClassifier(nn.Module):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(self, model_path, model_config_path, features=None, embedding_dim=768, sr=16000, n_classes=2, trainable=False):

        super().__init__()
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html    
        self.config = self.load_config(model_config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.trainable = trainable
        self.freeze_embedding_weights()
        # We will only train the classifier head
        self.audio_sr = sr
        self.features = features  
        self.embedding_dim = embedding_dim   

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def forward(self, sig):
        """
        Input
          sig (Tensor): (batch, time)
        Returns
          mean_embedding (Tensor): (batch, output_dim)
          logits (Tensor): (batch, n_classes)
        """
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        mean_embedding = out.mean(dim=1) #over time
        # return mean_embedding, logits
        return mean_embedding
    
    # Code to use while initially setting up the model
    def freeze_embedding_weights(self):
        """ Freeze weights in AVES embeddings for classification """
        # The convolutional layers should never be trainable
        self.model.feature_extractor.requires_grad_(False)
        self.model.feature_extractor.eval()
        # The transformers are optionally trainable
        for param in self.model.encoder.parameters():
            param.requires_grad = self.trainable

        if not self.trainable:
            # We also set layers without params (like dropout) to eval mode, so they do not change
            self.model.encoder.eval()
        
    def train(self, mode=True):
        super().train(mode)  # Call the original method to ensure standard behavior
        if mode:
           self.model.encoder.train()
        else:
            self.model.encoder.eval()

    def eval(self):
        self.train(False)  # This will call your overridden train method with mode=False



class AvesModule(LightningModule):
    def __init__(self, lr, weight_decay=0.001, trainable=True, query_time=None,file_path = None):
        super().__init__()

        model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
        model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
        self.model = AvesClassifier(model_path, model_config_path, trainable=trainable)
        # self.model = aves_model
        self.lr = lr
        self.weight_decay = weight_decay
        # self.macro_acc = torchmetrics.Accuracy(num_classes=2, average="macro")
        self.macro_f1 = torchmetrics.F1Score(num_classes=2, average="macro")

        self.save_hyperparameters()
        self.training_epoch_outputs = []
        self.training_epoch_labels = []
        self.protos = None
        self.loss_fn = SupConLoss()
        self.fusion_loss_fn = FusionLoss(alpha=0.3, beta=0.7)
        self.name_arr = np.array([])
        self.onset_arr = np.array([])
        self.offset_arr = np.array([])

        self.query_time = query_time
        self.file_path = file_path
        self.predict_csv_filepath = None
    
    def on_fit_start(self):
        # Generate random torch tensor prototypes
        if self.protos == None:
            self.protos = torch.rand(2, self.model.embedding_dim)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['Audio']
        y = batch['Label'].type(torch.cuda.LongTensor)

        # TODO - Apply augmentation here

        embedding = self.model(x)
        self.training_epoch_outputs.append(embedding.detach())
        self.training_epoch_labels.append(y.detach())

        # Proto loss
        # neg_proto = self.protos[0].to(embedding.device)
        # pos_proto = self.protos[1].to(embedding.device)

        # probability_pos = get_probability(neg_proto, pos_proto, embedding)
        # loss = F.binary_cross_entropy(probability_pos, y.float())


        # Supervised Contrastive Loss
        apply_augment = True
        if apply_augment:
            if x.shape[0] >= 16:
                n_augment = 4
            else:
                n_augment = 10 # TODO to parameters

            x_aug = apply_augmentations(x.unsqueeze(1).repeat(1, n_augment, 1), sample_rate=16000)

            # Pass each channel through the model by iterating over dimension 1
            embeddings = [self.model(x_aug[:,i,:].squeeze()).unsqueeze(1) for i in range(n_augment)]
            embed_array = torch.stack(embeddings, axis=1).squeeze()
            supconloss = self.loss_fn(embed_array, y)
        
        # else:

        #     # Assign positive prototype where y=1 and negative prototype where y=0
        #     prototypes_array = torch.where(y.unsqueeze(1) == 1, pos_proto, neg_proto).unsqueeze(1)

        #     # Skip the batch if size 1, loss cannot be computed
        #     if x.shape[0]==1:
        #         self.log_dict({'train_loss': 10, 'tr_SCL': 10}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #         return None
            
        #     # Supconloss
        #     supconloss = self.loss_fn(torch.stack([embedding.unsqueeze(1), prototypes_array], axis=1).squeeze(), y)

        #     # Try a fusion loss

        neg_idx = torch.where(y == 0)[0]
        pos_idx = torch.where(y == 1)[0]

        if len(neg_idx) == 0 or len(pos_idx) == 0:
            self.log_dict({'train_loss': 10}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            return None

    #     # update prototypes
        neg_batch_proto, pos_batch_proto = embedding[neg_idx].mean(axis=0).unsqueeze(0), embedding[pos_idx].mean(axis=0).unsqueeze(0)
        dist_proto = euclidean_dist(neg_batch_proto, pos_batch_proto)
        proto_dist_loss = 10*(1/dist_proto)

    # #  # Compute the total fusion loss
        loss = self.fusion_loss_fn(supconloss, proto_dist_loss)

        # Log training loss
        self.log_dict({'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    

    def on_train_epoch_end(self):
        # get index where label is 1 and label is 0
        embeddings = torch.cat(self.training_epoch_outputs)
        labels = torch.cat(self.training_epoch_labels)

        neg_idx = torch.where(labels == 0)[0]
        pos_idx = torch.where(labels == 1)[0]

        # update prototypes
        self.protos = torch.cat([embeddings[neg_idx].mean(axis=0).unsqueeze(0), embeddings[pos_idx].mean(axis=0).unsqueeze(0)], axis=0)

        self.training_epoch_outputs.clear()
        self.training_epoch_labels.clear()

        #Testing prototypes similarity
        dist_proto = euclidean_dist(embeddings[pos_idx].mean(axis=0).unsqueeze(0), embeddings[neg_idx].mean(axis=0).unsqueeze(0))
        print("Similarity between pos and neg proto is ", 1/(1+dist_proto))
        print("Where 0 is very different and 1 very similar")


    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['Audio']
        y = batch['Label'].type(torch.cuda.LongTensor)
        embedding = self.model(x)
               
        probability_pos = get_probability(self.protos[0].to(embedding.device), self.protos[1].to(embedding.device), embedding)
        proto_loss = F.binary_cross_entropy(probability_pos, y.float())

        neg_proto = self.protos[0].to(embedding.device)
        pos_proto = self.protos[1].to(embedding.device)

        prototypes_array = torch.where(y.unsqueeze(1) == 1, pos_proto, neg_proto).unsqueeze(1)

        supconloss = self.loss_fn(torch.stack([embedding.unsqueeze(1), prototypes_array], axis=1).squeeze(), y)

        # ce_loss = F.cross_entropy(pred, y)
        # Log validation loss
        # Could try the Focal loss as well

        # accuracy = self.macro_acc(pred, y)

        # Convert probability_pos to 0 and 1
        pred = torch.where(probability_pos > 0.5, torch.ones_like(y), torch.zeros_like(y))

        f1 = self.macro_f1(pred.type(torch.cuda.LongTensor), y)

        self.log_dict({'proto_loss': proto_loss, 'vd_SCL': supconloss, 'val_macro_f1': f1}, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        # Log validation loss
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return supconloss

    def test_step(self, batch, batch_idx):

        x = batch['Audio']
        embedding = self.model(x)

        neg_proto = self.protos[0].to(embedding.device)
        pos_proto = self.protos[1].to(embedding.device)
         # Create query set
                
        probability_pos = get_probability(neg_proto, pos_proto, embedding)            

        prob_final = np.array(probability_pos.cpu())
        thresh = 0.5
        
        krn = np.array([1, -1])
        prob_thresh = np.where(prob_final > thresh, 1, 0)

        # prob_pos_final = prob_final * prob_thresh
        
        changes = np.convolve(krn, prob_thresh)

        # onset = start of events, offset = end of events
        onset_frames = np.where(changes == 1)[0]
        offset_frames = np.where(changes == -1)[0]

        # Get onset and offset time in seconds (query time + batch_start_time + onsets/offsets)
        batch_size = x.shape[0]
        wind_dur = x.shape[1]/self.model.audio_sr

        onset = self.query_time + (batch_idx*batch_size + onset_frames)*wind_dur
        offset = self.query_time + (batch_idx*batch_size + offset_frames)*wind_dur

        assert len(onset) == len(offset)

        # Save predictions
        name = np.repeat(os.path.basename(self.file_path),len(onset))
        self.name_arr = np.append(self.name_arr,name)
        self.onset_arr = np.append(self.onset_arr,onset)
        self.offset_arr = np.append(self.offset_arr,offset)


    def on_test_end(self):

        df_out = pd.DataFrame({'Audiofilename':self.name_arr,'Starttime':self.onset_arr,'Endtime':self.offset_arr})
        df_out.sort_values(by=['Starttime'],inplace=True)

        prediction_path = self.predict_csv_filepath
        # if csv not exist, create it
        if not os.path.exists(prediction_path):
            df_out.to_csv(prediction_path,index=False)
        else:
            df_out.to_csv(prediction_path,mode='a',index=False,header=False)

        print("Prediction saved at ", prediction_path)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer




if __name__ == '__main__':
    # Trying to load from a checkpoint - working
    # checkpoint_path = '/home/reindert/Valentin_REVO/DCASE_2024/dcase-few-shot-bioacoustic/baselines/my_method/myresult/checkpoints/epoch=1-val_loss=0.05-val_macro_f1=1.00.ckpt'
    # # my_module = AvesModule.load_from_checkpoint(checkpoint_path)
    # model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    # model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
    # aves_model_empty = AvesClassifier(model_path=model_path, model_config_path=model_config_path, trainable=False)
    # # mymodule = AvesModule(aves_model_empty, lr=5e-6, weight_decay=0.01)
    # lightning_module = AvesModule.load_from_checkpoint(checkpoint_path)
    # model = lightning_module.model

    # # create random torch tensor
    # x = torch.randn(1, 32000)
    # test = model(x)
    # test2 = aves_model_empty(x)

    # print(test[0] == test2[0])

    # Testing to load BioLingual
    device = 'cuda'
    SR = 22500
    model = BioLingualClassifier(SR)
    model_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.pt"
    model_config_path = "/home/reindert/Valentin_REVO/Ressource/aves-base-bio.torchaudio.model_config.json"
    model2 = AvesClassifier(model_path, model_config_path)

    # Create a random numpy array
    x = np.random.rand(11025)
    x2 = torch.rand((8, 32000))
    x3 = x2.numpy()

    with torch.no_grad():
        features, logit = model(x)
        features2, logit2 = model2(x2)
    
    breakpoint()

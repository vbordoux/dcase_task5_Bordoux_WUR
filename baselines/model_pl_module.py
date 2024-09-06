
import torch
import pandas as pd
import numpy as np

from glob import glob
import numpy as np
import time
import datetime
import random
from Z_evaluate_proto import evaluate_prototypes_AVES
from pathlib import Path
from torchaudio.models import wav2vec2_model
import json
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule, seed_everything

import torchmetrics
from transformers import ClapModel, ClapProcessor


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
        self.dropout = nn.Dropout(0.5)
        self.classifier_head = nn.Linear(in_features=512, out_features=self.n_classes)

    def forward(self, x):
        np_x = x.cpu().squeeze().numpy()
        inputs = self.processor(audios=np_x, return_tensors="pt", sampling_rate=self.sr)
        inputs = inputs.to(x.device)
        mean_embedding = self.model.get_audio_features(**inputs)
        mean_embedding_after_drop = self.dropout(mean_embedding)
        logits = self.classifier_head(mean_embedding_after_drop)
        # return mean_embedding, logits
        return mean_embedding, logits



class AvesClassifier(nn.Module):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(self, model_path, model_config_path, features=None, embedding_dim=768, sr=48000, n_classes=2, trainable=False):

        super().__init__()
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html    
        self.config = self.load_config(model_config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.trainable = trainable
        self.freeze_embedding_weights()
        # We will only train the classifier head
        self.dropout = nn.Dropout(0.2)
        self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes)
        self.audio_sr = sr
        self.features = features       

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
        mean_embedding_after_drop = self.dropout(mean_embedding)
        logits = self.classifier_head(mean_embedding_after_drop)
        # return mean_embedding, logits
        return mean_embedding, logits
    
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

    # Code to use during training loop, to switch between eval and train mode
    def set_eval_aves(self):
        """ Set AVES-based classifier to eval mode. Takes into account whether we are training transformers """
        self.classifier_head.eval()
        self.dropout.eval()
        self.model.encoder.eval()

    def set_train_aves(self):
        """ Set AVES-based classifier to train mode. Takes into account whether we are training transformers """
        # Always train the classifier head
        self.classifier_head.train()
        self.dropout.train()
        # Optionally train the transformer of the model
        if self.trainable:
            self.model.encoder.train()



class AvesModule(LightningModule):
    def __init__(self, lr, weight_decay=0.001, trainable=True):
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

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['Audio']
        y = batch['Label'].type(torch.cuda.LongTensor)
        embedding, pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        # Log training loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = batch['Audio']
        y = batch['Label'].type(torch.cuda.LongTensor)
        embedding, pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        # accuracy = self.macro_acc(pred, y)
        f1 = self.macro_f1(pred, y)

        self.log_dict({'val_loss': loss, 'val_macro_f1': f1}, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        # Log validation loss
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def eval_set(self,batch, batch_idx):
        pass

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

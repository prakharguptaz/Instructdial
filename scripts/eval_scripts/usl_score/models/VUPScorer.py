
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, BertConfig
from collections import namedtuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VUPScorer(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=hparams.dropout, inplace=False)
        self.linear = torch.nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        last_hidden_state, _= self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        #max-pooling
        attention_mask = attention_mask.unsqueeze(-1).repeat((1,1,768))
        min_values = (torch.ones_like(attention_mask) * -100).type(torch.FloatTensor).to(device)
        hidden_state = attention_mask * last_hidden_state
        hidden_state = torch.where(attention_mask != 0, hidden_state, min_values)
        hidden_state, _ = hidden_state.max(dim=1)

        hidden_state = self.dropout(hidden_state)
        output = self.linear(hidden_state)
        return output

    def predict(self, x):
        instance = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=self.hparams.res_token_len,
            pad_to_max_length=True,
            return_tensors="pt",
            truncation=True
        )
        input_ids = instance['input_ids'].to(device)
        token_type_ids = instance['token_type_ids'].to(device)
        attention_mask = instance['attention_mask'].type(torch.FloatTensor).to(device)
        output = self(input_ids, token_type_ids, attention_mask)
        output = F.softmax(output, dim=1)
        output = output[:,1]
        return output.item()

    def training_step(self, batch, batch_nb):
        batch = [ x.to(device) for x in batch ]
        input_ids, token_type_ids, attention_mask, label = batch
        input_ids = input_ids.squeeze(1).to(device)
        token_type_ids = token_type_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).type(torch.FloatTensor).to(device)

        output = self(input_ids, token_type_ids, attention_mask)
        loss = F.cross_entropy(output, label)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        output = self.training_step(batch, batch_nb)
        loss = output['loss']
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        print ("val_loss: ", avg_loss)
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

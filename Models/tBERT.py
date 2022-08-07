import torch
from torch import nn
from transformers import BertModel, BertTokenizer, logging
from sklearn.decomposition import LatentDirichletAllocation as LDA

#Set the verbosity level
logging.set_verbosity_error()

torch.seed = 123

class tBERT(nn.Module):
    def __init__(self, num_topics=80, batch_size=64, corpus=['']):
        super(tBERT, self).__init__()

        # Loading Bert model
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_bert = BertModel.from_pretrained("bert-base-uncased")
        # Disable Bert grad
        for param in self.model_bert.parameters():
          param.requires_grad = False
        
        # LDA
        self.lda = LDA(n_components=num_topics, batch_size=batch_size,n_jobs=-1)
        corpus = self.tokenize(corpus)
        self.lda.fit(corpus)

        n_classes=1
        self.classifier = nn.Sequential(
            nn.Linear(768 + 2 * num_topics, 464, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(464, n_classes, bias=True),
            nn.Sigmoid()
        )

    def tokenize(self, sentence_to_tokenize):
      return self.tokenizer_bert(sentence_to_tokenize,return_tensors='pt',padding="max_length", pad_to_max_length=True,truncation=True)["input_ids"]

    def tokenize_2_sentences(self,sentence1,sentence2):
      return self.tokenizer_bert(sentence1,sentence2,return_tensors='pt',padding="max_length", pad_to_max_length=True,truncation=True)["input_ids"]

    def freeze_BERT(self,):
      for param in self.model_bert.parameters():
        param.requires_grad = False

    def forward(self, inputs):
      inputs1,inputs2 = inputs
      
      # Handle Bert part
      # tokenizer
      tokens_bert = self.tokenize_2_sentences(inputs1,inputs2).cuda()
      
      # features
      features_bert = self.model_bert(tokens_bert).last_hidden_state[:, 0, :]


      # Run LDA
      tokens_lda1 = self.tokenize(inputs1)
      features_lda1 = self.lda.transform(tokens_lda1)
      features_lda1 = torch.tensor(features_lda1).type(torch.FloatTensor).cuda()
      tokens_lda2 = self.tokenize(inputs2)
      features_lda2 = self.lda.transform(tokens_lda2)
      features_lda2 = torch.tensor(features_lda2).type(torch.FloatTensor).cuda()
      
      # concatenate lda features
      features_lda = torch.cat([features_lda1, features_lda2],dim=1)
      features_lda = features_lda

      # concatenate
      features = torch.cat([features_lda, features_bert],dim=1)
      self.aa = features
      self.bb = features_bert
      
      # classification
      outputs = self.classifier(features)
      return outputs 
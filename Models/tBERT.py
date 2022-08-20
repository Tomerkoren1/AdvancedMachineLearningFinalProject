import torch
from torch import nn
from transformers import BertModel, BertTokenizer, logging
from Models.LDA import LDA
from nltk.tokenize import word_tokenize

#Set the verbosity level
logging.set_verbosity_error()

torch.seed = 123

class tBERT(nn.Module):
    def __init__(self, num_topics=80, data_words=[''], use_lda=False, bert_update=True):
        super(tBERT, self).__init__()

        # Loading Bert model
        self.tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model_bert = BertModel.from_pretrained("bert-base-uncased")
        # Disable Bert grad
        if not bert_update:
          for param in self.model_bert.parameters():
            param.requires_grad = False
        
        # LDA
        self.use_lda = use_lda
        lda_part = 0
        if self.use_lda:
          self.lda_model = LDA(num_topics=num_topics, init_data_words=data_words)
          lda_part = 2 * num_topics

        n_classes=1
        self.classifier = nn.Sequential(
            nn.Linear(768 + lda_part, 464, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(464, n_classes, bias=True),
            nn.Sigmoid()
        )

    def tokenize(self, sentence_to_tokenize):
      return self.tokenizer_bert(sentence_to_tokenize,return_tensors='pt',padding="max_length", pad_to_max_length=True,truncation=True)["input_ids"]

    def tokenize_2_sentences(self,sentence1,sentence2):
      return self.tokenizer_bert(sentence1,sentence2,return_tensors='pt',padding="max_length", pad_to_max_length=True,truncation=True)["input_ids"]

    def forward(self, inputs):
      inputs1,inputs2 = inputs
      
      # Disable gradient calculations
      tokens_bert = self.tokenizer_bert(inputs1,inputs2,return_tensors='pt',padding=True, truncation=True)
      for t in tokens_bert:
        tokens_bert[t] = tokens_bert[t].cuda()

      # Get the model embeddings

      features_bert = self.model_bert(**tokens_bert)
      # features_bert.requires_grad = True
      CLSs = features_bert[0][:, 0, :]
      # normalize the CLS token embeddings
      features_bert = nn.functional.normalize(CLSs, p=2, dim=1)
      features = features_bert

      if self.use_lda:

        # preprocess and infer topics
        inputs1_p = [word_tokenize(st) for st in list(inputs1)]
        new_corpus_1, _, new_processed_texts_1 = self.lda_model.lda_preprocess(inputs1_p, delete_stopwords=True)
        topic_dist_1 = self.lda_model.infer_topic_dist(new_corpus_1)

        # preprocess and infer topics
        inputs2_p = [word_tokenize(st) for st in list(inputs2)]
        new_corpus_2, _, new_processed_texts_2 = self.lda_model.lda_preprocess(inputs2_p, delete_stopwords=True)
        topic_dist_2 = self.lda_model.infer_topic_dist(new_corpus_2)

        # sanity check
        assert(len(inputs1_p)==len(inputs2_p)==topic_dist_1.shape[0]==topic_dist_2.shape[0]) # same number of examples
        assert(topic_dist_1.shape[1]==topic_dist_2.shape[1]) # same number of topics
      
        # concatenate lda features
        features_lda = torch.cat([torch.tensor(topic_dist_1).type(torch.FloatTensor), torch.tensor(topic_dist_2).type(torch.FloatTensor)],dim=1)
        features_lda = features_lda.cuda()

        # concatenate
        features = torch.cat([features_lda, features_bert],dim=1)

      # classification
      outputs = self.classifier(features)
      return outputs
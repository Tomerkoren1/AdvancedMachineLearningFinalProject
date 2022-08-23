import pandas as  pd
from Dataset.PairSentencesDataset import PairSentencesDataset
from nlpaug.augmenter import word as naw
from tqdm import tqdm
tqdm.pandas()
import os


TOPK=20 #default=100
ACT = 'substitute' #"substitute" 'insert'
 
def load_data(use_aug=False):
    train_msrp = './Data/MSRP/MSRParaphraseCorpus/msr-para-train.tsv'
    val_msrp   = './Data/MSRP/MSRParaphraseCorpus/msr-para-val.tsv'
    test_msrp  = './Data/MSRP/MSRParaphraseCorpus/msr-para-test.tsv'

    train_msrp_df = pd.read_csv(train_msrp, sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()
    val_msrp_df   = pd.read_csv(val_msrp,   sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()
    test_msrp_df  = pd.read_csv(test_msrp,  sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()

    if use_aug:
        train_msrp_df = aug_data(train_msrp_df)

    return train_msrp_df, val_msrp_df, test_msrp_df

def create_datasets(train_msrp_df, val_msrp_df, test_msrp_df):
    train_msrp_dataset = PairSentencesDataset(train_msrp_df.string1.values, train_msrp_df.string2.values, train_msrp_df.label.values)
    val_msrp_dataset   = PairSentencesDataset(val_msrp_df.string1.values,   val_msrp_df.string2.values,   val_msrp_df.label.values)
    test_msrp_dataset  = PairSentencesDataset(test_msrp_df.string1.values,  test_msrp_df.string2.values,  test_msrp_df.label.values)

    return train_msrp_dataset, val_msrp_dataset, test_msrp_dataset

def aug_data(df):
    print("Run data augmentation")
    df_aug = df.copy()
    aug_bert = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', device='cuda', action=ACT, top_k=TOPK)

    df_aug['string1'] = aug_bert.augment(df_aug['string1'].tolist())
    df_aug['string2'] = aug_bert.augment(df_aug['string2'].tolist())
    df_aug = pd.concat([df_aug, df.dropna()], ignore_index=True)

    print("Finish data augmentation")
    del aug_bert
    return df_aug

import pandas as  pd
from Dataset.PairSentencesDataset import PairSentencesDataset


def load_data():
    train_msrp = './Data/MSRP/MSRParaphraseCorpus/msr-para-train.tsv'
    val_msrp   = './Data/MSRP/MSRParaphraseCorpus/msr-para-val.tsv'
    test_msrp  = './Data/MSRP/MSRParaphraseCorpus/msr-para-test.tsv'

    train_msrp_df = pd.read_csv(train_msrp, sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()
    val_msrp_df   = pd.read_csv(val_msrp,   sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()
    test_msrp_df  = pd.read_csv(test_msrp,  sep='\t', header=0, names=['label','id1','id2','string1','string2'],usecols=[0,3,4]).dropna()
    return train_msrp_df, val_msrp_df, test_msrp_df

def create_datasets(train_msrp_df, val_msrp_df, test_msrp_df):
    train_msrp_dataset = PairSentencesDataset(train_msrp_df.string1.values, train_msrp_df.string2.values, train_msrp_df.label.values)
    val_msrp_dataset   = PairSentencesDataset(val_msrp_df.string1.values,   val_msrp_df.string2.values,   val_msrp_df.label.values)
    test_msrp_dataset  = PairSentencesDataset(test_msrp_df.string1.values,  test_msrp_df.string2.values,  test_msrp_df.label.values)

    return train_msrp_dataset, val_msrp_dataset, test_msrp_dataset

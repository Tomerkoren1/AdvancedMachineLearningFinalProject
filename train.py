import pandas as  pd
from Models.tBERT import tBERT
from Trainer.tBertTrainer import tBertTrainer
from Dataset.PairSentencesDataset import PairSentencesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import wandb


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

def display_plots(df):
    sns.barplot(x="#Topics", y="F1", data=df)
    plt.show()
    sns.lineplot(x='Epoch', y="Train_Loss", hue="#Topics", data=df)
    plt.show()
    sns.lineplot(x='Epoch', y="Val_Loss", hue="#Topics", data=df)
    plt.show()

def train():
    # Params
    batch_size = 10
    lr = 3e-05
    epochs = 9
    use_lda = True

    use_wandb = False



    # Load data
    train_msrp_df, val_msrp_df, test_msrp_df = load_data()
    train_msrp_dataset, val_msrp_dataset, test_msrp_dataset = create_datasets(train_msrp_df, val_msrp_df, test_msrp_df)
    train_dataloader = DataLoader(train_msrp_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_msrp_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_msrp_dataset, batch_size=batch_size, shuffle=False)

    num_topics = [70, 75, 80, 85, 90]
    data_words = train_msrp_df.string1.values.tolist() + train_msrp_df.string2.values.tolist()
    df = pd.DataFrame(columns=['#Topics','F1','Train_Loss','Val_Loss','Epoch'])

    for n in num_topics:

        if use_wandb:
            config = {"learning_rate": lr, "epochs": epochs, "batch_size": batch_size, "num_topics":num_topics, "use_lda":use_lda}
            wandb.init(project="AML-FP", entity="tomerkoren", config=config)

        # model
        model = tBERT(data_words=data_words,num_topics=n,use_lda=use_lda).cuda()

        # train
        trainer = tBertTrainer(model, epochs=epochs, lr=lr, use_wandb=use_wandb)
        trainer.fit(train_dataloader, val_dataloader)

        # predict
        f1_score = trainer.test(test_dataloader)

        # save model results
        df_topic = pd.DataFrame({'#Topics':str(n),'F1':f1_score,'Train_Loss':trainer.loss_train,'Val_Loss':trainer.loss_val,'Epoch':range(epochs)})
        df = df.append(df_topic, ignore_index=True)

    # Plots
    display_plots(df)

if __name__ == '__main__':
    train()

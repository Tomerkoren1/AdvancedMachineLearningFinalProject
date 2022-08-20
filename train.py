import wandb
import pandas as  pd
from Models.tBERT import tBERT
from data import load_data, create_datasets
from Trainer.tBertTrainer import tBertTrainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")




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
            run = wandb.init(project="AML-FP", entity="tomerkoren", config=config, reinit=True)

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

        if use_wandb:
            wandb.log({'test_F1':f1_score})
            run.finish()

    # Plots
    display_plots(df)

if __name__ == '__main__':
    train()

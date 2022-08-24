import wandb
import pandas as  pd
from Models.tBERT import tBERT
from data import load_data, create_datasets
from Trainer.tBertTrainer import tBertTrainer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set_theme(style="whitegrid")

def cli():
    """ Handle argument parsing
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', type=float, default=3e-05,
                        help='learning rate')
    parser.add_argument('--epoch-num', type=int, default=9,
                        help='epoch number')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='batch_size')
    parser.add_argument('--use-wandb', action='store_true', help='use wandb')
    parser.add_argument('--wandb-entity', help='Enter your wandb entity')
    parser.add_argument('--use-aug', action='store_true', help='use augmentation')
    parser.add_argument('--use-lda', action='store_true', help='use LDA topic model')

    args = parser.parse_args()

    return args

def display_plots(df):
    sns.barplot(x="#Topics", y="F1", data=df)
    plt.show()
    sns.lineplot(x='Epoch', y="Train_Loss", hue="#Topics", data=df)
    plt.show()
    sns.lineplot(x='Epoch', y="Val_Loss", hue="#Topics", data=df)
    plt.show()

def train():
    # Parse command line
    userArgs = cli()
    print(vars(userArgs))

    # Running Parameters
    batch_size = userArgs.batch_size
    lr = userArgs.learning_rate
    epochs = userArgs.epoch_num
    use_lda = userArgs.use_lda
    use_wandb = userArgs.use_wandb
    use_aug =  userArgs.use_aug

    # Load datas
    train_msrp_df, val_msrp_df, test_msrp_df = load_data(use_aug)
    train_msrp_dataset, val_msrp_dataset, test_msrp_dataset = create_datasets(train_msrp_df, val_msrp_df, test_msrp_df)
    train_dataloader = DataLoader(train_msrp_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_msrp_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_msrp_dataset, batch_size=batch_size, shuffle=False)

    num_topics = [70, 75, 80, 85, 90]
    data_words = train_msrp_df.string1.values.tolist() + train_msrp_df.string2.values.tolist()

    df = pd.DataFrame(columns=['#Topics','F1','Train_Loss','Val_Loss','Epoch'])

    for n in num_topics:

        if use_wandb:
            run = wandb.init(project="AML-FP", entity=userArgs.wandb_entity, config=vars(userArgs), reinit=True)
            userArgs = run.config

        # model
        model = tBERT(data_words=data_words,num_topics=n,use_lda=use_lda).cuda()

        # train
        trainer = tBertTrainer(model, epochs=epochs, lr=lr, use_wandb=use_wandb)
        trainer.fit(train_dataloader, val_dataloader)

        # predict
        f1_score = trainer.test(test_dataloader)

        # save model results
        df_topic = pd.DataFrame({'#Topics':str(n),'F1':f1_score,'Train_Loss':trainer.loss_train,'Val_Loss':trainer.loss_val,'Epoch':range(epochs)})
        df = pd.concat([df,df_topic], ignore_index=True)

        if use_wandb:
            wandb.log({'test_F1':f1_score})
            run.finish()

    # Plots
    display_plots(df)

if __name__ == '__main__':
    train()

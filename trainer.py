from data_utils import *
from model import *


train_df = pd.read_csv('/path/to/train_file.csv')
df_train_concat = pd.read_csv('/path/to/train_file_concat.csv')
test_df = pd.read_csv('/path/to/test_file.csv')

train_df,test_df = read_and_clean_dataset(train_df,df_train_concat)

train_data=DataSet(train_df,0.2,0.2)
valid_data=DataSet(test_df,0.2,0.2)

train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
valid_loader = DataLoader(valid_data,batch_size=10,shuffle=True)

length = len(train_data)

def train_network(model,train_loader,loss_fn,optimizer,scheduler,n_epoch,device):

    torch.cuda.empty_cache()

    model.to(device)
    for epoch in range(1,n_epoch+1):
        train_losses = []
        accuracy = 0
        correct = 0
        for (img, csv_data),labels in tqdm(train_loader,"epoch {ep}: ".format(ep=epoch)):

            img_feat = torch.tensor(img,dtype=torch.float32).to(device)
            csv_feat = torch.tensor(csv_data,dtype=torch.float32).to(device)



            optimizer.zero_grad()

            output = model(img_feat,csv_feat)


            del img_feat,csv_feat

            label = torch.tensor(labels,dtype = torch.float32).to(device)

            loss = loss_fn(output,label.unsqueeze(1))
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())

            preds = torch.round(torch.sigmoid(output))

            correct += (preds.cpu() == label.cpu().unsqueeze(1)).sum().item()

            del label,output
            torch.cuda.empty_cache()


        train_acc = 100*(correct/length)

        print("Training Loss: ",torch.mean(torch.tensor(train_losses,dtype=torch.float32)))
        print("Training accuracy: ",train_acc)



def main():

    resnet = Resnet50Net(1,3)

    class_weight = torch.from_numpy((1 - (train_df['target'].value_counts(normalize=True).sort_index()).values)).float()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.AdamW(resnet.parameters(),lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(weight=class_weight)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_loader)),
                                                epochs=50,
                                                anneal_strategy='linear')

    train_network(resnet,train_loader,loss_fn,optimizer,scheduler,10,device)

if __name__ == "__main__":
    main()

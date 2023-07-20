from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score,confusion_matrix
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm.auto import tqdm

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_pred)
    prc = average_precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, roc, prc, f1

def compute_metrics(model,dataset,dev):

  data_len = len(dataset)
  preds = []
  true = []
  model.eval()
  for i in tqdm(range(data_len),"performing metrics evaluation: "):


    (img,csv_data), labels = dataset.__getitem__(i)

    img_feat = torch.tensor(img,dtype=torch.float32).unsqueeze(0).to(dev)
    csv_feat = torch.tensor(csv_data,dtype=torch.float32).unsqueeze(0).to(dev)

    out = int(torch.round(torch.sigmoid(model(img_feat,csv_feat))))

    del img_feat,csv_feat

    preds.append(out)
    true.append(labels)

  acc, roc, prc, f1 = metrics(true,preds)

  return acc,roc,prc,f1,preds,true
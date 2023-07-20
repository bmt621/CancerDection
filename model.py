import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, vgg16,  vgg19, inception_v3, densenet161


class Resnet50Net(nn.Module):
    def __init__(self,output_size,n_columns):
        super(Resnet50Net,self).__init__()
        self.no_columns, self.output_size = n_columns, output_size

        #resnet50
        self.resnet_feat=resnet50(pretrained=True)

        self.csv_feat = nn.Sequential(nn.Linear(self.no_columns,250),
                                      nn.BatchNorm1d(250),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),

                                      nn.Linear(250,200),
                                      nn.BatchNorm1d(200),
                                      nn.ReLU(),
                                      nn.Dropout(0.2))

        self.out = nn.Linear(1200,self.output_size)


    def forward(self,image,csv_data):

        image_feat = self.resnet_feat(image)
        csv_feat = self.csv_feat(csv_data)

        concat_feat = torch.cat((image_feat,csv_feat),dim=1)

        out = self.out(concat_feat)

        return out




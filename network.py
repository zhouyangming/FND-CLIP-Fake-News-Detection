from random import random
import torch
import torch.nn as nn
import random
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

manualseed = 64
random.seed(manualseed)
np.random.seed(manualseed)
torch.manual_seed(manualseed)
torch.cuda.manual_seed(manualseed)
cudnn.deterministic = True

class UnimodalDetection(nn.Module):
        def __init__(self, shared_dim=256, prime_dim = 64):
            super(UnimodalDetection, self).__init__()
            
            self.text_uni = nn.Sequential(
                nn.Linear(1280, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

            self.image_uni = nn.Sequential(
                nn.Linear(1512, shared_dim),
                nn.BatchNorm1d(shared_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(shared_dim, prime_dim),
                nn.BatchNorm1d(prime_dim),
                nn.ReLU())

        def forward(self, text_encoding, image_encoding):
            text_prime = self.text_uni(text_encoding)
            image_prime = self.image_uni(image_encoding)
            return text_prime, image_prime

class CrossModule(nn.Module):
    def __init__(
            self,
            corre_out_dim=64):
        super(CrossModule, self).__init__()
        self.corre_dim = 1024
        self.c_specific = nn.Sequential(
            nn.Linear(self.corre_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        correlation = torch.cat((text, image),1)
        
        correlation_out = self.c_specific(correlation.float())
        return correlation_out


class MultiModal(nn.Module):
    def __init__(
            self,
            feature_dim = 64,
            h_dim = 64
            ):
        super(MultiModal, self).__init__()
        self.weights = nn.Parameter(torch.rand(13, 1))
        #SENET
        self.senet = nn.Sequential(
                nn.Linear(3, 3),
                nn.GELU(),
                nn.Linear(3, 3),
        )
        self.sigmoid = nn.Sigmoid()

        self.w = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(1))

        self.avepooling =  nn.AvgPool1d(64, stride=1)
        self.maxpooling =  nn.MaxPool1d(64, stride=1)

        self.resnet101 = torchvision.models.resnet101(pretrained=True).cuda()

        self.uni_repre = UnimodalDetection()
        self.cross_module = CrossModule()
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 2)
        )
    def forward(self, input_ids, all_hidden_states, image_raw, text, image):
        # Process image
        image_raw = self.resnet101(image_raw)

        # Process text
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(13, input_ids.shape[0], 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        text_raw = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])

        # Unimodal processing
        text_prime, image_prime = self.uni_repre(torch.cat([text_raw, text], 1), torch.cat([image_raw, image], 1))
        
        # Cross-modal processing
        correlation = self.cross_module(text, image)

        # Calculate similarity weights
        sim = torch.div(torch.sum(text * image, 1), torch.sqrt(torch.sum(torch.pow(text, 2), 1)) * torch.sqrt(torch.sum(torch.pow(image, 2), 1)))
        sim = sim * self.w + self.b
        mweight = sim.unsqueeze(1)

        # Apply correlation weights
        correlation = correlation * mweight

        # Combine features
        final_feature = torch.cat([text_prime.unsqueeze(1), image_prime.unsqueeze(1), correlation.unsqueeze(1)], 1)

        # Pooling and transformation
        s1 = self.avepooling(final_feature)
        s2 = self.maxpooling(final_feature)
        s1 = s1.view(s1.size(0), -1)
        s2 = s2.view(s2.size(0), -1)
        s1 = self.senet(s1)
        s2 = self.senet(s2)
        s = self.sigmoid(s1 + s2)
        s = s.view(s.size(0), s.size(1), 1)

        # Apply pooling weights
        final_feature = s * final_feature

        # Classification
        pre_label = self.classifier_corre(final_feature[:, 0, :] + final_feature[:, 1, :] + final_feature[:, 2, :])

        return pre_label
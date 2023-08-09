import os
from torch.utils.data import DataLoader
from network import MultiModal
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import logging, BertConfig
from torch.autograd import Variable
from tqdm import tqdm
from weibo_dataset import *

# Set logging verbosity for transformers library
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set CUDA device if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BERT model
model_name = 'bert-base-chinese'
config = BertConfig.from_pretrained(model_name, num_labels=3)
config.output_hidden_states = True
BERT = BertModel.from_pretrained(model_name, config=config).cuda()

# Freeze the parameters of the BERT model
for param in BERT.parameters():
    param.requires_grad = False

def train():
    batch_size = 64
    lr = 1e-3
    l2 = 0  
    
    # Create training and testing datasets
    train_set = weibo_dataset(is_train=True)
    test_set = weibo_dataset(is_train=False)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn,
        shuffle=True)

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, shuffle=False)

    # Initialize the MultiModal network
    rumor_module = MultiModal()
    rumor_module.to(device)

    # Define the loss function for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optim_task = torch.optim.Adam(
        rumor_module.parameters(), lr=lr, weight_decay=l2)

    # Training loop
    for epoch in range(100):
        
        rumor_module.train()
        corrects_pre_rumor = 0
        loss_total = 0
        rumor_count = 0

        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in tqdm(enumerate(train_loader)):  
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(label)
        
            with torch.no_grad():
                # Extract features from BERT
                BERT_feature = BERT(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids) 
                last_hidden_states = BERT_feature['last_hidden_state']
                all_hidden_states =  BERT_feature['hidden_states']
                
                # Encode image and text using CLIP model
                image_clip = clipmodel.encode_image(imageclip) 
                text_clip = clipmodel.encode_text(textclip)    

            # Forward pass through the MultiModal network
            pre_rumor = rumor_module(last_hidden_states, all_hidden_states, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)

            optim_task.zero_grad()
            loss_rumor.backward()
            optim_task.step()
    
            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()
            
            loss_total += loss_rumor.item() * last_hidden_states.shape[0]
            rumor_count += last_hidden_states.shape[0]

        loss_rumor_train = loss_total / rumor_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        acc_rumor_test,  loss_rumor_test,  conf_rumor = test(rumor_module, test_loader)
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f ||  loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test))
   
        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def test(rumor_module, test_loader):
    rumor_module.eval()

    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(test_loader): 
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(label)
        
            # Extract features from BERT
            BERT_feature = BERT(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
            last_hidden_states = BERT_feature['last_hidden_state']
            all_hidden_states =  BERT_feature['hidden_states']
            
            # Encode image and text using CLIP model
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal network
            pre_rumor = rumor_module(last_hidden_states, all_hidden_states, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)

            pre_label_rumor = pre_rumor.argmax(1)
            loss_total += loss_rumor.item() * last_hidden_states.shape[0]
            rumor_count += last_hidden_states.shape[0]

            # Store predicted and true labels for evaluation
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        # Calculate accuracy and confusion matrix
        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)
        acc_rumor_test = accuracy_score(rumor_pre_label_all, rumor_label_all)
        conf_rumor = confusion_matrix(rumor_pre_label_all, rumor_label_all)

    return  acc_rumor_test, loss_rumor_test, conf_rumor

if __name__ == "__main__":
    train()

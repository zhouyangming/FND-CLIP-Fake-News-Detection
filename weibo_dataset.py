import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import pandas
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import clip

# Check if CUDA is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and preprocessing function
clipmodel, preprocess = clip.load('ViT-B/32', device)

# Freeze the parameters of the CLIP model
for param in clipmodel.parameters():
    param.requires_grad = False

# Function to read an image
def read_img(imgs, root_path, LABLEF):
    GT_path = imgs[np.random.randint(0, len(imgs))]
    if '/' in GT_path:
        GT_path = GT_path[GT_path.rfind('/')+1:]
    GT_path = "{}/{}/{}".format(root_path, LABLEF, GT_path)
    img_GT = Image.open(GT_path).convert('RGB')
    return img_GT

# Custom dataset class
class weibo_dataset(data.Dataset):

    def __init__(self, root_path='/data/ymzhou/weibo', image_size=224, is_train=True):
        super(weibo_dataset, self).__init__()
        self.is_train = is_train
        self.root_path = root_path
        self.index = 0
        self.label_dict = []
        self.preprocess = preprocess
        self.image_size = image_size
        self.local_path = '/data/ymzhou/dataset'

        # Read data from CSV file
        wb = pandas.read_csv(self.local_path+'/{}_weibo.csv'.format('train' if is_train else 'test'))

        # Store relevant information in label_dict
        for i in tqdm(range(len(wb))):
            images_name = str(wb.iloc[i, 2]).lower()
            label = int(wb.iloc[i, 3])
            content = str(wb.iloc[i, 1])
            sum_content = str(wb.iloc[i, 4])
            record = {}
            record['images'] = images_name
            record['label'] = label
            record['content'] = content
            record['sum_content'] = sum_content
            self.label_dict.append(record)

        assert len(self.label_dict) != 0, 'Error: GT path is empty.'

    def __getitem__(self, index):
        record = self.label_dict[index]
        images, label, content, sum_content = record['images'], record['label'], record['content'], record['sum_content']

        # Determine the label folder
        if label == 0:
            LABLEF = 'rumor_images'
        else:
            LABLEF = 'nonrumor_images'
        
        imgs = images.split('|')
        try:
            img_GT = read_img(imgs, self.root_path, LABLEF)
        except Exception:
            raise IOError("Load {} Error {}".format(imgs, record['images']))

        return (content, self.preprocess(img_GT), sum_content), label

    def __len__(self):
        return len(self.label_dict)

# Load BERT tokenizer
token = BertTokenizer.from_pretrained('bert-base-chinese')

# Custom collate function
def collate_fn(data):
    sents = [i[0][0] for i in data]
    images = [i[0][1] for i in data]
    textclip = [i[0][2] for i in data]
    labels = [i[1] for i in data]

    # Tokenize text data using BERT tokenizer
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                    truncation=True,
                                    padding='max_length',
                                    max_length=300,
                                    return_tensors='pt',
                                    return_length=True)

    # Tokenize text data using CLIP tokenizer
    textclip = clip.tokenize(textclip, truncate=True)
    
    # Prepare input data for the model
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    image = torch.stack(images)
    imageclip = torch.stack(images)
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, image, imageclip, textclip, labels

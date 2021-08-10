import torch
import json
import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from multiprocessing import Pool
from PIL import Image
import clip
from googletrans import Translator
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator


translator = Translator()



class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

device = torch.device('cuda:0')
def load_model():
    model, _ = clip.load('ViT-B/32', device=device, jit=False)
    model.eot_token = clip._tokenizer.encoder["<|endoftext|>"]
    model.eval()
    
    return model.to(device)


input_size = 224
def transform(string):
    img = Image.open(string)
    img = F.resize(img, input_size, Image.BICUBIC)
    img = F.center_crop(img, (input_size,input_size))
    img = img.convert("RGB")
    tensor = F.to_tensor(img)
    ret = F.normalize(tensor, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    return ret 

class MyDataset(Dataset):
    def __init__(self, datalist, transform=transform):
        super(MyDataset, self).__init__()
        self.transform = transform
        self.dataset = datalist

    def __getitem__(self, index):
        filename = self.dataset[index]

        sample = None
        if self.transform is not None:
            sample = self.transform(filename)
        return sample

    def __len__(self):
        return len(self.dataset)

# model = torch.nn.DataParallel(model)

@torch.no_grad()
def get_sentence_feature(sent, model, tokenizer):
    batchsize = 1000
    ret = []
    for start in tqdm(range(0, len(sent), batchsize)):
        input_ids = tokenizer(sent[start:start+batchsize]).to(device)
        _,text_features = model.encode_text(input_ids)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        ret.append(text_features.cpu())
    ret = torch.cat(ret, dim=0)
    assert len(sent)==ret.size()[0]
    return ret.float()

@torch.no_grad()
def get_imgs_feature(filelist,model,transform):
    #/share/wanghaofan/clip_features/features_path.pth

    dataset = MyDataset(filelist,transform)
    dataloader = DataLoaderX(dataset,batch_size=100,num_workers=20,prefetch_factor=8)   
    feature = []
    for images in tqdm(dataloader):
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        feature.append(image_features.cpu())
    feature = torch.cat(feature, dim=0)

    return feature.float()

def preprocess(dataset='./coco2014_val_caption_new.json'):
    # obj = json.load(open(dataset))
    obj = eval(open(dataset).read())["data"]
    sentens= dict()
    imgs_path = dict()
    positive = list()
    translater = dict()
    for pair in obj:
        if pair['ch_caption'] not in translater:
            translater[pair['ch_caption']]=pair['en_caption']    # 英文到中文的mapping
        if pair['img'] not in imgs_path:
            imgs_path[pair['img']]=len(imgs_path)
        if pair['ch_caption'] not in sentens:
            sentens[pair['ch_caption']]=len(sentens)
        #if pair['label']==1:
        positive.append((sentens[pair['ch_caption']],imgs_path[pair['img']]))

    gtmat = torch.zeros([len(sentens),len(imgs_path)],dtype=torch.bool)
    for one in positive:
        gtmat[one[0],one[1]]=1

    ord_sentens = list(sentens.items())
    ord_sentens.sort(key=lambda ele: ele[-1])
    ord_sentens = [one[0] for one in ord_sentens]

    chinese_sentens= [translater[line] for line in ord_sentens]

    ord_imgs = list(imgs_path.items())
    ord_imgs.sort(key=lambda ele: ele[-1])
    ord_imgs = [one[0] for one in ord_imgs]

    return ord_sentens,ord_imgs,gtmat,chinese_sentens


def preprocess_coco(dataset='./data/captions_val2014.json'):
    data = json.load(open(dataset))
    print(data.keys())
    sentens= dict()
    imgs_path = dict()
    positive = list()
    translater = dict()
    
    id2sentence = dict()
    id2imgs_path = dict()
    
    # 解析句子描述，一张图片可以对应多个描述
    for data_item in data["annotations"]:
        id = data_item["image_id"]
        if id in id2sentence:
            id2sentence[id].append(data_item["caption"])
        else:
            id2sentence[id] = [data_item["caption"]]
    
    # 组成句子-图片pair
    all_data_pair = []
    for data_item in data["images"]:
        id = data_item["id"]
        path = '/dataset/COCO/val2017/' + data_item["file_name"]
        for sent in id2sentence[id]:
            all_data_pair.append({"img" : path, "caption":sent})
            
    # 构造ground truth矩阵    
    for pair in all_data_pair:
        if pair['img'] not in imgs_path:
            imgs_path[pair['img']]=len(imgs_path)
        if pair['caption'] not in sentens:
            sentens[pair['caption']]=len(sentens)
        positive.append((sentens[pair['caption']],imgs_path[pair['img']]))
    gtmat = torch.zeros([len(sentens),len(imgs_path)],dtype=torch.bool)
    for one in positive:
        gtmat[one[0],one[1]]=1

    ord_sentens = list(sentens.items())
    ord_sentens.sort(key=lambda ele: ele[-1])
    ord_sentens = [one[0] for one in ord_sentens]

    # chinese_sentens= [translater[line] for line in ord_sentens]

    ord_imgs = list(imgs_path.items())
    ord_imgs.sort(key=lambda ele: ele[-1])
    ord_imgs = [one[0] for one in ord_imgs]

    return ord_sentens,ord_imgs,gtmat,# chinese_sentens


def recall_k(sentens,imgs,gtmat,k):
    index = torch.sum(gtmat,dim=1)>0
    score = torch.matmul(sentens, imgs.T)
    values,indices = torch.topk(score,k=k,dim=-1)
    gt = torch.gather(gtmat, dim=-1, index=indices)
    recall = (gt[index].sum(dim=-1)>0).float().mean()
    
    return recall,indices,gt



def batch_translate(ord_sentens):
    input_str = ""
    trans_result = []
    for sentence in tqdm(ord_sentens):
        if len(input_str) > 4000:
            input_str = input_str.strip(" .\n")
            out_str = translator.translate(input_str, src="zh-cn", dest="en").text.strip(" .\n")
            print("Input: ", input_str)
            print("Output: ", out_str)
            out_str = out_str.split("\n")
            out_str = [ss for ss in out_str if ss != ""]
            trans_result += out_str
            
            input_str = ""

        input_str = input_str + sentence + "\n"
        
    input_str = input_str.strip(" .\n")
    out_str = translator.translate(input_str, src="zh-cn", dest="en").text.strip(" .\n")
    out_str = out_str.split("\n")
    out_str = [ss for ss in out_str if ss != ""]
    
    trans_result += out_str
    
    assert len(trans_result) == len(ord_sentens)
    
    return trans_result

def preprocess_aic(path):
    datas = json.loads(open(path).read())
    images=[]
    sentens=[]
    image2sentence={}
    sentens2index={}
    image_base_path='/dataset/ai_challange/caption/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/'
    for data in datas[:10000]:
        data['image_id']=image_base_path+data['image_id']
        images.append(data['image_id'])
        for sen in data['caption']:
            sentens2index[sen]=len(sentens)
            sentens.append(sen)
        image2sentence[data['image_id']]=data['caption']
    gt=torch.zeros((len(images),len(sentens)))
    for i,image in enumerate(images):
        indexes=[sentens2index[sen] for sen in image2sentence[image]]

        gt[i][indexes]=1
    return sentens,images,gt.to(torch.bool)

def recall_k_score(sentens,imgs,gtmat,k):
    recall,topk_index,gt= recall_k(sentens,imgs,gtmat,k)

    print('recall:',k,recall)
    

def eval_in_coco():
    model = load_model().float()  # 替换蒸馏模型
    
    global input_size 
    input_size = model.visual.input_resolution
    print("Image resize to: ", input_size)
    
    ord_sentens,ord_imgs,gtmat, eng_sentens = preprocess()
    ord_sentens=eng_sentens
    
    print(gtmat.size())
    print(gtmat)
    print("ord_sentens: \n", ord_sentens[0], len(ord_sentens))
    print("ord_imgs: \n", ord_imgs[0], len(ord_imgs))
    
    imgs = get_imgs_feature(ord_imgs,model,transform)
    sentens=get_sentence_feature(ord_sentens,model,clip.tokenize)
    
    print(sentens.shape,imgs.shape)
    print(gtmat.T.shape)
    
    recall_k_score(sentens,imgs,gtmat,1)
    recall_k_score(sentens,imgs,gtmat,5)
    recall_k_score(sentens,imgs,gtmat,10)
    sentens,imgs,gtmat=imgs,sentens,gtmat.T
    recall_k_score(sentens,imgs,gtmat,1)
    recall_k_score(sentens,imgs,gtmat,5)
    recall_k_score(sentens,imgs,gtmat,10)

    """
    pairs=[]
    for idx,line in enumerate(chinese_sentens):
        for k,img_index in enumerate(topk_index[idx].tolist()):
            pairs.append((line,ord_imgs[img_index],k,gt[idx,k].item()))

    json.dump(pairs, open('clip_topk_show.json', 'w'), indent=4, ensure_ascii=False)
    """
  
if __name__=="__main__":
    eval_in_coco()

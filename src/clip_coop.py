from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
from tqdm import tqdm_notebook
import numpy as np
import torch
import torch.nn as nn

import clip
import ds 
from ds import prepare_coco_dataloaders, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
from tqdm import tqdm
from losses import *
from utils import *
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import metrics
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# import linear regression model
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

class CoOp_PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, dtype):
        super().__init__()
        self.dtype = dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.n_cls = len(classnames)
        print('Number of classes: {:}'.format(self.n_cls))
        ctx_init = 'a photo of a'
        self.n_ctx = len(ctx_init.split(" "))

        if ctx_init:
            # use given words to initialize context vectors
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :].cuda()
            self.prompt_prefix = ctx_init       
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            self.prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)
        self.get_prefix_suffix_token(classnames, clip_model)

        print('Initial context: {:}, Number of context words (tokens): {:}'.format(self.prompt_prefix, self.n_ctx))
        
    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        _tokenizer = _Tokenizer()
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        # a photo of a class_name.
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        # print(ctx.shape)
        # print(ctx)

        # print(self.token_prefix.shape)
        # print(self.token_suffix.shape)
        # print(ctx.shape)

        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        return prompts
    
class CoOpCLIP(nn.Module):

    def __init__(self, cfg, clip_model, device, dtype, tokeniser):
        super().__init__()
        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = torch.ones([]) * cfg['logit_scale']
        # self.logit_scale = nn.Parameter(torch.ones([]) * cfg['logit_scale'])

        # self.dtype = torch.cuda.FloatTensor()
        self.adapter = CoOp_PromptLearner(cfg['classnames'], clip_model, dtype)
        self.text_encoder = TextEncoder(clip_model, dtype).cuda()

        # get the default clip embeddings
        with torch.no_grad():
            query = [f"a photo of a {c}" for c in cfg['classnames']]
            # caption_targets = clip_model.token_embedding(query)
            caption_targets = tokeniser(query)
            caption_targets = caption_targets.to(device)
            self.original_feats = clip_model.encode_text(caption_targets)

        # self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad_(True)

        # self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # self.param_group = []
        # for k, v in self.adapter.named_parameters():
        #     v.requires_grad_(True)
        # self.mean_score = nn.Parameter(torch.ones([]) * cfg['mean_score'])
            
    def forward(self, image_features):
        # x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        # image_features = image_features[0]

        prompts = self.adapter()
        tokenized_prompts = self.adapter.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print(text_features[5])

        logits = image_features @ text_features.t()

        # logits = logits - self.mean_score
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * logits

        return logits.squeeze(-1)
    
    def return_clip_feats(self):
        # x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        # image_features = image_features[0]

        prompts = self.adapter()
        tokenized_prompts = self.adapter.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def forward_original_feats(self, image_features, query_features):
        # x = self.adapter(image_features)

        # image_features = self.ratio * x + (1 - self.ratio) * image_features
        image_features = image_features[0]

        # prompts = self.adapter()
        # tokenized_prompts = self.adapter.tokenized_prompts
        # text_features = self.text_encoder(prompts, tokenized_prompts)

        text_features = self.original_feats

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # print(text_features[5])

        logits = image_features @ text_features.t()

        # logits = logits - self.mean_score
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * logits

        return logits.squeeze(-1)
    
    # def element_wise_sim(self, image_features, text_features):
    #     # x = self.adapter(image_features)

    #     # image_features = self.ratio * x + (1 - self.ratio) * image_features

    #     prompts = self.adapter()
    #     tokenized_prompts = self.adapter.tokenized_prompts
    #     text_features = self.text_encoder(prompts, tokenized_prompts)

    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)


    #     logits = image_features * text_features

    #     # logits = logits - self.mean_score
    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * logits

    #     return logits.squeeze(-1)
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model, dtype):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = dtype
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print(x.shape)
        # print(x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)].shape)
        # print(tokenized_prompts.argmax(dim=-1))
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        return x
    
def loss_entropy(input_, average=True):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    if entropy.dim() == 1:
        entropy = torch.sum(entropy)
        return entropy

    if average:
        entropy = torch.sum(entropy, dim=1).mean()
    else:
        entropy = torch.sum(entropy, dim=1)
    return entropy 

def loss_entropy_wei(input_, weight):
    epsilon = 1e-5
    entropy = torch.sum(-input_ * torch.log(input_ + epsilon), dim=1)
    entropy = torch.sum(entropy * weight) / torch.sum(weight)
    return entropy
    
def train_coop_ueo(
    CLIP_Net,
    train_loader,
    eval_loader,
    device='cuda',
    lr=1e-3,
    dtype=torch.cuda.FloatTensor(),
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cfg=None,
    tokeniser=None,
):
    # create output dir
    os.makedirs(ckpt_path, exist_ok=True)   

    # start the writer
    writer = SummaryWriter(ckpt_path+'logs')   

    adapter = CoOpCLIP(cfg, CLIP_Net, device, dtype, tokeniser)
    # params = adapter.param_group

    classes = cfg['classnames']

    for k, v in adapter.named_parameters():
        if v.requires_grad:
            print(k)

    optimizer = torch.optim.AdamW(adapter.adapter.parameters(), lr=lr, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # use bce with logit loss
    # loss_fn = nn.BCEWithLogitsLoss()

    # cross entropy loss with logits
    loss_fn = nn.CrossEntropyLoss()

    # get the original logits
    with torch.no_grad():
        caption_query = [f'An image of a {concept}' for concept in classes]
        caption_targets = tokeniser(caption_query)
        caption_targets = caption_targets.to(device)
        original_queries = CLIP_Net.encode_text(caption_targets)
    
    score = 1e-8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        adapter.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):

                tepoch.set_description('Epoch {}'.format(eph))
                
                gt_classes = []
                img_embeddings = []
                for i in range(batch[0].shape[0]):
                    query_embedding = batch[0][i,:,:].to(device)
                    gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                    captions = batch[2][i]
                    img_embedding = batch[3][i,:,:].to(device)
                    seen = ~batch[4][i]
                    
                    # get the class name
                    gt_class_name = captions.replace("An image of a ", "")

                    # get the index of the class name
                    gt_class_idx = classes.index(gt_class_name)

                    # add to the embeddings
                    img_embeddings.append(img_embedding)
                    gt_classes.append(gt_class_idx)

                # stack the image embeddings
                img_embeddings = torch.stack(img_embeddings).squeeze(1)

                # run adapter on the image embeddings
                logits = adapter(img_embeddings)

                # apply the softmax
                outputs = torch.nn.Softmax(dim=1)(logits)

                # get the weights
                img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                orig_logits = img_embeddings @ original_queries.t()
                target_outputs = torch.nn.Softmax(dim=1)(orig_logits)
                weight, _ = torch.max(target_outputs, dim=1)
                mean_outputs = torch.mm(torch.diag(1 / weight), outputs).sum(dim=0) / torch.sum(1 / weight)
                loss = loss_entropy_wei(outputs, weight) - 1.0 * loss_entropy(mean_outputs)

                # # create the targets tensor
                # targets = torch.zeros_like(logits)
                # for i, gt_class_idx in enumerate(gt_classes):
                #     targets[i, gt_class_idx] = 1

                # # # stack logits and targets
                # # logits = torch.stack(logits)
                # # targets = torch.stack(targets)

                # # calculate the loss
                # loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                # calculate the loss
                loss.backward()
                b = list(adapter.parameters())[0].grad
                # print(b)
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            # print('Avg. loss: {}'.format(eph_loss))
            # add to the writer
            writer.add_scalar('train/loss', eph_loss, eph)
        
        # evaluate and save the models
        torch.save(adapter.adapter.ctx, ckpt_path+'last.pth')
        if eph%eval_every == 0:
            curr_score = eval_coop(
                adapter,
                eval_loader,
                device=device,
                epoch=eph,
                writer=writer,
                classes=classes
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score >= score:
                score = curr_score
                torch.save(adapter.adapter.ctx, ckpt_path+'best.pth')
            torch.save(adapter.adapter.ctx, ckpt_path+f'epoch{eph}.pth')

    optim_scheduler.step()
    writer.flush()
    writer.close()

def train_coop(
    CLIP_Net,
    train_loader,
    eval_loader,
    device='cuda',
    lr=1e-3,
    dtype=torch.cuda.FloatTensor(),
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cfg=None,
    tokeniser=None,
):
    # create output dir
    os.makedirs(ckpt_path, exist_ok=True)   

    # start the writer
    writer = SummaryWriter(ckpt_path+'logs')   

    adapter = CoOpCLIP(cfg, CLIP_Net, device, dtype, tokeniser)
    # params = adapter.param_group

    classes = cfg['classnames']

    for k, v in adapter.named_parameters():
        if v.requires_grad:
            print(k)

    optimizer = torch.optim.AdamW(adapter.adapter.parameters(), lr=lr, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # use bce with logit loss
    # loss_fn = nn.BCEWithLogitsLoss()

    # cross entropy loss with logits
    loss_fn = nn.CrossEntropyLoss()
    
    score = 1e-8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        adapter.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):

                tepoch.set_description('Epoch {}'.format(eph))
                
                gt_classes = []
                img_embeddings = []
                for i in range(batch[0].shape[0]):
                    query_embedding = batch[0][i,:,:].to(device)
                    gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                    captions = batch[2][i]
                    img_embedding = batch[3][i,:,:].to(device)
                    seen = ~batch[4][i]
                    
                    # get the class name
                    gt_class_name = captions.replace("An image of a ", "")

                    # get the index of the class name
                    gt_class_idx = classes.index(gt_class_name)

                    # add to the embeddings
                    img_embeddings.append(img_embedding)
                    gt_classes.append(gt_class_idx)

                # stack the image embeddings
                img_embeddings = torch.stack(img_embeddings).squeeze(1)

                # run adapter on the image embeddings
                logits = adapter(img_embeddings)

                # create the targets tensor
                targets = torch.zeros_like(logits)
                for i, gt_class_idx in enumerate(gt_classes):
                    targets[i, gt_class_idx] = 1

                # # stack logits and targets
                # logits = torch.stack(logits)
                # targets = torch.stack(targets)

                # calculate the loss
                loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                # calculate the loss
                loss.backward()
                b = list(adapter.parameters())[0].grad
                # print(b)
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            # print('Avg. loss: {}'.format(eph_loss))
            # add to the writer
            writer.add_scalar('train/loss', eph_loss, eph)
        
        # evaluate and save the models
        torch.save(adapter.adapter.ctx, ckpt_path+'last.pth')
        if eph%(eval_every+1) == 0:
            curr_score = eval_coop(
                adapter,
                eval_loader,
                device=device,
                epoch=eph,
                writer=writer,
                classes=classes
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score >= score:
                score = curr_score
                torch.save(adapter.adapter.ctx, ckpt_path+'best.pth')
            torch.save(adapter.adapter.ctx, ckpt_path+f'epoch{eph}.pth')

    optim_scheduler.step()
    writer.flush()
    writer.close()

def eval_coop(
    adapter,
    eval_loader,
    device='cuda',
    epoch=0,
    writer=None,
    classes=None
):
    adapter.eval()

    recall_results = {'seen':[], 'unseen':[], 'total':[]}
    # recall_results = {'total':[]}
    score_results = {'seen':[], 'unseen':[], 'total':[]}
    # score_results = {'total':[]}
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):

            tepoch.set_description('Validating')

            gt_classes = []
            img_embeddings = []
            for i in range(batch[0].shape[0]):
                query_embedding = batch[0][i,:,:].to(device)
                gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                captions = batch[2][i]
                img_embedding = batch[3][i,:,:].to(device)
                seen = ~batch[4][i]
                
                # get the class name
                gt_class_name = captions.replace("An image of a ", "")

                # get the index of the class name
                gt_class_idx = classes.index(gt_class_name)

                # append
                img_embeddings.append(img_embedding)
                gt_classes.append(gt_class_idx)

            # stack the image embeddings
            img_embeddings = torch.stack(img_embeddings).squeeze(1)

            # run adapter on the image embeddings
            with torch.no_grad():
                # logits = adapter.forward_original_feats(img_embeddings, query_embedding)
                logits = adapter(img_embeddings)

            # print(logits)
            # # create target of zeros the same shape as img_embeddings
            # target = torch.zeros_like(logits)
            # # gt_idxs = [idx+1 for idx in gt_idxs]
            # target[gt_idxs] = 1
            # print(target)

            # iterate through each gt_idx
            for i, gt_idx in enumerate(gt_classes):

                logit = logits[i]
                gt_idxs = [gt_idx]

                # get the max
                max_vals, max_idxs = torch.max(logit, dim=0)

                # print the class with max val
                # print(classes[max_idxs.cpu().numpy()])

                # append to the results
                recall_results['total'].append(max_idxs.cpu().numpy() in gt_idxs)
                score_results['total'].append(max_vals.cpu().numpy())

                # if seen
                if seen:
                    recall_results['seen'].append(max_idxs.cpu().numpy() in gt_idxs)
                    score_results['seen'].append(max_vals.cpu().numpy())
                else:
                    recall_results['unseen'].append(max_idxs.cpu().numpy() in gt_idxs)
                    score_results['unseen'].append(max_vals.cpu().numpy())
        
    for key in ['seen']:
    
        # print the accuracy on the recall task
        print(f'Scene Recall - {key}: {np.mean(recall_results[key])}')

        # write the results to tensorboard
        writer.add_scalar(f'{key}/recall', np.mean(recall_results[key]), epoch)

        # plot the ROC curve
        auroc_src, fpr_scr, tpr_scr = assess_classification(recall_results[key], score_results[key])

        # print the AUROC
        print(f'AUROC - {key}: {auroc_src}')

        # write the results to tensorboard
        writer.add_scalar(f'{key}/auroc', auroc_src, epoch)

    return np.mean(recall_results['seen'])

def assess_classification(gt, predictions, roc_point = None, title = False):
    # Calculate the false positive rate (FPR) and true positive rate (TPR)
    fpr, tpr, thresholds = metrics.roc_curve(gt, predictions)
    # Calculate the area under the ROC curve (AUC)
    roc_auc = metrics.auc(fpr, tpr)
    
    precision, recall, thresholds = metrics.precision_recall_curve(gt, predictions)
    average_precision = metrics.average_precision_score(gt, predictions)

    # get precision at 95 recall
    idx = np.argmax(recall >= 0.95)
    precision_at_target_recall = precision[idx]
    idx = np.argmax(precision>= 0.95)
    # recall_at_target_propen-vocab-clustering/classify_query_accuracy.py @ 95 Recall: ', precision_at_target_recall)
    # print('Recall @ 95 Precision: ', recall_at_target_precision)

    # Plot the precision-recall curve
    if title:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        if roc_point is not None:
            plt.scatter(roc_point[1], roc_point[0], color='red', s=100, zorder=5, label=f'Point of negative queries')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {title}')
        plt.legend(loc="lower right")
        plt.grid(True)

    return roc_auc, fpr, tpr
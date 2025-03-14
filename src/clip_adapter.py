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

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):

    def __init__(self, cfg, clip_model, device):
        super().__init__()
        # self.logit_scale = clip_model.logit_scale
        self.logit_scale = torch.ones([]) * cfg['logit_scale']
        # self.logit_scale = nn.Parameter(torch.ones([]) * cfg['logit_scale'])

        # self.dtype = torch.cuda.FloatTensor()
        self.adapter = Adapter(cfg['embedding_dim'], 2).to(device)
        self.ratio = cfg['ratio']

        self.mean_score = cfg['mean_score']
        # self.mean_score = nn.Parameter(torch.ones([]) * cfg['mean_score'])
            
    def forward(self, image_features, text_features):
        x = self.adapter(image_features)

        image_features = self.ratio * x + (1 - self.ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()

        logits = logits - self.mean_score
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * logits

        return logits.squeeze(-1)
    
    def element_wise_sim(self, image_features, text_features):
        x = self.adapter(image_features)

        image_features = self.ratio * x + (1 - self.ratio) * image_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features * text_features

        # logits = logits - self.mean_score
        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * logits

        return logits.squeeze(-1)


def train_clip_adapter(
    CLIP_Net,
    train_loader,
    eval_loader,
    device='cuda',
    lr=1e-3,
    dtype=torch.cuda.FloatTensor(),
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cfg=None
):
    # create output dir
    os.makedirs(ckpt_path, exist_ok=True)   

    # start the writer
    writer = SummaryWriter(ckpt_path+'logs')   

    adapter = CustomCLIP(cfg, CLIP_Net, device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=1e-4)
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # use bce with logit loss
    loss_fn = nn.BCEWithLogitsLoss()

    # cross entropy loss with logits
    # loss_fn = nn.CrossEntropyLoss()
    
    score = 1e-8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        adapter.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):

                tepoch.set_description('Epoch {}'.format(eph))
                
                logits = []
                targets = []
                for i in range(batch[0].shape[0]):
                    query_embedding = batch[0][i,:,:].to(device)
                    gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                    captions = batch[2][i]
                    img_embeddings = batch[3][i,:,:].to(device)
                    seen = ~batch[4][i]

                    # get max value of each row
                    # max_row = torch.max(img_embeddings, 1)[0]
                    # get the length of the image embeddings
                    # n_objects = img_embeddings.shape[0]
                    # remove all image embeddings with max value of 0
                    # img_embeddings = img_embeddings[max_row > 0.01]

                    # run adapter on the image embeddings
                    logit = adapter(img_embeddings, query_embedding)

                    # pad the logit to the length of the original image embeddings
                    # logit = F.pad(logit, (0, n_objects - logit.shape[0]), value=-10)
                    # print(logit)

                    # remove idx larger than the length of the logit
                    gt_idxs = [idx for idx in gt_idxs if idx < logit.shape[0]]

                    # create target of zeros the same shape as img_embeddings
                    target = torch.zeros_like(logit)
                    # gt_idxs = [idx+1 for idx in gt_idxs]
                    target[gt_idxs] = 1

                    logits.append(logit)
                    targets.append(target)

                # stack logits and targets
                logits = torch.stack(logits)
                targets = torch.stack(targets)

                # calculate the loss
                loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                # calculate the loss
                loss.backward()
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
        torch.save(adapter.state_dict(), ckpt_path+'last.pth')
        if eph%eval_every == 0:
            curr_score = eval_clip_adapter(
                adapter,
                eval_loader,
                device=device,
                epoch=eph,
                writer=writer
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score >= score:
                score = curr_score
                torch.save(adapter.state_dict(), ckpt_path+'best.pth')

            # save every eval    
            torch.save(adapter.state_dict(), ckpt_path+f'epoch{eph}.pth')

    optim_scheduler.step()
    writer.flush()
    writer.close()

def eval_clip_adapter(
    adapter,
    eval_loader,
    device='cuda',
    epoch=0,
    writer=None,
):
    adapter.eval()

    recall_results = {'seen':[]}
    # recall_results = {'total':[]}
    score_results = {'seen':[]}
    # score_results = {'total':[]}
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):

            tepoch.set_description('Validating')

            for i in range(batch[0].shape[0]):
                # stack the queries and the image embeddings to perform dot product
                query_embedding = batch[0][i,:,:].to(device)
                gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                captions = batch[2][i]
                img_embeddings = batch[3][i,:,:].to(device)
                seen = ~batch[4][i]

                # if seen:
                #     print(captions)

                # get max value of each row
                max_row = torch.max(img_embeddings, 1)[0]
                # remove all image embeddings with max value of 0
                img_embeddings = img_embeddings[max_row > 0.01]

                # # normalise the embeddings and calculate the dot product
                # img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                # query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                # logits = torch.matmul(query_embedding, img_embeddings.T).squeeze(0)

                # run adapter on the image embeddings
                with torch.no_grad():
                    logits = adapter(img_embeddings, query_embedding)

                # print(logits)
                # # create target of zeros the same shape as img_embeddings
                # target = torch.zeros_like(logits)
                # # gt_idxs = [idx+1 for idx in gt_idxs]
                # target[gt_idxs] = 1
                # print(target)

                # get the max
                max_vals, max_idxs = torch.max(logits, dim=0)

                # # append to the results
                # recall_results['total'].append(max_idxs.cpu().numpy() in gt_idxs)
                # score_results['total'].append(max_vals.cpu().numpy())

                # if seen
                if seen:
                    recall_results['seen'].append(max_idxs.cpu().numpy() in gt_idxs)
                    score_results['seen'].append(max_vals.cpu().numpy())
                # else:
                #     recall_results['unseen'].append(max_idxs.cpu().numpy() in gt_idxs)
                #     score_results['unseen'].append(max_vals.cpu().numpy())
        
    for key in recall_results.keys():
    
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

def test_clip_adapter(
    CLIP_Net,
    cfg,
    eval_loader,
    train_loader,
    device='cuda',
    classifier='element_sim',
    checkpoint_path=None
):
    # if checkpoint path is not provided, use the last checkpoint
    if checkpoint_path is not None:
        adapter = CustomCLIP(cfg, CLIP_Net, device)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        adapter.load_state_dict(checkpoint)
    else:
        adapter = None

    # get training data to fit model 
    train_data = {'X': [], 'Y': []}
    with tqdm(train_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
                    tepoch.set_description('Validating ...')

                    # # get scene names and class index
                    # for i in range(len(batch)):
                    #     print(len(batch[i]))
                    #     print(batch[i][0])

                    for i in range(batch[0].shape[0]):
                        # stack the queries and the image embeddings to perform dot product
                        query_embedding = batch[0][i,:,:]
                        gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                        captions = batch[2][i]
                        img_embeddings = batch[3][i,:,:]

                        # get max value of each row
                        max_row = torch.max(img_embeddings, 1)[0]
                        # remove all image embeddings with max value of 0
                        img_embeddings = img_embeddings[max_row > 0.01]

                        # put query_embedding and img_embeddings on the same device
                        query_embedding = query_embedding.to(device)
                        img_embeddings = img_embeddings.to(device)

                        if adapter is not None:
                            # run adapter on the image embeddings
                            with torch.no_grad():
                                dot_product = adapter(img_embeddings, query_embedding)
                                dot_product = dot_product.unsqueeze(0)
                
                        else:
                            # normalise the queries and image embeddings
                            query_embedding = query_embedding / torch.norm(query_embedding, dim=1).unsqueeze(1)
                            img_embeddings = img_embeddings / torch.norm(img_embeddings, dim=1).unsqueeze(1)

                            # get the dot product between the two
                            dot_product = torch.matmul(query_embedding, img_embeddings.T)

                        # remove nan from the dot product
                        dot_product = torch.nan_to_num(dot_product)

                        # get the max value and index for each query
                        max_vals, max_idxs = torch.max(dot_product, dim=1)

                        # if using the element wise similarity
                        if classifier == 'element_sim':
                            # get the img_embeddings for the max object
                            matched_object_embeddings = img_embeddings[max_idxs.item()]

                            # pass to the adapter
                            if adapter is not None:
                                with torch.no_grad():
                                    similarity = adapter.element_wise_sim(matched_object_embeddings, query_embedding)
                                    similarity = similarity.squeeze(0).cpu().numpy()
                            else:
                                # element wise multiplication
                                similarity = query_embedding * matched_object_embeddings
                                # squeeze and convert to numpy
                                similarity = similarity.squeeze(0).cpu().numpy()

                            # add to the training data
                            train_data['X'].append(similarity)
                            train_data['Y'].append(max_idxs.item() in gt_idxs)
                        
                        elif classifier == 'top_5':
                            # get the top 5 objects
                            _, top_idxs = torch.topk(dot_product, min(5,dot_product.shape[1]), dim=1)
                            # iterate through the top 5 objects
                            for i in range(top_idxs.shape[1]):
                                obj_idx = top_idxs[0,i].item()
                                # get the embeddings
                                matched_object_embedding = img_embeddings[obj_idx,:]

                                # pass to the adapter
                                if adapter is not None:
                                    with torch.no_grad():
                                        similarity = adapter.element_wise_sim(matched_object_embedding, query_embedding)
                                        similarity = similarity.squeeze(0).cpu().numpy()
                                else:
                                    # element wise multiplication
                                    similarity = query_embedding * matched_object_embedding
                                    # squeeze and convert to numpy
                                    similarity = similarity.squeeze(0).cpu().numpy()

                                # if similarity is nan, skip
                                if np.isnan(similarity).any():
                                    continue

                                # add to the training data
                                train_data['X'].append(similarity)
                                train_data['Y'].append(obj_idx in gt_idxs)


                        # if using sim_scores
                        elif classifier == 'sim_scores':
                            # clamp dot product to between 0.05 and 0.45
                            dot_product = dot_product[(dot_product >= 0.05) & (dot_product <= 0.45)]

                            score_kde = gaussian_kde(dot_product.squeeze(0).cpu().numpy(), bw_method="scott")
                            x = np.linspace(0.05, 0.45, 100)
                            pdf = score_kde(x)

                            # get mean, median, min, max, variance, q1, q3, skewness, kurtosis
                            mean = torch.mean(dot_product).item()
                            median = torch.median(dot_product).item()
                            min_val = torch.min(dot_product).item()
                            max_val = torch.max(dot_product).item()
                            std_dev = torch.std(dot_product).item()
                            q1 = torch.quantile(dot_product, 0.25).item()
                            q3 = torch.quantile(dot_product, 0.75).item()
                            n = dot_product.shape[0]
                            skewness = (n * torch.sum(((dot_product - mean) / std_dev) ** 3)) / ((n - 1) * (n - 2))
                            kurtosis = (n * (n + 1) * torch.sum(((dot_product - mean) / std_dev) ** 4) / ((n - 1) * (n - 2) * (n - 3))) - (3 * (n - 1)**2 / ((n - 2) * (n - 3)))
                            X = [mean, median, min_val, max_val, std_dev, q1, q3, skewness.item(), kurtosis.item()]

                            # plt.figure()
                            # plt.plot(x, pdf)
                            # plt.show()

                            train_data['X'].append(X)
                            train_data['Y'].append(max_idxs.item() in gt_idxs)

                            # # normalise the mean and target
                            # mean = query_embedding / torch.norm(query_embedding)
                            # target = matched_object_embeddings / torch.norm(matched_object_embeddings)

                            # # get the dot product between the two
                            # similarity = torch.matmul(mean, target.T)

                            # # squeeze and convert to numpy
                            # similarity = similarity.squeeze(0).cpu().numpy()

                            # # add to the training data
                            # train_data['X'].append(similarity)
                            # train_data['Y'].append(max_idxs.item() in gt_idxs)
    
    # fit a linear regression to the training data
    X = np.array(train_data['X'])
    Y = np.array(train_data['Y'])
    clf = LinearRegression()
    clf.fit(X, Y)

    recall_results = {'seen':[], 'unseen':[], 'total':[]}
    score_results = {'seen':[], 'unseen':[], 'total':[]}
    classifier_results = {'seen':[], 'unseen':[], 'total':[]}
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')

            # # get scene names and class index
            # for i in range(len(batch)):
            #     print(len(batch[i]))
            #     print(batch[i][0])

            for i in range(batch[0].shape[0]):
                # stack the queries and the image embeddings to perform dot product
                query_embedding = batch[0][i,:,:]
                gt_idxs = [int(obj_idx) for obj_idx in batch[1][i].split('_')]
                captions = batch[2][i]
                img_embeddings = batch[3][i,:,:]
                seen = ~batch[4][i]

                # get max value of each row
                max_row = torch.max(img_embeddings, 1)[0]
                # remove all image embeddings with max value of 0
                img_embeddings = img_embeddings[max_row > 0.01]

                # put query and image embeddings on the same device
                query_embedding = query_embedding.to(device)
                img_embeddings = img_embeddings.to(device)
                
                if adapter is not None:

                    # # normalise the embeddings and calculate the dot product
                    # img_embeddings = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                    # query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                    # logits = torch.matmul(query_embedding, img_embeddings.T).squeeze(0)

                    # run adapter on the image embeddings
                    with torch.no_grad():
                        dot_product = adapter(img_embeddings, query_embedding)
                        dot_product = dot_product.unsqueeze(0)
                
                else:
                    # normalise the queries and image embeddings
                    query_embedding = query_embedding / torch.norm(query_embedding, dim=1).unsqueeze(1)
                    img_embeddings = img_embeddings / torch.norm(img_embeddings, dim=1).unsqueeze(1)

                    # get the dot product between the two
                    dot_product = torch.matmul(query_embedding, img_embeddings.T)

                # remove nan from the dot product
                dot_product = torch.nan_to_num(dot_product)

                # get the max value and index for each query
                max_vals, max_idxs = torch.max(dot_product, dim=1)

                # if using the element wise similarity
                if classifier == 'element_sim' or classifier == 'top_5':
                    # get the img_embeddings for the max object
                    matched_object_embeddings = img_embeddings[max_idxs.item()]

                    # pass to the adapter
                    if adapter is not None:
                        with torch.no_grad():
                            similarity = adapter.element_wise_sim(matched_object_embeddings, query_embedding)
                        similarity = similarity.squeeze(0).cpu().numpy()

                    else:
                        # element wise multiplication
                        similarity = query_embedding * matched_object_embeddings

                        # squeeze and convert to numpy
                        similarity = similarity.squeeze(0).cpu().numpy()

                    # if similarity is nan, skip
                    if np.isnan(similarity).any():
                        continue

                    clf_score = clf.predict([similarity])[0]
                    # clf_score = clf.predict([query_embedding.squeeze(0).cpu().numpy()])[0]

                # elif classifier == 'top_5':
                #     # get the img_embeddings for the max object
                #     matched_object_embedding = img_embeddings[max_idxs.item(),:].cpu().numpy()

                #     clf_score = clf.predict([matched_object_embedding])[0]

                # if using sim_scores
                elif classifier == 'sim_scores':
                    # clamp dot product to between 0.05 and 0.45
                    dot_product = dot_product[(dot_product >= 0.05) & (dot_product <= 0.45)]

                    score_kde = gaussian_kde(dot_product.squeeze(0).cpu().numpy(), bw_method="scott")
                    x = np.linspace(0.05, 0.45, 100)
                    pdf = score_kde(x)

                    # get mean, median, min, max, variance, q1, q3, skewness, kurtosis
                    mean = torch.mean(dot_product).item()
                    median = torch.median(dot_product).item()
                    min_val = torch.min(dot_product).item()
                    max_val = torch.max(dot_product).item()
                    std_dev = torch.std(dot_product).item()
                    q1 = torch.quantile(dot_product, 0.25).item()
                    q3 = torch.quantile(dot_product, 0.75).item()
                    n = dot_product.shape[0]
                    skewness = (n * torch.sum(((dot_product - mean) / std_dev) ** 3)) / ((n - 1) * (n - 2))
                    kurtosis = (n * (n + 1) * torch.sum(((dot_product - mean) / std_dev) ** 4) / ((n - 1) * (n - 2) * (n - 3))) - (3 * (n - 1)**2 / ((n - 2) * (n - 3)))
                    X = [mean, median, min_val, max_val, std_dev, q1, q3, skewness.item(), kurtosis.item()]

                    clf_score = clf.predict([X])[0]

                    # plt.figure()
                    # plt.plot(x, pdf)
                    # plt.show()


                if seen:
                    recall_results['seen'].append(max_idxs.item() in gt_idxs)
                    score_results['seen'].append(max_vals.item())
                    classifier_results['seen'].append(clf_score)
                else:
                    recall_results['unseen'].append(max_idxs.item() in gt_idxs)
                    score_results['unseen'].append(max_vals.item())
                    classifier_results['unseen'].append(clf_score)
                
                recall_results['total'].append(max_idxs.item() in gt_idxs)
                score_results['total'].append(max_vals.item())
                classifier_results['total'].append(clf_score)

    for key in recall_results.keys():
    
        # print the accuracy on the recall task
        print(f'Scene Recall - {key}: {np.mean(recall_results[key])}')

        # plot the ROC curve
        auroc_src, fpr_scr, tpr_scr = assess_classification(recall_results[key], score_results[key])

        print(f'AUROC - {key}: {auroc_src}')

        # # plot the ROC curve
        # auroc_clf, fpr_clf, tpr_clf = assess_classification(recall_results[key], classifier_results[key])

        # plt.figure()
        # plt.plot(fpr_scr, tpr_scr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc_src:.2f})')
        # plt.plot(fpr_clf, tpr_clf, color='green', lw=2, label=f'ROC curve (area = {auroc_clf:.2f})')
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # # if roc_point is not None:
        # #     plt.scatter(roc_point[1], roc_point[0], color='red', s=100, zorder=5, label=f'Point of negative queries')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title(f'Receiver Operating Characteristic (ROC) Curve - {key}')
        # plt.legend(loc="lower right")
        # plt.grid(True)
        # plt.show()

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
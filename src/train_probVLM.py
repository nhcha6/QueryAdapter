import os

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

def train_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    train_loader,
    eval_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor(),
    init_lr=1e-4,
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cross_modal_lambda=1e-4,
    T1=1e0,
    T2=5e-2
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    ##
    BayesCap_Net.to(device)
    BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    ##

    optimizer = torch.optim.Adam(
        list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=init_lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    score = 1e8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):

                if idx>2000:
                    break
                tepoch.set_description('Epoch {}'.format(eph))
                
                # when using the clip model during training
                # xI, xT  = batch[0].to(device), batch[1].to(device)
                # with torch.no_grad():
                #     xfI, xfT = CLIP_Net(xI, xT)

                # when precomputing the clip features
                xfI, xfT = batch[0].to(device), batch[1].to(device)

                # xI, xT = xI.type(dtype), xT.type(dtype)
                # pass them through the network
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)

                # fits a generalised normal distribution across each dimension of the CLIP feature space
                # this has a mean, shape (essentially sharpness of the distribution) and scale (essentially the variance of the distribution)
                # print(img_mu.shape, img_1alpha.shape, img_beta.shape)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
                loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                # print(loss)
                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            print('Avg. loss: {}'.format(eph_loss))
        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), ckpt_path+'_last.pth')
        if eph%eval_every == 0:
            curr_score = eval_ProbVLM(
                CLIP_Net,
                BayesCap_Net,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score <= score:
                score = curr_score
                torch.save(BayesCap_Net.state_dict(), ckpt_path+'_best.pth')
    optim_scheduler.step()
    
def eval_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    # build up a datastructure to perform object recall in the scene
    scene_recall = {}

    mean_mse = 0
    mean_mae = 0
    num_imgs = 0
    # list_error = []
    # list_var = []
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')

            # when using the clip model during training
            # xI, xT  = batch[0].to(device), batch[1].to(device)
            # with torch.no_grad():
            #     xfI, xfT = CLIP_Net(xI, xT)

            # get scene names and class index
            scene_names = batch[4]
            cls_nums = batch[2]
            img_names = batch[5]
            crops = batch[6]
            captions = batch[7]

            # when precomputing the clip features
            xfI, xfT = batch[0].to(device), batch[1].to(device)

            # xI, xT = xI.type(dtype), xT.type(dtype)
            # pass them through the network
            with torch.no_grad():
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)

            n_batch = img_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(img_mu[j], xfI[j]) + emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(img_mu[j], xfI[j]) + emb_mae(txt_mu[j], xfT[j])

                if scene_names[j] not in scene_recall:
                    scene_recall[scene_names[j]] = {
                        'img_mu': [],
                        'txt_mu': [],
                        'img_1alpha': [],
                        'txt_1alpha': [],
                        'img_beta': [],
                        'txt_beta': [],
                        'img_clip': [],
                        'txt_clip': [],
                        'img_names': [],
                        'crops': [],
                        'captions': [],
                        'cls_nums': []
                    }
                
                scene_recall[scene_names[j]]['img_mu'].append(img_mu[j])
                scene_recall[scene_names[j]]['txt_mu'].append(txt_mu[j])
                scene_recall[scene_names[j]]['img_1alpha'].append(img_1alpha[j])
                scene_recall[scene_names[j]]['txt_1alpha'].append(txt_1alpha[j])
                scene_recall[scene_names[j]]['img_beta'].append(img_beta[j])
                scene_recall[scene_names[j]]['txt_beta'].append(txt_beta[j])
                scene_recall[scene_names[j]]['img_clip'].append(xfI[j])
                scene_recall[scene_names[j]]['txt_clip'].append(xfT[j])
                scene_recall[scene_names[j]]['img_names'].append(img_names[j])
                scene_recall[scene_names[j]]['crops'].append(crops[j])
                scene_recall[scene_names[j]]['captions'].append(captions[j])
                scene_recall[scene_names[j]]['cls_nums'].append(cls_nums[j].item())

        # calculate the recall for each scene
        recall_results = []
        for scene_name in scene_recall.keys():
            # get the unique cls_nums
            unique_cls_nums = np.unique(scene_recall[scene_name]['cls_nums'])
            
            # iterate through each class num
            queries = []
            gt_idxs = []
            for cls_num in unique_cls_nums:
                # get the indices of the cls_num
                idxs = np.where(np.array(scene_recall[scene_name]['cls_nums']) == cls_num)[0]

                # # show the images for debugging
                # for idx in idxs:
                #     x, y, w, h = scene_recall[scene_name]['crops'][idx].numpy()

                #     print(scene_recall[scene_name]['captions'][idx])
                #     # print the class number
                #     print(scene_recall[scene_name]['cls_nums'][idx])

                #     # open the image
                #     img = Image.open(scene_recall[scene_name]['img_names'][idx])
                #     img = img.crop((x, y, x+w, y+h))
                #     # display the image
                #     img = np.array(img)
                #     plt.imshow(img)
                #     plt.show()
                
                # get the embeddings for the class
                query_embedding = scene_recall[scene_name]['txt_mu'][idxs[0]]
                # query_embedding = scene_recall[scene_name]['txt_clip'][idxs[0]]

                # update the query and gt lists
                queries.append(query_embedding)
                gt_idxs.append(idxs)

            # stack the queries and the image embeddings
            queries = torch.stack(queries).squeeze(1)
            img_embeddings = torch.stack(scene_recall[scene_name]['img_mu']).squeeze(1)
            # img_embeddings = torch.stack(scene_recall[scene_name]['img_clip']).squeeze(1)

            # normalise the queries and image embeddings
            queries = queries / torch.norm(queries, dim=1).unsqueeze(1)
            img_embeddings = img_embeddings / torch.norm(img_embeddings, dim=1).unsqueeze(1)

            # print(queries.shape, img_embeddings.shape)
            # get the dot product between the two
            dot_product = torch.matmul(queries, img_embeddings.T)

            # get the max value and index for each query
            max_vals, max_idxs = torch.max(dot_product, dim=1)
            # print(max_vals.shape)

            # print(max_idxs)
            # print(gt_idxs)
            # true if max_idx is in the gt_idxs, false otherwise
            recall = [max_idx.item() in gt_idxs[i] for i, max_idx in enumerate(max_idxs)]
            recall_results.extend(recall)

        # print the accuracy on the recall task
        print('Scene Recall: {}'.format(np.mean(recall_results)))

        # print the embedding reconstruction accuracy
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(
            'Avg. MSE: {} | Avg. MAE: {}'.format
            (
                mean_mse, mean_mae 
            )
        )

    return mean_mae

def train_ProbVLM_HF(
    CLIP_Net,
    BayesCap_Net,
    train_loader,
    eval_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor(),
    init_lr=1e-4,
    num_epochs=100,
    eval_every=1,
    ckpt_path='../ckpt/ProbVLM',
    cross_modal_lambda=1e-4,
    T1=1e0,
    T2=5e-2
):
    CLIP_Net['img'].to(device)
    CLIP_Net['txt'].to(device)
    CLIP_Net['img'].eval()
    CLIP_Net['txt'].eval()
    ##
    BayesCap_Net.to(device)
    # BayesCap_Net.img_BayesCap.train()
    BayesCap_Net.txt_BayesCap.train()
    ##
    optimizer = torch.optim.Adam(
        # list(BayesCap_Net.img_BayesCap.parameters())+list(BayesCap_Net.txt_BayesCap.parameters()), 
        list(BayesCap_Net.txt_BayesCap.parameters()), 
        lr=init_lr
    )
    optim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    score = 1e8
    all_loss = []
    for eph in range(num_epochs):
        eph_loss = 0
        BayesCap_Net.train()
        with tqdm(train_loader, unit='batch') as tepoch:
            for (idx, batch) in enumerate(tepoch):
                if idx>2000:
                    break
                tepoch.set_description('Epoch {}'.format(eph))
                ##
                xI, xT  = batch[0].to(device), batch[1].to(device)
                with torch.no_grad():
                    xfI = CLIP_Net['img'](xI)
                    xfT = CLIP_Net['txt'](xT)
                    # print('dbg1: ', xfI, xfT)
                    xfI = xfI['last_hidden_state']
                    xfT = xfT['last_hidden_state']

                    inb, inc, ind = xfI.shape
                    tnb, tnc, tnd = xfT.shape

                    xfI = xfI.reshape(inb, -1)
                    xfT = xfT.reshape(tnb, -1) 

                    # print('dbg', xfI.shape, xfT.shape)

                    # print('dbg2: ', xfI.shape, xfT.shape)
                # pass them through the network
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)
                
                optimizer.zero_grad()
                loss_i = Cri(img_mu, img_1alpha, img_beta, xfI, T1=T1, T2=T2)
                loss_t = Cri(txt_mu, txt_1alpha, txt_beta, xfT, T1=T1, T2=T2)
                #cross modal terms
                loss_i4t = Cri(img_mu, img_1alpha, img_beta, xfT, T1=T1, T2=T2)
                loss_t4i = Cri(txt_mu, txt_1alpha, txt_beta, xfI, T1=T1, T2=T2)
                loss = loss_i + loss_t + cross_modal_lambda*(loss_i4t + loss_t4i)
                loss = loss_i + loss_t
                loss.backward()
                optimizer.step()
                ##
                eph_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
            eph_loss /= len(train_loader)
            all_loss.append(eph_loss)
            print('Avg. loss: {}'.format(eph_loss))
        # evaluate and save the models
        torch.save(BayesCap_Net.state_dict(), ckpt_path+'_last.pth')
        if eph%eval_every == 0:
            curr_score = eval_ProbVLM_HF(
                CLIP_Net,
                BayesCap_Net,
                eval_loader,
                device=device,
                dtype=dtype,
            )
            print('current score: {} | Last best score: {}'.format(curr_score, score))
            if curr_score <= score:
                score = curr_score
                torch.save(BayesCap_Net.state_dict(), ckpt_path+'_best.pth')
    optim_scheduler.step()

def eval_ProbVLM_HF(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor,
):
    CLIP_Net['img'].to(device)
    CLIP_Net['txt'].to(device)
    CLIP_Net['img'].eval()
    CLIP_Net['txt'].eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    mean_mse = 0
    mean_mae = 0
    num_imgs = 0
    list_error = []
    list_var = []
    with tqdm(eval_loader, unit='batch') as tepoch:
        for (idx, batch) in enumerate(tepoch):
            tepoch.set_description('Validating ...')
            ##
            xI, xT  = batch[0].to(device), batch[1].to(device)

            # pass them through the network
            with torch.no_grad():
                xfI = CLIP_Net['img'](xI)
                xfT = CLIP_Net['txt'](xT)
                xfI = xfI['last_hidden_state']
                xfT = xfT['last_hidden_state']
                
                inb, inc, ind = xfI.shape
                tnb, tnc, tnd = xfT.shape
                
                xfI = xfI.reshape(inb, -1)
                xfT = xfT.reshape(tnb, -1)

                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = BayesCap_Net(xfI, xfT)

            n_batch = txt_mu.shape[0]
            for j in range(n_batch):
                num_imgs += 1
                mean_mse += emb_mse(txt_mu[j], xfT[j])
                mean_mae += emb_mae(txt_mu[j], xfT[j])
        mean_mse /= num_imgs
        mean_mae /= num_imgs
        print(
            'Avg. MSE: {} | Avg. MAE: {}'.format
            (
                mean_mse, mean_mae 
            )
        )
    return mean_mae
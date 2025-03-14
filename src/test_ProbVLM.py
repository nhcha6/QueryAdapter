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
from scipy.stats import gennorm
import scipy.special as sc

# import linear regression model
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde


class GenNormLikelihood(nn.Module):
    def __init__(
        self, reduction='mean',
		alpha_eps = 1e-4, beta_eps=1e-4,
		resi_min = 1e-4, resi_max=1e3
    ) -> None:
        super(GenNormLikelihood, self).__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
    
    def likelihood(
        self, 
        mean: Tensor, one_over_alpha: Tensor, beta: Tensor, target: Tensor
    ):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps
        x = mean - target

        logpdf = torch.log(0.5*beta*one_over_alpha1) - torch.lgamma(torch.pow(beta1, -1)) - torch.pow(abs(x)*one_over_alpha1, beta)
        pdf = torch.exp(logpdf)

        return pdf.mean()

    def variance(
        self,
        one_over_alpha: Tensor, beta: Tensor
    ):
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps

        # convert to numpy
        one_over_alpha1 = one_over_alpha1.cpu().numpy()
        beta1 = beta1.cpu().numpy()

        # t1 = torch.log(torch.pow(one_over_alpha1,-2)) 
        # t2 = torch.lgamma(3.0*torch.pow(beta1, -1)) 
        # t3 = torch.lgamma(torch.pow(beta1, -1)) 

        # logvar = t1+t2-t3
        # # logvar = logvar.clamp(min=-float('inf'), max=1)
        # var = torch.exp(logvar)

        # # remove all inf values
        # var = var[~torch.isinf(var)]

        var = np.power(one_over_alpha1, -2) * sc.gamma(3.0/beta1) / sc.gamma(1.0/beta1)
        # remove all nan
        var = var[~np.isnan(var)]

        # # clamp the var
        # var = var.clip(max=1)

        # for i in range(len(var[0])):
        #     # check if inf
        #     if var[0][i] == float('inf'):
        #         print(logvar[0][i])

        return var.mean()

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

def test_ProbVLM(
    CLIP_Net,
    BayesCap_Net,
    eval_loader,
    device='cuda',
    dtype=torch.cuda.FloatTensor,
    checkpoint_path=None
):
    CLIP_Net.to(device)
    CLIP_Net.eval()
    BayesCap_Net.to(device)
    BayesCap_Net.eval()

    # get the likelihood function
    gen_norm = GenNormLikelihood()

    # load checkpoint if it exists
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        BayesCap_Net.load_state_dict(checkpoint)
        print('Loaded last checkpoint')

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
        score_results = []
        uncertainty_results = []
        for scene_name in scene_recall.keys():
            # get the unique cls_nums
            unique_cls_nums = np.unique(scene_recall[scene_name]['cls_nums'])
            
            # iterate through each class num
            queries = []
            queries_lalpha = []
            queries_beta = []
            queries_caption = []
            queries_clip = []
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
                query_lalpha = scene_recall[scene_name]['txt_1alpha'][idxs[0]]
                query_beta = scene_recall[scene_name]['txt_beta'][idxs[0]]
                query_clip = scene_recall[scene_name]['txt_clip'][idxs[0]]

                # update the query and gt lists
                queries.append(query_embedding)
                queries_lalpha.append(query_lalpha)
                queries_beta.append(query_beta)
                queries_clip.append(query_clip)
                queries_caption.append(scene_recall[scene_name]['captions'][idxs[0]])
                
                gt_idxs.append(idxs)

            # stack the queries and the image embeddings to perform dot product
            queries = torch.stack(queries).squeeze(1)
            img_embeddings = torch.stack(scene_recall[scene_name]['img_mu']).squeeze(1)
            
            # alternatively, we can use the original clip embeddings
            clip_queries = torch.stack(queries_clip).squeeze(1)
            clip_img_embeddings = torch.stack(scene_recall[scene_name]['img_clip']).squeeze(1)

            # normalise the queries and image embeddings
            queries_norm = queries / torch.norm(queries, dim=1).unsqueeze(1)
            img_embeddings_norm = img_embeddings / torch.norm(img_embeddings, dim=1).unsqueeze(1)

            # print(queries.shape, img_embeddings.shape)
            # get the dot product between the two
            dot_product = torch.matmul(queries_norm, img_embeddings_norm.T)

            # get the max value and index for each query
            max_vals, max_idxs = torch.max(dot_product, dim=1)
            # print(max_vals.shape)

            # get the img_embeddings for the max objects
            matched_object_embeddings = clip_img_embeddings[max_idxs]

            # print(matched_object_embeddings.shape)
            # print(clip_queries.shape)

            # true if max_idx is in the gt_idxs, false otherwise
            recall = [max_idx.item() in gt_idxs[i] for i, max_idx in enumerate(max_idxs)]
            score = [max_val.item() for max_val in max_vals]

            uncertainty_data = get_features_uncertainty(BayesCap_Net, matched_object_embeddings, clip_queries)
            # print(uncertainty_data['t_u'][0].shape)
            # print(len(uncertainty_data['t_u']))
            uncertainty_t = [geometric_mean(uncertainty_data['t_u'][i]) for i in range(len(uncertainty_data['t_u']))]
            uncertainty_i = [geometric_mean(uncertainty_data['i_u'][i]) for i in range(len(uncertainty_data['i_u']))]
            # get the geometric mean of the uncertainties
            uncertainty = [geometric_mean(torch.tensor([uncertainty_t[i], uncertainty_i[i]])) for i in range(len(uncertainty_t))]
            # uncertainty = score

            # check the uncertainty data
            mean_img = [uncertainty_data['ir_f'][i] / torch.norm(uncertainty_data['ir_f'][i]) for i in range(len(uncertainty_data['ir_f']))]
            mean_txt = [uncertainty_data['tr_f'][i] / torch.norm(uncertainty_data['tr_f'][i]) for i in range(len(uncertainty_data['tr_f']))]
            score = [torch.matmul(txt, img.T).item() for txt, img in zip(mean_txt, mean_img)]

            recall_results.extend(recall)
            score_results.extend(score)
            uncertainty_results.extend(uncertainty_i)

            # score = []
            # uncertainty = []
            # for query_idx in range(len(queries)):
            #     # print the caption
            #     print('Caption: ', queries_caption[query_idx])
            #     object_idx = max_idxs[query_idx].item()

            #     # get the matched text and image embeddings
            #     text_embedding = queries[query_idx].unsqueeze(0)
            #     img_embedding = scene_recall[scene_name]['img_mu'][object_idx]
               
            #     # print('Query distribution: ', queries_lalpha[query_idx].shape, queries_beta[query_idx].shape)
            #     # print('Object distribution: ', scene_recall[scene_name]['img_1alpha'][object_idx].shape, scene_recall[scene_name]['img_beta'][object_idx].shape)

            #     # using text distribution
            #     # lalpha = torch.ones(queries_lalpha[query_idx].shape).to('cuda')
            #     lalpha = queries_lalpha[query_idx].unsqueeze(0)
            #     # beta = torch.ones(queries_beta[query_idx].shape).to('cuda')
            #     beta = queries_beta[query_idx].unsqueeze(0)
            #     mean = queries[query_idx].unsqueeze(0)
            #     target = scene_recall[scene_name]['img_mu'][object_idx]

            #     # # using image distribution
            #     # lalpha = scene_recall[scene_name]['img_1alpha'][object_idx]
            #     # beta = scene_recall[scene_name]['img_beta'][object_idx]
            #     # mean = scene_recall[scene_name]['img_mu'][object_idx]
            #     # target = queries[query_idx].unsqueeze(0)

            #     lhood = gen_norm.likelihood(
            #         mean, lalpha, beta, target
            #     )
            #     print('Likelihood', lhood.item())

            #     var = gen_norm.variance(
            #         lalpha, beta
            #     )
            #     print('Variance: ', var.item())

            #     uncertainty.append(var.item())

            #     # normalise the mean and target
            #     mean = mean / torch.norm(mean)
            #     target = target / torch.norm(target)
            #     similarity = torch.matmul(mean, target.T)
            #     score.append(similarity.item())
            
            # recall_results.extend(recall)
            # score_results.extend(score)
            # uncertainty_results.extend(uncertainty)
        
        # plot score vs uncertainty
        # plt.figure()
        # plt.scatter(score_results, uncertainty_results)
        # plt.show()

        # print the accuracy on the recall task
        print('Scene Recall: {}'.format(np.mean(recall_results)))

        # plot the ROC curve
        auroc, fpr, tpr = assess_classification(recall_results, score_results, title = 'Similarity Score Reliability')
        plt.show()

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
    
def test_simple_methods(
    eval_loader,
    train_loader,
    device='cuda',
    classifier='element_sim'
):
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

                        # put query_embedding and img_embeddings on the same device
                        query_embedding = query_embedding.to(device)
                        img_embeddings = img_embeddings.to(device)
                        
                        # normalise the queries and image embeddings
                        query_embedding = query_embedding / torch.norm(query_embedding, dim=1).unsqueeze(1)
                        img_embeddings = img_embeddings / torch.norm(img_embeddings, dim=1).unsqueeze(1)

                        # print(queries.shape, img_embeddings.shape)
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

                            # element wise multiplication
                            similarity = query_embedding * matched_object_embeddings

                            # squeeze and convert to numpy
                            similarity = similarity.squeeze(0).cpu().numpy()

                            # squeeze matched object embedding

                            # add to the training data
                            train_data['X'].append(similarity)
                            train_data['Y'].append(max_idxs.item() in gt_idxs)
                        
                        elif classifier == 'top_5':
                            # get the top 5 objects
                            _, top_idxs = torch.topk(dot_product, 5, dim=1)
                            # iterate through the top 5 objects
                            for i in range(top_idxs.shape[1]):
                                obj_idx = top_idxs[0,i].item()
                                # get the embeddings
                                matched_object_embedding = img_embeddings[obj_idx,:]

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

                # put query_embedding and img_embeddings on the same device
                query_embedding = query_embedding.to(device)
                img_embeddings = img_embeddings.to(device)
                
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

                    # element wise multiplication
                    similarity = query_embedding * matched_object_embeddings

                    # squeeze and convert to numpy
                    similarity = similarity.squeeze(0).cpu().numpy()

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

        # plot the ROC curve
        auroc_clf, fpr_clf, tpr_clf = assess_classification(recall_results[key], classifier_results[key])

        plt.figure()
        plt.plot(fpr_scr, tpr_scr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc_src:.2f})')
        plt.plot(fpr_clf, tpr_clf, color='green', lw=2, label=f'ROC curve (area = {auroc_clf:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # if roc_point is not None:
        #     plt.scatter(roc_point[1], roc_point[0], color='red', s=100, zorder=5, label=f'Point of negative queries')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {key}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
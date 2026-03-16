import numpy as np
import matplotlib.pyplot as plt
import monai
from tqdm import tqdm
import cv2
import random
import sys
import os
from PIL import Image

import torch
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch import nn

import torchvision.transforms.functional as F

from model import DINOCell
from utils import lr_lambda_with_warmup, full_dataset_list, pc_bf_non_tissue_whole_cell_only_dataset, fluorescent_dataset_list, tissue_dataset_list, get_metrics
from dataset import get_dataset, DINOCellOODDataset
from pipeline import get_pipeline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.test.pca_visualization import run_pca_visualization_DINOCell



def convert_label_to_rainbow(label):
  label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
  for cell in np.unique(label):
      if cell == 0:
          continue #background
      label_rainbow[label == cell] = np.random.rand(3) * 255

  return label_rainbow


def get_dataset_list_from_dataset_type(dataset_type):
  if dataset_type == 'full_dataset_list':
    return full_dataset_list
  elif dataset_type == 'pc_bf_non_tissue_whole_cell_only':
    return pc_bf_non_tissue_whole_cell_only_dataset
  elif dataset_type == 'fluorescent':
    return fluorescent_dataset_list
  elif dataset_type == 'tissue':
    return tissue_dataset_list



def train(model, train_dataloader, optimizer, scheduler, objective_type, model_save_path, train_state_save_path, epoch, device):
  model.train()
  loss_list = []

  # train model for 1 epoch

  batch_num = 0
  for batch in tqdm(train_dataloader):
    # forward pass + loss calculation
    ground_truth_data = batch["gt_data"] #.float().to(device)

    #remove center data after 15 epochs for direct mask prediction
    if objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
      if epoch >= 15:
        ground_truth_data['gt_centers'] = None
        pred_index = 1
      else:
        pred_index = 0
    else:
      pred_index = 0

    preds, loss = model(x=batch["pixel_values"].to(device), targets=ground_truth_data, ignore_masks=ground_truth_data['ignore_mask'])

    if objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
      if batch_num == 3:
        if len(preds[pred_index][0]) > 0:
          # Display the output
          fig, ax = plt.subplots(1, 2, figsize=(10, 10))
          #remove axes
          for i in range(2):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            #set 1 in borders
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['left'].set_visible(False)

          ax[0].imshow(batch['pixel_values'][0].permute(1, 2, 0), cmap='gray')
          ax[1].imshow(preds[pred_index][0][0].detach().cpu().numpy())
          plt.savefig(f'{log_path}/objectness_vis/epoch_{epoch}.png')
          plt.close()
        else:
          # Display the output
          fig, ax = plt.subplots(1, 2, figsize=(10, 10))
          #remove axes
          for i in range(2):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            #set 1 in borders
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['left'].set_visible(False)

          ax[0].imshow(batch['pixel_values'][0].permute(1, 2, 0), cmap='gray')
          ax[1].imshow(np.zeros_like(batch['pixel_values'][0].permute(1, 2, 0)), cmap='gray')
          plt.savefig(f'{log_path}/objectness_vis/epoch_{epoch}.png')
          plt.close()

      
    # backward pass (compute gradients of parameters w.r.t. loss)
    optimizer.zero_grad()
    loss.backward()

    # optimize
    optimizer.step()
    scheduler.step()
    loss_list.append(loss.item())

    batch_num += 1

  print(f'epoch {epoch} training checkpoint created')
  torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'scheduler_state_dict': scheduler.state_dict(),
    }, 
  train_state_save_path)

  print('saving model to {}...'.format(model_save_path))
  torch.save(model.state_dict(), model_save_path)

  return np.mean(loss_list)

def convert_label_to_rainbow(label):
  label_rainbow = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
  for cell in np.unique(label):
      if cell == 0:
          continue #background
      label_rainbow[label == cell] = np.random.rand(3) * 255

  return label_rainbow


def val(model, patch_size, val_dataloader, pipeline, dataset_root_path, dataset_type, objective_type, log_path, epoch, device):
  #validate on image
  model.eval()
  loss_list = []
  # metrics_list = {'MMA': [], 'AP@50': [], 'SEG': [],'DET': [], 'CCR': [], 'CCS': [], 'ShapeScore': []}
  # metrics_list = {'MMA': []}
  # metrics_list = {'SEG': [], 'DET': []}
  metrics_list = {'MMA': [], 'AP@50': []}
  mmas = []
  # validate model for 1 epoch
  for batch in tqdm(val_dataloader):
    with torch.no_grad():
      # forward pass + loss calculation + accuracy calculation
      ground_truth_data = batch["gt_data"] #.float().to(device)

      #remove center data after 15 epochs for direct mask prediction
      if objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
        if epoch >= 15:
          ground_truth_data['gt_centers'] = None
          pred_index = 1
        else:
          pred_index = 0
      else:
        pred_index = 0
      
      preds, loss, metrics = model(x=batch["pixel_values"].to(device), targets=ground_truth_data, ignore_masks=ground_truth_data['ignore_mask'], labels=ground_truth_data['gt_mask'])
      
      loss_list.append(loss.item())
      if metrics is not None:
        for key in metrics.keys():
          metrics_list[key].append(metrics[key])

  #validate model on full image
  with torch.no_grad():
    pipeline.set_model(model)
    if dataset_type == 'full_dataset_list':
      val_image = np.array(Image.open(f'{dataset_root_path}/cellpose_cyto/non_tissue_fluorescent_images/val/imgs/img_424.png'))
      gt_label = np.load(f'{dataset_root_path}/cellpose_cyto/non_tissue_fluorescent_images/val/anns/ann_424.npy')
    else:
      val_image = np.array(Image.open(f'{dataset_root_path}/LiveCell/val/imgs/img_0.png'))
      gt_label = np.load(f'{dataset_root_path}/LiveCell/val/anns/ann_0.npy')

    output, val_image, gt_label = pipeline.run(val_image, gt_label)

    fig_name = f'{log_path}/pca_vis/epoch_{epoch}.png'
    run_pca_visualization_DINOCell(model, patch_size, fig_name)

    output_rgb = convert_label_to_rainbow(output)
    ann_rgb = convert_label_to_rainbow(gt_label)
    val_image = val_image.astype(np.uint8)
    output_rgb[output == 0] = cv2.cvtColor(val_image, cv2.COLOR_GRAY2RGB)[output == 0]
    ann_rgb[gt_label == 0] = cv2.cvtColor(val_image, cv2.COLOR_GRAY2RGB)[gt_label == 0]

    # Display the output
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    #remove axes
    for i in range(3):
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      #set 1 in borders
      ax[i].spines['top'].set_visible(False)
      ax[i].spines['right'].set_visible(False)
      ax[i].spines['bottom'].set_visible(False)
      ax[i].spines['left'].set_visible(False)

    ax[0].imshow(val_image, cmap='gray')
    ax[1].imshow(output_rgb, cmap='gray')
    ax[2].imshow(ann_rgb, cmap='gray')
    plt.savefig(f'{log_path}/seg_vis/epoch_{epoch}.png')
    plt.close()

  avg_metrics = {}
  for key in metrics_list.keys():
    avg_metrics[key] = np.mean(metrics_list[key])

  return np.mean(loss_list), avg_metrics


def ood_test(model, ood_dataloader_bbbc030, ood_dataloader_hek, ood_dataloader_n2a, pipeline, objective_type, log_path, epoch, device):
  model.eval()
  #test on bbbc030 data
  # bbbc030_metrics_list = {'MMA': [], 'AP@50': [], 'SEG': [],'DET': [], 'CCR': [], 'CCS': [], 'ShapeScore': []}
  # bbbc030_metrics_list = {'MMA': []}
  # bbbc030_metrics_list = {'SEG': [],'DET': []}
  bbbc030_metrics_list = {'MMA': [], 'AP@50': []}
  # test model for 1 epoch
  for batch in tqdm(ood_dataloader_bbbc030):
    with torch.no_grad():
      # forward pass + loss calculation + accuracy calculation
      ground_truth_data = batch["gt_data"] #.float().to(device)
      preds, metrics = model(x=batch["pixel_values"].to(device), labels=ground_truth_data['gt_mask'])
      for key in metrics.keys():
        bbbc030_metrics_list[key].append(metrics[key])

  #test on hek image
  # hek_metrics_list = {'MMA': [], 'AP@50': [], 'SEG': [],'DET': [], 'CCR': [], 'CCS': [], 'ShapeScore': []}
  # hek_metrics_list = {'MMA': []}
  # hek_metrics_list = {'SEG': [],'DET': []}
  hek_metrics_list = {'MMA': [], 'AP@50': []}
  # test model for 1 epoch
  for batch in tqdm(ood_dataloader_hek):
    with torch.no_grad():
      # forward pass + loss calculation + accuracy calculation
      ground_truth_data = batch["gt_data"] #.float().to(device)
      preds, metrics = model(x=batch["pixel_values"].to(device), labels=ground_truth_data['gt_mask'])
      for key in metrics.keys():
        hek_metrics_list[key].append(metrics[key])

  #test on n2a image
  # n2a_metrics_list = {'MMA': [], 'AP@50': [], 'SEG': [],'DET': [], 'CCR': [], 'CCS': [], 'ShapeScore': []}
  # n2a_metrics_list = {'MMA': []}
  # n2a_metrics_list = {'SEG': [],'DET': []}
  n2a_metrics_list = {'MMA': [], 'AP@50': []}
  # test model for 1 epoch
  for batch in tqdm(ood_dataloader_n2a):
    with torch.no_grad():
      # forward pass + loss calculation + accuracy calculation
      ground_truth_data = batch["gt_data"] #.float().to(device)
      preds, metrics = model(x=batch["pixel_values"].to(device), labels=ground_truth_data['gt_mask'])
      for key in metrics.keys():
        n2a_metrics_list[key].append(metrics[key])

  #test model on full bbbc030 image
  with torch.no_grad():
    pipeline.set_model(model)
    
    test_image_bbbc030 = np.array(Image.open(f'../../Dataset/Full_Segmentation_Dataset/BBBC030/train/imgs/img_50.png'))
    gt_label_bbbc030 = np.load(f'../../Dataset/Full_Segmentation_Dataset/BBBC030/train/anns/ann_50.npy')

    output, test_image, gt_label = pipeline.run(test_image_bbbc030, gt_label_bbbc030)

    output_rgb = convert_label_to_rainbow(output)
    ann_rgb = convert_label_to_rainbow(gt_label)
    test_image = test_image.astype(np.uint8)
    output_rgb[output == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[output == 0]
    ann_rgb[gt_label == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[gt_label == 0]

    # Display the output
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    #remove axes
    for i in range(3):
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      #set 1 in borders
      ax[i].spines['top'].set_visible(False)
      ax[i].spines['right'].set_visible(False)
      ax[i].spines['bottom'].set_visible(False)
      ax[i].spines['left'].set_visible(False)

    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(output_rgb, cmap='gray')
    ax[2].imshow(ann_rgb, cmap='gray')
    plt.savefig(f'{log_path}/ood_bbbc030_seg_vis/epoch_{epoch}.png')
    plt.close()

  #test model on full HEK image
  with torch.no_grad():
    pipeline.set_model(model)
    
    test_image_hek = np.array(Image.open(f'../../Dataset/Full_Segmentation_Dataset/PBL/HEK/imgs/img_0.png'))
    gt_label_hek = np.load(f'../../Dataset/Full_Segmentation_Dataset/PBL/HEK/anns/ann_0.npy')

    output, test_image, gt_label = pipeline.run(test_image_hek, gt_label_hek)

    #output = np.load('../../Dataset/LiveCell_Dataset/train/anns.npy')[0][0]
    output_rgb = convert_label_to_rainbow(output)
    ann_rgb = convert_label_to_rainbow(gt_label)
    test_image = test_image.astype(np.uint8)
    output_rgb[output == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[output == 0]
    ann_rgb[gt_label == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[gt_label == 0]

    # Display the output
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    #remove axes
    for i in range(3):
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      #set 1 in borders
      ax[i].spines['top'].set_visible(False)
      ax[i].spines['right'].set_visible(False)
      ax[i].spines['bottom'].set_visible(False)
      ax[i].spines['left'].set_visible(False)

    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(output_rgb, cmap='gray')
    ax[2].imshow(ann_rgb, cmap='gray')
    plt.savefig(f'{log_path}/ood_hek_seg_vis/epoch_{epoch}.png')
    plt.close()

  #test model on full N2A image
  with torch.no_grad():
    pipeline.set_model(model)

    test_image_n2a = np.array(Image.open(f'../../Dataset/Full_Segmentation_Dataset/PBL/N2A/imgs/img_0.png'))
    gt_label_n2a = np.load(f'../../Dataset/Full_Segmentation_Dataset/PBL/N2A/anns/ann_0.npy')

    output, test_image, gt_label = pipeline.run(test_image_n2a, gt_label_n2a)

    #output = np.load('../../Dataset/LiveCell_Dataset/train/anns.npy')[0][0]
    output_rgb = convert_label_to_rainbow(output)
    ann_rgb = convert_label_to_rainbow(gt_label)
    test_image = test_image.astype(np.uint8)
    output_rgb[output == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[output == 0]
    ann_rgb[gt_label == 0] = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)[gt_label == 0]

    # Display the output
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    #remove axes
    for i in range(3):
      ax[i].set_xticks([])
      ax[i].set_yticks([])
      #set 1 in borders
      ax[i].spines['top'].set_visible(False)
      ax[i].spines['right'].set_visible(False)
      ax[i].spines['bottom'].set_visible(False)
      ax[i].spines['left'].set_visible(False)

    ax[0].imshow(test_image, cmap='gray')
    ax[1].imshow(output_rgb, cmap='gray')
    ax[2].imshow(ann_rgb, cmap='gray')
    plt.savefig(f'{log_path}/ood_n2a_seg_vis/epoch_{epoch}.png')
    plt.close()

  
  avg_bbbc030_metrics = {}
  for key in bbbc030_metrics_list.keys():
    avg_bbbc030_metrics[key] = np.mean(bbbc030_metrics_list[key])

  avg_hek_metrics = {}
  for key in hek_metrics_list.keys():
    avg_hek_metrics[key] = np.mean(hek_metrics_list[key])

  avg_n2a_metrics = {}
  for key in n2a_metrics_list.keys():
    avg_n2a_metrics[key] = np.mean(n2a_metrics_list[key])

  return avg_bbbc030_metrics, avg_hek_metrics, avg_n2a_metrics





if __name__ == '__main__':
  #set seed for train consistency
  def set_seed(seed):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)  # if using multi-GPU
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

  set_seed(42)

  #train options setup
  num_epochs = 80 #40, 80 #CHANGE
  patch_size = 14 #8, 14 CHANGE
  crop_size = 512 #512, 256 CHANGE
  finetune_vision = True 
  lower_lr_encoder = True # CHANGE 
  delayed_encoder_unfreezing = False 
  hyperparameter_type = 'hyperparam_4' #original, cellpose_sam, gpt_suggested, hyperparam_4, hyperparam_5, hyperparam_6 #CHANGE
  dropout_in_encoder = True
  use_ignore_masks = True
  ignore_all = True
  use_advanced_augmentations = False #CHANGE
  include_silver_data = False
  use_dino_weights = True # True, False CHANGE
  resume_training = True #CHANGE
  dataset_root_path = '../../Dataset/Full_Segmentation_Dataset'
  dataset_type = 'pc_bf_non_tissue_whole_cell_only' #full_dataset_list, fluorescent, pc_bf_non_tissue_whole_cell_only, tissue CHANGE
  dino_type = f'domain_adaption_patch_size_{patch_size}' #CHANGE
  # dino_type = f'random_init_patch_size_{patch_size}'
  # dino_type = f'pretrained_but_not_domain_adapted'
  dino_model = f'../pretrained_dino_models/dino_{dino_type}_checkpoint_19999.pth' # CHANGE
  # dino_model = f'../pretrained_dino_models/dino_{dino_type}_checkpoint_24799.pth'
  # dino_model = f'../pretrained_dino_models/dino_{dino_type}_checkpoint_99999.pth'
  if dino_type == f'random_init_patch_size_{patch_size}':
    dino_model = '../pretrained_dino_models/dino_random_init_patch_size_14_checkpoint_34399.pth'
  elif dino_type == 'pretrained_but_not_domain_adapted':
    dino_model = f'../pretrained_dino_models/dino_{dino_type}.pth'
  decoder_type = 'upsample' # upsample, convolutional, SAM
  objective_type = 'flows' #'distance_map', 'directional_distance_map', 'flows', 'direct_mask_prediction', 'direct_mask_prediction_independent_encoding' #CHANGE
  ignore_mask_weight = 0.05 #0.2, 0.02, 0.05 CHANGE
  silver_data_weight = 0.3 #0.1, 0.3, 0.5, 1.0
  log_modifier = f'piw_{str(int(ignore_mask_weight * 100))}_isd_{str(include_silver_data)}_hpt_{hyperparameter_type}_die_{dropout_in_encoder}_ftv_{str(finetune_vision)}_llre_{str(lower_lr_encoder)}_deu_{str(delayed_encoder_unfreezing)}_cs_{crop_size}_sdw_{str(int(silver_data_weight * 100))}_ne_{num_epochs}'
  # log_modifier += f'_udw_{use_dino_weights}'
  # log_modifier += '_longer_domain_adaption' #CHANGE
  # log_modifier += '_advanced_augmentations'
  #piw = partial_ignore_weight, isd = include_silver_data, hpt = hyper_parameter_type, die = dropout_in_encoder, ftv = finetune_vision, llre = lower_lr_encoder, deu = delayed_encoder_unfreezing, cs = crop_size, sdw = silver_data_weight, ne = num_epochs,

  #logging setup
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/pca_vis', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/seg_vis', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/ood_bbbc030_seg_vis', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/ood_hek_seg_vis', exist_ok=True)
  os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/ood_n2a_seg_vis', exist_ok=True)
  if objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
    os.makedirs(f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}/objectness_vis', exist_ok=True)
    
  log_path = f'../logs/dataset_{dataset_type}/dino_{dino_type}/decoder_{decoder_type}/objective_{objective_type}/modifier_{log_modifier}'

  #dataset setup
  dataset_list = get_dataset_list_from_dataset_type(dataset_type)
    
  #ood Paths
  ood_dataset_path_bbbc030 = '../../Dataset/Full_Segmentation_Dataset/BBBC030/combined'
  pbl_dataset_path = '../../Dataset/Full_Segmentation_Dataset/PBL'
  ood_dataset_path_hek = f'{pbl_dataset_path}/HEK'
  ood_dataset_path_n2a = f'{pbl_dataset_path}/N2A'


  print('loading dataset...')
  train_dataset = get_dataset(objective_type, dataset_root_path, dataset_list, split='train', include_silver_data=include_silver_data, use_advanced_augmentations=use_advanced_augmentations, ignore_all=ignore_all, ignore_mask_weight=ignore_mask_weight, silver_data_weight=silver_data_weight, crop_size=crop_size)
  val_dataset = get_dataset(objective_type, dataset_root_path, dataset_list, split='val', include_silver_data=include_silver_data, use_advanced_augmentations=use_advanced_augmentations, ignore_all=ignore_all, ignore_mask_weight=ignore_mask_weight, silver_data_weight=silver_data_weight, crop_size=crop_size)
  # test_dataset = get_dataset(objective_type, dataset_root_path, dataset_list, split='test', use_advanced_augmentations=use_advanced_augmentations, ignore_all=ignore_all, ignore_mask_weight=ignore_mask_weight, silver_data_weight=silver_data_weight, crop_size=crop_size)
  ood_dataset_bbbc030 = DINOCellOODDataset(ood_dataset_path_bbbc030, use_advanced_augmentations, crop_size)
  ood_dataset_hek = DINOCellOODDataset(ood_dataset_path_hek, use_advanced_augmentations, crop_size)
  ood_dataset_n2a = DINOCellOODDataset(ood_dataset_path_n2a, use_advanced_augmentations, crop_size)
  if objective_type == 'direct_mask_prediction_independent_encoding':
    batch_size = 4
  else:
    batch_size = 8
  train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
  # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
  ood_dataloader_bbbc030 = DataLoader(ood_dataset_bbbc030, batch_size=1, shuffle=True)
  ood_dataloader_hek = DataLoader(ood_dataset_hek, batch_size=1, shuffle=True)
  ood_dataloader_n2a = DataLoader(ood_dataset_n2a, batch_size=1, shuffle=True)
  print(f'loaded {len(train_dataset)} train images and {len(val_dataset)} val images')

  #hyperparameter setup
  if hyperparameter_type == 'original':
    lr = 1e-5
    weight_decay = 1e-3
    drop_rate = 0.0
  elif hyperparameter_type == 'cellpose_sam':
    lr = 5e-5
    weight_decay = 0.1
    drop_rate = 0.4
  elif hyperparameter_type == 'gpt_suggested':
    lr = 1e-5
    weight_decay = 1e-4
    drop_rate = 0.1
  elif hyperparameter_type == 'hyperparam_4':
    lr = 3e-5
    weight_decay = 1e-4
    drop_rate = 0.02
  elif hyperparameter_type == 'hyperparam_5':
    lr = 2e-5
    weight_decay = 1e-4
    drop_rate = 0.05
  elif hyperparameter_type == 'hyperparam_6':
    lr = 1e-5
    weight_decay = 1e-4
    drop_rate = 0.05


  #model setup
  sam_model = 'facebook/sam-vit-base' #deprecated but leaving for now
  if objective_type == 'direct_mask_prediction' or objective_type == 'direct_mask_prediction_independent_encoding':
    flows_model_modifier = f'piw_{str(int(ignore_mask_weight * 100))}_isd_{str(include_silver_data)}_hpt_{hyperparameter_type}_die_{dropout_in_encoder}_ftv_{str(finetune_vision)}_llre_{str(lower_lr_encoder)}_deu_{str(delayed_encoder_unfreezing)}_cs_{crop_size}_sdw_{str(int(silver_data_weight * 100))}_ne_{num_epochs}'
    flows_model = f'../models/DINOCell_{dataset_type}_{dino_type}_{decoder_type}_flows_{flows_model_modifier}_train_checkpoint.pth'
  else:
    flows_model = None
  if delayed_encoder_unfreezing:
    finetune_vision = False
  print('loading model...') # CHANGE - loss function for direct mask prediction
  model = DINOCell(dino_model, sam_model, flows_model, decoder_type, objective_type, use_dino_weights, patch_size, feat_size=64, crop_size=crop_size, drop_rate=drop_rate, dropout_in_encoder=dropout_in_encoder, finetune_vision=finetune_vision, finetune_decoder=True, finetune_prediction_head=True) 

  #train setup
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model.to(device)

  if lower_lr_encoder:
    optimizer = AdamW([
      {
        "params":model.encoder.parameters(), 
        "lr":lr / 10, 
        "weight_decay":weight_decay, 
        "betas":(0.9, 0.999)
      },
      {
        "params":model.decoder.parameters(), 
        "lr":lr, 
        "weight_decay":weight_decay, 
        "betas":(0.9, 0.999)
      },
      {
        "params":model.prediction_head.parameters(), 
        "lr":lr, 
        "weight_decay":weight_decay, 
        "betas":(0.9, 0.999)
      }
    ]) 
  else:
    optimizer = AdamW(
      model.parameters(), 
      lr=lr, 
      weight_decay=weight_decay, 
      betas=(0.9, 0.999)
    ) 
    
    #normal lr: 1e-5, lower: 1e-6, old weight_decay: 0.1, now: 1e-3
  scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda_with_warmup(step, warmup_steps=(num_epochs * len(train_dataset)) // 5, total_steps=(num_epochs * len(train_dataset))))

  pipeline = get_pipeline(objective_type, model, device, crop_size=crop_size, use_advanced_augmentations=use_advanced_augmentations)

  #resume/save training
  model_save_path = f'../models/DINOCell_{dataset_type}_{dino_type}_{decoder_type}_{objective_type}_{log_modifier}.pt'
  train_state_save_path = f'../models/DINOCell_{dataset_type}_{dino_type}_{decoder_type}_{objective_type}_{log_modifier}_train_checkpoint.pth'
  if resume_training:
    checkpoint = torch.load(train_state_save_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    model.to(device)
    print(f'Resuming from Epoch {start_epoch}...')
  else:
    start_epoch = 0
    with open(f'{log_path}/metric_log.txt', "w") as file:
      file.write('')
  
  #CHANGE
  # val_loss, val_metrics = val(model=model, patch_size=patch_size, val_dataloader=val_dataloader, pipeline=pipeline, dataset_root_path=dataset_root_path, dataset_type=dataset_type, objective_type=objective_type, log_path=log_path, epoch=99, device=device)
  # ood_test_bbbc030_metrics, ood_test_hek_metrics, ood_test_n2a_metrics = ood_test(model=model, ood_dataloader_bbbc030=ood_dataloader_bbbc030, ood_dataloader_hek=ood_dataloader_hek, ood_dataloader_n2a=ood_dataloader_n2a, pipeline=pipeline, objective_type=objective_type, log_path=log_path, epoch=99, device=device)


  print('training...')
  for epoch in range(start_epoch, num_epochs):
    #unfreeze encoder if doing delayed encoder unfreezing
    if delayed_encoder_unfreezing:
      if epoch >= 10:
        print("Unfreezing image encoder")
        for p in model.encoder.parameters():
            p.requires_grad = True

    if objective_type == 'direct_mask_prediction':
      if epoch >= 15:
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.decoder.parameters():
            p.requires_grad = True
        for p in model.prediction_head.parameters():
            p.requires_grad = True

    #train step
    train_loss = train(model=model, train_dataloader=train_dataloader, optimizer=optimizer, scheduler=scheduler, objective_type=objective_type, model_save_path=model_save_path, train_state_save_path=train_state_save_path, epoch=epoch, device=device)

    #val step
    val_loss, val_metrics = val(model=model, patch_size=patch_size, val_dataloader=val_dataloader, pipeline=pipeline, dataset_root_path=dataset_root_path, dataset_type=dataset_type, objective_type=objective_type, log_path=log_path, epoch=epoch, device=device)

    #test step
    # test_loss, test_metrics = test(model=model, patch_size=patch_size, test_dataloader=test_dataloader, pipeline=pipeline, dataset_root_path=dataset_root_path, dataset_type=dataset_type, log_path=log_path, epoch=epoch, device=device)

    #ood_test step
    ood_test_bbbc030_metrics, ood_test_hek_metrics, ood_test_n2a_metrics = ood_test(model=model, ood_dataloader_bbbc030=ood_dataloader_bbbc030, ood_dataloader_hek=ood_dataloader_hek, ood_dataloader_n2a=ood_dataloader_n2a, pipeline=pipeline, objective_type=objective_type, log_path=log_path, epoch=epoch, device=device)

    log_string = f'Epoch {epoch}: Train Loss - {train_loss} || Val Loss - {val_loss}'
    for key in val_metrics.keys():
      log_string += f' || Val {key} - {val_metrics[key]}'
    for key in ood_test_bbbc030_metrics.keys():
      log_string += f' || OOD-BBBC030 {key} - {ood_test_bbbc030_metrics[key]}'
    for key in ood_test_hek_metrics.keys():
      log_string += f' || OOD-HEK {key} - {ood_test_hek_metrics[key]}'
    for key in ood_test_n2a_metrics.keys():
      log_string += f' || OOD-N2A {key} - {ood_test_n2a_metrics[key]}'

    print(log_string)

    with open(f'{log_path}/metric_log.txt', 'a') as f:
        f.write(f'{log_string}\n')



  
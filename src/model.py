import cv2
import math
import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random

import itertools
import numpy as np
import wandb
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Type

from torchvision.ops import sigmoid_focal_loss
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt

from utils import get_dist_map_metrics, get_flows_metrics, get_direct_mask_metrics
from cellpose.dynamics import compute_masks

from transformers import SamModel, SamConfig, SamMaskDecoderConfig
from transformers.modeling_outputs import BaseModelOutput

from utils import LayerNorm2d, SoftDiceLoss, get_IoU

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.models.vision_transformer import vit_base

from transformer import TwoWayTransformer

import matplotlib.pyplot as plt

class DINOCell(nn.Module):
    def __init__(self, dino_model, sam_model, flows_model, decoder_type, objective_type, use_dino_weights, patch_size=14, feat_size=64, crop_size=256, drop_rate=0.0, dropout_in_encoder=False, finetune_vision=True, finetune_decoder=True, finetune_prediction_head=True, finetune_direct_mask_encoder=True):
        super().__init__()
        
        self.decoder_type = decoder_type
        self.objective_type = objective_type
        self.use_dino_weights = use_dino_weights
        self.embed_dim = 768
        self.patch_size = patch_size
        self.feat_size = 896 // self.patch_size
        self.drop_rate = drop_rate
        self.dropout_in_encoder = dropout_in_encoder


        if self.decoder_type != 'SAM':
            self.decoder_dim = 512

            # Initialize dino encoder
            self.encoder = DINO_Encoder(dino_model, self.use_dino_weights, self.decoder_type, self.patch_size, self.embed_dim, self.decoder_dim, self.drop_rate, self.dropout_in_encoder)

            self.num_upsamples = int(math.sqrt(crop_size // self.feat_size))

            # Initialize decoder
            self.decoder = DINOCell_Decoder(decoder_type, self.num_upsamples, crop_size, self.decoder_dim, self.drop_rate)
            
            self.feat_dim = self.decoder_dim // (2**(self.num_upsamples + 1))

            # Initialize prediction head
            self.prediction_head = DINOCell_Prediction_Head(self.objective_type, self.feat_dim, self.drop_rate)

            if self.objective_type == 'direct_mask_prediction' or self.objective_type == 'direct_mask_prediction_independent_encoding':
                #Turn off gradients for encoder, decoder, and flows head
                finetune_vision = False
                finetune_decoder = False
                finetune_prediction_head = False


                #Load pretrained flows model weights into encoder, decoder, and flows prediction head
                flows_model_weights = torch.load(flows_model, map_location="cpu", weights_only=False)
                encoder_weights = {}
                decoder_weights = {}
                prediction_head_weights = {}
                for weight_category in flows_model_weights['model_state_dict'].keys():
                    if 'encoder' in weight_category:
                        encoder_weights[weight_category[8:]] = flows_model_weights['model_state_dict'][weight_category]
                    elif 'decoder' in weight_category:
                        decoder_weights[weight_category[8:]] = flows_model_weights['model_state_dict'][weight_category]
                    elif 'prediction_head' in weight_category:
                        prediction_head_weights[weight_category[16:]] = flows_model_weights['model_state_dict'][weight_category]
                
                self.encoder.load_state_dict(encoder_weights)
                self.decoder.load_state_dict(decoder_weights)
                self.prediction_head.load_state_dict(prediction_head_weights)

                #Initialize Prompt Decoder
                self.prompt_encoder_and_mask_decoder = DINOCell_Prompt_Encoder_and_Mask_Decoder((crop_size, crop_size), (self.feat_size, self.feat_size))

                if self.objective_type == 'direct_mask_prediction_independent_encoding':
                    #initialize an independent encoder for the direct mask prediction
                    self.direct_mask_prediction_image_encoder = DINO_Encoder(dino_model, self.decoder_type, self.use_dino_weights, self.patch_size, self.embed_dim, self.decoder_dim, self.drop_rate, self.dropout_in_encoder)

                    if not finetune_direct_mask_encoder:
                        for param in self.direct_mask_prediction_image_encoder.parameters():
                            param.requires_grad_(False)
                    else:
                        for param in self.direct_mask_prediction_image_encoder.parameters():
                            param.requires_grad_(True)

            #freeze required layers
            if not finetune_vision:
                for param in self.encoder.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.encoder.parameters():
                    param.requires_grad_(True)
            
            if not finetune_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.decoder.parameters():
                    param.requires_grad_(True)

            if not finetune_prediction_head:
                for param in self.prediction_head.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.prediction_head.parameters():
                    param.requires_grad_(True)
        else:
            # Load pretrained SAM model
            config = SamConfig.from_pretrained(sam_model, ignore_mismatched_sizes=True)
            self.model = SamModel(config)

            self.dino_encoder = DINO_Encoder(dino_model, self.decoder_type, self.embed_dim, 256)

            self.model.vision_encoder = self.dino_encoder

            #freeze required layers
            if not finetune_vision:
                for param in self.model.vision_encoder.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.model.vision_encoder.parameters():
                    param.requires_grad_(True)
                
            if not finetune_prompt:
                for param in self.model.prompt_encoder.parameters():
                    param.requires_grad_(False)
            
            if not finetune_prediction_head:
                for param in self.model.mask_decoder.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.model.mask_decoder.parameters():
                    param.requires_grad_(True)

    def forward(self, x, targets=None, ignore_masks=None, labels=None):
        if self.decoder_type == 'SAM':
            dist_map_pred = self.model(x)['pred_masks'].squeeze(1)[:, 0, : , :]
            preds = [dist_map_pred]

            if targets is not None:
                loss_fn = nn.MSELoss(reduction='none')

                loss = loss_fn(dist_map_pred, targets['gt_dist_map'].to(x.device))

                example_weights = targets['example_weight'].view(-1, 1, 1).to(x.device)
                
                if ignore_masks is not None:
                    pixel_weights = ignore_masks.to(x.device).float() * example_weights
                    weighted_loss = loss * pixel_weights
                    loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)
                else:
                    pseudo_ignore_masks = np.ones_like(loss)
                    pixel_weights = ignore_masks.to(x.device).float() * example_weights
                    weighted_loss = loss * pixel_weights
                    loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)

                if labels is not None:
                    metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                    return preds, loss, metrics
                else:
                    return preds, loss
            else:
                if labels is not None:
                    metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                    return preds, metrics
                else:
                    return preds
        elif self.objective_type == 'direct_mask_prediction':
            img_embed = self.encoder(x)
            decoded_img_embed = self.decoder(img_embed)

            if targets is not None:
                if targets['gt_centers'] is None:
                    flows_pred, flows_loss = self.prediction_head(decoded_img_embed, targets, ignore_masks)

                    if labels is not None:
                        mask_preds, iou_preds, mask_loss, metrics = self.prompt_encoder_and_mask_decoder(img_embed, flows_pred, targets, labels)
                    else:
                        mask_preds, iou_preds, mask_loss = self.prompt_encoder_and_mask_decoder(img_embed, flows_pred, targets, labels)
                    
                    loss = (0.2 * flows_loss) + mask_loss

                    preds = [flows_pred, mask_preds, iou_preds]
                else:
                    if labels is not None:
                        mask_preds, iou_preds, mask_loss, metrics = self.prompt_encoder_and_mask_decoder(img_embed, None, targets, labels)
                    else:
                        mask_preds, iou_preds, mask_loss = self.prompt_encoder_and_mask_decoder(img_embed, None, targets, labels)

                    loss = mask_loss

                    preds = [mask_preds, iou_preds]

                if labels is None:
                    return preds, loss
                else: 
                    return preds, loss, metrics
            else:
                flows_pred = self.prediction_head(decoded_img_embed, targets, ignore_masks)

                if labels is None:
                    mask_preds, iou_preds = self.prompt_encoder_and_mask_decoder(img_embed, flows_pred, targets, labels)
                    preds = [flows_pred, mask_preds, iou_preds]
                    return preds
                else: 
                    mask_preds, iou_preds, metrics = self.prompt_encoder_and_mask_decoder(img_embed, flows_pred, targets, labels)
                    preds = [flows_pred, mask_preds, iou_preds]
                    return preds, metrics
        elif self.objective_type == 'direct_mask_prediction_independent_encoding':
            direct_mask_img_embed = self.direct_mask_prediction_image_encoder(x)

            if targets is not None:
                if targets['gt_centers'] is None:
                    img_embed = self.encoder(x)
                    decoded_img_embed = self.decoder(img_embed)
                    flows_pred, _ = self.prediction_head(decoded_img_embed, targets, ignore_masks)

                    if labels is not None:
                        mask_preds, iou_preds, mask_loss, metrics = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, flows_pred, targets, labels)
                    else:
                        mask_preds, iou_preds, mask_loss = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, flows_pred, targets, labels)
                    
                    loss = mask_loss

                    preds = [flows_pred, mask_preds, iou_preds]
                else:
                    if labels is not None:
                        mask_preds, iou_preds, mask_loss, metrics = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, None, targets, labels)
                    else:
                        mask_preds, iou_preds, mask_loss = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, None, targets, labels)

                    loss = mask_loss

                    preds = [mask_preds, iou_preds]

                if labels is None:
                    return preds, loss
                else: 
                    return preds, loss, metrics
            else:
                img_embed = self.encoder(x)
                decoded_img_embed = self.decoder(img_embed)
                flows_pred = self.prediction_head(decoded_img_embed, targets, ignore_masks)

                if labels is None:
                    mask_preds, iou_preds = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, flows_pred, targets, labels)
                    preds = [flows_pred, mask_preds, iou_preds]
                    return preds
                else: 
                    mask_preds, iou_preds, metrics = self.prompt_encoder_and_mask_decoder(direct_mask_img_embed, flows_pred, targets, labels)
                    preds = [flows_pred, mask_preds, iou_preds]
                    return preds, metrics
        else:
            x = self.encoder(x)
            x = self.decoder(x)
            if targets is not None:
                if labels is not None:
                    preds, loss, metrics = self.prediction_head(x, targets, ignore_masks, labels)
                    return preds, loss, metrics
                else:
                    preds, loss = self.prediction_head(x, targets, ignore_masks, labels)
                    return preds, loss
            else:
                if labels is not None:
                    preds, metrics = self.prediction_head(x, targets, ignore_masks, labels)
                    return preds, metrics
                else:
                    preds = self.prediction_head(x, targets, ignore_masks, labels)
                    return preds
        
                

##################################
#            ENCODER
##################################

class DINO_Encoder(nn.Module):
    def __init__(self, dino_model, use_dino_weights, decoder_type, patch_size, embed_dim, decoder_dim, drop_rate, dropout_in_encoder):
        super().__init__()

        self.decoder_type = decoder_type
        self.patch_size = patch_size
        self.num_patches = 896 // self.patch_size

        if use_dino_weights:
            # Process DINO weights
            if 'not_domain_adapted' in dino_model:
                checkpoint = torch.load(dino_model, map_location="cpu", weights_only=False)
                dino_checkpoint = {}
                for key in checkpoint.keys():
                    new_key = f'backbone.{key}'
                    dino_checkpoint[new_key] = checkpoint[key]
            else:
                dino_checkpoint = torch.load(dino_model, map_location="cpu", weights_only=False)['teacher']
            dino_backbone_weights = {}
            for key in dino_checkpoint.keys():
                if 'backbone' in key:
                    weight_key = key[9:]
                    dino_backbone_weights[weight_key] = dino_checkpoint[key]

            # Interpolate DINO pos_embed weights to 64x64 (from 37x37) or 112x112 (from 64x64)
            dino_pos_embed = dino_backbone_weights['pos_embed']
            cls_token_pos_embed = dino_pos_embed[:, :1, :] # Shape: (1, 1, embed_dim)
            patch_pos_embed_2d = dino_pos_embed[:, 1:, :] # Shape: (1, 1369, embed_dim)

            # Calculate original grid dimensions (37x37)
            orig_grid_h = orig_grid_w = int(patch_pos_embed_2d.shape[1]**0.5) # Should be 37 or 64

            # Reshape from (1, H*W, C) to (1, C, H, W)
            patch_pos_embed_2d = patch_pos_embed_2d.permute(0, 2, 1).reshape(
                1, -1, orig_grid_h, orig_grid_w
            ) # Shape: (1, embed_dim, 37, 37) or (1, embed_dim, 64, 64)

            target_grid_h = target_grid_w = self.num_patches # For 896x896 image with 14x14 patches or 8x8 patches

            interpolated_patch_pos_embed = F.interpolate(
                patch_pos_embed_2d,
                size=(target_grid_h, target_grid_w),
                mode='bicubic', 
                align_corners=False 
            ) # Shape: (1, embed_dim, self.num_patches, self.num_patches)

            # Flatten back to (1, H_new*W_new, C)
            interpolated_patch_pos_embed = interpolated_patch_pos_embed.flatten(2).permute(0, 2, 1) # Shape: (1, 4096, embed_dim)

            # Concatenate the CLS token back
            interpolated_pos_embed_with_cls = torch.cat([cls_token_pos_embed, interpolated_patch_pos_embed], dim=1)

            # Insert interpolated position embedding weights back into DINO model weights
            dino_backbone_weights['pos_embed'] = interpolated_pos_embed_with_cls

            # 2) Interpolate DINO patch.proj weights to 14x14 to 8x8
            if (patch_size == 8 and 'not_domain_adapted' in dino_model) or (patch_size == 8 and 'random_init' in dino_model):
                dino_patch_proj_weight = dino_backbone_weights['patch_embed.proj.weight']

                target_patch_h = target_patch_w = 8

                interpolated_dino_patch_proj_weight = F.interpolate(
                    dino_patch_proj_weight,
                    size=(target_patch_h, target_patch_w),
                    mode='bicubic', 
                    align_corners=False 
                ) # Shape: (embed_dim, 3, 8, 8)

                # Insert interpolated patch projection weights back into DINO model weights
                dino_backbone_weights['patch_embed.proj.weight'] = interpolated_dino_patch_proj_weight

        if not dropout_in_encoder:
            encoder_drop_rate = 0.0
        else:
            encoder_drop_rate = drop_rate

        #create DINO base vit model
        self.dino_encoder = vit_base(
            img_size=896, #896
            patch_size=self.patch_size,
            block_chunks=0,
            num_register_tokens=4,
            init_values=1e-5,
            drop_path_rate=encoder_drop_rate
        )

        if use_dino_weights:
            # load trained weights in DINO model
            self.dino_encoder.load_state_dict(dino_backbone_weights)

        self.vision_neck = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_dim, kernel_size=(1,1), bias=False),
            LayerNorm2d(decoder_dim),
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=(3, 3), padding=1, bias=False),
            LayerNorm2d(decoder_dim)
        )

    def forward(self, x, **kwargs):
        # embed patches
        x = self.dino_encoder(x)['x_norm_patchtokens'].to(x.device)
        
        B, N, D = x.shape
        x = x.reshape(
            B, self.num_patches, self.num_patches, -1
        ).permute(0, 3, 1, 2) # Shape: (B, embed_dim, self.num_patches, self.num_patches)

        x = self.vision_neck(x).to(x.device) # Shape: (B, decoder_dim, self.num_path, self.num_patches)

        if self.decoder_type != 'SAM':
            return x
        else:
            return BaseModelOutput(last_hidden_state=x)

##################################
#           DECODERS
##################################

class DINOCell_Decoder(nn.Module):
    def __init__(self, decoder_type, num_upsamples, crop_size, decoder_dim, drop_rate):
        super().__init__()

        if decoder_type == 'convolutional':
            self.decoder = Convolutional_Decoder(num_upsamples, crop_size, decoder_dim, drop_rate)
        elif decoder_type == 'upsample':
            self.decoder = Upsample_Decoder(num_upsamples, crop_size, decoder_dim, drop_rate)
    
    def forward(self, x):
        x = self.decoder(x)

        return x



class Upsample_Decoder(nn.Module):
    def __init__(self, num_upsamples, crop_size, decoder_dim, drop_rate):
        super().__init__()

        self.decoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_dim // (2**i), decoder_dim // (2**(i+1)), kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(drop_rate)
                )
                for i in range(num_upsamples)
            ] 
        )
        self.decoder.append(nn.Sequential(
            nn.Conv2d(decoder_dim // (2**num_upsamples), decoder_dim // (2**(num_upsamples+1)), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=crop_size, mode='bilinear')
        ))

    def forward(self, x):
        x = self.decoder(x)

        return x


class Convolutional_Decoder(nn.Module):
    def __init__(self, num_upsamples, crop_size, decoder_dim, drop_rate):
        super().__init__()

        self.decoder = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(decoder_dim // (2**i), decoder_dim // (2**(i+1)), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(drop_rate),
                    nn.Upsample(scale_factor=2, mode='bilinear')
                )
                for i in range(num_upsamples)
            ] 
        )

        self.decoder.append(nn.Sequential(
            nn.Conv2d(decoder_dim // (2**num_upsamples), decoder_dim // (2**(num_upsamples+1)), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=crop_size, mode='bilinear')
        ))

    def forward(self, x):
        x = self.decoder(x)

        return x




##################################
#       PREDICTION HEADS
##################################


class DINOCell_Prediction_Head(nn.Module):
    def __init__(self, objective_type, feat_dim, drop_rate):
        super().__init__()

        if objective_type == 'distance_map':
            self.prediction_head = DistanceMapPredictionHead(feat_dim, drop_rate)
        elif objective_type == 'directional_distance_map':
            self.prediction_head = DirectionalDistanceMapPredictionHead(feat_dim, drop_rate)
        elif objective_type == 'flows' or objective_type == 'direct_mask_prediction' or  objective_type == 'direct_mask_prediction_independent_encoding':
            self.prediction_head = FlowsPredictionHead(feat_dim, drop_rate)
        

    def forward(self, x, targets=None, ignore_masks=None, labels=None):
        if targets is not None:
            if labels is not None:
                preds, loss, metrics = self.prediction_head(x, targets, ignore_masks, labels)
                return preds, loss, metrics
            else:
                preds, loss = self.prediction_head(x, targets, ignore_masks, labels)
                return preds, loss
        else:
            if labels is not None:
                preds, metrics = self.prediction_head(x, targets, ignore_masks, labels)
                return preds, metrics
            else:
                preds = self.prediction_head(x, targets, ignore_masks, labels)
                return preds

class DistanceMapPredictionHead(nn.Module):
    def __init__(self, feat_dim, drop_rate):
        super().__init__()

        self.dist_map_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )


    def forward(self, x, targets=None, ignore_masks=None, labels=None):
        dist_map_pred = self.dist_map_pred_head(x).squeeze(1)

        preds = [dist_map_pred]

        if targets is not None:
            loss_fn = nn.MSELoss(reduction='none')

            loss = loss_fn(dist_map_pred, targets['gt_dist_map'].to(x.device))
            
            # if ignore_masks is not None:
            #     masked_loss = loss * ignore_masks.to(x.device).float()
            #     loss = masked_loss.sum() / (ignore_masks.sum() + 1e-6)
            # else:
            #     loss = loss.mean()

            example_weights = targets['example_weight'].view(-1, 1, 1).to(x.device)

            if ignore_masks is not None:
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)
            else:
                pseudo_ignore_masks = np.ones_like(loss)
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)

            if labels is not None:
                metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, loss, metrics
            else:
                return preds, loss
        else:
            if labels is not None:
                metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, metrics
            else:
                return preds


class DirectionalDistanceMapPredictionHead(nn.Module):
    def __init__(self, feat_dim, drop_rate):
        super().__init__()

        self.dist_map_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.dist_map_top_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.dist_map_bottom_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.dist_map_left_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.dist_map_right_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

    def forward(self, x, targets=None, ignore_masks=None, labels=None):
        dist_map_pred = self.dist_map_pred_head(x).squeeze(1)
        dist_map_top_pred = self.dist_map_top_pred_head(x).squeeze(1)
        dist_map_bottom_pred = self.dist_map_bottom_pred_head(x).squeeze(1)
        dist_map_left_pred = self.dist_map_left_pred_head(x).squeeze(1)
        dist_map_right_pred = self.dist_map_right_pred_head(x).squeeze(1)

        preds = [dist_map_pred, dist_map_top_pred, dist_map_bottom_pred, dist_map_left_pred, dist_map_right_pred]

        if targets is not None:
            loss_fn = nn.MSELoss(reduction='none')

            dist_map_loss = loss_fn(dist_map_pred, targets['gt_dist_map'].to(x.device))
            dist_map_top_loss = loss_fn(dist_map_top_pred, targets['gt_dist_map_top'].to(x.device))
            dist_map_bottom_loss = loss_fn(dist_map_bottom_pred, targets['gt_dist_map_bottom'].to(x.device))
            dist_map_left_loss = loss_fn(dist_map_left_pred, targets['gt_dist_map_left'].to(x.device))
            dist_map_right_loss = loss_fn(dist_map_right_pred, targets['gt_dist_map_right'].to(x.device))

            loss = dist_map_loss + (0.5 * dist_map_top_loss) + (0.5 * dist_map_bottom_loss) + (0.5 * dist_map_left_loss) + (0.5 * dist_map_right_loss)
            
            example_weights = targets['example_weight'].view(-1, 1, 1).to(x.device)

            if ignore_masks is not None:
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)
            else:
                pseudo_ignore_masks = np.ones_like(loss)
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)

            if labels is not None:
                metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, loss, metrics
            else:
                return preds, loss
        else:
            if labels is not None:
                metrics = get_dist_map_metrics(dist_map_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, metrics
            else:
                return preds


class FlowsPredictionHead(nn.Module):
    def __init__(self, feat_dim, drop_rate):
        super().__init__()


        # self.dist_map_pred_head = nn.Sequential(
        #     nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        # )

        self.flows_dx_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.flows_dy_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.cell_prob_pred_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim // 2, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop_rate),
            nn.Conv2d(feat_dim // 2, feat_dim // 4, kernel_size=(3,3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 4, 1, kernel_size=(1,1))
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x, targets=None, ignore_masks=None, labels=None):
        # dist_map_pred = self.dist_map_pred_head(x).squeeze(1)
        flows_dx_pred = self.flows_dx_pred_head(x).squeeze(1)
        flows_dy_pred = self.flows_dy_pred_head(x).squeeze(1)
        cell_prob_pred = self.sigmoid(self.cell_prob_pred_head(x).squeeze(1))

        # preds = [dist_map_pred, flows_dx_pred, flows_dy_pred, cell_prob_pred]
        preds = [flows_dx_pred, flows_dy_pred, cell_prob_pred]


        if targets is not None:
            loss_fn = nn.MSELoss(reduction='none')
            loss_fn_bce = nn.BCELoss(reduction='none')

            # dist_map_loss = loss_fn(dist_map_pred, targets['gt_dist_map'].to(x.device))
            flows_dx_loss = loss_fn(flows_dx_pred, 5 * targets['gt_flows_dx'].to(x.device))
            flows_dy_loss = loss_fn(flows_dy_pred, 5 * targets['gt_flows_dy'].to(x.device))
            cell_prob_loss = loss_fn_bce(cell_prob_pred, targets['gt_cell_prob'].to(x.device))

            # loss = dist_map_loss + flows_dx_loss + flows_dy_loss + cell_prob_loss
            loss = flows_dx_loss + flows_dy_loss + cell_prob_loss

            example_weights = targets['example_weight'].view(-1, 1, 1).to(x.device)
            
            if ignore_masks is not None:
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)
            else:
                pseudo_ignore_masks = np.ones_like(loss)
                pixel_weights = ignore_masks.to(x.device).float() * example_weights
                weighted_loss = loss * pixel_weights
                loss =  weighted_loss.sum() / (pixel_weights.sum() + 1e-6)

            if labels is not None:
                metrics = get_flows_metrics(flows_dx_pred.detach().cpu().numpy(), flows_dy_pred.detach().cpu().numpy(), cell_prob_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, loss, metrics
            else:
                return preds, loss
        else:
            if labels is not None:
                metrics = get_flows_metrics(flows_dx_pred.detach().cpu().numpy(), flows_dy_pred.detach().cpu().numpy(), cell_prob_pred.detach().cpu().numpy(), labels.detach().cpu().numpy(), ignore_masks)
                return preds, metrics
            else:
                return preds


class DINOCell_Prompt_Decoder(nn.Module):
    def __init__(self, feat_dim, num_queries=128, num_layers=3):
        super().__init__()

        self.num_queries = num_queries

        #Initialize learnable queries
        self.query_embeddings = nn.Parameter(
            torch.randn(self.num_queries, feat_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feat_dim,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.objectness_head = nn.Linear(feat_dim, 1)
        self.coord_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 2),
            nn.Sigmoid()  # normalize to [0,1]
        )

    def get_normalized_true_center_coords(self, true_center):
        center_coords = torch.argwhere(true_center == 1) / true_center.shape[0]
        
        return center_coords

    def get_hungarian_matching_loss(self, objectness, coords, true_centers):
        total_loss = 0.0
        obj_criterion = nn.BCEWithLogitsLoss()
        dist_criterion = nn.L1Loss()

        for i in range(true_centers.shape[0]):
            pred_coords = coords[i]
            true_center_coords = self.get_normalized_true_center_coords(true_centers[i]).to('cuda')
            if true_center_coords.shape[0] == 0:
                continue

            cost_dist = torch.linalg.norm(
                true_center_coords[:, None, :] - pred_coords[None, : , :],
                axis=-1
            )
            
            pred_objectness = objectness[i]
            cost_obj = -torch.log(torch.sigmoid(pred_objectness))

            #Penalty if coordinate not in cell mask??
            cost = cost_dist + (0.5 * cost_obj)

            gt_indices, pred_indices = linear_sum_assignment(cost.detach().cpu().numpy())

            true_objectness = torch.zeros(self.num_queries, device='cuda')
            for index in pred_indices:
                true_objectness[index] = 1
            
            L_obj = obj_criterion(pred_objectness.to('cuda'), true_objectness)
            L_dist = 0.0
            for j in range(gt_indices.shape[0]):
                L_dist += dist_criterion(pred_coords[pred_indices[j]], true_center_coords[gt_indices[j]])
            L_dist = L_dist / gt_indices.shape[0]

            #Loss penalty for not being in mask??
            total_loss += (L_obj + L_dist)
        
        total_loss = total_loss / true_centers.shape[0]

        return total_loss

            


    def forward(self, image_encoding, true_centers=None):
        #Flatten image encoding from (B, D, self.num_patches, self.num_patches) to (B, self.num_patches**2, D)
        image_encoding = image_encoding.flatten(2)
        image_encoding = image_encoding.permute(0, 2, 1)

        B = image_encoding.size(0)

        # Expand learned queries across batch
        queries = self.query_embeddings.unsqueeze(0).repeat(B, 1, 1)
        # (B, Nq, D)

        # Cross-attend queries to image
        decoded_queries = self.decoder(
            tgt=queries,
            memory=image_encoding
        )
        # (B, Nq, D)

        objectness = self.objectness_head(decoded_queries).squeeze(-1) # (B, Nq)
        coords = self.coord_head(decoded_queries) # (B, Nq, 2)

        preds = [objectness, coords]

        if true_centers is not None:
            #Ignore masking???
            loss = self.get_hungarian_matching_loss(objectness, coords, true_centers)

            return preds, loss
        else:
            return preds





class DINOCell_Prompt_Encoder_and_Mask_Decoder(nn.Module):
    def __init__(self, img_size, img_embed_size, prompt_embed_dim=512):
        super().__init__()

        self.crop_size = img_size[0]

        # self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        self.bce_loss_fn = nn.BCELoss(reduction='mean')

        self.dice_loss_fn = SoftDiceLoss()


        self.prompt_encoder = PromptEncoder(prompt_embed_dim, img_embed_size, img_size)

        self.mask_decoder = MaskDecoder(
            crop_size=self.crop_size,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

    def get_centers_from_dist_map(self, dist_maps):
        pred_center_maps = []
        for dist_map in dist_maps:
            dist_map = dist_map.detach().cpu().numpy()
            cells_max = dist_map > 0.5
            cell_fill = dist_map > 0.05
            #find centroids of connected components
            contours, _ = cv2.findContours(cells_max.astype(np.uint8), 0, cv2.CHAIN_APPROX_SIMPLE)
            centers_map = np.zeros(dist_map.shape, dtype=np.int32)
            # for i, contour in enumerate(contours):
            #     contour = np.flip(contour, axis=2)
            #     mask[tuple(contour.T)] = i + 1

            for i, contour in enumerate(contours):
                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    # Handle cases where the moment is zero to avoid division by zero
                    cX, cY = 0, 0

                #set closest pixel to centroid
                centers_map[int(cY)-2:int(cY)+2, int(cX)-2:int(cX)+2] = i + 1

            pred_center_maps.append(centers_map)

        return torch.from_numpy(np.array(pred_center_maps))

    
    def cells_from_flows(self, flows_dx, flows_dy, cell_prob, cellprob_threshold=0.5):
        """
        flows_3ch => shape (3, H, W) => (dx, dy, cell_prob).
        We interpret channel 0 => dx, 1 => dy, 2 => cell_prob
        Then we call cellpose's compute_masks(dP, cellprob_threshold=whatever).
        """

        masks = []

        for i in range(len(flows_dx)):
            curr_flows_x = flows_dx[i].detach().cpu().numpy()
            curr_flows_y = flows_dy[i].detach().cpu().numpy()
            curr_cell_prob = cell_prob[i].detach().cpu().numpy()

            # assemble dP with shape (2,H,W)
            dP = np.stack([curr_flows_y, curr_flows_x], axis=0)

            mask = compute_masks(
                dP=dP,
                cellprob=curr_cell_prob,
                niter=200,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=0,
                do_3D=False,
                min_size=15,
                max_size_fraction=0.4,
                device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
            masks.append(mask)
            
        return masks

    def get_centers_from_flows(self, flows, ignore_masks=None):
        cell_masks = self.cells_from_flows(flows[0], flows[1], flows[2])

        centers = []
        for i in range(len(cell_masks)):
            img_masks = cell_masks[i]
            # mask_dists = cv2.distanceTransform(img_masks, cv2.DIST_L2, 3)
            boundaries = find_boundaries(img_masks, mode="inner")
            dist = distance_transform_edt(~boundaries)
            dist[img_masks == 0] = 0

            centers_map = np.zeros(img_masks.shape, dtype=np.int32)

            # local_peaks = h_maxima(dist, h=2.0)
            # prompt_coords = np.column_stack(np.where(local_peaks))
            prompt_coords = peak_local_max(
                dist,
                min_distance=15,
                threshold_rel=0.2
            )

            for j in range(len(prompt_coords)):
                centers_map[prompt_coords[j][0], prompt_coords[j][1]] = j + 1

            if ignore_masks is not None:
                centers_map = centers_map * ignore_masks[i] #don't include prompt points that are in ignore mask areas

            centers.append(centers_map)

        return torch.from_numpy(np.array(centers))


    def forward(self, image_encoding, flows_pred, targets=None, labels=None):
        
        total_pixel_loss = 0.0
        total_dice_loss = 0.0
        total_iou_loss = 0.0
        num_losses = 0

        batch_masks = []
        batch_iou_preds = []

        B = image_encoding.size(0)

        ignore_masks = None
        if targets is not None:
            gt_centers = targets['gt_centers']
            gt_masks = targets['gt_mask']
            ignore_masks = targets['ignore_mask'].detach().cpu().numpy()
            cell_centers = gt_centers

        if targets is None or cell_centers is None:
            cell_centers = self.get_centers_from_flows(flows_pred, ignore_masks)

        for i in range(B):
            img_masks = []
            img_iou_preds = []

            curr_image_encoding = image_encoding[i]
            curr_centers = cell_centers[i].to('cuda')
            if targets is not None:
                curr_gt_mask = gt_masks[i].to('cuda')

            center_coords = torch.argwhere(curr_centers != 0)

            k = 8

            if targets is not None and center_coords.shape[0] > k:
                indices = random.sample(range(0, center_coords.shape[0]), k=k)
            else:
                indices = range(center_coords.shape[0])
            
            # indices = range(center_coords.shape[0]) # TEMP

            for index in indices:
                prompt_labels = torch.ones(1).to('cuda')

                prompt_coords = center_coords[index].unsqueeze(0).unsqueeze(0)
                prompt_labels = prompt_labels.unsqueeze(0)

                prompt_encoding = self.prompt_encoder((prompt_coords, prompt_labels))

                mask_preds, iou_pred = self.mask_decoder(
                    image_embeddings=curr_image_encoding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=prompt_encoding,
                    dense_prompt_embeddings=None,
                    multimask_output=True
                )

                mask_preds = mask_preds.squeeze(0)
                iou_pred = iou_pred.squeeze(0)

                if targets is not None:
                    coords = center_coords[index]
                    coord_x = int(coords[0])
                    coord_y = int(coords[1])
                    gt_mask_label = curr_centers[coord_x, coord_y]

                    gt_mask = torch.where(curr_gt_mask == gt_mask_label, 1.0, 0.0)

                    with torch.no_grad():
                        binary_mask_preds = torch.where(mask_preds > 0.5, 1, 0)
                        target_ious = get_IoU(binary_mask_preds, gt_mask.repeat(3, 1, 1))

                    best_mask_index = torch.argmax(target_ious)
                    mask_pred = mask_preds[best_mask_index]
                    mask_pred_iou_pred = iou_pred[best_mask_index]
                    img_masks.append(mask_pred.detach().cpu())
                    img_iou_preds.append(mask_pred_iou_pred.detach().cpu())


                    iou_loss = F.l1_loss(mask_pred_iou_pred, target_ious[best_mask_index])

                    # CHANGE
                    pixel_loss = sigmoid_focal_loss(
                        mask_pred,
                        gt_mask,
                        alpha=0.25,
                        gamma=2,
                        reduction="mean"
                    )

                    # pixel_loss = self.bce_loss_fn(mask_pred, gt_mask)

                    dice_loss = self.dice_loss_fn(mask_pred, gt_mask)

                    total_pixel_loss += pixel_loss
                    total_dice_loss += dice_loss
                    total_iou_loss += iou_loss
                    num_losses += 1
                else:
                    max_iou_pred_index = torch.argmax(iou_pred)

                    mask_pred = mask_preds[max_iou_pred_index] #.to(torch.half)
                    mask_pred_iou_pred = iou_pred[max_iou_pred_index]
                    img_masks.append(mask_pred.detach().cpu())
                    img_iou_preds.append(mask_pred_iou_pred.detach().cpu())
            
            batch_masks.append(img_masks)
            batch_iou_preds.append(img_iou_preds)


            # for index in indices:
            #     prompt_labels = torch.ones(len(center_coords)).to('cuda')
            #     prompt_labels[index] = 0

            #     prompt_coords = center_coords.unsqueeze(0)
            #     prompt_labels = prompt_labels.unsqueeze(0)


            #     prompt_encoding = self.prompt_encoder((prompt_coords, prompt_labels))

            #     mask_preds, iou_pred = self.mask_decoder(
            #         image_embeddings=curr_image_encoding.unsqueeze(0),
            #         image_pe=self.prompt_encoder.get_dense_pe(),
            #         sparse_prompt_embeddings=prompt_encoding,
            #         dense_prompt_embeddings=None,
            #         multimask_output=True
            #     )

            #     mask_preds = mask_preds.squeeze(0)

            #     max_iou_pred_index = torch.argmax(iou_pred)

            #     mask_pred = mask_preds[max_iou_pred_index] #.to(torch.half)
            #     img_masks.append(mask_pred.detach().cpu())

            #     if targets is not None:
            #         coords = center_coords[index]
            #         coord_x = int(coords[0])
            #         coord_y = int(coords[1])
            #         gt_mask_label = curr_centers[coord_x, coord_y]

            #         gt_mask = torch.where(curr_gt_mask == gt_mask_label, 1.0, 0.0)

            #         mask_loss = self.loss_fn(mask_pred, gt_mask)

            #         total_loss += mask_loss
            #         num_losses += 1
            
            # batch_masks.append(img_masks)

            


            # prompt_coords = []
            # curr_gt_masks = []
            # for index in indices:
            #     prompt_coords.append(center_coords[index])

            #     if targets is not None:
            #         coords = center_coords[index]
            #         coord_x = int(coords[0])
            #         coord_y = int(coords[1])
            #         gt_mask_label = curr_centers[coord_x, coord_y]

            #         gt_mask = torch.where(curr_gt_mask == gt_mask_label, 1.0, 0.0)

            #         curr_gt_masks.append(gt_mask)
            
            # K = len(indices)

            # if K > 0:

            #     prompt_labels = torch.ones((K, 1)).to('cuda')

            #     img_prompt_coords = torch.stack(prompt_coords).view(K, 1, 2)

            #     prompt_encoding = self.prompt_encoder((img_prompt_coords, prompt_labels))

            #     mask_preds, iou_pred = self.mask_decoder(
            #         image_embeddings=curr_image_encoding.unsqueeze(0),
            #         image_pe=self.prompt_encoder.get_dense_pe(),
            #         sparse_prompt_embeddings=prompt_encoding,
            #         dense_prompt_embeddings=None,
            #         multimask_output=True
            #     )

            #     max_iou_pred_indices = torch.argmax(iou_pred, dim=1)
            #     batch_indices = torch.arange(K, device='cuda')

            #     img_mask_preds = mask_preds[batch_indices, max_iou_pred_indices, :, :] #.to(torch.half)
            #     batch_masks.append(img_mask_preds)

            #     img_gt_masks = torch.stack(curr_gt_masks)

            #     mask_loss = loss_fn_bce(img_mask_preds, img_gt_masks)

            #     losses.append(mask_loss)
            # else:
            #     batch_masks.append([])
            #     losses.append(torch.tensor(0.0, device='cuda'))
            
        
        if targets is not None:
            
            if num_losses > 0:
                mean_pixel_loss = total_pixel_loss / num_losses
                mean_dice_loss = total_dice_loss / num_losses
                mean_iou_loss = total_iou_loss / num_losses

                total_loss = (((20 * mean_pixel_loss) + mean_dice_loss) + (0.1 * mean_iou_loss))
            else:
                total_loss = targets['gt_mask'].sum() * 0

            if labels is not None:
                metrics = get_direct_mask_metrics(batch_masks, batch_iou_preds, labels.detach().cpu().numpy(), ignore_masks, crop_size=self.crop_size)

                return batch_masks, batch_iou_preds, total_loss, metrics
            else:
                return batch_masks, batch_iou_preds, total_loss
        else:
            if labels is not None:
                metrics = get_direct_mask_metrics(batch_masks, batch_iou_preds, labels.detach().cpu().numpy(), ignore_masks, crop_size=self.crop_size)

                return batch_masks, batch_iou_preds, metrics
            else:
                return batch_masks, batch_iou_preds


                


            



                





class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_embedding_size,
        input_image_size,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # pos/neg point
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)


    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding


    def _get_batch_size(
        self,
        points
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points
    ):
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.

        Returns:
          torch.Tensor: sparse embeddings for the points, with shape
            BxNx(embed_dim), where N is determined by the number of input points.
        """
        bs = self._get_batch_size(points)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=True)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        return sparse_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C





class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        crop_size: int,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          crop_size (int): the size of the image crop (and how large the mask output should be)
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.crop_size = crop_size

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
            nn.Upsample(size=self.crop_size, mode='bilinear'),
            activation()
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        if dense_prompt_embeddings is not None:
            src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # I ADDED THIS
        masks = torch.sigmoid(masks)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x






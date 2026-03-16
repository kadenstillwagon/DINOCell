import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np

from utils import get_dist_map_metrics, get_flows_metrics

from transformers.modeling_outputs import BaseModelOutput

from utils import LayerNorm2d

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from dinov2.dinov2.models.vision_transformer import vit_base

import matplotlib.pyplot as plt

class DINOCell(nn.Module):
    def __init__(self, dino_model, decoder_type, objective_type, use_dino_weights, patch_size=14, feat_size=64, crop_size=256, drop_rate=0.0, dropout_in_encoder=False, finetune_vision=True, finetune_decoder=True, finetune_prediction_head=True):
        super().__init__()
        
        self.decoder_type = decoder_type
        self.objective_type = objective_type
        self.use_dino_weights = use_dino_weights
        self.embed_dim = 768
        self.patch_size = patch_size
        self.feat_size = 896 // self.patch_size
        self.drop_rate = drop_rate
        self.dropout_in_encoder = dropout_in_encoder


        self.decoder_dim = 512

        # Initialize dino encoder
        self.encoder = DINO_Encoder(dino_model, self.use_dino_weights, self.decoder_type, self.patch_size, self.embed_dim, self.decoder_dim, self.drop_rate, self.dropout_in_encoder)

        self.num_upsamples = int(math.sqrt(crop_size // self.feat_size))

        # Initialize decoder
        self.decoder = DINOCell_Decoder(decoder_type, self.num_upsamples, crop_size, self.decoder_dim, self.drop_rate)
        
        self.feat_dim = self.decoder_dim // (2**(self.num_upsamples + 1))

        # Initialize prediction head
        self.prediction_head = DINOCell_Prediction_Head(self.objective_type, self.feat_dim, self.drop_rate)

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

    def forward(self, x, targets=None, ignore_masks=None, labels=None):
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


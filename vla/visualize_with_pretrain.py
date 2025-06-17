import torch
import requests
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig
from transformers import Trainer
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration,PaliGemmaPreTrainedModel
from torch import nn
import torch.nn as nn
import sys
sys.path.append("/media/casia/data2/lpy/3D_VLA/code/RVT")
from rvt_our.mvt.raft_utils import ConvexUpSample
import rvt_our.mvt.utils as mvt_utils
from einops import rearrange, repeat
import json 
import os
import random
from tqdm import tqdm  
from PIL import Image, ImageDraw
import datetime
from safetensors import safe_open
import json
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import ast
from itertools import cycle
'''
这个文件是在之前的基础上进一步添加了目标检测的lvis数据集
'''
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import glob
import math
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import collections


def check_channel_sums(q_trans, tol=1e-6):
    """
    检查每个通道的和是否为1（考虑浮点误差）
    
    参数：
        q_trans : numpy数组，形状 (H, W, C)
        tol : 允许的误差容忍度
    
    返回：
        bool : 是否所有通道的和都≈1
        details : 各通道的实际和
    """
    channel_sums = np.sum(q_trans, axis=(0, 1))  # 计算每个通道的和，形状 (3,)
    is_valid = np.allclose(channel_sums, 1.0, atol=tol)
    return is_valid, channel_sums

def remove_prefix(checkpoint):
    # 创建新字典
    new_checkpoint = collections.OrderedDict()

    # 键名替换逻辑
    for old_key in checkpoint:
        # 按优先级处理前缀
        if old_key.startswith("mvt1.model."):
            new_key = old_key[len("mvt1.model."):]
        elif old_key.startswith("mvt1."):
            new_key = old_key[len("mvt1."):]
        else:
            new_key = old_key  # 保留非目标前缀的键
        
        # 添加新键值对
        new_checkpoint[new_key] = checkpoint[old_key]
    return new_checkpoint

def channel_softmax(x):
    """
    对每个通道单独进行softmax运算
    参数：
        x : numpy数组，形状为(H, W, C)
    返回：
        softmax后的数组，每个通道的和为1
    """
    result = np.zeros_like(x)
    for c in range(x.shape[2]):
        # 获取当前通道数据
        channel = x[:, :, c]
        # 数值稳定性处理（减去最大值）
        exp_channel = np.exp(channel - np.max(channel))
        # 计算softmax
        result[:, :, c] = exp_channel / np.sum(exp_channel)
    return result

def visualize_comparison(images, heatmaps, save_path):
    """
    可视化对比函数
    参数：
        images : 包含3个PIL.Image对象的列表，每个图像大小为616x616
        heatmaps: 224,224,3的numpy对象，三个通道分别对应三张输入图像的heatmap
        save_path: 存储路径
    """
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 转换图像为numpy数组
    img_arrays = [np.array(img) for img in images]
    
    # 计算缩放比例
    scale_h = img_arrays[0].shape[0] / heatmaps.shape[0]  # 616/224
    scale_w = img_arrays[0].shape[1] / heatmaps.shape[1]  # 616/224
    
    # 为每个图像找到heatmap最大值位置并保存单独的图像
    for i in range(3):
        # 获取当前图像的heatmap
        heatmap = heatmaps[:, :, i]
        max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # 将heatmap坐标缩放到原始图像大小
        scaled_x = int(max_pos[1] * scale_w)
        scaled_y = int(max_pos[0] * scale_h)
        
        # 创建新的图像
        plt.figure(figsize=(8, 8))
        plt.imshow(img_arrays[i])
        plt.scatter(scaled_x, scaled_y, 
                   c='lime', s=40, edgecolors='black',
                   linewidths=1, zorder=2)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, f'rgb_{i+1}_pretrain.png'), 
                   bbox_inches='tight', pad_inches=0)
        plt.close()
    
    # 创建2x3的对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # 第一行：预训练模型的预测结果
    for i in range(3):
        ax = axes[0, i]
        img = plt.imread(os.path.join(save_path, f'rgb_{i+1}_pretrain.png'))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Pretrain Prediction {i+1}')
    
    # 第二行：原始标注结果
    for i in range(3):
        ax = axes[1, i]
        img = plt.imread(os.path.join(save_path, f'rgb_{i+1}_annotated.png'))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Original Annotation {i+1}')
    
    # 保存对比图
    plt.savefig(os.path.join(save_path, 'comparison.png'), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()




def visualize_pali_gemma_results(pixel_values, heatmaps, save_path="/opt/tiger/3D_OpenVLA/3d_policy/RoboPoint/robopoint/train/debug.png",num_samples=5):
    """
    可视化 PaLI-Gemma 预处理结果与热力图
    参数：
        pixel_values : Tensor (num_img, 3, 224, 224) - 预处理后的图像张量
        heatmaps     : Tensor (num_img, 224, 224)    - 热力图
        num_samples  : int - 最多显示样本数（默认5）
    """
    # 设备处理与数据转换
    device = pixel_values.device
    pixel_values = pixel_values.detach().cpu().numpy()
    heatmaps = heatmaps.detach().cpu().numpy()
    
    # 限制显示样本数量
    num_samples = min(num_samples, pixel_values.shape[0])
    
    # 反归一化参数（假设使用标准ImageNet参数）
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,3,1,1)
    
    # 反归一化并转换通道顺序
    denorm_images = np.clip((pixel_values * std + mean) * 255, 0, 255)
    denorm_images = denorm_images.transpose(0,2,3,1).astype(np.uint8) # (N,224,224,3)
    
    # 创建画布
    fig = plt.figure(figsize=(2*num_samples, 6))
    
    for idx in range(num_samples):
        img = denorm_images[idx]
        heatmap = heatmaps[idx]
        
        # 原始图像 --------------------------------------------------
        ax = fig.add_subplot(3, num_samples, idx+1)
        ax.imshow(img)
        ax.set_title(f'Sample {idx+1}\nOriginal', fontsize=8)
        ax.axis('off')
        
        # 热力图 ---------------------------------------------------
        ax = fig.add_subplot(3, num_samples, num_samples + idx+1)
        heatmap_display = ax.imshow(heatmap, cmap='viridis')
        plt.colorbar(heatmap_display, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Heatmap', fontsize=8)
        ax.axis('off')
        
        # 叠加显示 --------------------------------------------------
        ax = fig.add_subplot(3, num_samples, 2*num_samples + idx+1)
        ax.imshow(img)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        ax.set_title('Overlay', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)


def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    对每个样本的非零区域进行独立softmax计算
    
    参数：
    heatmap : Tensor - 形状为 (bs, H, W) 的浮点型张量
    eps     : float - 数值稳定系数（默认1e-8）
    
    返回：
    soft_heatmap : Tensor - 处理后的heatmap，非零区域和为1
    """
    # 创建非零掩码
    mask = (heatmap != 0).float()
    
    # 数值稳定处理：移除无效零值的影响
    stable_input = heatmap * mask  # 确保零值不参与计算
    
    # 计算指数值并屏蔽非关注区域
    exp_vals = torch.exp(stable_input) * mask
    
    # 计算各样本的归一化分母
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)  # 形状 (bs, 1, 1)
    
    # 执行归一化（含数值保护）
    soft_heatmap = exp_vals / (sum_exp + eps)
    
    return soft_heatmap




def masked_mean(tensor):
    """
    对形状为 (bs, W, H) 的张量做逐位置加权平均
    每个位置的权重是该位置非零元素的数量
    输入值范围需在 [0, 1]
    """
    # 计算非零元素的掩码
    mask = (tensor != 0).float()
    
    # 计算每个位置的非零数量（分母）
    count = mask.sum(dim=0, keepdim=True)
    
    # 防止除零错误：将零分母的位置设为1
    count = torch.where(count == 0, torch.ones_like(count), count)
    
    # 计算加权平均
    summed = tensor.sum(dim=0, keepdim=True) / count

    return summed


def denormalize_points(points, width, height):
    """将归一化坐标转换为实际像素坐标"""
    return [(x * width, y * height) for (x, y) in points]

def normalize_points(points, new_width, new_height):
    """将实际像素坐标转换回归一化坐标"""
    return [(x / new_width, y / new_height) for (x, y) in points]




class RoboPoint_Paligemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config,layer=-1):
        super().__init__(config)
        # 直接使用继承的模型主体
        self.vlm_dim=config.hidden_size
        self.up0 = ConvexUpSample(
                in_dim=config.hidden_size,
                out_dim=1,
                up_ratio=14,  # hardcode  img_patch_size
            )  
        self.num_pat_img = 16 # hardcode  224/14
        self.processor = AutoProcessor.from_pretrained("/home/guangyun/Project/FiveAges/3d_vla_code/huggingface_ckpt/paligemma-3b-pt-224/")
        # self.custom_layer = nn.Linear(config.hidden_size, config.hidden_size)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.layer_index=layer
        print(f"You are using the {self.layer_index} layer to pretrain the model.")
    
    def forward(self, input_ids, pixel_values,attention_mask,raw_label,flag):
        '''
        input_ids: bs,seq_len   
        pixel_values: bs*num_img,3,224,224
        attention_mask: bs,seq_len
        raw_label: [[],[]...]  字符格式的点
        flag: [flag1,flag2,...]  每个样本的flag，究竟来自robopoint数据还是detection数据
        '''
        # huggingface会自动管理设备
        # device=self.device
        # input_ids = input_ids.to(device)
        # pixel_values = pixel_values.to(device)
        # attention_mask = attention_mask.to(device)
        # device=input_ids.device

        # 自定义前向逻辑
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states 

        x = hidden_states[self.layer_index] #hardcode ，使用倒数第二层的特征进行预训练，然后在RLBench上微调时使用倒数第四层的特征。
        # 确保所有中间张量参与计算
        dummy_loss = 0.0
        dummy_loss+=(torch.sum(hidden_states[-1])-torch.sum(hidden_states[-1]))*0.0
        
        
        # # 批量处理非零token (假设attention_mask已正确标记)
        # non_zero_mask = (attention_mask != 0)
        # image_tokens = x[non_zero_mask][:, :256]  # 每个样本取前256个token

        image_tokens= []
        bs_num_img,img_feat_dim, h, w = pixel_values.shape #这里的num_img 确认好
        bs=input_ids.shape[0]
        num_img=bs_num_img//bs
        assert h==w
        assert num_img==3
        raw_label=ast.literal_eval(raw_label)
        # 对每个批次进行处理  Todo Vectorize
        for i in range(bs):
            # 获取当前批次的ids和output
            current_ids =attention_mask[i]
            current_output = x[i]
            
            # 提取非零id对应的token
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]  # 找到非零id的索引
            non_zero_output = current_output[non_zero_indices]  # 提取这些非零id对应的token输出
            
            # 取出前256个token（如果非零token的数量大于256，则截取前256个）
            assert non_zero_output.shape[0] > 256*num_img
            non_zero_output = non_zero_output[:256*num_img]
            
            # 将处理后的output加入新output列表
            image_tokens.append(non_zero_output)

        # 将新output合并为一个张量
        image_tokens = torch.stack(image_tokens) # bs,256,vlm_dim
        # x: bs,vlm_dim,num_img,h,w
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=num_img, h1=self.num_pat_img, h2=self.num_pat_img)  # 能否直接这样reshape？
        # feat = []
        # _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]# bs,vlm_dim,1
        # _feat = _feat.view(bs, -1)
        # feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32) 
        
        trans = self.up0(x) # bs*num_img,1,224,224
        # trans=torch.zeros_like(trans)   
        trans = trans.view(bs, num_img, h, w) # bs,num_img,224,224
        assert h==w
        q_trans=trans.view(bs,num_img,h*w).transpose(1,2)# bs,50176,num_img
        # get action_trans
        action_trans=[]
        for i in range(bs):
            flag_now=flag[i]
            raw_label_now=raw_label[i]
            if flag_now=="points":
                assert False 
                answer_points=ast.literal_eval(raw_label_now)

                assert type(answer_points[0]) is tuple and len(answer_points[0])==2
                labels=torch.tensor([ [answer_point[0],answer_point[1]] for answer_point in answer_points ])
                action_trans_all = mvt_utils.generate_hm_from_pt(
                                        labels.reshape(-1, 2) * h ,# hardcode,
                                        (224, 224),
                                        sigma=2, # hardcode
                                        thres_sigma_times=3,  # hardcode
                                        )  # check it carefully    （bs,h,w)
                # fuse the action_trans
                summed = action_trans_all.sum(dim=0, keepdim=True)  # 不再除以重叠的数量  
                action_trans_now = masked_softmax(summed) # 只对本来的非零值进行softmax
                action_trans.append(action_trans_now)
            elif flag_now=="detection_1":
                answer_points=np.array(raw_label_now)
                assert answer_points.shape==(num_img,1,2)
                labels=torch.tensor(answer_points)
                action_trans_now = mvt_utils.generate_hm_from_pt(
                labels.reshape(-1, 2) * h,
                (w, h),
                sigma=2, # hardcode
                thres_sigma_times=3,  # hardcode
                )  # check it carefully   
                action_trans_now = action_trans_now.view(num_img, h, w)
                action_trans.append(action_trans_now)                       
            elif flag_now=="detection_2":
                answer_points=raw_label_now
                labels=[[ [answer_point[0],answer_point[1]] for answer_point in answer_points_single ] for answer_points_single in answer_points] # (num_img,_, 2)
                # 由于经过平移或者旋转之后，有的图像的ground truth的点会减少，因此无法直接构成张量来处理，需要单独处理
                action_trans_all=[]
                for i in range(num_img):
                    labels_now=torch.tensor(labels[i])# 单张图像上的所有ground truth点
                    action_trans_all_now = mvt_utils.generate_hm_from_pt(
                                            labels_now.reshape(-1, 2)*h,
                                            (w, h),
                                            sigma=2, # hardcode
                                            thres_sigma_times=3,  # hardcode
                                            )  # check it carefully  
                    action_trans_all_now=action_trans_all_now.view(len(labels_now), h , w)  


                    action_trans_now=masked_mean(action_trans_all_now) 
                    action_trans_now=masked_softmax(action_trans_now)  # (1,w,h)
                    action_trans_all.append(action_trans_now.squeeze(0)) 
                action_trans_all=torch.stack(action_trans_all)
                action_trans.append(action_trans_all)   
            else:
                assert False

        action_trans=torch.stack(action_trans) #()
        action_trans = action_trans.view(bs,num_img, h * w).transpose(1, 2).clone().to(q_trans.device)
        # visualize
        # 将pixel values还原进行可视化  作为第一排  将heatmap作为第二排
        # index=0
        # vis_pixel_values=pixel_values.view(bs,num_img,img_feat_dim,h,w)[index] # num_img,3,224,224
        # vis_heatmap=action_trans[index].transpose(0,1).view(num_img,h,w) # num_img,224,224
        # visualize_pali_gemma_results(vis_pixel_values, vis_heatmap, save_path="/opt/tiger/3D_OpenVLA/3d_policy/RoboPoint/robopoint/train/debug.png",num_samples=5)



        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()

        # print("dummy_loss: ",dummy_loss)
        return {"loss": trans_loss + dummy_loss}  # 合并损失
        # return {"loss": trans_loss}


    def forward_eval(self, input_ids, pixel_values,attention_mask):
        '''
        input_ids: bs,seq_len   
        pixel_values: bs*num_img,3,224,224
        attention_mask: bs,seq_len
        '''
        # huggingface会自动管理设备
        device=self.device
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        attention_mask = attention_mask.to(device)
        device=input_ids.device

        # 自定义前向逻辑
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states 

        x = hidden_states[self.layer_index] #hardcode ，使用倒数第四层的特征
        # 确保所有中间张量参与计算
        dummy_loss = 0.0
        dummy_loss+=(torch.sum(hidden_states[-1])-torch.sum(hidden_states[-1]))*0.0   
        
        # # 批量处理非零token (假设attention_mask已正确标记)
        # non_zero_mask = (attention_mask != 0)
        # image_tokens = x[non_zero_mask][:, :256]  # 每个样本取前256个token

        image_tokens= []
        bs_num_img,img_feat_dim, h, w = pixel_values.shape  #这里的num_img 确认好
        bs=input_ids.shape[0]
        num_img=bs_num_img//bs
        assert h==w
        assert num_img==1
        # 对每个批次进行处理  Todo Vectorize
        for i in range(bs):
            # 获取当前批次的ids和output
            current_ids =attention_mask[i]
            current_output = x[i]
            
            # 提取非零id对应的token
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]  # 找到非零id的索引
            non_zero_output = current_output[non_zero_indices]  # 提取这些非零id对应的token输出
            
            # 取出前256个token（如果非零token的数量大于256，则截取前256个）
            assert non_zero_output.shape[0] > 256*num_img
            non_zero_output = non_zero_output[:256*num_img]
            
            # 将处理后的output加入新output列表
            image_tokens.append(non_zero_output)

        # 将新output合并为一个张量
        image_tokens = torch.stack(image_tokens) # bs,256,vlm_dim
        # x: bs,vlm_dim,num_img,h,w
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=num_img, h1=self.num_pat_img, h2=self.num_pat_img)  # 能否直接这样reshape？
        # feat = []
        # _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]# bs,vlm_dim,1
        # _feat = _feat.view(bs, -1)
        # feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32) 
        
        trans = self.up0(x) # bs*num_img,1,224,224
        # trans=torch.zeros_like(trans)   
        trans = trans.view(bs, num_img, h, w) # bs,num_img,224,224
        assert h==w
        q_trans=trans.view(bs,num_img,h*w).transpose(1,2)# bs,50176,num_img
        return {"q_trans":q_trans}  # 合并损失
        # return {"loss": trans_loss}




def load_all_params(checkpoint_dir):
    """
    安全地加载模型参数
    """
    try:
        # 首先尝试使用 safetensors 加载
        with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
            index = json.load(f)
        
        all_params = {}
        for shard_file in set(index["weight_map"].values()):
            with safe_open(f"{checkpoint_dir}/{shard_file}", framework="pt") as f:
                for key in f.keys():
                    clean_key = key.replace("module.", "")
                    all_params[clean_key] = f.get_tensor(key)
        return all_params
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        # 如果 safetensors 加载失败，尝试加载 .pth 文件
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    checkpoint = checkpoint["model_state"]
                return checkpoint
            except Exception as e:
                print(f"Error loading pth file: {e}")
                raise
        else:
            raise FileNotFoundError(f"No model file found at {checkpoint_path}")

class Pretrain_RoboPoint_Palligemma:
    def __init__(self,pretrain,debug=False):
        self.model_id = "/home/guangyun/Project/FiveAges/3d_vla_code/huggingface_ckpt/paligemma-3b-pt-224/"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if not pretrain:
            self.device="cuda"
            checkpoint_dir="/media/casia/data2/lpy/3D_VLA/code/checkpoints/pretrain/one_image_layer1_pretrain_3824"
            
            # 加载模型参数
            all_params = load_all_params(checkpoint_dir)
            all_params = remove_prefix(all_params)
            
            # 初始化模型
            self.pretrained_model = RoboPoint_Paligemma.from_pretrained(self.model_id, trust_remote_code=True)
            
            # 加载参数
            missing_keys, unexpected_keys = self.pretrained_model.load_state_dict(all_params, strict=False)
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)
            
            # 清理内存
            del all_params
            
            # 将模型移到GPU
            self.pretrained_model.to(self.device)

    def test_inference(self,image_folder,text):
        save_path=image_folder
        texts=[text]
        images=[]
        for i in range(1,4):
            image_path=os.path.join(image_folder,f"rgb_{i}.png")
            image=Image.open(image_path).convert("RGB")
            images.append(image)
        q_transes=[]
        for image in images:
            tokens = self.processor(text=texts, images=[image],return_tensors="pt", padding="longest")
            tokens=tokens.to(self.device)
            self.pretrained_model.eval()
            output_dict=self.pretrained_model.forward_eval(**tokens)

            q_trans=output_dict["q_trans"].view(224,224,1).detach().cpu().numpy()
            q_trans=channel_softmax(q_trans)
            is_valid_q,sums=check_channel_sums(q_trans, tol=1e-9)
            is_valid_action,sums=check_channel_sums(q_trans, tol=1e-9)
            assert is_valid_q 
            assert is_valid_action
            q_transes.append(q_trans)
        # Stack the q_transes list into a single array with shape (224, 224, 3)
        q_transes = np.concatenate([q.reshape(224, 224, 1) for q in q_transes], axis=2)
        # visualize
        visualize_comparison(images, q_transes,save_path)
            



if __name__=="__main__":

    image_folder="/media/casia/data2/lpy/3D_VLA/code/RVT/rvt_our/debug_visualize_with_pretrain/57/stage1"
    text="find all instances of bottle"
    text="place the bottle in the green plate"
    pipeline=Pretrain_RoboPoint_Palligemma(pretrain=False,debug=True)
    pipeline.test_inference(image_folder=image_folder,text=text)



import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from einops import rearrange
from amplify.utils.cfg_utils import get_device

class VisionEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, frozen=False, lr_multiplier=1.0, get_patches=False, get_cls_and_patches=False, use_depth=False, patch_pooling=None, img_size=224):
        super(VisionEncoder, self).__init__()
        self.preprocess = None
        self.model_name = model_name
        self.get_patches = get_patches
        self.get_cls_and_patches = get_cls_and_patches
        self.use_depth = use_depth
        self.patch_pooling = patch_pooling
        self.img_size = img_size
        self.resize = T.Resize((self.img_size, self.img_size), antialias=False)
        self.device = get_device()

        if "clip" in model_name:
            try:
                import clip
            except ImportError:
                raise ImportError(
                    "CLIP is required. Install with:\n"
                    "pip install git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
                )

        if "sam" in model_name:
            try:
                from segment_anything import (
                    SamAutomaticMaskGenerator,
                    SamPredictor,
                    build_sam,
                    build_sam_vit_b,
                    build_sam_vit_h,
                    build_sam_vit_l,
                )
            except ImportError:
                raise ImportError(
                    "Segment Anything is required. Install with:\n"
                    "pip install git+https://github.com/facebookresearch/segment-anything.git@6fdee8f2727f4506cfbbe553e23b895e27956588"
                )
        

        # ViT
        if model_name == "vit-tiny":
            self.model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 192
            self.patch_size = 16

        elif model_name == "vit-small":
            self.model = timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 384
            self.patch_size = 16

        elif model_name == "vit-base":
            self.model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 768
            self.patch_size = 16

        elif model_name == "vit-mae":
            self.model = timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained, num_classes=0)
            self.embed_dim = 768
            self.patch_size = 16

        elif model_name == "vit-large":
            self.model = timm.create_model("vit_large_patch16_224", pretrained=pretrained, num_classes=0)
            self.embed_dim = 1024
            self.patch_size = 16

        # ResNet
        elif model_name == "resnet18":
            if self.get_patches:
                # get the feature map and flatten
                model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
                self.model = nn.Sequential(*list(model.children())[:-2])
                # print("resnet18 model: ", self.model)
                self.embed_dim = 512
                self.patch_size = 32
            else:
                self.model = timm.create_model("resnet18", pretrained=pretrained, num_classes=0)
                self.embed_dim = 512
        elif model_name == "resnet50":
            if self.get_patches:
                # get the feature map and flatten
                model = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
                self.model = nn.Sequential(*list(model.children())[:-2])
                # print("resnet18 model: ", self.model)
                self.embed_dim = 2048
                self.patch_size = 32
            else:
                self.model = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
                self.embed_dim = 2048
        elif model_name == "resnet101":
            self.model = timm.create_model("resnet101", pretrained=pretrained, num_classes=0)
            self.embed_dim = 2048
        elif model_name == "resnet152":
            self.model = timm.create_model("resnet152", pretrained=pretrained, num_classes=0)
            self.embed_dim = 2048

        # CLIP
        elif model_name == "clip-base":
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.embed_dim = 768
            self.patch_size = 32
        elif model_name == "clip-large":
            self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
            self.embed_dim = 1024
            self.patch_size = 14
        elif model_name == "clip-resnet50":
            self.model, self.preprocess= clip.load("RN50", device=device)
            self.embed_dim = 512

        # DINOv2
        elif model_name == "dinov2-small":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.embed_dim = 384
            self.patch_size = 14
        elif model_name == "dinov2-base":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.embed_dim = 768
            self.patch_size = 14
        elif model_name == "dinov2-large":
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.embed_dim = 1024
            self.patch_size = 14

        # Voltron
        elif model_name == "voltron-base":
            from voltron import instantiate_extractor, load
            self.model, self.preprocess = load("v-cond-base", device=self.device, freeze=True, cache="../voltron-robotics/cache")
            self.embed_dim = 768
            self.vector_extractor = instantiate_extractor(self.model)().to(self.device)

        # VIP
        elif model_name == "vip":
            from vip import load_vip
            self.model = load_vip()
            self.embed_dim = 2048

        # SAM
        elif model_name == "sam-base":
            sam_path = "sam_checkpoints/sam_vit_b_01ec64.pth" # Change this to your own sam checkpoint path
            self.model = build_sam_vit_b(checkpoint=sam_path)
            self.predictor = SamPredictor(self.model)
            # self.preprocess = self.model.preprocess
            # self.preprocess = self.predictor.set_image
            self.embed_dim = 768

        else:
            raise ValueError(f"Model name {model_name} not recognized.")

        # self.patch_size = self.model.patch_size if hasattr(self.model, "patch_size") else None
        if self.get_patches and not self.patch_pooling:
            self.seq_len = (self.img_size // self.patch_size) ** 2
        elif self.get_cls_and_patches:
            self.seq_len = (self.img_size // self.patch_size) ** 2 + 1
        else:
            self.seq_len = 1

        print("venc seq_len: ", self.seq_len)

        if self.preprocess is None:
            self.preprocess = self.preprocess_np_or_torch

        # applying lr multiplier
        if lr_multiplier != 1.0:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.parameters():
                param.requires_grad = True
                param.data *= lr_multiplier
            print("MULTIPLYING VISION ENCODER LR BY ", lr_multiplier)

        if frozen:
            print("FREEZING VISION ENCODER")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            print("NOT FREEZING VISION ENCODER")

        print("Vision encoder trainable parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        if use_depth:
            print("modifying vision encoder for depth input")
            # Modify the patch embedding layer to accept 4 channels (RGBD)
            original_patch_embedding = self.model.patch_embed.proj # (768, 3, patch_size, patch_size)
            print("original_patch_embedding shape: ", original_patch_embedding.weight.shape)
            self.model.patch_embed.proj = nn.Conv2d(4, self.model.embed_dim, kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size), bias=False) # (768, 4, 16, 16). Conv with stride=16 is equivalent to having a linear transformation for each patch

            # Initialize the depth channel weights by averaging the weights from the RGB channels
            with torch.no_grad():
                self.model.patch_embed.proj.weight[:, :3] = original_patch_embedding.weight.clone()
                self.model.patch_embed.proj.weight[:, 3] = original_patch_embedding.weight.mean(dim=1) # Average the weights from the RGB channels
            print("modified_patch_embedding shape: ", self.model.patch_embed.proj.weight.shape)

            # new patch_embed layer should never be frozen, even if the rest of the model is
            for param in self.model.patch_embed.parameters():
                param.requires_grad = True


    def preprocess_np_or_torch(self, images):
        '''
        Preprocesses numpy or torch images, channel first or channel last. uint8 with values between 0 and 255
        returns: torch tensor with shape (3, self.img_size, self.img_size) and floats between 0 and 1
        '''
        if images.shape[-1] <= 4: # if (H, W, C)
            images = images.permute(2, 0, 1) # (C, H, W)

        images = T.Resize((self.img_size, self.img_size), antialias=False)(images)

        return images


    def preprocess_batch(self, images):
        '''
        Input: b h w c
        Output: b c 224 224
        '''
        images = images.permute(0, 3, 1, 2) # (b, c, h, w)
        images = self.resize(images)
        return images


    def forward(self, images):
        """
        images should be tensor with shape (batch_size, height, width, channels) and values between 0 and 1 (float32)
        """
        if images.ndim == 3:
            images = images[None, ...] # add batch dim

        if "voltron" in self.model_name:
            # convert to tensor using torchvision
            # images = self.preprocess(torch.tensor(images).permute(0, 3, 1, 2).to(self.device))
            images = self.preprocess(torch.tensor(images).to(images.device)) # NOTE: needs to be channel first

        elif "clip" in self.model_name:
            # Convert tensor to uint8 numpy array for PIL
            # Input tensor is in range [0, 1], PIL expects [0, 255] uint8
            images_np = (images.cpu().detach().numpy() * 255).astype(np.uint8)
            images = [self.preprocess(PIL.Image.fromarray(image)) for image in images_np]
            images = torch.stack(images).to(self.device)
            return self.model.encode_image(images)
        elif "sam" in self.model_name:
            embs = []
            for image in images:
                self.predictor.set_image(image)
                patch_embs = self.predictor.get_image_embedding()

                if self.cfg is not None and self.cfg.v_enc_patch:
                    embs.append(patch_embs)
                else:
                    avgpooled_embs = patch_embs.mean(dim=(-1, -2))
                    embs.append(avgpooled_embs)

            return torch.stack(embs).squeeze(0)
        else:
            images = self.preprocess_batch(images).to(images.device)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        assert images.ndim == 4, f"images should have 4 dimensions, but has {images.ndim} dimensions"
        assert images.dtype == torch.float32, f"images should have dtype float32, but has dtype {images.dtype}"
        assert images.min() >= 0.0 and images.max() <= 1.01 and images.max() > 1./255., f"images should have values between 1e-2 and 1, but has values between {images.min()} and {images.max()}"
        assert images.shape[1] <= 4, f"images should have 4 channels or less, but has {images.shape[1]} channels. Be sure shape is (batch_size, channels, height, width)"

        if "clip" in self.model_name:
            return self.model.encode_image(images)
        elif "voltron" in self.model_name:
            dense_emb =  self.model(images, mode="visual")
            return self.vector_extractor(dense_emb)
        else: # ViT, ResNet, DINOv2
            if self.get_patches:
                if "dinov2" in self.model_name:
                    embs = self.model.forward_features(images)['x_norm_patchtokens']
                elif "resnet" in self.model_name:
                    embs = self.model(images)
                    embs = rearrange(embs, 'b c h w -> b (h w) c')
                else:
                    embs = self.model.forward_features(images)[:, 1:]
                if self.patch_pooling == "avg":
                    embs = embs.mean(dim=1)
            elif self.get_cls_and_patches:
                assert self.patch_pooling is None, "patch_pooling not implemented for get_cls_and_patches"
                if "dinov2" in self.model_name:
                    cls_emb = self.model.forward_features(images)['x_norm_clstoken']
                    patch_embs = self.model.forward_features(images)['x_norm_patchtokens']
                    embs = torch.cat([cls_emb, patch_embs], dim=1)
                else:
                    embs = self.model.forward_features(images)
            else:
                embs = self.model(images)
                if 'resnet' in self.model_name:
                    embs = embs.unsqueeze(1)
        return embs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vit-base", help="Name of the model to use.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 224
    batch_size = 4

    encoder = VisionEncoder(args.model_name, device, get_patches=False, img_size=img_size).to(device)
    print("model: ", encoder.model)
    print("preprocess: ", encoder.preprocess)

    image = np.ones((batch_size, img_size, img_size, 3), dtype=np.float32) # (b h w c)
    image = torch.from_numpy(image).to(device)

    emb = encoder(image)
    print("embedding shape: ", emb.shape)

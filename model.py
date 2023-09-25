import logging
import random
import os
import torch
import torch.nn as nn
import numpy as np

from .clip import load, tokenize
import torchvision
import open_clip
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertForSequenceClassification

from training.distributed import is_master
from training.projection import DINOHead
import training.transforms

from loss import NEED_LOGIT_SCALE, NEED_PROTOTYPE_LAYER
from contextlib import suppress

AVALIABLE_TEXT_MODEL_BUILDER = ['openclip', 'chineseclip', 'huggingface', 'sbert']
AVALIABLE_IMAGE_MODEL_BUILDER = ['openclip', 'chineseclip', 'torchvision', "torchhub"]


def mean_pooling(hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
    return torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CLIP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model, self.preprocess = load(args.text_model, device="cpu", jit=False)
        self.device = args.device
        self.model.to(self.device)
        self._freeze_parameters(args)

    def _freeze_parameters(self, args):
        # Set 'param.required_grad' to implement partial finetune
        for name, param in self.model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False if args.lock_text_model else True
                if args.lock_text_partial != '':
                    for keyword in args.lock_text_partial.split(','):
                        if keyword.replace('!', '') in name:
                            if '!' in keyword:
                                param.requires_grad = True
                                if args.lock_text_model:
                                    break
                            else:
                                param.requires_grad = False
                                if not args.lock_text_model:
                                    break

        for name, param in self.model.visual.named_parameters():
            param.requires_grad = False if args.lock_image_model else True
            if args.lock_image_partial != '':
                for keyword in args.lock_image_partial.split(','):
                    if keyword.replace('!', '') in name:
                        if '!' in keyword:
                            param.requires_grad = True
                            if args.lock_image_model:
                                break
                        else:
                            param.requires_grad = False
                            if not args.lock_image_model:
                                break

    def reinit_logit_scale(self, logit_scale):
        self.model.logit_scale = nn.Parameter(torch.ones(1) * np.log(1 / logit_scale))#.to(self.device)
        #self.logit_scale.to(self.device)
        self.model.to(self.device)

    def encode_image(self, images, projection=False):
        return self.model.encode_image(images)

    def encode_text(self, texts, projection=False, use_pooler=True):
        texts = tokenize(texts, context_length=77, truncate=True).to(self.device)
        return self.model.encode_text(texts)

    def forward(self, images, texts, text_only):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        texts:  torch.tensor (batchs_size, token_indexs)
        """
        text_features = self.encode_text(texts)

        if text_only: # skip image forward for efficient teacher caching 
            image_features = text_features
        else:
            image_features = self.encode_image(images)

        return image_features, text_features, self.model.logit_scale.exp()


def get_model(args):
    logging.info(f'Builing model for rank {args.rank}')

    # === text model === #
    if is_master(args):
        logging.info(f'Loading [{args.text_model}] as text model via [{args.text_model_builder}]. Pretrained={args.pretrained_text_model}')
    
#     if args.text_model_builder=='openclip':
#         CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#             model_name=args.text_model,
#             pretrained=args.text_model_tag if args.pretrained_text_model else '',
#             precision=args.precision,
#             device=args.device,
#             jit=args.torchscript,
#             force_quick_gelu=args.force_quick_gelu,
#             cache_dir=os.path.join(args.cache_dir, 'open_clip')
#         )
#         CLIP_model.visual = None
#         text_backbone = CLIP_model
#         tokenizer = open_clip.tokenize
#         args.text_width, args.text_dim = text_backbone.text_projection.size()
#         text_backbone.layers = open_clip.get_model_config(args.text_model)['text_cfg']['layers']
                    
#         if args.adapter is not None:
#             raise RuntimeError(f'Adapter {args.adapter} is not avaliable for {args.text_model_builder} models!')
    
#     # === image model === #
#     if is_master(args):
#         logging.info(f'Loading [{args.image_model}] as image model via [{args.image_model_builder}]. Pretrained={args.pretrained_image_model}')
    
#     if args.image_model_builder == 'openclip':
#         CLIP_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#             model_name=args.image_model,
#             pretrained=args.image_model_tag if args.pretrained_image_model else '',
#             precision=args.precision,
#             device=args.device,
#             jit=args.torchscript,
#             force_quick_gelu=args.force_quick_gelu,
#             cache_dir=os.path.join(args.cache_dir, 'open_clip')
#         )
#         image_backbone = CLIP_model.visual
#         args.image_dim = image_backbone.output_dim
#         image_backbone.layers = open_clip.get_model_config(args.image_model)['vision_cfg']['layers']
#         if type(image_backbone.layers) == list:
#             image_backbone.layers = len(image_backbone.layers)
#         if 'RN' in args.image_model:
#             image_backbone.arch = 'ResNet'
#             image_backbone.layers += 2 # stem and attention pooling accont for two layers
#         elif 'ViT' in args.image_model:
#             image_backbone.arch = 'ViT'
#         else:
#             raise RuntimeError(f'Unrecognized image backbone architechture')

#     # Set 'param.required_grad' to implement partial finetune
#     for name, param in text_backbone.named_parameters():
#         param.requires_grad = False if args.lock_text_model else True
#         if args.lock_text_partial != '':
#             for keyword in args.lock_text_partial.split(','):
#                 if keyword.replace('!', '') in name:
#                     if '!' in keyword:
#                         param.requires_grad = True
#                         if args.lock_text_model:
#                             break
#                     else:
#                         param.requires_grad = False
#                         if not args.lock_text_model:
#                             break
                    
#     for name, param in image_backbone.named_parameters():
#         param.requires_grad = False if args.lock_image_model else True
#         if args.lock_image_partial != '':
#             for keyword in args.lock_image_partial.split(','):
#                 if keyword.replace('!', '') in name:
#                     if '!' in keyword:
#                         param.requires_grad = True
#                         if args.lock_image_model:
#                             break
#                     else:
#                         param.requires_grad = False
#                         if not args.lock_image_model:
#                             break

#     model = ItraModel(
#         text_backbone=text_backbone, 
#         image_backbone=image_backbone, 
#         tokenizer=tokenizer, 
#         args=args
#         )
    model = CLIP(args)
    preprocess_train = preprocess_val = preprocess_val = model.preprocess
    return model, preprocess_train, preprocess_val, preprocess_val



class ItraModel(nn.Module):
    def __init__(self, text_backbone, image_backbone, tokenizer, args) -> None:
        super().__init__()
        self.device = args.device
        self.text_model = args.text_model
    
    # text backbone
        self.text_backbone = text_backbone
        self.text_pooler = args.text_pooler
        if self.text_pooler!= 'cls':
            self.text_backbone.pooler = nn.Identity()
        self.text_dim = args.text_dim
        self.text_width = args.text_dim
        self.tokenizer = tokenizer        
        self.text_model_builder = args.text_model_builder
        self.image_model_builder = args.image_model_builder
        self.max_seq_length = args.max_seq_length
            
        self.image_context = torch.no_grad if (
            args.lock_image_model and 
            '!' not in args.lock_image_partial
            ) else suppress 
            
        self.text_context = torch.no_grad if (
            args.lock_text_model and 
            '!' not in args.lock_text_partial and 
            args.adapter is None and
            not args.prompt
            ) else suppress
        
        if is_master(args):
            logging.info(f'Calculate gradients for image backbone?\t{self.image_context==suppress}')
            logging.info(f'Calculate gradients for text backbone?\t{self.text_context==suppress}')
        
        # TODO: CoOp text prompt
        if args.prompt:
            assert args.text_model_builder=='openclip' # CoOp style prompt only supports OpenCLIP models
            self.prompt = nn.Parameter(torch.empty(args.n_prompt, args.text_width))
            torch.nn.init.normal_(self.prompt, std=0.02)
            self.n_prompt = args.n_prompt
        else:
            self.prompt = None

    # image backbone
        self.image_backbone = image_backbone
        self.image_dim = image_backbone.output_dim
        self.image_model_tag = args.image_model_tag

    
    # text projection head
        if args.text_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.image_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.image_dim # adaption layer
            self.text_projection_head = DINOHead(
                in_dim=self.text_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.text_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            
            # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher=='text':
                for param in self.text_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.text_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Text backbone do not append projection head, so set args.joint_projection_dim = self.text_dim')
            args.joint_projection_dim = self.text_dim

    # image projection head
        if args.image_head_n_layers > 0 or args.loss in NEED_PROTOTYPE_LAYER:
            if args.text_head_n_layers==0 and args.joint_projection_dim<0:
                args.joint_projection_dim = self.text_dim # adaption layer
            self.image_projection_head = DINOHead(
                in_dim=self.image_dim, out_dim=65536, bottleneck_dim=args.joint_projection_dim,
                nlayers=args.image_head_n_layers, skip_last_layer=args.loss not in NEED_PROTOTYPE_LAYER
                ).to(args.device)
            # FIXME? # DINO & ProtoCPC copy student's learnable prototype to teacher, so teacher's prototype should not be optimized
            if args.loss in NEED_PROTOTYPE_LAYER and args.teacher=='image':
                for param in self.image_projection_head.parameters():
                    param.requires_grad = False
        else:
            self.image_projection_head = nn.Identity()
            if is_master(args):
                logging.info('Image backbone do not append projection head so set args.joint_projection_dim = self.image_dim')
            args.joint_projection_dim = self.image_dim

        if args.loss in NEED_LOGIT_SCALE:
            if hasattr(self.text_backbone, 'logit_scale'):
                self.logit_scale = self.text_backbone.logit_scale 
                self.text_backbone.logit_scale = None
            else:
                self.logit_scale = torch.autograd.Variable(torch.ones(1) * np.log(1 / args.logit_scale)).to(self.device)
            self.logit_scale = nn.Parameter(self.logit_scale)
            self.logit_scale.requires_grad = True
        else:
            self.logit_scale = torch.zeros(1)
        self.to(self.device)

    def reinit_logit_scale(self, logit_scale):
        self.logit_scale = torch.nn.parameter.Parameter(torch.ones(1) * np.log(1 / logit_scale))#.to(self.device)
        #self.logit_scale.to(self.device)
        self.to(self.device)

    def encode_image(self, images, projection=False):
        with self.image_context():
            image_features = self.image_backbone(images)
            if 'vicregl' in self.image_model_tag:
                image_features = image_features[1]
        if projection:
            image_features = self.image_projection_head(image_features)
        return image_features.float()

    # sentence-transformers API
    def encode(self, sentences, batch_size=32, show_progress_bar=None, convert_to_numpy=True, convert_to_tensor=True, use_pooler=False):
        with torch.no_grad():
            def _text_length(text):
                if isinstance(text, dict):              #{key: value} case
                    return len(next(iter(text.values())))
                elif not hasattr(text, '__len__'):      #Object has no len() method
                    return 1
                elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
                    return len(text)
                else:
                    return sum([len(t) for t in text])      #Sum of length of individual strings

            all_embeddings = []
            length_sorted_idx = np.argsort([_text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in range(0, len(sentences), batch_size):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                embeddings = self.encode_text(sentences_batch, projection=True, use_pooler=use_pooler).cpu()
                all_embeddings.extend(embeddings)
            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                all_embeddings = torch.stack(all_embeddings)
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        return all_embeddings
    
    def encode_text(self, texts, projection=False, use_pooler=True):
        with self.text_context():
            if self.text_model_builder in ['openclip']:
                # TODO: support CoOp-style prompting (CoOp for retrieval finetuning?)
                context_length = (77 - self.n_prompt) if self.prompt is not None else 77
                texts = self.tokenizer(texts, context_length=context_length).to(self.device)
                def open_clip_forward(texts):
                    x = self.text_backbone.token_embedding(texts)  # [batch_size, n_ctx, d_model] (bs, 77-args.n_prompts, 512)
                    if self.prompt is not None:
                        batch_prompt = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
                        x = torch.cat([x[:, :1, :], batch_prompt, x[:, 1:, :]], dim=1)
                    x = x + self.text_backbone.positional_embedding
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = self.text_backbone.transformer(x, attn_mask=self.text_backbone.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = self.text_backbone.ln_final(x) # [batch_size, n_ctx, transformer.width]
                    # take features from the eot embedding (eot_token is the highest number in each sequence)
                    x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_backbone.text_projection
                    return x
                text_features = open_clip_forward(texts)

        if projection:
            text_features = self.text_projection_head(text_features)

        return text_features

    def forward(self, images, texts, text_only):
        """
        images: torch.tensor (batchs_size, preprocessed image)
        texts:  torch.tensor (batchs_size, token_indexs)
        """
        text_features = self.encode_text(texts, projection=True)

        if text_only: # skip image forward for efficient teacher caching 
            image_features = text_features
        else:
            image_features = self.encode_image(images, projection=True)

        return image_features, text_features, self.logit_scale.exp()

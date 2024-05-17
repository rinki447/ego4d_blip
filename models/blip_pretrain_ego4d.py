'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
import transformers
transformers.logging.set_verbosity_error()
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.blip_ego import init_tokenizer, load_checkpoint

class BLIP_Ego4d(nn.Module):
    def __init__(self,num_frames,verb_classes,noun_classes,vision_width = 512,med_config = 'configs/bert_config.json'):                
                 
                 
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        self.noun_classes = noun_classes
        self.verb_classes = verb_classes
        self.num_frames = num_frames
               
        self.tokenizer = init_tokenizer()   
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased',config=encoder_config, add_pooling_layer=False)
        self.text_encoder.resize_token_embeddings(len(self.tokenizer)) 

        text_width = self.text_encoder.config.hidden_size
        self.text_width = text_width
        
        # self.vision_proj = nn.Linear(vision_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)

        self.noun_head = nn.Linear(text_width, self.noun_classes) 
        self.verb_head = nn.Linear(text_width, self.verb_classes)
        
        
        
        # create the decoder
        # decoder_config = BertConfig.from_json_file(med_config)
        # decoder_config.encoder_width = vision_width        
        # self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)    
        # self.text_decoder.resize_token_embeddings(len(self.tokenizer)) 
        # tie_encoder_decoder_weights(self.text_encoder,self.text_decoder.bert,'','/attention')
        
        
    def forward(self, caption, noun_labels, verb_labels,device,vid_feature=None):
        
        # image_embeds = self.visual_encoder(image) 
        # image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        # image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)          
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                              return_tensors="pt").to(device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = text_output.last_hidden_state[:,0,:] # bs*num_frames x 768

        
        batch_size = text_feat.shape[0] // self.num_frames
        text_feat = text_feat.view(batch_size,self.num_frames,self.text_width)
        
        ## Temporally mean pooled text feature
        text_feat_pooled = text_feat.mean(1) # bs x 768

        #print(text_feat.shape)
        #print(text_feat_pooled.shape)
        
        #Rinki########################################## check shape
        
        noun_cls_logits = self.noun_head(text_feat_pooled).to(torch.float32)   
        verb_cls_logits = self.verb_head(text_feat_pooled).to(torch.float32)         

        noun_labels=noun_labels.to(torch.int64)
        verb_labels=verb_labels.to(torch.int64)
        
        loss_noun = F.cross_entropy(noun_cls_logits, noun_labels)  
        loss_verb = F.cross_entropy(verb_cls_logits, verb_labels)  

        
        ##================= LM ========================##     
        # decoder_input_ids = text.input_ids.clone()      
        # decoder_input_ids[:,0] = self.tokenizer.bos_token_id
        # decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100) 

        # decoder_output = self.text_decoder(decoder_input_ids, 
        #                                    attention_mask = text.attention_mask, 
        #                                    encoder_hidden_states = image_embeds,
        #                                    encoder_attention_mask = image_atts,                  
        #                                    labels = decoder_targets,
        #                                    return_dict = True,   
        #                                   )   
          
        # loss_lm = decoder_output.loss 
        ##=================================================================##   
        
        #Rinki########################################## make prediction classes from scores of noun and verbs
        noun_probs = torch.softmax(noun_cls_logits, dim=1)
        #predicted_noun_class = torch.argmax(noun_probs, dim=1)
        #predicted_noun_class=predicted_noun_class.numpy()


        verb_probs = torch.softmax(verb_cls_logits, dim=1)
        #predicted_verb_class = torch.argmax(verb_probs, dim=1)
        #predicted_verb_class=predicted_verb_class.numpy()
        #print(predicted_noun_class)
        
        prediction = {'predicted_verb_probab': noun_probs,
                      'predicted_noun_probab': verb_probs}    
        
                   
        return prediction, loss_noun, loss_verb
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

                        
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 


def blip_pretrain_ego4d(pretrained='',**kwargs):
    model = BLIP_Ego4d(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        assert(len(msg.missing_keys)==0)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output     


from typing import List
def tie_encoder_decoder_weights(encoder: nn.Module, decoder: nn.Module, base_model_prefix: str, skip_key:str):
    uninitialized_encoder_weights: List[str] = []
    if decoder.__class__ != encoder.__class__:
        logger.info(
            f"{decoder.__class__} and {encoder.__class__} are not equal. In this case make sure that all encoder weights are correctly initialized."
        )

    def tie_encoder_to_decoder_recursively(
        decoder_pointer: nn.Module,
        encoder_pointer: nn.Module,
        module_name: str,
        uninitialized_encoder_weights: List[str],
        skip_key: str,
        depth=0,
    ):
        assert isinstance(decoder_pointer, nn.Module) and isinstance(
            encoder_pointer, nn.Module
        ), f"{decoder_pointer} and {encoder_pointer} have to be of type torch.nn.Module"
        if hasattr(decoder_pointer, "weight") and skip_key not in module_name:
            assert hasattr(encoder_pointer, "weight")
            encoder_pointer.weight = decoder_pointer.weight
            if hasattr(decoder_pointer, "bias"):
                assert hasattr(encoder_pointer, "bias")
                encoder_pointer.bias = decoder_pointer.bias                
            print(module_name+' is tied')    
            return

        encoder_modules = encoder_pointer._modules
        decoder_modules = decoder_pointer._modules
        if len(decoder_modules) > 0:
            assert (
                len(encoder_modules) > 0
            ), f"Encoder module {encoder_pointer} does not match decoder module {decoder_pointer}"

            all_encoder_weights = set([module_name + "/" + sub_name for sub_name in encoder_modules.keys()])
            encoder_layer_pos = 0
            for name, module in decoder_modules.items():
                if name.isdigit():
                    encoder_name = str(int(name) + encoder_layer_pos)
                    decoder_name = name
                    if not isinstance(decoder_modules[decoder_name], type(encoder_modules[encoder_name])) and len(
                        encoder_modules
                    ) != len(decoder_modules):
                        # this can happen if the name corresponds to the position in a list module list of layers
                        # in this case the decoder has added a cross-attention that the encoder does not have
                        # thus skip this step and subtract one layer pos from encoder
                        encoder_layer_pos -= 1
                        continue
                elif name not in encoder_modules:
                    continue
                elif depth > 500:
                    raise ValueError(
                        "Max depth of recursive function `tie_encoder_to_decoder` reached. It seems that there is a circular dependency between two or more `nn.Modules` of your model."
                    )
                else:
                    decoder_name = encoder_name = name
                tie_encoder_to_decoder_recursively(
                    decoder_modules[decoder_name],
                    encoder_modules[encoder_name],
                    module_name + "/" + name,
                    uninitialized_encoder_weights,
                    skip_key,
                    depth=depth + 1,
                )
                all_encoder_weights.remove(module_name + "/" + encoder_name)

            uninitialized_encoder_weights += list(all_encoder_weights)

    # tie weights recursively
    tie_encoder_to_decoder_recursively(decoder, encoder, base_model_prefix, uninitialized_encoder_weights, skip_key)  

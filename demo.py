from models.blip_pretrain_ego4d import blip_pretrain_ego4d
import torch

'''
def blip_pretrain_ego4d(**kwargs):
    model = BLIP_Ego4d(**kwargs)
    return model '''

num_frames=16
verb_classes=117
noun_classes=483
caption=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","tweleve","thirteen","fourteen","fifteen","sixteen","one","two","three","four","five","six","seven","eight","nine","ten","eleven","tweleve","thirteen","fourteen","fifteen","sixteen"]
noun_labels=torch.tensor([2,3]) #"box"
verb_labels=torch.tensor([3,5]) #"put"
gpu_device="cuda:0"
device=torch.device(gpu_device)


model = blip_pretrain_ego4d(num_frames=num_frames, verb_classes=verb_classes, noun_classes=noun_classes, vision_width=512, med_config='configs/bert_config.json', embed_dim=256, queue_size=57600, momentum=0.995)

out=model(caption, noun_labels, verb_labels,device,vid_feature=None)
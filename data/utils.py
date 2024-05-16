import re
import json
import os

import torch
import torch.distributed as dist

import utils

def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,utils.get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if utils.is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(utils.get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file

def collate_fn(batch):
    #meta_list = []

    batch_size = len(batch)

    #vid_ft_dim = batch[0][0].shape[-1]
    #tag_ft_dim = batch[0][2].shape[-1]
    #max_props_num = batch[0][3]['temporal_span']
    vid_id_dim=batch[0][0].shape[-1]
    caption_dim=batch[0][1].shape[-1]
    gt_verb_dim=batch[0][2].shape[-1]
    gt_noun_dim=batch[0][3].shape[-1]
   
    #vid_fts = torch.zeros(batch_size, max_props_num, vid_ft_dim)
    #tag_fts = torch.zeros(batch_size, max_props_num, tag_ft_dim)
    #masks = torch.zeros(batch_size, max_props_num)
    #sampled_frames_logical = torch.zeros(batch_size, max_props_num, vid_ft_dim)
    vid_id = torch.zeros(batch_size, vid_id_dim)
    caption = torch.zeros(batch_size, caption_dim)
    gt_verb = torch.zeros(batch_size, gt_verb_dim)
    gt_noun = torch.zeros(batch_size, gt_noun_dim)

    for i, sample in enumerate(batch):
        #vid_fts[i, :sample[0].shape[0], :] = sample[0]
        #masks[i, :sample[1].shape[0]] = sample[1]
        #tag_fts[i, :sample[2].shape[0], :] = sample[2]
        #sampled_frames_logical[i, :sample[2].shape[0],:] = 1


        vid_id[i, :sample[0].shape[0]] = sample[0]
        caption[i, :sample[1].shape[0]] = sample[1]
        gt_verb[i, :sample[2].shape[0]] = sample[2]
        gt_noun[i, :sample[2].shape[0]] = sample[3]

        #meta_list.append(sample[3])
   
    # masks = masks.to(torch.bool)
    #return vid_fts, masks, tag_fts, sampled_frames_logical, meta_list
    return vid_id, caption, gt_verb, gt_noun





'''from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url

def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval'''
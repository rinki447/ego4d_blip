'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_pretrain_ego4d import BLIP_Ego4d
import utils
from utils import cosine_lr_schedule
from data.ego4d import Ego4dDataset
from data import create_dataset, create_sampler, create_loader
from data.utils import collate_fn
# from torch.utils.data import DataLoader, Dataset
from utils import compute_acc


def accuracy(predictions,noun_labels,verb_labels):

    noun_acc=compute_acc(predictions['predicted_noun_probab'], torch.tensor(noun_labels), reduction='mean')
    verb_acc=compute_acc(predictions['predicted_verb_probab'], torch.tensor(verb_labels), reduction='mean')


    #total_accuracy needs the following steps

    verb_probs=predictions['predicted_verb_probab']
    noun_probs=predictions['predicted_noun_probab']
    predicted_noun_class = torch.argmax(noun_probs, dim=1).numpy()
    predicted_verb_class = torch.argmax(verb_probs, dim=1).numpy()
    same_verb=np.where(predictions['predicted_verb_label']==noun_labels)
    same_noun=np.where(predictions['predicted_noun_label']==verb_labels)
    total_video=len(noun_labels)
    both_hit=np.where(same_verb[0]==same_noun[0])[0]
    total_acc=len(both_hit)/total_video

   
    return noun_acc,verb_acc,total_acc


def train(model, data_loader_train, optimizer, epoch, device, config):
    # train
    
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_noun', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_verb', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50



    for i,(vid_id,caption,verb_labels,noun_labels) in enumerate(metric_logger.log_every(data_loader_train, print_freq, header)): 
        
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)
        _, loss_noun, loss_verb = model(caption, noun_labels, verb_labels,vid_feature=None)             
        loss = loss_noun + loss_verb
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_noun=loss_noun.item())
        metric_logger.update(loss_verb=loss_verb.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


@torch.no_grad()
def evaluation(model, data_loader_test, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Action Reco performance:'
    print_freq = 50
    metric_logger.add_meter('t_acc', utils.SmoothedValue(window_size=1,fmt='{value:.2f}'))
    metric_logger.add_meter('n_acc', utils.SmoothedValue(window_size=1,fmt='{value:.2f}'))
    metric_logger.add_meter('v_acc', utils.SmoothedValue(window_size=1,fmt='{value:.2f}'))
    
    # metric_logger.meters['t_acc'] = AverageMeter()
    # metric_logger.meters['n_acc'] = AverageMeter()
    # metric_logger.meters['v_acc'] = AverageMeter()


    result = []
    
    start_time = time.time()  

    for i,(vid_ids,caption,verb_labels,noun_labels) in enumerate(metric_logger.log_every(data_loader_test, print_freq, header)): 
        
        verb_labels = verb_labels.to(device)
        noun_labels = noun_labels.to(device)
        predictions, _, _ = model(caption,noun_labels,verb_labels,vid_feature=None) 
        
        noun_acc,verb_acc,total_acc=accuracy(predictions,noun_labels,verb_labels)
        

        #metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0)) # taken from line 87 of train_nlvr.py
      
        metric_logger.meters['t_acc'].update(total_acc.item(), n=vid_ids.shape[0])
        metric_logger.meters['n_acc'].update(noun_acc.item(), n=vid_ids.shape[0])
        metric_logger.meters['v_acc'].update(verb_acc.item(), n=vid_ids.shape[0])

    metric_logger.synchronize_between_processes()           
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    print("Averaged stats:", metric_logger.global_avg()) 
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


  
def main(args, config):

       
    # train_batch_size=2
    # test_batch_size=2

    annots_dir_train=config.annots_dir_train
    annots_dir_test=config.annots_dir_test
    taxonomy_path=config.taxonomy_path
    llava_captions_path=config.llava_captions_path
    short_annot_train_path=config.short_annot_train_path
    short_annot_train_path=config.short_annot_test_path

    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
   
    
    print("Creating action reco dataset")
    #train_dataset = Ego4dDataset(mode='train',annots_dir_train,taxonomy_path,llava_captions_path) #Rinki->should put in config the annot paths
    #test_dataset = Ego4dDataset(mode='test',annots_dir_test,taxonomy_path,llava_captions_path)

    train_dataset = Ego4dDataset(mode='train',annots_path=annots_dir_train,taxonomy_path=taxonomy_path,llava_captions_path=llava_captions_path, short_annot_path=short_annot_train_path) 
    test_dataset = Ego4dDataset(mode='test',annots_path=annots_dir_test,taxonomy_path=taxonomy_path,llava_captions_path=llava_captions_path, short_annot_path=short_annot_test_path) 

    
    #train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset, test_dataset], [True,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    collate_fn_train=collate_fn()
    collate_fn_test=collate_fn()
    
    train_loader,test_loader = create_loader([train_dataset, test_dataset],samplers,
                                                          batch_size=[config['train_batch_size'],config['test_batch_size']],
                                                          num_workers=[4,4],
                                                          is_trains=[True, False], 
                                                          collate_fns=[collate_fn_train,collate_fn_test]) 

    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=shuffle)
    # test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

   

    #### Model #### 
    print("Creating model")
    #model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             #vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             #queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = BLIP_Ego4d(num_frames=config['num_frames'], 
                       verb_classes=config['verb_classes'],
                       noun_classes=config['noun_classes'], 
                       vision_width=512, 
                       med_config=config['med_config'])#'configs/bert_config.json')

    model = model.to(device)   
    
    model_without_ddp = model


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    

    
    for epoch in range(0, config['max_epoch']):    
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
            
            train_stats = train(model, train_loader, optimizer, epoch, device, config)  
        
        else:             
            test_stats=evaluation(model, test_loader, device, config)  
            
            if utils.is_main_process():                
                log_stats = {
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")  
            break
       
        if epoch%2==0:
            test_stats=evaluation(model, test_loader, device, config) 
            
    
            if utils.is_main_process():       
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            }
                
                

                if float(test_stats['t_acc'])>best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    
                    print(f'New best t_acc= {test_stats['t_acc']}, 
                            with noun acc = {test_stats['n_acc']} and verb acc = {test_stats['v_acc']} at epoch={epoch} > 
                            previous best t_acc = {best} at epoch={best_epoch}')
                    
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth')) 
                    best = float(test_stats['t_acc'])
                    best_epoch = epoch
                        
        else:
            if utils.is_main_process():       
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            }

        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")
                   
         
        dist.barrier()   
        torch.cuda.empty_cache()
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)      
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()   
    
    parser.add_argument('--config', default='./configs/ego4d.yaml') 
    parser.add_argument('--output_dir', default='/data/AmitRoyChowdhury/Rinki/BLIP/')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    parser.add_argument('--annots_dir_train', default='/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json') 
    parser.add_argument('--annots_dir_test', default='/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_test.json') 
    parser.add_argument('--short_annots_dir_train', default='/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_short_train.json') 
    parser.add_argument('--short_annots_dir_test', default='/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_short_test.json') 
    parser.add_argument('--lava_captions_path', default='/data/AmitRoyChowdhury/Anirudh/llava_object_responses/') 
    parser.add_argument('--taxonomy_path', default='/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_taxonomy.json') 
    parser.add_argument('--train_batch_size', default=1)
    parser.add_argument('--test_batch_size', default=1)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
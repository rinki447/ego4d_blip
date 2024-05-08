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

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader


def train(model, data_loader, optimizer, epoch, device, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_noun', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_verb', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(caption, vid_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): #Rinki loader must have everything needed for model
        
        _, loss_noun, loss_verb = model()#Rinki: input to your blip_ego model#                
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
def evaluation(model, data_loader, device, config):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Action Reco performance:'
    print_freq = 50

    result = []
    
    
    start_time = time.time()  
    for i,(caption, vid_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)): 
        predictions, _, _ = model()#Rinki: input to your blip_ego model#      predictions should be np array
        
        #Rinki you know the predictions dictionary , so calculated noun, verb, action accuracy  
  

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return result


  
def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

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
        
        #Rinki evaluate after every 2 epochs 
        # score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
        # score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
    
        if utils.is_main_process():  
      
            
            print(val_result)
                                
            if val_result['r_mean']>best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))  
                best = val_result['r_mean']        
                best_epoch = epoch  
                
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt) 
                print(test_result)
            
            if args.evaluate:                
                log_stats = {**{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},                  
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in val_result.items()},
                             **{f'test_{k}': v for k, v in test_result.items()},  
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")   
                    
        if args.evaluate: 
            break

        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_flickr.yaml') # rinki path to your config
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
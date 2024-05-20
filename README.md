demo.py can be usee to run models/blip_pretrain_ego4d.py

** config file:  /configs/ego4d.yaml  
** main file:  /train_ego4d_2.py  
** model file:  /model/ego4d.py  
** dataset file: /data/blip_pretrain_ego4d.py  

** output file: /data/AmitRoyChowdhury/Rinki/BLIP/log.txt  


# main file -Sayak please check
** line 43 to 61  
** line 136-138, n=len(vid_ids)   
** line 222-223 (find_unused_parameters=True)  
***line 290 of train_ego4d_2.py was giving error,(log_stats was not introduced before),so added log_stats={} in line 236


# main file -Rinki please check


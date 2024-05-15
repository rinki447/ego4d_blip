demo.py runs models/blip_pretrain_ego4d.py


# main file -Sayak please check
Updated config file config/retrieval_flickr_ego4d.yaml by deleting unneccessary parameters, added some paths to the config inside train_ego4d_2.py  in line 307 to 314

doubt:
line 29 of train_ego4d_2.py:  from data import create_dataset, create_sampler, create_loader

inside data folder there is no create_dataset, create_loader or create_sampler function;
that is why I was using line 189 and 190 (you have commented out) instead of line 176 to 187

# main file -Rinki please check
Done: [[[[[[ ok(I have addressed all the concerns above);
done(You need to create a config file);
done(in the train code check wherever there is Rinki-> written and address those concerns);
done(also please add a dataset call in demo.ipynb I need to check it before doing collate_fn);]]]]]

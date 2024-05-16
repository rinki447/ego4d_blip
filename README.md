demo.py runs models/blip_pretrain_ego4d.py


# main file -Sayak please check
added collate fn and modified it within data/utils.py: line 78 to line 116

included the collate_fn in train_ego4d_2.py: check line number 183 to 190

doubt1: what should be passed as argument within collate_fn in line number 183 of train_ego4d_2.py

doubt2:line 303, should we keep action="store_true" in  parser.add_argument('--evaluate', action='store_true')


# main file -Rinki please check
Done: [[[[[[ ok(I have addressed all the concerns above);
done(You need to create a config file);
done(in the train code check wherever there is Rinki-> written and address those concerns);
done(also please add a dataset call in demo.ipynb I need to check it before doing collate_fn);]]]]]

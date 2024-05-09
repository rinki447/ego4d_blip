from torch.utils.data import Dataset
import os
import re
import json
import pickle

class Ego4dDataset(Dataset):
    def __init__(
            self, 
            annots_path, 
            taxonomy_path,
            llava_captions_path, 
        ):

        self.llava_captions_path = llava_captions_path
        with open(taxonomy_path, "r") as f:
            lta_taxo = json.load(f)
        
        self.lta_nouns = lta_taxo['nouns']
        self.lta_verbs = lta_taxo['verbs']
        
        with open(annots_path, "r") as f:
            self.annots = json.load(f)['clips']

        self.valid_files = self.comp_valid_files()

    def get_seg_start_end_frame(self, seg_name):
        """Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
            start_frame= XX and end_frame=YY respectively.
        """
        start_frame = re.search(r'start_frame_(\d+)', seg_name).group(1)
        end_frame = re.search(r'end_frame_(\d+)', seg_name).group(1)

        return int(start_frame), int(end_frame)

    def comp_valid_files(self):
        
        tot_valid_files = []

        llava_files = os.listdir(self.llava_captions_path)

        for annot in self.annots:
            clip_id = annot['clip_uid']
            start_frame = annot['action_clip_start_frame']
            end_frame = annot['action_clip_end_frame']

            seg_file = "{}_start_frame_{}_end_frame_{}.pkl".format(clip_id, start_frame, end_frame)

            if seg_file in llava_files:
                tot_valid_files.append(seg_file)

        return tot_valid_files

    def __len__(self):
        return len(self.valid_files)

    def get_gt_caption(self, seg_file):
        """Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
            corresponding verb_label and noun_label respectively.
        """
        clip_id = seg_file.split("_st")[0]
        start_frame, end_frame = self.get_seg_start_end_frame(seg_file)

        for seg_annot in self.annots:
            seg_start_frame = seg_annot["action_clip_start_frame"]
            seg_end_frame = seg_annot["action_clip_end_frame"]
            seg_clip_id = seg_annot['clip_uid']

            if seg_clip_id == clip_id and seg_start_frame == start_frame and\
                seg_end_frame == end_frame:
                return seg_annot['verb_label'], seg_annot['noun_label']

    def __getitem__(self, idx):
        seg_file = self.valid_files[idx]

        with open(self.llava_captions_path + seg_file, "rb") as f:
            llava_caps = pickle.load(f)

        gt_verb_label, gt_noun_label = self.get_gt_caption(seg_file)

        return seg_file, llava_caps, gt_verb_label, gt_noun_label

if __name__ == "__main__":
    annots_dir = "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/"
    llava_captions_path = " /data/AmitRoyChowdhury/Anirudh/llava_object_responses/"
    taxonomy_path = annots_dir + "fho_lta_taxonomy.json"

    train_dataset = Ego4dDataset(

    )
    
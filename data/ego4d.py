import torch
from torch.utils.data import Dataset
import os
import re
import json
import pickle
from data.utils import pre_caption
import numpy as np
class Ego4dDataset(Dataset):
	def __init__(
			self,
   			mode, 
			annots_path, 
			taxonomy_path,
			llava_captions_path, 
			short_annot_path
		):
		self.idx=0
		self.llava_captions_path = llava_captions_path
		self.short_annot_path=short_annot_path
		self.taxonomy_path=taxonomy_path

		with open(taxonomy_path, "r") as f:
			lta_taxo = json.load(f)
		
		self.lta_nouns = lta_taxo['nouns']
		self.lta_verbs = lta_taxo['verbs']
		
		
		with open(self.short_annot_path, "r") as f:
			self.annots = list(json.load(f).keys()) 

		#with open(annots_path, "r") as f:
			#self.annots = json.load(f)['clips']

		self.valid_files = self.comp_valid_files() #list of segment names collected from train_annotation_path
		print("total valid training files",len(self.valid_files))
		
		#print(f'{mode}ing dataset with {self.__len__()} video segments')	
		

	'''def get_seg_start_end_frame(self, seg_name):
		"""Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
			start_frame= XX and end_frame=YY respectively.
		"""
		start_frame = re.search(r'start_frame_(\d+)', seg_name).group(1)
		end_frame = re.search(r'end_frame_(\d+)', seg_name).group(1)

		return int(start_frame), int(end_frame)'''

	def comp_valid_files(self):
		
		tot_valid_files = []

		llava_files = os.listdir(self.llava_captions_path)
		#print(len(llava_files))

		for i,annot in enumerate(self.annots):
			
			if i==50:#for debugging
				break
    
			seg_file=annot+".pkl"
			if seg_file in llava_files:
				tot_valid_files.append(annot)


		return tot_valid_files

	def __len__(self):
		return len(self.valid_files)

	'''def get_gt_caption(self, seg_file):
		"""Given a segment in the format "clip_name_start_frame_XX_end_frame_YY", get the
			corresponding verb_label and noun_label respectively.
		"""
		#clip_id = seg_file.split("_st")[0]
		#start_frame, end_frame = self.get_seg_start_end_frame(seg_file)

		for seg_annot in self.annots:
			seg_start_frame = seg_annot["action_clip_start_frame"]
			seg_end_frame = seg_annot["action_clip_end_frame"]
			seg_clip_id = seg_annot['clip_uid']

			if seg_clip_id == clip_id and seg_start_frame == start_frame and\
				seg_end_frame == end_frame:
				return seg_annot['verb_label'], seg_annot['noun_label']'''
		

		

	def __getitem__(self, idx):
		
		
			seg_file=self.llava_captions_path + self.valid_files[idx]+".pkl"
			#print("llava file name",seg_file)
			with open(seg_file, "rb") as f:
				llava_caps = pickle.load(f)

			frames=list(llava_caps.keys())
			caps=[]
			for i in frames:
				caps.append(pre_caption(llava_caps[i],max_words=20)) #Rinki-> check what is being outputed by getitem() in demo.ipynb
				# caps.append(llava_caps[i].strip('"').strip("\n").strip("."))
			with open(self.short_annot_path, "r") as f:
				file=json.load(f)

			vid_name=self.valid_files[idx]
			gt_noun_label=file[vid_name]["noun"]
			gt_verb_label=file[vid_name]["verb"]

			noun_list=np.array(self.lta_nouns)
			verb_list=np.array(self.lta_verbs)

			#print("gt_noun",f"{gt_noun_label}")
			#print("gt_verb",f"{gt_verb_label}")
			#print(noun_list)
			if f"{gt_noun_label}" in noun_list:
				noun_label=torch.LongTensor(np.where(noun_list==f"{gt_noun_label}")[0])
			if f"{gt_verb_label}" in verb_list:
				verb_label=torch.LongTensor(np.where(verb_list==f"{gt_verb_label}")[0])

			#print("labels",noun_label,verb_label)
			
			
			return seg_file, caps, verb_label, noun_label




'''
if __name__ == "__main__":
	annots_dir_train = "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"
	annots_dir_test = "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_test.json"
	llava_captions_path = "/data/AmitRoyChowdhury/Anirudh/llava_object_responses/"
	taxonomy_path = "/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_taxonomy.json"

	train_dataset = Ego4dDataset(annots_dir_train,taxonomy_path,llava_captions_path)

	for i,(seg_file,caps,gt_verb_label,gt_noun_label) in enumerate(train_dataset):
		print(i)
		print(caps)
		exit()
'''	

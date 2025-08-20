import os
import numpy as np
import argparse
from collections import namedtuple

import torch

from train import Trainer
from eval import Evaluator
from hparams import *


def set_random_seed(hparams):
	# set random seed
	np.random.seed(hparams.random_seed)
	torch.manual_seed(hparams.random_seed)
	torch.cuda.manual_seed_all(hparams.random_seed)
	torch.set_default_tensor_type('torch.FloatTensor')
	return

def arg_parser():
	arg_parser = argparse.ArgumentParser(description="Retriever")
	arg_parser.add_argument("--task_name", dest="task_name", default="Document Retrieval" , type=str, required=True, help="")

	# running type - train, eval, predict
	arg_parser.add_argument("--evaluate", default = "" , dest="evaluate", type=str, help="eval checkpoint path") # resources/bert-base-uncased
	arg_parser.add_argument("--predict", default = "" ,  dest="predict", type=str, help="predict checkpoint path") # resources/bert-base-uncased

	# model name
		# wisenut_korean, electra_korean, bert_korean의 경우 bert_pretrained 및 tokenizer setting이 되어 있음
	arg_parser.add_argument("--encoder_model", dest="encoder_model", type=str, required=True,
							default="bert", help="bert, electra, sentence_transformer") 	

	# dir
	arg_parser.add_argument("--data_type", dest="data_type", type=str, required=True,
							help="kibo, aihub_paper, aihub_patent") 
	arg_parser.add_argument("--root_dir", default = "./", type=str)
	args = arg_parser.parse_args()
	return args

def train_model(hparams):
	model = Trainer(hparams)
	model.train()

def evaluate_model(hparams):
	model = Evaluator(hparams)
	model.run_evaluate(hparams.evaluate)

def main():
	args = arg_parser()
	hparams = vars(args)
	
	hparams.update(DATASET_MAP[args.data_type])
	hparams.update(BASE_PARAMS)
	hparams.update(PRETRAINED_MODEL_MAP[args.encoder_model])

	named_tuple_ = namedtuple('named_tuple_' , list(hparams.keys()))
	hparams = named_tuple_(*list(hparams.values()))
	set_random_seed(hparams)

	if hparams.evaluate:
		evaluate_model(hparams)

	elif hparams.predict:
		predict_model(hparams)
		
	else:
		train_model(hparams)


if __name__ == '__main__':
	main()
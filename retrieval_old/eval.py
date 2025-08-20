import os
import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RetrieverDataset
from utils import init_logger, batch_to_device

from model.tokenizer import *
from hparams import *
from dataset import RetrieverDataset
from loss import BiEncoderNllLoss
from model.utils.scorer import calculate_candidates_ranking, logits_mrr, \
	logits_recall_at_k, precision_at_one, mean_average_precision


class Evaluator(object):
	def __init__(self, hparams, tokenizer=None, model=None):

		self.hparams = hparams
		self.tokenizer = tokenizer
		self.model = model
		self._logger = logging.getLogger(__name__)
		self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
					   if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))
		self._build_dataloader()

	def _build_dataloader(self): 
		self.eval_dataset = RetrieverDataset(
			self.hparams,
			self.tokenizer,
			split= "valid",
		)

		self.eval_dataloader = DataLoader(
			self.eval_dataset,
			batch_size=self.hparams.eval_batch_size,
			num_workers=self.hparams.cpu_workers,
			shuffle=True,
			drop_last=False,
		)

	def build_model(self, evaluation_path):

		if self.hparams.parameter_sharing == True:
			self.encoder = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
			self.tokenizer = TOKENIZER_MAP[self.hparams.encoder_model]
		else: 
			self.q_encoder = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
			self.d_encoder = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
			self.tokenizer = TOKENIZER_MAP[self.hparams.encoder_model]

	def _build_creterion(self):
		# =============================================================================
		#   CRITERION
		# =============================================================================
		self.iterations = len(self.eval_dataset)
		self.criterion = BiEncoderNllLoss()

	def load_model(self, evaluation_path):
		self.model = self.model.from_pretrained(
			pretrained_model_name_or_path=evaluation_path, # fine-tuned checkpoints
			config=self.pretrained_config, # pre-trained config (ex, BertConfig, ElectraConfig)
			hparams=self.hparams # hparams
		)
		self.model = self.model.to(self.device)
		print("Model Load Completes")

	def run_evaluate(self, evaluation_path):
		self._logger.info("Evaluation")
		if not self.model:
			print("Build and Load fine-tuned checkpoint")
			self.build_model(evaluation_path)
			self.load_model(evaluation_path)

		self._build_creterion()
		k_list = self.hparams.recall_k_list
		# total_mrr, total_prec_at_one, total_map = 0, 0, 0
		# total_examples, total_correct = 0, 0
		accu_loss = accu_cnt = eval_cnt =0
		total_acc_topk = 0
		self.model.eval()

		with torch.no_grad():
			print(f"self.model : {self.model}")
			for batch_idx, batch in enumerate(tqdm(self.eval_dataloader)):
				
				with torch.cuda.amp.autocast():
					if self.hparams.encoder_model == "sentence_transformer":
						field1 = batch['field1']
						field2 = batch['field2']

						field1_emb = self.model(field1, batch_size = self.hparams.eval_batch_size)
						field2_emb = self.model(field2, batch_size = self.hparams.eval_batch_size)
					else:
						field1 = batch_to_device(batch['field1'], self.device)
						field2 = batch_to_device(batch['field2'], self.device)
					
						field1_emb = self.model(*field1).last_hidden_state[:,0,:]
						field2_emb = self.model(*field2).last_hidden_state[:,0,:]

					if 'label' not in batch:
						loss, logits, correct_predictions_count = self.criterion.calc(field1_emb, field2_emb)
						
					else:
						logits = torch.sum(field1_emb * field2_emb, dim=-1)
						label = batch['label']
						loss = self.criterion(logits, label)

				pred = np.array(logits.to("cpu").tolist()) # |batch_size, batch_size|
				batch_acc_topk = np.zeros_like(k_list) # [1, 2, 5, 10]
				accu_loss += loss
				accu_cnt += 1

				for answer_idx, similarities in enumerate(pred):
					pred_rank = similarities.argsort()[::-1] # [top1_index, top2_index ... topk_index]
					rank_of_answer = np.where(pred_rank == answer_idx)[0][0]+1 # find the rank of answer document
					batch_acc_topk += np.where(k_list//rank_of_answer>0 , 1 , 0)

				total_acc_topk += batch_acc_topk/len(pred)


			total_acc_topk = total_acc_topk/accu_cnt

		acc_results = ""
		for i, topk in enumerate(k_list):
			acc_results += "top@%s: %.4f |" % (topk, total_acc_topk[i])
		
		print(f"[Evaluation][Eval data count : {eval_cnt}][Iter cnt : {len(self.eval_dataloader)}][Loss : {accu_loss/accu_cnt:.4f}][Accuracy: {acc_results}]")
		print(f"total_acc_topk: {total_acc_topk}")
		self._logger.info(f"[Evaluation][Eval data count : {eval_cnt}][Iter cnt : {len(self.eval_dataloader)}][Loss : {accu_loss/accu_cnt:.4f}][Accuracy: {acc_results}]")	
			# 	for key in batch:
			# 		buffer_batch[key] = buffer_batch[key].to(self.device)

			# 	logits, loss = self.model(buffer_batch)

			# 	pred = logits.to("cpu").tolist()

			# 	rank_by_pred, pos_index, stack_scores = \
			# 		calculate_candidates_ranking(np.array(pred),
			# 									 np.array(buffer_batch["label"].to("cpu").tolist()),
			# 									 self.hparams.evaluate_candidates_num)

			# 	num_correct = logits_recall_at_k(pos_index, k_list)

			# 	# if multiple labels exist
			# 	if self.hparams.task_name in ["douban", "kakao"]:
			# 		total_prec_at_one += precision_at_one(rank_by_pred)
			# 		total_map += mean_average_precision(pos_index)
			# 		for pred in rank_by_pred:
			# 			if sum(pred) == 0:
			# 				total_examples -= 1

			# 	total_mrr += logits_mrr(pos_index)

			# 	total_correct = np.add(total_correct, num_correct)
			# 	total_examples += math.ceil(
			# 		buffer_batch["label"].size()[0] / self.hparams.evaluate_candidates_num)

			# 	recall_result = ""
			# 	if (batch_idx + 1) % self.hparams.evaluate_print_step == 0:
			# 		for i in range(len(k_list)):
			# 			recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % (
			# 						(total_correct[i] / total_examples) * 100)
			# 		else:
			# 			print("%d[th] | %s | MRR : %.3f | P@1 : %.3f | MAP : %.3f" %
			# 				  (batch_idx + 1, recall_result, float(total_mrr / total_examples),
			# 				   float(total_prec_at_one / total_examples), float(total_map / total_examples)))
			# 		self._logger.info("%d[th] | %s | MRR : %.3f | P@1 : %.3f | MAP : %.3f" %
			# 						  (batch_idx + 1, recall_result, float(total_mrr / total_examples),
			# 						   float(total_prec_at_one / total_examples), float(total_map / total_examples)))

			# avg_mrr = float(total_mrr / total_examples)
			# avg_prec_at_one = float(total_prec_at_one / total_examples)
			# avg_map = float(total_map / total_examples)
			# recall_result = ""

			# for i in range(len(k_list)):
			# 	recall_result += "Recall@%s : " % k_list[i] + "%.2f%% | " % ((total_correct[i] / total_examples) * 100)
			# print(recall_result)
			# print("MRR: %.4f" % avg_mrr)
			# print("P@1: %.4f" % avg_prec_at_one)
			# print("MAP: %.4f" % avg_map)

			# self._logger.info(recall_result)
			# self._logger.info("MRR: %.4f" % avg_mrr)
			# self._logger.info("P@1: %.4f" % avg_prec_at_one)
			# self._logger.info("MAP: %.4f" % avg_map)

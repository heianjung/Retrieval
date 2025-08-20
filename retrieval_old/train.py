import os
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from model.tokenizer import *
from hparams import *
from dataset import RetrieverDataset
from loss import BiEncoderNllLoss

from eval import Evaluator
from utils import init_logger, batch_to_device
from model.utils.checkpointing import CheckpointManager, load_checkpoint

class Trainer(object):
	def __init__(self, hparams):
		self.hparams = hparams

		# get logger
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		log_dir = os.path.join(hparams.root_dir, "logs", hparams.task_name, "%s/" % timestamp)
		self._logger = init_logger(log_dir)
		self._logger.info("Hyper-parameters: %s" % str(hparams))

	def _build_dataloader(self):
		# =============================================================================
		#   SETUP DATASET, DATALOADER
		# =============================================================================

		self.train_dataset = RetrieverDataset(self.hparams, self.tokenizer, split="train")
		self.train_dataloader = DataLoader(
			self.train_dataset,
			batch_size=self.hparams.train_batch_size,
			num_workers=self.hparams.cpu_workers,
			shuffle=True,
			drop_last=True
		)

		print(""" 
			# -------------------------------------------------------------------------
			#   DATALOADER FINISHED
			# -------------------------------------------------------------------------
			""")

	def _build_model(self):
		# =============================================================================
		#   MODEL
		# =============================================================================
		print(f"self.hparams.device: {self.hparams.device}")
		# @! 
		# if self.hparams.parameter_sharing == True:
		# 	self.model = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
		# 	self.tokenizer = TOKENIZER_MAP[self.hparams.encoder_model]
		# else: 
		# 	self.q_encoder = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
		# 	self.d_encoder = ENCODER_MAP[self.hparams.encoder_model](self.hparams, self.hparams.device)
		# 	self.tokenizer = TOKENIZER_MAP[self.hparams.encoder_model]
		
		# @@!
		# "klue/roberta-base"
		# self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
		# self.model = RobertaModel.from_pretrained("klue/roberta-base")

		# # "klue/bert-base"
		# self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
		# self.model = BertModel.from_pretrained("klue/bert-base")

		# sbert - 2) huggingface
		self.tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
		self.model = AutoModel.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')


		print("""
			# -------------------------------------------------------------------------
			#   MODEL BUILD FINISHED
			# -------------------------------------------------------------------------
			""")
		

	def _build_creterion(self):
		# =============================================================================
		#   CRITERION
		# =============================================================================
		self.iterations = len(self.train_dataset)
		self.criterion = BiEncoderNllLoss()

		# Prepare optimizer and schedule (linear warmup and decay)
		if self.hparams.optimizer_type == "Adam":
			self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate)
		elif self.hparams.optimizer_type == "AdamW":
			no_decay = ['bias', 'LayerNorm.weight']
			optimizer_grouped_parameters = [
				{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
				 'weight_decay': self.hparams.weight_decay},
				{'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
				 'weight_decay': 0.0}
			]
			self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
								   eps=self.hparams.adam_epsilon)
			self.scheduler = get_linear_schedule_with_warmup(
				self.optimizer, num_warmup_steps=self.hparams.warmup_steps,
				num_training_steps=self.iterations * self.hparams.num_epochs)

		print("""
			# -------------------------------------------------------------------------
			#   CRITERION SET FINISHED
			# -------------------------------------------------------------------------
			""")

	def _setup_checkpoint_manager(self):
		self.start_epoch = 1
		if self.hparams.save_dirpath == 'checkpoints/':
			self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
		self.checkpoint_manager = CheckpointManager(self.model, self.optimizer, self.save_dirpath, hparams=self.hparams)

		print(
			"""
			# -------------------------------------------------------------------------
			#   Setup Training Finished
			# -------------------------------------------------------------------------
			"""
		)

	def train(self):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self._build_model()
		self._build_dataloader()
		self._build_creterion()
		self._setup_checkpoint_manager()

		self.model = self.model.to(self.device)
		self._logger.info("Start train model at %s" % datetime.now().strftime('%H:%M:%S'))
		
		print(f"train model : {self.model}")
		train_begin = datetime.utcnow()  # New
		total_loss = 0
		global_iteration_step = 0

		self._logger.info(self.checkpoint_manager.ckpt_dirpath)

		# -------------------------------------------------------------------------
		#   학습안한모델 성능 찍어보려고
		# -------------------------------------------------------------------------
		epoch = 0
		self.checkpoint_manager.step(epoch)
		self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.bin" % (epoch))
		self._logger.info(self.previous_model_path)

		torch.cuda.empty_cache()
		self._logger.info("Evaluation after %d epoch" % epoch)
		
		# Evaluation Setup
		evaluation = Evaluator(self.hparams, tokenizer = self.tokenizer, model=self.model)
		evaluation.run_evaluate(self.previous_model_path)
		torch.cuda.empty_cache()



		for epoch in range(self.start_epoch, self.hparams.num_epochs + 1):
			self.model.train()
			tqdm_batch_iterator = tqdm(self.train_dataloader)
			for batch_idx, batch in enumerate(tqdm_batch_iterator):
								
				with torch.cuda.amp.autocast():
					# if self.hparams.encoder_model == "sentence_transformer":
					# 	field1 = batch['field1']
					# 	field2 = batch['field2']
					# 	bs = len(field1)

					# 	field1_emb = self.model(field1, batch_size = self.hparams.train_batch_size)
					# 	field2_emb = self.model(field2, batch_size = self.hparams.train_batch_size)

					# else:
					field1 = batch_to_device(batch['field1'], self.device)
					field2 = batch_to_device(batch['field2'], self.device)
					bs = len(field1[0])

					field1_emb = self.model(*field1).last_hidden_state[:,0,:]
					field2_emb = self.model(*field2).last_hidden_state[:,0,:]

					if 'label' not in batch:
						loss, _, _ = self.criterion.calc(field1_emb, field2_emb)
					else:
						logits = torch.sum(field1_emb * field2_emb, dim=-1)
						label = batch['label']
						loss = self.criterion(logits, label)
					
					total_loss+= loss

				nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.max_gradient_norm)
				loss.backward()
				self.optimizer.step()
				if self.hparams.optimizer_type == "AdamW":
					self.scheduler.step()

				global_iteration_step += 1
				self.optimizer.zero_grad()
				if global_iteration_step%100 ==0:
					description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
						datetime.utcnow() - train_begin,
						epoch,
						global_iteration_step, total_loss / global_iteration_step, self.optimizer.param_groups[0]['lr'])
					tqdm_batch_iterator.set_description(description)

				# tensorboard
				if global_iteration_step % self.hparams.tensorboard_step == 0:
					self._logger.info(description)

			# -------------------------------------------------------------------------
			#   ON EPOCH END  (checkpointing and validation)
			# -------------------------------------------------------------------------
			self.checkpoint_manager.step(epoch)
			self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.bin" % (epoch))
			self._logger.info(self.previous_model_path)

			torch.cuda.empty_cache()
			self._logger.info("Evaluation after %d epoch" % epoch)
			
			# Evaluation Setup
			evaluation = Evaluator(self.hparams, tokenizer = self.tokenizer, model=self.model)
			evaluation.run_evaluate(self.previous_model_path)
			torch.cuda.empty_cache()

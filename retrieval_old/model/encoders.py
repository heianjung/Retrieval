# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torch import Tensor

from typing import Union, List, Dict
from transformers import BertPreTrainedModel
from sentence_transformers import SentenceTransformer, util
from transformers import BertPreTrainedModel
# 박아놓음
from transformers import AutoTokenizer, AutoModel, RobertaModel
import torch

class KRSentenceBERT0(nn.Module):
    def __init__(self,
                 hidden_size: int = 768,
                 projection_size: int = 128,
                 dropout_p: float = .0,
                 device: str = "cuda"
        ):
        super().__init__()
        self.sbert = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device)

    def forward(self, sentences: Union[str, List[str]], batch_size: int = 32) -> Tensor:
        # print(f"[KRSentenceBERT] batch_size : {batch_size}")
        # print(f"[KRSentenceBERT] self.sbert.device : {self.sbert.device}")

        embedding = self.sbert.encode(sentences, convert_to_tensor=True, batch_size=batch_size,show_progress_bar = False)
        # print(f"encoder.py KRSentenceBERT0 embedding : {embedding.shape}")
        return embedding
    
class TWOTOWER_ENCODER(BertPreTrainedModel):
# 아직 손보지 않음 (황태선 대리님 작업 내용 그대로)
	def __init__(self, pretrained_config, hparams):
		super().__init__(pretrained_config)
		self.hparams = hparams
		self.pretrained_config = pretrained_config

		self.query_model = self.hparams.pretrained_model(pretrained_config)
		self.query_proj = nn.Linear(hparams.hidden_size, hparams.projection_size)

		self.document_model = self.hparams.pretrained_model(pretrained_config)
		self.document_proj = nn.Linear(hparams.hidden_size, hparams.projection_size)

		self.init_weights()

	def forward(self, batch):

		candidate_num = self.hparams.evaluate_candidates_num
		loss = None

		# dialogue
		if self.hparams.do_contrastive_learning and not self.training:
			# merge same dialogue context
			# bs * cand_num, max_seq_len
			batch_size = batch["d_input_ids"].shape[0] // candidate_num
			input_ids = batch["d_input_ids"].reshape(batch_size, candidate_num, -1)[:,0]
			segment_ids = batch["d_segment_ids"].reshape(batch_size, candidate_num, -1)[:,0]
			attention_mask = batch["d_attention_mask"].reshape(batch_size, candidate_num, -1)[:,0]

			dialogue_outputs = self.dialogue_model(
				input_ids,
				token_type_ids=segment_ids,
				attention_mask=attention_mask
			) # bs / cand_num, bert_output_size

		else:
			dialogue_outputs = self.dialogue_model(
				batch["d_input_ids"],
				token_type_ids=batch["d_segment_ids"],
				attention_mask=batch["d_attention_mask"]
			)
		dialogue_cls = dialogue_outputs[0][:, 0, :] # bs, hidden_dim
		dialogue_cls = self.dialogue_proj(dialogue_cls) # bs, proj_dim

		# response
		response_outputs = self.response_model(
			batch["r_input_ids"],
			token_type_ids=batch["r_segment_ids"],
			attention_mask=batch["r_attention_mask"]
		)
		response_cls= response_outputs[0][:, 0, :]  # bs, hidden_dim
		response_cls = self.response_proj(response_cls) # bs, proj_dim

		if self.hparams.do_contrastive_learning and self.training:
			logits = torch.matmul(dialogue_cls, response_cls.t())  # bs, bs
			# label should be [[1,0,0...], [0,1,0..], [0,0,1...]...]
			labels = torch.arange(dialogue_cls.shape[0], dtype=torch.long).to(logits.device)
			loss_fn = nn.CrossEntropyLoss()

			# symmetric loss function
			r_loss = loss_fn(logits, labels)
			d_loss = loss_fn(logits.t(), labels)
			loss = (r_loss + d_loss) / 2

		elif not self.hparams.do_contrastive_learning and self.training:
			logits = torch.sum(dialogue_cls * response_cls, dim=-1)  # [bs]
			loss_fn = nn.BCEWithLogitsLoss()
			loss = loss_fn(logits, batch["label"])

		else:
			# [unique_batch, 1, hidden] -> [unique_batch, candidate_num, hidden] -> [unique_batch * candidate_num, hidden]
			expanded_dialogue = dialogue_cls.unsqueeze(1).expand(-1, candidate_num, -1)
			# inner product
			logits = torch.sum(expanded_dialogue.reshape(response_cls.shape[0], -1) * response_cls, dim=-1)  # [bs]

		return logits, loss

class SIAMESE_ENCODER(BertPreTrainedModel):
	def __init__(self, pretrained_config, hparams):
		super().__init__(pretrained_config)
		self.hparams = hparams
		self.pretrained_config = pretrained_config

		self.model = self.hparams.pretrained_model(pretrained_config)
		self.proj = nn.Linear(hparams.hidden_size, hparams.projection_size)
		self.init_weights()

	def get_embeddings(self, batch):
		# dialogue
		outputs = self.model(
			batch["input_ids"],
			token_type_ids=batch["token_type_ids"],
			attention_mask=batch["attention_mask"],
		) 
		cls = outputs[0][:, 0, :] # bs, hidden_dim
		cls = self.proj(cls) # bs, proj_dim
		return cls

	def forward(self, batch):
		
		field1_cls = self.get_embeddings(batch["field1"])
		field2_cls = self.get_embeddings(batch["field2"])
		
		if self.hparams.do_contrastive_learning:
		# if self.hparams.do_contrastive_learning and self.training:
			logits = torch.matmul(field1_cls, field2_cls.t())  # bs, bs
			# label should be [[1,0,0...], [0,1,0..], [0,0,1...]...]
			labels = torch.arange(field1_cls.shape[0], dtype=torch.long).to(logits.device)
			loss_fn = nn.CrossEntropyLoss()

			# symmetric loss function
			r_loss = loss_fn(logits, labels)
			d_loss = loss_fn(logits.t(), labels)
			loss = (r_loss + d_loss) / 2

		else:
		# elif not self.hparams.do_contrastive_learning and self.training:
			logits = torch.sum(field1_cls * field2_cls, dim=-1)  # [bs]
			loss_fn = nn.BCEWithLogitsLoss()
			loss = loss_fn(logits, batch["label"])

		# else:
		# 	# [unique_batch, 1, hidden] -> [unique_batch, candidate_num, hidden] -> [unique_batch * candidate_num, hidden]
		# 	expanded_dialogue = dialogue_cls.unsqueeze(1).expand(-1, candidate_num, -1)
		# 	# inner product
		# 	logits = torch.sum(expanded_dialogue.reshape(response_cls.shape[0], -1) * response_cls, dim=-1)  # [bs]

		return logits, loss

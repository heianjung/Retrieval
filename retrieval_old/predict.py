import os
import logging
import time

import faiss
import numpy as np
from datetime import datetime
from collections import namedtuple

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from hparams import *
from model.tokenizer import *
from dataset import RetrieverDataset, CashingDataset
from utils import init_logger, batch_to_device
# from wisekmapy.wisekma import Wisekma


class Predictor(object):
	def __init__(self, hparams, tokenizer=None, pred_ckpt_path=None):

		self.hparams = hparams
		self.tokenizer = tokenizer
		self.pred_ckpt_path = pred_ckpt_path
		self._logger = logging.getLogger(__name__)

		# get logger
		timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
		log_dir = os.path.join(hparams.root_dir, "logs", hparams.task_name, "%s/" % timestamp)
		self._logger = init_logger(log_dir)
		self._logger.info("Hyper-parameters: %s" % str(hparams))

		self.device = (torch.device("cuda", self.hparams.gpu_ids[0])
					   if self.hparams.gpu_ids[0] >= 0 else torch.device("cpu"))

		self.build_model(pred_ckpt_path)
		# self.load_model(pred_ckpt_path)

	def build_dataloader(self, txt_path):
		self.test_dataset = CashingDataset(
										self.hparams, 
										self.tokenizer,
										txt_path)
		self.dataloader = DataLoader(
			self.test_dataset,
			batch_size=self.hparams.pred_batch_size,
			num_workers=self.hparams.cpu_workers,
			drop_last=False
		)


	def build_model(self, pred_ckpt_path:str=None):
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

		# "klue/bert-base"
		self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
		self.model = BertModel.from_pretrained("klue/bert-base")

		# self.tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
		# self.model = AutoModel.from_pretrained('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

		# @@@ default  - shinhan_ressel
		# # pre-trained ckpt의 config
		# config_path = os.path.dirname(pred_ckpt_path)
		# self.pretrained_config = self.hparams.pretrained_config.from_pretrained(config_path)
		# self.model = BiEncoder(self.pretrained_config, self.hparams)
		# print(self.pretrained_config)
		# print("Model Build Completes")

	def load_model(self, pred_ckpt_path):
		self.model = self.model.from_pretrained(
			pretrained_model_name_or_path=pred_ckpt_path, # fine-tuned checkpoints
			# config=self.pretrained_config, # pre-trained config (ex, BertConfig, ElectraConfig)
			# hparams=self.hparams # hparams
		)
		self.model = self.model.to(self.device)

		# Use Multi-GPUs
		if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
			self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

		print("Model Load Completes")

	def load_and_cache_responses(self, test_pool:list=None):
		if test_pool:
			self.text_dict=dict()
			self.embedding_dict=dict()

			for idx, txt in enumerate(test_pool):
				self.text_dict[idx] = txt
				self.embedding_dict[idx]= self.get_query_embedding(txt)


		else:
			txt_path = os.path.join(self.hparams.data_dir,
									self.hparams.txt_path)
			cached_path = os.path.join(self.hparams.data_dir,
									self.hparams.cached_path)
			if not os.path.exists(cached_path):
				print("Response Caching")

				self.build_dataloader(txt_path)
				self.model.eval()

				field_emb = dict()

				with torch.no_grad():
					for batch_idx, batch in enumerate(tqdm(self.dataloader)):
						field_id = batch['field_id']
						batch = batch['field_tokenized']
						batch = batch_to_device(batch, self.device)

						field_emb = self.model(*batch).last_hidden_state[:,0,:]

						for i, f_emb in zip(field_id, field_emb):
							field_emb[i]= f_emb.cpu().numpy()

				print("Saving embeddings..")
				with open(cached_path, 'wb') as f:
					pickle.dump(field_emb, f, pickle.HIGHEST_PROTOCOL)

				print("Save Completes!")
			else:
				print('Loading embeddings..')
				with open('data.pickle', 'rb') as f:
					field_emb = pickle.load(cached_path)

			self.embedding_dict = field_emb

	def get_query_embedding(self, query_txt):
		inputs = self.tokenizer(query_txt,
								padding = 'max_length', 
								truncation = True, 
								return_tensors = "pt")
		
		token_ids = torch.squeeze(inputs['input_ids']) # tensor type
		attention_mask = torch.squeeze(inputs['attention_mask']) # tensor type
		token_type_ids = torch.squeeze(inputs['token_type_ids']) # tensor type

		embedding = self.model(**inputs).last_hidden_state[:,0,:]
		embedding = embedding.detach().numpy()
		return embedding

	def get_faiss_index(self, search_type="l2"):
		dim = 768 # self.hparams.emb_size  # dimension
		# if self.hparams.cpu_predict:
		# self.device = torch.device("cpu")
		# self.model = self.model.to(self.device)

		# faiss search
		# self.load_and_cache_responses()

		ids, embeddings = [], []
		for k,v in self.embedding_dict.items():
			ids.append(k)
			if len(v.shape) == 1 and v.shape[0] == dim:
				v = np.expand_dims(v, axis=0)
			embeddings.append(v)
		embeddings = np.concatenate(embeddings, axis=0)
		self.mapping_table = {i: _id for i, _id in enumerate(ids)}
		# faiss search by ANNS algorithm
		k = 5  # k-NN size

		if search_type == "l2":	
			self.faiss_index = faiss.IndexFlatL2(dim)
			self.faiss_index.add(embeddings)  # add dataset

		elif search_type == "ivf":
			nlist = 50  # clusters

			quantizer = faiss.IndexFlatL2(dim)
			self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)  # index
			self.faiss_index.train(embeddings)  # train
			self.faiss_index.add(embeddings)  # add dataset

		elif search_type == "l2":
			self.faiss_index = faiss.IndexFlatL2(dim)
			print(self.faiss_index.is_trained)
			self.faiss_index.add(embeddings)  # add dataset

		else:
			raise NotImplementedError
		
	def inference(self, query:str, k:int=10):
		# encode query
		query_emb = self.get_query_embedding(query)

		# model infrence -> response search
		t = time.time()
		topk_distances, topk_indexes = self.faiss_index.search(query_emb, k)
		topk_doc_ids = [[self.mapping_table[i] for i in indexes] for indexes in topk_indexes]
		topk_distances = [list(map(float, dists)) for dists in topk_distances]

		if len(topk_doc_ids) ==1 and len(topk_distances) ==1:
			topk_doc_ids, topk_distances = topk_doc_ids[0], topk_distances[0]

		print('total time: %.2f ms' % ((time.time() - t) * 1000))

		retrieved_responses = []
		if self.text_dict:
			for d_id, dist in zip(topk_doc_ids, topk_distances):
				retrieved_responses.append({"query" : query, "distance" : dist, "response" :self.text_dict[d_id]})

		return retrieved_responses
	
def arg_parser():
	arg_parser = argparse.ArgumentParser(description="Retriever")
	arg_parser.add_argument("--root_dir", dest="root_dir", default="./" , type=str, help="")
	arg_parser.add_argument("--task_name", dest="task_name", default="dr" , type=str, help="")

	# running type - train, eval, predict
	arg_parser.add_argument("--encoder_model", dest="encoder_model", default = "bert_korean" ,  type=str, help="encdoer model type") 
	arg_parser.add_argument("--predict", default = "" ,  dest="predict", type=str, help="predict checkpoint path") # resources/bert-base-uncased
	args = arg_parser.parse_args()
	return args
	
def main(query, test_pool):
	args = arg_parser()
	hparams = vars(args)
	# hparams.update(DATASET_MAP[args.data_type])
	hparams.update(BASE_PARAMS)
	hparams.update(PRETRAINED_MODEL_MAP[args.encoder_model])
	named_tuple_ = namedtuple('named_tuple_' , list(hparams.keys()))
	hparams = named_tuple_(*list(hparams.values()))
	
	predictor = Predictor(hparams, tokenizer=None, pred_ckpt_path=None)	
	predictor.load_and_cache_responses(test_pool=test_pool)
	predictor.get_faiss_index()
	response = predictor.inference(query, k =10)

	for d in response:
		print(d, "\n")
		

if __name__ == "__main__":
	query = "미세 플라스틱 관련 정책이 있을까?"
	test_pool = ["2018년은 바로 서울시가 전국 최초 ‘플라스틱 프리 도시’가 되겠다며 ‘일회용 플라스틱 없는 서울 종합계획’을 발표한 해다. 당시 수도권 지역을 중심으로 쓰레기 수거가 중단된 쓰레기 대란이 발생하면서 내놓은 대책이었다.",
	      		"코로나19 유행 첫해인 2020년 한국인은 276억 개의 비닐봉투를 사용한 것으로 분석됐다. 사회적 거리두기와 배달 증가로 플라스틱 폐기물 양은 코로나19 팬데믹 이전보다 크게 증가했다.",
				"연세대 의대 예방의학교실 조재림 교수 공동 연구팀은 미세먼지 등 대기오염 물질이 대뇌피질의 두께를 얇게 만들어 알츠하이머 치매 위험도를 높인다고 밝혔다.",
				"전세계 바다에 떠다니는 미세플라스틱 입자가 171조개에 달하고 총 무게만 230만톤(t)으로 추정된다는 연구 결과가 나왔다. 연구진은 2005년 이후 해양 플라스틱 오염이 전례없이 증가하고 있어 현재의 상태가 지속된다면 2040년에는 바다로 유입되는 미세플라스틱의 양이 거의 3배에 달할 수 있다고 경고했다.",
				"일교차가 줄고 낮 기온이 15도 이상을 웃돌면서 ‘운동하기 좋은 시기’가 찾아왔다. 특히 최근 건강을 챙기는 문화가 자리잡음과 동시에 ‘오운완(오늘 하루 운동 완료)’이 하나의 트렌드로 영향을 끼치고 있다. ",
				"오는 4월초 출시 예정인 ‘무신사 스탠다드 스포츠’는 운동을 취미로 즐기거나 이제 막 시작하려는 고객을 위한 고품질의 소재로 제작된 기능성 스포츠 웨어다. 특정 종목에 관계없이 다양한 야외 활동과 운동을 즐기면서도 패션과 스타일링에 관심이 높은 고객을 타겟으로 한다.",
				"구글은 21일(현지시간) 블로그에 질문을 하면 답변을 제공하는 독립 웹페이지로 챗봇 '바드'를 출시했다고 발표했다. 다만 이번에 출시한 '바드'는 검색과는 무관한 AI 챗봇 서비스라는 점을 분명히 했다. 그렇지만 구글은 답변 창 아래에 별도의 검색 버튼을 배치해 필요한 부분은 구글 검색도 가능하게 했다. 또 채팅창 하단에는 '잘못된 답변이 나올 수 있다'는 경고 메시지도 배치했다. 다른 답변을 볼 수 있도록 하는 기능도 넣었다.",
				"연세대 에브리타임에서는 “연세우유 크림빵 수익금 덕분”이라는 글이 호응을 얻었다. 연세우유는 상품 홈페이지에 “연세대학교가 운영하는 비영리 학교법인으로, 수익을 모두 장학사업에 사용한다”는 내용을 홍보하고 있다. 이를 근거로 “크림빵으로 돈을 많이 벌었기 때문에 가난한 친구들에게 250만원씩 도와줄 수 있게 됐다”는 글이 설득력을 얻게 된 것이다. 편의점 CU가 연세우유와 협업해 지난해 1월 출시한 연세우유 크림빵 시리즈는 흥행에 성공하면서 출시 1년 만에 누적 판매량 1900만개를 넘었다.",
				"캐나다 통계청에 따르면 올해 1월 1일 기준 캐나다 인구는 전년 대비 105만명 늘어 3957만명을 기록했다. 캐나다 인구가 1년 동안 100만명 이상 증가한 것은 이번이 처음이다. 인구 증가율은 2.7%로 주요 7국(G7) 중 가장 가파르다. 통계청은 “증가율을 유지한다면 약 26년 후엔 인구가 두 배로 늘어날 것”이라고 예상했다. 늘어난 인구의 약 96%는 이민자로 집계됐다.",
				"미국 실리콘밸리은행(SVB)과 시그니처은행 파산으로 금융 시장 전반에 불안감이 퍼지는 가운데 재닛 옐런 미 재무장관이 22일(현지시간) 모든 은행 예금을 보호하려는 것은 아니란 입장을 밝혔다. 전날 '예금 전액 보장 가능성'을 시사했던 것에서 슬그머니 말을 바꾼 태도에 미국 증시는 얼어붙었다.",
				"UCITS ETF’도 순환경제 관련 상품이다. 자원 재생, 제품 생애 연장, 신재생에너지 등 순환경제 비즈니스 모델을 보유한 기업 중 ESPI 리서치의 긍정적인 ESG 등급을 받은 기업들에 투자한다."]
	main(query, test_pool)
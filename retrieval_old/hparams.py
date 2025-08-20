import random
from collections import defaultdict

from transformers import BertModel, BertConfig
from transformers import ElectraModel, ElectraConfig
from sentence_transformers import SentenceTransformer, util

from model.encoders import *
from model.tokenizer import *
# ---------------------------------------
# DATASET
# ---------------------------------------

# # datasets
DATA_KIBO_PARAMS = defaultdict(
	recall_k_list=[1, 2, 5, 10],
	evaluate_data_type="test",
	language="korean",
    data_dir = "/home/wisenut/hijung/DR/data/kibo/30"
    
)

DATA_PAPER_PARAMS = defaultdict(
	recall_k_list=[1, 2, 5, 10, 50],
	language="korean",
    data_dir = "/home/wisenut/hijung/DR/data/aihub/paper"
)

DATA_PATENT_PARAMS = defaultdict(
	num_epochs=10,
	recall_k_list=[1, 2, 5, 10],
	language="korean",
    data_dir = "/home/wisenut/hijung/DR/data/aihub/patent"
)

# mapping
DATASET_MAP = {
	"kibo": DATA_KIBO_PARAMS,
	"aihub_paper": DATA_PAPER_PARAMS,
	"aihub_patent": DATA_PATENT_PARAMS
}


# ---------------------------------------
# ENCODER PARAMS
# ---------------------------------------
BI_ENCODER_PARAMS = defaultdict(
	train_batch_size=64,
	virtual_batch_size=64,
	eval_batch_size=5000,  # 1000
	pred_response_batch_size=15000,

	do_contrastive_learning=True,

	# num_epochs=10,
	learning_rate=5e-05,  # 2e-05,
	projection_size=128,
)

# ---------------------------------------
# PREDICT PARAMS
# ---------------------------------------
PREDICT_PARAMS = defaultdict(
	pred_response_batch_size=15000,
	cached_responses_path="cached_response_emb",
	faiss_search_type="ivf" # ivf / l2
)

# ---------------------------------------
# PRETRAINED MODEL PARAMS (KOREAN)
# ---------------------------------------
WISENUT_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=ElectraConfig,
	pretrained_model=ElectraModel,
	bert_pretrained_dir="resources",
	bert_pretrained="wisenut-koelectra-base-v2-1.7M", # wisenut-koelectra-base-v2-1.7M, wisenut-koelectra-base-v3-788k
	bert_checkpoint_path="electra_base_512_model.bin",
	tokenizer_type="wisenut",
)

WISENUT_SYLLABLE_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=ElectraConfig,
	pretrained_model=ElectraModel,
	bert_pretrained_dir="resources",
	bert_pretrained="electra-base-v4-1400k",
	bert_checkpoint_path="electra_base_512_model.bin",
	tokenizer_type="ko-syllable",
)

BERT_FP_FROM_WISENUT_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=BertConfig,
	pretrained_model=BertModel,
	bert_pretrained_dir="resources",
	bert_pretrained="bert-fp-from-wisenut",
	bert_checkpoint_path="bert.pt",
	tokenizer_type="wisenut",
)

ELECTRA_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=ElectraConfig,
	pretrained_model=ElectraModel,
	bert_pretrained_dir="",
	bert_pretrained="monologg/koelectra-base-discriminator",
	bert_checkpoint_path="",
	tokenizer_type="electra",
)

BERT_FP_FROM_ELECTRA_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=BertConfig,
	pretrained_model=BertModel,
	bert_pretrained_dir="resources",
	bert_pretrained="bert-fp-from-electra",
	bert_checkpoint_path="bert.pt",
	tokenizer_type="electra",
)

BERT_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=BertConfig,
	pretrained_model=BertModel,
	bert_pretrained_dir="",
	bert_pretrained="klue/bert-base",
	bert_checkpoint_path="",
	tokenizer_type="bert",
)

BERT_FP_FROM_BERT_KOREAN_MODEL_PARAMS = defaultdict(
	pretrained_config=BertConfig,
	pretrained_model=BertModel,
	bert_pretrained_dir="resources",
	bert_pretrained="bert-fp-from-bert",
	bert_checkpoint_path="bert.pt",
	tokenizer_type="bert",
)


SENTENCE_TRANSFORMER_MODEL_PARAMS = defaultdict(
	bert_pretrained_dir="",
	bert_pretrained = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
	bert_checkpoint_path="",
	max_seq_length  = 512,
	pretrained_model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')
)


# only for korean
# 영어 관련 test 시에는 추가해서 사용해 주시면 됩니다.
PRETRAINED_MODEL_MAP = {
	"wisenut_korean": WISENUT_KOREAN_MODEL_PARAMS,
	"wisenut_syllable_korean": WISENUT_SYLLABLE_KOREAN_MODEL_PARAMS,
	"electra_korean": ELECTRA_KOREAN_MODEL_PARAMS,
	"bert_korean": BERT_KOREAN_MODEL_PARAMS,
	"bert_fp_from_electra_korean": BERT_FP_FROM_ELECTRA_KOREAN_MODEL_PARAMS,
	"bert_fp_from_wisenut_korean": BERT_FP_FROM_WISENUT_KOREAN_MODEL_PARAMS,
	"bert_fp_from_bert_korean": BERT_FP_FROM_BERT_KOREAN_MODEL_PARAMS,
	"sentence_transformer": SENTENCE_TRANSFORMER_MODEL_PARAMS
}

# encoder mapping (23.03.10 hyein)

ENCODER_MAP = {
    'bert' : SIAMESE_ENCODER,
    'sentence_transformer' : KRSentenceBERT0

}




# -------------------------------------------------------------------------
#  BASE PARAMETER MAPPING
# -------------------------------------------------------------------------
BASE_PARAMS = defaultdict(

	# GPU params
	gpu_ids=[0],  # [0,1,2,3]
	device = "cuda",
	# Input params
	train_batch_size=16,  # 32
	eval_batch_size=100,  # 1000
    pred_batch_size=16,
	virtual_batch_size=32,
	# pred_response_batch_size=15000

	#data
	train_cnt = 0,
    valid_cnt = 0,

	# Training BERT params
	learning_rate=2e-05,  # 2e-05
	dropout_keep_prob=0.8,
	num_epochs=20,
	max_gradient_norm=5,
	adam_epsilon=1e-8,
	weight_decay=0.0,
	warmup_steps=0,
	optimizer_type="AdamW",  # Adam, AdamW

	pad_idx=0,
	max_position_embeddings=512,
	num_hidden_layers=12,
	num_attention_heads=12,
	intermediate_size=3072,
	bert_hidden_dim=768,
	attention_probs_dropout_prob=0.1,
	layer_norm_eps=1e-12,

	# Train Model Config
	in_batch = True,
	parade_yn = False,
	do_shuffle_ressel=False,
	parameter_sharing=True,
	task_name="ubuntu",
	do_eot=True,

	max_sequence_len=512,
	max_dialogue_len=320,
	max_response_len=40,

	save_dirpath='checkpoints/',  # /path/to/checkpoints

	load_pthpath="",
	cpu_workers=4,
	tensorboard_step=100,
	evaluate_print_step=10,
    
	recall_k_list=[1, 2, 5, 10],
    random_seed = 4112
	# random_seed=random.sample(range(1000, 10000), 1)[0],  # 3143
)





import pickle5 as pickle
from transformers import RobertaModel, BertModel, AutoModel, AutoConfig
import torch
import torch.nn as nn

def load_pt_model(model_path):
    # model = AutoModel.from_config(config)
    model = AutoModel.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS") # "klue/bert-base", "snunlp/KR-SBERT-V40K-klueNLI-augSTS", "klue/roberta-base"
    model.load_state_dict(torch.load(model_path))
    return model

def load_pickle_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

if __name__ == "__main__":
    klue_bert = "/home/wisenut/hijung/easy_dr/ckpt_pkls/kluebert_dr_1"
    sbert = "/home/wisenut/hijung/easy_dr/ckpt_pkls/sbert_dr_6"
    roberta = "/home/wisenut/hijung/easy_dr/ckpt_pkls/roberta_dr_4"

    pt_model  = load_pt_model(sbert+".pt")
    print(f"pt_model : {pt_model}")

    # # save
    # with open(sbert+".pkl", 'wb') as f:
    #     pickle.dump(pt_model, f, pickle.HIGHEST_PROTOCOL)


    pkl_model  = load_pickle_model(sbert+".pkl")    
    print(f"pkl_model : {pkl_model}")

    text = "음식물처리기는 보통 온도와 습도가 높은 여름철에 잘 팔리지만, 지난달에는 기상관측 이후 가장 따뜻한 봄 날씨가 이어지면서 찾는 손길이 늘어난 것으로 보입니다."
    print(f" pt_model : {pt_model(text)}")
    print(f" pkl_model : {pkl_model(text)}")


import os
import random
import torch
import pickle5 as pickle
from torch.utils.data import Dataset

# in-batch
class RetrieverDataset(Dataset):
    def __init__(self, hparams, tokenizer, split):
        super().__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        self.split = split # train / valid

        # load data
        doc_id_path = os.path.join(hparams.data_dir, "%s_docids.pickle" % (self.split)) # paper_train_docids.pickle
        field_txt_path = os.path.join(hparams.data_dir, "fieldtxts.pickle") # paper_fieldtxts.pickle
        
        with open(doc_id_path, 'rb') as fd:
            docid = pickle.load(fd)
        with open(field_txt_path, 'rb') as ff:
            field_txt = pickle.load(ff)
        
        if self.hparams.train_cnt or self.hparams.valid_cnt:
            max_cnt = self.hparams.train_cnt if self.split == "train" else self.hparams.valid_cnt
        else:
            max_cnt = len(docid)
        self.docid = docid[:max_cnt]
        self.field1, self.field2 = self.get_input_text(self.docid, field_txt)

    def __len__(self):
        return len(self.field1)

    def __getitem__(self, index):
        if not self.hparams.parade_yn:
            field1 = self.text_to_inputids_basic(self.field1[index]) # |batch_size, embedding_size|
            field2 = self.text_to_inputids_basic(self.field2[index])

        else:
            field1 = self.text_to_inputids_parade(self.field1[index]) # |batch_size, max_paragraph_n, embedding_size|
            field2 = self.text_to_inputids_parade(self.field2[index])           

        return {'field1': field1, 'field2': field2}

    def get_input_text(self, docid, field_txt):
        field1, field2 = [], []
        for i in range(len(docid)):
            doc_id_ = docid[i]
            category_filed_path = os.path.join(self.hparams.data_dir, "category.pickle") # paper_train_category.pickle            
            with open(category_filed_path,   'rb') as fc:
                category = pickle.load(fc)

            ###############################  1  ########################################
            # field_types = random.sample(["_" + f for f in list(category.keys())], 2)
            # fid_1, fid_2 = doc_id_ + field_types[0], doc_id_ + field_types[1]

            # if (field_txt[fid_1]) and (field_txt[fid_2]):
            #     field1.append(field_txt[fid_1])
            #     field2.append(field_txt[fid_2])

            # -----------------------------------------------------
            # 2. filed[0], [1]이 모두 존재할때까지 random sampling
            count = 400
            while count > 0:
                # category = {'A': 'abstract', 'C': 'conclustion', 'B': 'bodytext'}
                field_types = random.sample(["_" + f for f in list(category.keys())], 2)
                fid_1, fid_2 = doc_id_ + field_types[0], doc_id_ + field_types[1]
                try:
                    field1_txt = field_txt[fid_1]
                    field2_txt = field_txt[fid_2]
                    field1.append(field1_txt)
                    field2.append(field2_txt)
                    break
                except:
                    count-=1

        print(f"------------- field1 cnt : {len(field1)}, field2 cnt: {len(field2)}")


        return field1, field2


    def text_to_inputids_basic(self, text):
        if self.tokenizer:
            inputs = self.tokenizer(text,
                                    max_length = self.hparams.max_sequence_len, # for not parade : 512
                                    padding = 'max_length', 
                                    truncation = True, 
                                    return_tensors = "pt")

            token_ids = torch.squeeze(inputs['input_ids']) # tensor type
            attention_mask = torch.squeeze(inputs['attention_mask']) # tensor type
            token_type_ids = torch.squeeze(inputs['token_type_ids']) # tensor type
            # return {"input_ids":token_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}
            return token_ids, attention_mask, token_type_ids

        else:
            return text

    def text_to_inputids_parade(self, text):
        inputs = self.tokenizer(text, 
                                max_length= self.hparams.seq_window_len, 
                                stride= self.hparams.seq_stride_len,
                                padding='max_length', 
                                truncation=True , 
                                return_overflowing_tokens=True,
                                return_tensors = 'pt')

        paragraph_n = len(inputs['input_ids'])
        max_pragraph_n = self.hparams.max_pragraph_n

        input_ids = inputs['input_ids'] # tensor type
        attention_mask = inputs['attention_mask'] # tensor type
        token_type_ids = inputs['token_type_ids'] # tensor type

        # zero padding 
        if paragraph_n < max_pragraph_n :
            added = torch.cat(tuple([torch.zeros(1,self.hparams.seq_window_len)]*(max_pragraph_n - paragraph_n)), 0).long()  # |3, 225|
            input_ids = torch.cat((input_ids, added), 0)
            attention_mask = torch.cat((attention_mask, added), 0)
            token_type_ids = torch.cat((token_type_ids, added), 0)
        else:
            input_ids = torch.squeeze(input_ids[:max_pragraph_n]) # tensor type
            attention_mask = torch.squeeze(attention_mask[:max_pragraph_n]) # tensor type
            token_type_ids = torch.squeeze(token_type_ids[:max_pragraph_n]) # tensor type
        
        return {"input_ids":input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}


class CashingDataset(Dataset):
    """ 
    - input : field_text {field_id : field_txt}
    -> output : field_embeddings {field_id : field_embedding}

    """
    def __init__(self, hparams, tokenizer, txt_path):
        super().__init__()

        self.hparams = hparams
        self.tokenizer = tokenizer
        self.txt_path = txt_path

        # load data
        with open(txt_path, 'rb') as ff:
            field_txt = pickle.load(ff)
        self.filed_id = list(field_txt.keys())
        self.filed_txt = list(field_txt.values())

    def __len__(self):
        return len(self.filed_id)

    def __getitem__(self, index):
        field_id = self.filed_id[index]
        field_tokenized = self.text_to_inputids_basic(self.filed_txt[index])           

        return {'field_id': field_id, 'field_tokenized': field_tokenized}

    def text_to_inputids_basic(self, text):
        if self.tokenizer:
            inputs = self.tokenizer(text,
                                    max_length = self.hparams.max_sequence_len, # for not parade : 512
                                    padding = 'max_length', 
                                    truncation = True, 
                                    return_tensors = "pt")

            token_ids = torch.squeeze(inputs['input_ids']) # tensor type
            attention_mask = torch.squeeze(inputs['attention_mask']) # tensor type
            token_type_ids = torch.squeeze(inputs['token_type_ids']) # tensor type
            # return {"input_ids":token_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids}
            return token_ids, attention_mask, token_type_ids

        else:
            return text
        
if __name__ == "__main__":
    print("a")
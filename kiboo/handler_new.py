from ts.torch_handler.base_handler import BaseHandler
from collections import defaultdict, OrderedDict
from inspect import signature
from typing import Any, Type, Dict, Optional, List, Tuple, Union
from tqdm import tqdm
from model import *
from collections import defaultdict

import pickle5 as pickle
import numpy as np
import torch
import faiss
import re
import requests
import logging
import json
import os


logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.ERROR)

JSON = Union[Dict[str, Any], List[Any], int, str, float, bool, Type[None]]

_get_document_name = lambda x: x.rsplit("_", 1)[0]
_distance_mappings = OrderedDict([
    ("l2", lambda x: x['distance']),
])


def adjust_layer_name(module):
    return OrderedDict({re.sub("^sbert\.", "", k): v for k, v in module.items()})


class KiboDocRecommendHandler(BaseHandler):
    def __init__(self):
        self.faiss_index = None
        self.model_dir = None
        self.serialized_file = None
        self.data_dir = None
        self.emb_file = None
        self.txt_file = None
        self.manifest = None
        self.request_dest = None
        self.inference_batch_size = 4
        self.faiss_below_to_cpu = None
        self._logger = logging.getLogger(__name__)
        super().__init__()

    @torch.no_grad()
    def _encode(self, texts: List[str],
                batch_size: int = 128,
                return_tensors: str = "pt") -> Union[np.ndarray, torch.Tensor]:

        print(f"[_ENCODE] texts : {texts}]")
        print(f"[_ENCODE] self.model : \n {self.model}]")
        print(f"[_ENCODE] encoding.... ")

        embedding = self.model(texts, batch_size=batch_size)
        
        print("[_ENCODE] device : ", self.device)
        print("[_ENCODE] embedding.device : ", embedding.device)

        print("[_ENCODE] self.model.device : ", self.model.device)

        # embedding = torch.tensor(embedding)
        # embedding = torch.nn.functional.normalize(embedding, dim=-1) # for cosine similarity
        if return_tensors == "np":
            if self.device.type == "cpu":
                return embedding.numpy()
            return embedding.cpu().numpy()
      
        return embedding

    def _get_model(self, model_dir: str, serialized_file: str, model_config: str, device: Optional[str] = None):

        self._logger.info('Get Model Start')
        model_pt_path = os.path.join(model_dir, serialized_file)

        with open(model_pt_path, 'rb') as intp:
            self.model = pickle.load(intp)

        self._logger.info(f"::::MODEL::::\n{self.model}")
        self.model.eval()
        self._logger.info(f"[_GET_MODEL] Model Loaded From {model_pt_path}")

        self._logger.info(f"[_GET_MODEL] Set Model Device : {device}")
        self.model.device = device 
        self._logger.info(f"[_GET_MODEL] Device Setting Finished --> self.model : {self.model.device}")


    def _get_embs(self, data_dir: str, emb_file: str, txt_file: str, batch_size: int):
        self._logger.info('Get Embed Start')
        self._logger.info(f'Batch Size of EMBEDDINGS: {batch_size}')
        emb_path = os.path.join(data_dir, emb_file)
        txt_path = os.path.join(data_dir, txt_file)

        if os.path.isfile(emb_path):  # embs load(if there is embs file)
            with open(emb_path, 'rb') as f:
                self.embedding_dict = pickle.load(f)
            self._logger.info("Embedding File Loaded From {0}".format(emb_path))

        elif os.path.isfile(txt_path):  # embs making(if there is txt file)
            with open(txt_path, 'rb') as f:
                field_texts = pickle.load(f)
            field_emb_dict = {}
            self._logger.info("Text File Loaded From {0}".format(txt_path))
            self._logger.info("Text File to Embed File  {0}".format(txt_path))
            
            
            # f_ids, texts = list(zip(*list(field_texts.items())))
            # outputs = self._encode(texts, return_tensors='np', batch_size=batch_size)
            # for k,v in zip(f_ids, outputs):
            #     field_emb_dict[k] = v
        
            def batch_collator(data, batch_size:int):
                ids, texts = [],[]
                for k,v in data.items():
                    ids.append(k)
                    texts.append(v)
                    if len(ids) == batch_size:
                        yield ids, texts
                        ids, texts = [], []
                if ids:
                    yield ids, texts
            #print("###########################################################################")
            #print("BATCH_SIZE;" , batch_size)
            from math import ceil
            _gen = batch_collator(field_texts, batch_size=1000) #@ TODO : to model_config.json
            for f_ids, texts in tqdm(_gen, total=int(ceil(len(field_texts)/1000))):
                #print("DATA LENGTH:", len(f_ids))
                outputs = self._encode(texts, return_tensors='np', batch_size=batch_size)
                #print("OUTPUTS:", len(outputs))
                #raise ValueError
                for k,v in zip(f_ids, outputs):
                    field_emb_dict[k] = np.expand_dims(v, axis=0)
            
            print(f"--------field_emb_dict : {len(field_emb_dict)}")

            try :
                emb_path = os.path.join(data_dir, emb_file)
                with open(emb_path, 'wb') as f:
                    pickle.dump(field_emb_dict, f, pickle.HIGHEST_PROTOCOL)  # overwrite new

            except :
                print(field_emb_dict.shape)
                self._logger.error("Emb File Save Fail {0}".format(emb_path))


            # emb_path = os.path.join(data_dir, emb_file)
            # with open(emb_path, 'wb') as f:
            #     pickle.dump(field_emb_dict, f, pickle.HIGHEST_PROTOCOL)  # overwrite new

            self.embedding_dict = field_emb_dict
            self._logger.info("Embed File Saved at {0}".format(emb_path))

        else:  # if there is no both
            raise RuntimeError(f"Invalid txt File : {txt_path}")

    # def _get_faiss_index(self, dim: int = 128, device: Optional[str] = None) -> None:
    #     device = device if device is not None else self.map_location
    #     self._logger.info('Start Document Indexing')
    #     self.faiss_index = faiss.IndexFlatL2(dim)  # embeddings.shape[1] is number of features in feature vector

    #     if device == "gpu":
    #         self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)

    #     ids, embeddings = [], []
    #     for k,v in self.embedding_dict.items():
    #         ids.append(k)
    #         if len(v.shape) == 1 and v.shape[0] == dim:
    #             v = np.expand_dims(v, axis=0)
    #         embeddings.append(v)

    #     embeddings = np.concatenate(embeddings, axis=0)

    #     self.mapping_table = {i: _id for i, _id in enumerate(ids)}

    #     if embeddings.ndim > 2:
    #         embeddings = embeddings.squeeze()

    #     self.faiss_index.add(embeddings)  # indexing ready

    #     self.document_pool_size, *_ = embeddings.shape
    #     self._logger.info(f"Indexing Finished : {embeddings.shape}")

    def _get_faiss_index(self, dim: int = 768, device: Optional[str] = None) -> None:
        device = device if device is not None else self.map_location
        self._logger.info('Start Document Indexing')
        self.faiss_index = {}
        cat_ids = defaultdict(list)
        cat_embeddings = defaultdict(list)
        for k, v in self.embedding_dict.items():
            cat = k.split("_")[0]
            if len(v.shape) == 1 and v.shape[0] == dim:
                v = np.expand_dims(v, axis=0)
            cat_ids[cat].append(k)
            cat_embeddings[cat].append(v)

        #faiss_index2 = {}
        #faiss_index3 = {}
        self.mapping_table, self.document_pool_size = {}, 0
        for cat in cat_embeddings:
            self.mapping_table[cat] = {i: _id for i, _id in enumerate(cat_ids[cat])}
            embeddings = np.concatenate(cat_embeddings[cat], axis=0)
            if embeddings.ndim > 2:
                embeddings = embeddings.squeeze()
            if cat not in self.faiss_index:
                self.faiss_index[cat] = faiss.IndexFlatL2(dim)
                #faiss_index2[cat] = faiss.IndexFlatL2(dim)
                #faiss_index3[cat] = faiss.IndexFlatL2(dim)

            self.faiss_index[cat].add(embeddings)
            #faiss_index2[cat].add(embeddings)
            #faiss_index3[cat].add(embeddings)
            self._logger.info(f"[_get_faiss_index] Indexing Finished at the category '{cat}': {embeddings.shape}")
            document_pool_size, *_ = embeddings.shape
            self.document_pool_size += document_pool_size

        if device == "gpu" or self.device.type.startswith("cuda"):
            self._logger.info('Change faiss cpu to gpu...')
            self._logger.info(f'[_get_faiss_index] self.device : {self.device}')
            self._logger.info(f'[_get_faiss_index] device : {device}')
            self._logger.info(f'[_get_faiss_index] torch.cuda.is_available() : {torch.cuda.is_available()}')
            self._logger.info(f'[_get_faiss_index] faiss.get_num_gpus() : {faiss.get_num_gpus()}')
            #previous_dumps_for_raise_errors = {k: faiss.index_cpu_to_all_gpus(v) for k, v in self.faiss_index.items()}
            for k, v in sorted(self.faiss_index.items(), key=lambda x: x[1].ntotal, reverse=True):
                #print("category:", k)
                #print("num_vectors:", v.ntotal)
                try:
                    self.faiss_index[k] = faiss.index_cpu_to_all_gpus(v)
                    self._logger.info(f"faiss index moving to the GPU at the category '{k}'!")
                except:
                    self._logger.info(f"faiss index moving to the CPU at the category '{k}'!")
                    continue


    def initialize(self, context: Any) -> None:
        if context == 'poc_inference' or context == 'poc_explain':
            self.manifest = context.manifest  # torch archive info
            properties = context.system_properties  # config.properties / model / version /configs

            self.map_location = "cuda" if torch.cuda.is_available() \
                                          and properties.get("gpu_id") is not None else "cpu"


            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
                if torch.cuda.is_available() and properties.get("gpu_id") is not None
                else self.map_location
            )


            root_dir = properties.get('model_dir')
            self._logger.info(f"TorchServe tmp Directory Path : {root_dir}")

        else:
            self._logger.info(f"[INITIALIZE] context  ----> {context}")

            # context.manifest = {'createdOn': '26/04/2023 13:33:56', 'runtime': 'python', 
            #                     'model': {'modelName': 'demo', 'handler': 'handler_new.py', 'modelFile': 'model.py', 'modelVersion': '1.0'}, 
            #                     'archiverVersion': '0.6.1'}
            self.manifest = context.manifest  # torch archive info
            #  properties = {'model_dir': '/tmp/models/329b5120dfdb49be985403a7e9b27540', 'gpu_id': 0, 
            #                'batch_size': 1, 'server_name': 'MMS', 'server_version': '0.5.2', 'limit_max_image_pixels': True}
            properties = context.system_properties  # config.properties / model / version /configs

            # self.map_location = "cuda" / "cpu"
            self.map_location = "cuda" if torch.cuda.is_available() \
                                          and properties.get("gpu_id") is not None else "cpu"

            self._logger.info(f"[INITIALIZE] properties.get('gpu_id')  ----> {str(properties.get('gpu_id'))}")

            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id"))
                if torch.cuda.is_available() and properties.get("gpu_id") is not None
                else self.map_location
            )
            self._logger.info(f"[INITIALIZE] self.device : {self.device}")
            root_dir = properties.get('model_dir')

    def server_init(self,
                    model_dir: str,
                    serialized_file: str,
                    data_dir: str,
                    emb_file: str,
                    txt_file: str,
                    model_config: dict,
                    batch_size: int = 128) -> List[JSON]:
        try:

            # print(f"data_dir:{data_dir}, emb_file: {emb_file}, txt_file : {txt_file}")
            self.inference_batch_size = batch_size
            self._logger.info(f'Batch Size of SERVER_INIT: {batch_size}')
            device = model_config.pop("device", None)
            self._get_model(model_dir, serialized_file, model_config=model_config, device=device)  # model load
            self._get_embs(data_dir, emb_file, txt_file, batch_size=batch_size)  # embs load
            self._get_faiss_index(dim=768, device=device)  # embs indexing

            self.model_dir = model_dir
            self.serialized_file = serialized_file
            self.data_dir = data_dir
            self.emb_file = emb_file
            self.txt_file = txt_file

            #@
            # get text dict
            # txt_path = os.path.join(self.data_dir, self.txt_file)
            # self._logger.info(f"txt_path: {txt_path}")
            # if os.path.isfile(txt_path):
            #     with open(txt_path, 'rb') as f:
            #         self.txts_dict = pickle.load(f)
            # self._logger.info("Text File Loaded From {0}".format(txt_path))

            responses = [
                json.dumps({
                    "success": 'True',
                    "updated_paths": {
                        "model_dir": model_dir,
                        "serialized_file": serialized_file,
                        "data_dir": data_dir,
                        "emb_file": emb_file,
                        "txt_file": txt_file
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            responses = [json.dumps({"success": 'False'}, ensure_ascii=False)]
        finally:
            return responses

    def model_load(self, model_dir: str, serialized_file: str, model_config: dict, device: Optional[str] = None) -> List[JSON]:
        try:
            self._get_model(model_dir, serialized_file, model_config=model_config, device=device)

            self.model_dir = model_dir
            self.serialized_file = serialized_file

            responses = [
                json.dumps({
                    "success": 'True',
                    "updated_paths": {
                        "model_dir": model_dir,
                        "serialized_file": serialized_file
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            responses = [json.dumps({"success": 'False'}, ensure_ascii=False)]
        finally:
            return responses

    def indexing(self, mode, data_dir, emb_file, txt_file, request_dest: Optional[str] = None, batch_size: int = 32) -> List[JSON]:
        self._logger.info(f'Batch Size of INDEXING: {batch_size}')

        try:
            self._get_embs(data_dir, emb_file, txt_file, batch_size=batch_size)  # embs load
            self._get_faiss_index(dim=768)  # embs indexing

            if request_dest is not None:
                requests.post(
                    request_dest, data=json.dumps({
                        "success": 'True',
                        "mode": mode,
                        "document_pool_size": '{}'.format(len(self.embedding_dict)),
                        "updated_paths": {
                            "data_dir": data_dir,
                            "emb_file": emb_file,
                            "txt_file": txt_file
                        }
                    }, ensure_ascii=False)
                )
                self._logger.info("Sending Indexing Success Request Successfully completed!!")

            self.data_dir = data_dir
            self.emb_file = emb_file
            self.txt_file = txt_file

            responses = [json.dumps({"result_cd": "000",
                                     "result_msg": "SUCCESS"
                                     }, ensure_ascii=False)]

        except Exception as e:
            self._logger.error(e)

            if request_dest is not None:
                requests.post(
                    request_dest, data=json.dumps({"success": 'False'}, ensure_ascii=False)
                )

                self._logger.info("Sending Indexing Fail Request Successfully completed!!")

            responses = [json.dumps({"result_cd": "500",
                                     "result_msg": "FAIL"
                                     }, ensure_ascii=False)]


        finally:
            return responses

    def inference(self, inputs: Tuple[List[int], List[int], List[str], List[int], List[List[str]]], mode=None, batch_size=128) -> dict or List[JSON]:
        self._logger.info(f'Batch Size of INFERENCE: {batch_size}')

        try:
            batch_ids, query_ids, sentences, top_k, categories = inputs
            self._logger.info(f'[INFERENCE] encoding queries....')
            query_embeddings = self._encode(
                sentences,
                batch_size=min(self.inference_batch_size, batch_size),
                return_tensors="np"
            )
            unique_categories = list(set(c for cats in categories for c in cats))
            topk_distances, topk_doc_ids = [{} for _ in range(len(sentences))], [{} for _ in range(len(sentences))]

            self._logger.info(f'[INFERENCE] faiss searching...')
            for cat in unique_categories:
                if self.faiss_index and cat not in self.faiss_index:
                    for i in range(len(sentences)):
                        topk_distances[i][cat] = []
                        topk_doc_ids[i][cat] = []
                    continue
                _dist, _ids = self.faiss_index[cat].search(query_embeddings, k=min(self.document_pool_size, 2048))

                _ids = [[self.mapping_table[cat][si] for si in sids if si > 0] for sids in _ids]

                # self._logger.info(f'[INFERENCE] faiss search results : {_ids} ')

                _dist = [list(map(float, dists)) for dists in _dist]
                for i, (_dist, _id) in enumerate(zip(_dist, _ids)):
                    topk_distances[i][cat] = _dist
                    topk_doc_ids[i][cat] = _id
            outputs = defaultdict(list)

            self._logger.info(f'[INFERENCE] category filtering... ')

            for batch_id, query_id, topk_id, topk_dist, k, category in zip(
                    batch_ids, query_ids, topk_doc_ids, topk_distances, top_k, categories
            ):

                #print(batch_id, query_id, topk_id, topk_dist, k, category, sep=" / ")
                field_results, unique_results = defaultdict(dict), defaultdict(dict)
                for cat in category:
                    field_results[cat]['result'] = [
                        {"field_id": _i, "distance": _d} for _i, _d in zip(topk_id[cat], topk_dist[cat])
                        if topk_dist != -1
                    ]
                    min_results = {}
                    for dist_id in field_results[cat]['result']:
                        uid = _get_document_name(dist_id["field_id"])
                        if uid in min_results and min_results[uid]['distance'] < dist_id['distance']:
                            continue
                        min_results[uid] = {"unique_id": uid, "distance": dist_id['distance']}

                    if "result" not in unique_results[cat]:
                        unique_results[cat]['result'] = []

                    unique_results[cat]['result'] += list(min_results.values())

                    unique_results[cat]['result'] = sorted(unique_results[cat]['result'], key=_distance_mappings["l2"])[:k]
                    unique_ids = set(u["unique_id"] for u in unique_results[cat]['result'])

                    field_results[cat]['result'] = [v for v in field_results[cat]['result'] if _get_document_name(v["field_id"]) in unique_ids]
                    field_results[cat]['result'] = sorted(field_results[cat]['result'], key=_distance_mappings["l2"])

                    # add_count
                    field_results[cat]['count'] = len(field_results[cat]['result'])
                    unique_results[cat]['count'] = len(unique_results[cat]['result'])

                outputs[batch_id].append({
                    "category_list": category,
                    "query_id": query_id,
                    "topk": k,
                    "field_result": dict(field_results),
                    "unique_result": dict(unique_results)
                })

            return [{"success": "True", "result": v} for k, v in sorted(outputs.items(), key=lambda x: x[0])]

        except Exception as e:
            self._logger.info(f'[INFERENCE] exception occured : {e} ')
            self._logger.error(e)
            return [{"success": 'False'} for _ in range(max(batch_ids) + 1)]

    def preprocess(self, data: List[dict]) -> Tuple[List[int], List[int], List[str], List[int]]:
        batch_ids, doc_ids, sentences, top_k, categories = [], [], [], [], []
        b_append, d_append, s_append, t_append, c_append = batch_ids.append, doc_ids.append, sentences.append, top_k.append, categories.append
        for i, d in enumerate(data):
            k = d.get('top_k', 10)
            c = d.get('categories', '')
            for doc in d.get('documents', []):
                doc_id = doc.get("doc_id")
                doc_content = doc.get("document")
                if doc_id and doc_content:
                    b_append(i)
                    d_append(doc_id)
                    s_append(doc_content)
                    t_append(k)
                    c_append(c)

        return batch_ids, doc_ids, sentences, top_k, categories

    def postprocess(self, results: Union[dict, List[JSON]]) -> List[JSON]:
        result = [json.dumps(result, ensure_ascii=False) for result in results]  # return type must list
        return result

    def add(self, added: List[dict], batch_size: int = 128) -> List[JSON]:
        self._logger.info('Add Embed, Txt Start')
        self._logger.info(f'Batch Size of ADD: {batch_size}')

        batch_ids, field_ids, sentences = [], [], []
        b_append, f_append, s_append = batch_ids.append, field_ids.append, sentences.append

        for i, d in enumerate(added):
            for doc in d.get('documents', []):
                field_id = doc.get("field_id")
                field_text = doc.get("field_text")
                if field_id and field_text:
                    b_append(i)
                    f_append(field_id)
                    s_append(field_text)

        # get embedding dict
        emb_path = os.path.join(self.data_dir, self.emb_file)
        self._logger.info(f"emb_path: {emb_path}")
        if not self.embedding_dict:
            self._get_embs(self.data_dir, self.emb_file, self.txt_file, batch_size=batch_size)

        # get text dict
        txt_path = os.path.join(self.data_dir, self.txt_file)
        self._logger.info(f"txt_path: {txt_path}")
        if os.path.isfile(txt_path):
            with open(txt_path, 'rb') as f:
                txts_dict = pickle.load(f)
            self._logger.info("Text File Loaded From {0}".format(txt_path))

        try:
            added_embeddings = self._encode(sentences, batch_size=batch_size, return_tensors="np")

            if len(added_embeddings.shape) == 2:
                added_embeddings = np.expand_dims(added_embeddings, axis=1)

            self._logger.info('Added Text Encoding Finished')
            self._logger.info("Added Field Id List :  {0}".format(field_ids))

            valid_ids, duplicated = [], []
            for i, fid in enumerate(field_ids):
                if fid not in self.embedding_dict:
                    self.embedding_dict[fid] = added_embeddings[i]
                    txts_dict[fid] = sentences[i]
                    valid_ids.append(fid)
                else:
                    duplicated.append(str(fid))

            with open(emb_path, 'wb') as f:
                pickle.dump(self.embedding_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Embed File Saved at {0}".format(emb_path))

            with open(txt_path, 'wb') as f:
                pickle.dump(txts_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Txt File Saved at {0}".format(txt_path))

            return [
                json.dumps({
                    "success": 'True',
                    "added_count": len(valid_ids),
                    "duplication_failure": ", ".join(duplicated),
                    "document_pool_size": '{}'.format(len(self.embedding_dict)),
                    "updated_paths": {
                        "data_dir": self.data_dir,
                        "emb_file": emb_path,
                        "txt_file": txt_path
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            return [json.dumps({"success": 'False'}, ensure_ascii=False)]

    def delete(self, deleted: List[dict], batch_size:int = 32) -> List[JSON]:
        self._logger.info('Delete Embed, Txt Start')
        self._logger.info(f'Batch Size of DELETE: {batch_size}')
        batch_ids, doc_ids = [], []
        b_append, d_append = batch_ids.append, doc_ids.append

        for i, d in enumerate(deleted):
            doc_ids = d.get('doc_ids', [])
            for doc_id in doc_ids:
                b_append(i)
                d_append(doc_id)


        # get embedding dict
        emb_path = os.path.join(self.data_dir, self.emb_file)
        self._logger.info(f"emb_path: {emb_path}")
        if not self.embedding_dict:
            self._get_embs(self.data_dir, self.emb_file, self.txt_file, batch_size=batch_size)


        # get text dict
        txt_path = os.path.join(self.data_dir, self.txt_file)
        if os.path.isfile(txt_path):
            with open(txt_path, 'rb') as f:
                txts_dict = pickle.load(f)
            self._logger.info("Text File Loaded From {0}".format(txt_path))

        try:
            total_field_ids = list(self.embedding_dict.keys())
            related_field_ids = [id for id in total_field_ids if _get_document_name(id) in doc_ids]
            deleted_ids = []
            for f_id in related_field_ids:
                try:
                    del self.embedding_dict[f_id]
                    del txts_dict[f_id]
                    deleted_ids.append(f_id)
                except:
                    pass

            self._logger.info("Deleted Field Id List :  {0}".format(deleted_ids))

            with open(emb_path, 'wb') as f:
                pickle.dump(self.embedding_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Embed File Saved at {0}".format(emb_path))

            with open(txt_path, 'wb') as f:
                pickle.dump(txts_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Txt File Saved at {0}".format(txt_path))

            return [
                json.dumps({
                    "success": 'True',
                    "deleted_count": len(deleted_ids),
                    "document_pool_size": '{}'.format(len(self.embedding_dict)),
                    "updated_paths": {
                        "data_dir": self.data_dir,
                        "emb_file": emb_path,
                        "txt_file": txt_path
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            return [json.dumps({"success": 'False'}, ensure_ascii=False)]

    #@ hijung 12.12
    def add_file(self, add_file_path, added: List[dict], batch_size: int = 128) -> List[JSON]:
        self._logger.info('Add File - Embed, Txt Start')
        self._logger.info(f'Batch Size of ADD: {batch_size}')
        self._logger.info(f"faiss_below_to_cpu:{self.faiss_below_to_cpu}")
        if self.faiss_below_to_cpu:
            for k,v in self.faiss_index.items():
                try:
                    self.faiss_index[k] = faiss.index_gpu_to_cpu(v)
                    self._logger.info(f"The Faiss Index({k}) has been allocated from GPU to CPU")
                except:
                    continue

        with open(add_file_path, 'rb') as f:
            added = pickle.load(f)
        self._logger.info("Add File Loaded From {0}".format(add_file_path))

        #@ hijung 12.12 이하 add와 동일
        field_ids, sentences = [], []
        f_append, s_append = field_ids.append, sentences.append


        for doc in added.get('documents', []):
            field_id = doc.get("field_id")
            field_text = doc.get("field_text")
            if field_id and field_text:
                f_append(field_id)
                s_append(field_text)

        # get embedding dict
        emb_path = os.path.join(self.data_dir, self.emb_file)
        self._logger.info(f"emb_path: {emb_path}")
        if not self.embedding_dict:
            self._get_embs(self.data_dir, self.emb_file, batch_size=batch_size)

        # get text dict
        txt_path = os.path.join(self.data_dir, self.txt_file)
        self._logger.info(f"txt_path: {txt_path}")
        if os.path.isfile(txt_path):
            with open(txt_path, 'rb') as f:
                txts_dict = pickle.load(f)
            self._logger.info("Text File Loaded From {0}".format(txt_path))

        try:
            added_embeddings = self._encode(sentences, batch_size=batch_size, return_tensors="np")

            if len(added_embeddings.shape) == 2:
                added_embeddings = np.expand_dims(added_embeddings, axis=1)

            self._logger.info('Added Text Encoding Finished')
            self._logger.info("Added Field Id List :  {0}".format(field_ids))

            valid_ids, duplicated = [], []
            for i, fid in enumerate(field_ids):
                if fid not in self.embedding_dict:
                    self.embedding_dict[fid] = added_embeddings[i]
                    txts_dict[fid] = sentences[i]
                    valid_ids.append(fid)
                else:
                    duplicated.append(str(fid))

            with open(emb_path, 'wb') as f:
                pickle.dump(self.embedding_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Embed File Saved at {0}".format(emb_path))

            with open(txt_path, 'wb') as f:
                pickle.dump(txts_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Txt File Saved at {0}".format(txt_path))

            if self.faiss_below_to_cpu:
                for k, v in self.faiss_index.items():
                    try:
                        self.faiss_index[k] = faiss.index_cpu_to_all_gpus(v)
                        self._logger.info(f"The Faiss Index({k}) has been allocated from CPU to GPU")
                    except:
                        continue

            return [
                json.dumps({
                    "success": 'True',
                    "added_count": len(valid_ids),
                    "duplication_failure": ", ".join(duplicated),
                    "document_pool_size": '{}'.format(len(self.embedding_dict)),
                    "updated_paths": {
                        "data_dir": self.data_dir,
                        "emb_file": emb_path,
                        "txt_file": txt_path
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            return [json.dumps({"success": 'False'}, ensure_ascii=False)]

    #@ hijung 12.12
    def delete_file(self, delete_file_path, deleted: List[dict], batch_size:int = 32) -> List[JSON]:
        self._logger.info('Delete File - Embed, Txt Start')
        self._logger.info(f'Batch Size of DELETE: {batch_size}')

        with open(delete_file_path, 'rb') as f:
            deleted = pickle.load(f)
        self._logger.info("Delete File Loaded From {0}".format(delete_file_path))

        
        doc_ids = []
        deleted = deleted.get('doc_ids', [])
        for doc_id in deleted:
            doc_ids.append(doc_id)


        # get embedding dict
        emb_path = os.path.join(self.data_dir, self.emb_file)
        self._logger.info(f"emb_path: {emb_path}")
        if not self.embedding_dict:
            self._get_embs(self.data_dir, self.emb_file, batch_size=batch_size)


        # get text dict
        txt_path = os.path.join(self.data_dir, self.txt_file)
        if os.path.isfile(txt_path):
            with open(txt_path, 'rb') as f:
                txts_dict = pickle.load(f)
            self._logger.info("Text File Loaded From {0}".format(txt_path))

        try:
            total_field_ids = list(self.embedding_dict.keys())
            related_field_ids = [id for id in total_field_ids if id.rsplit("_", 1)[0] in doc_ids]
            deleted_ids = []
            for f_id in related_field_ids:
                try:
                    del self.embedding_dict[f_id]
                    del txts_dict[f_id]
                    deleted_ids.append(f_id)
                except:
                    pass

            self._logger.info("Deleted Field Id List :  {0}".format(deleted_ids))

            with open(emb_path, 'wb') as f:
                pickle.dump(self.embedding_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Embed File Saved at {0}".format(emb_path))

            with open(txt_path, 'wb') as f:
                pickle.dump(txts_dict, f, pickle.HIGHEST_PROTOCOL)
            self._logger.info("Txt File Saved at {0}".format(txt_path))

            return [
                json.dumps({
                    "success": 'True',
                    "deleted_count": len(deleted_ids),
                    "document_pool_size": '{}'.format(len(self.embedding_dict)),
                    "updated_paths": {
                        "data_dir": self.data_dir,
                        "emb_file": emb_path,
                        "txt_file": txt_path
                    }
                }, ensure_ascii=False)
            ]

        except Exception as e:
            self._logger.error(e)
            return [json.dumps({"success": 'False'}, ensure_ascii=False)]




    def request_decode(self, data: List[Union[dict, JSON]]) -> List[dict]:
        if isinstance(data[0]['body'], bytearray):
            if isinstance(data[0], str):
                data = [d['body'] for d in data]
            else:
                data = [d['body'].decode(encoding='utf-8') for d in data]
        else:
            return [d['body'] for d in data]
        return [json.loads(d) for d in data]

    def handle(self, data: List[Any], context: Any) -> List[JSON]:
        self.context = context
        data = self.request_decode(data)

        try:
            response = None
            if not self._is_explain() or context == 'poc_inference':
                self._logger.info("Call Preprocessing, Inference, Postprocessing API")
                data = self.preprocess(data)
                data = self.inference(data, batch_size=self.inference_batch_size)
                response = self.postprocess(data)

            elif self._is_explain() or context == 'poc_explanation':
                mode = data[0]["mode"]
                batch_size = data[0].get("batch_size", 32)

                if mode == "add":
                    self._logger.info("Call Document Add API")
                    response = self.add(data, batch_size=min(self.inference_batch_size, batch_size))

                elif mode == "delete":
                    self._logger.info("Call Document Delete API")
                    response = self.delete(data, batch_size=min(self.inference_batch_size, batch_size))

                elif mode == "add_file":
                    add_file_path = data[0].get("add_file", 32) # @hijung 12.12
                    self._logger.info("Call Document Add File API")
                    response = self.add_file(
                        add_file_path,
                        data,
                        batch_size=min(self.inference_batch_size, batch_size)
                    )

                elif mode == "delete_file":
                    delete_file_path = data[0].get("delete_file", 32) # @hijung 12.12
                    self._logger.info("Call Document Delete File API")
                    response = self.delete_file(delete_file_path, data, batch_size=min(self.inference_batch_size, batch_size))
                else:
                    config = data[0]["config"]
                    model_dir = config.get("ckpt_dir")
                    model_config = config.get("model_config", {})

                    if model_config is None:
                        model_config = {}

                    if isinstance(model_config, str):
                        path = os.path.join(model_dir, model_config)
                        if os.path.isfile(path):
                            with open(path, "r", encoding="utf-8") as r:
                                model_config = json.load(r)
                        else:
                            raise FileNotFoundError(f"The config file not exist in {path}")

                    data_dir = config.get("data_dir")
                    request_dest = model_config.get("request_dest", None)
                    init_batch_size = model_config.get("batch_size", 32)
                    serialized_file = config.get("ckpt_file")
                    emb_file, txt_file = config.get("emb_file"), config.get("txt_file")
                    self.faiss_below_to_cpu = model_config.get("faiss_below_to_cpu", False)

                    if mode == "server_init":
                        self._logger.info("Call Server Init API")
                        response = self.server_init(
                            model_dir=model_dir,
                            serialized_file=serialized_file,
                            data_dir=data_dir,
                            emb_file=emb_file,
                            txt_file=txt_file,
                            model_config=model_config,
                            batch_size=init_batch_size
                        )

                    elif mode == "model_load":
                        self._logger.info("Call Model Load API")
                        response = self.model_load(
                            model_dir=model_dir,
                            serialized_file=serialized_file,
                            model_config=model_config,
                            device=model_config.get("device"),
                            batch_size=min(self.inference_batch_size, batch_size)
                        )

                    elif mode in ("indexing", "indexing_deploy"):

                        self._logger.info("Call Document Indexing API")
                        response = self.indexing(
                            mode,
                            data_dir=data_dir,
                            emb_file=emb_file,
                            txt_file=txt_file,
                            request_dest=request_dest,
                            batch_size=min(self.inference_batch_size, batch_size),
                        )


                    else:
                        self._logger.error("Invalid Explanation Config")
                        response = [json.dumps(data[0], ensure_ascii=False)]

            return response

        except Exception as e:
            self._logger.error(e)
            return [json.dumps({"success": 'False'}, ensure_ascii=False)]


if __name__ =="__main__":
    print("A")

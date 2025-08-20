import requests
import json
import pprint
pp = pprint.PrettyPrinter(indent=4)

# ----------------  < CHANGE VARIABLES HERE > ----------------------------------------------------------------
# model
model_config = "model_config.json"
ckpt_dir = "/data/kibo_external/deployment_1208/ckpt_dir"
ckpt_file = "KRSentenceBERT0.pkl"

# data
data_dir = "/home/wisenut/kibo_test/deployment/data_dir/small_data"  # small data : 1만
# data_dir = "/data/kibo_external/data_deployment_1202/deploy_data2" # big data : 300만
emb_file = 'emb.pickle'
txt_file = 'text.pickle'
new_emb_file = 'emb.pickle' # indexing_deploy 모드에서 생성할 emb 파일

add_delete_dir = "/home/wisenut/kibo_test/deployment/data_dir/small_data"
add_file_dir = add_delete_dir + "/add.pickle"
delete_file_dir = add_delete_dir + "/delete.pickle"

# etc
PORT = "9090" # <주의> config.properties의 설정과 일치해야함
mar_name = "demo" # (e.g) {demo_kibo_38}.mar 

# batch_size
inference_bs = 128
indexing_bs = 128
add_bs = 64
delete_bs = 1

# #######################################################################################
# # 0. server init test
# #######################################################################################
print("======== 0. server init test ========", end="\n\n")
data = json.dumps({
                     "mode": "server_init",
                     "config": {
                                 "ckpt_dir" : ckpt_dir,
                                 "model_config": model_config,
                                 "ckpt_file": ckpt_file,
                                 "data_dir": data_dir,
                                 "emb_file": emb_file,
                                 "txt_file": txt_file
                                 }
                 })

response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
pp.pprint(response.json())

#######################################################################################
# 1. inference test (1)
#######################################################################################
print("\n")
print("======== 1-(1) inference test ========")
input_q = json.dumps({
                    "top_k": 3,
                    "batch_size": inference_bs,
                    "categories": ["ntisProject", "ntisRpaper", "patentNormal", "patentTrust", "patentUM"],
                    "documents": [
                                    {
                                        "doc_id": 'patentTrust_1020100036011_1',
                                        "document": "평형기능 보완용 두피 접촉식 진동 자극장치"
                                    },
                                    {
                                        "doc_id": 'ntisProject_1345149431-314-82-09226_6',
                                        "document": "본 연구과제는 다양한 환경적 요인이 유기박막트랜지스터(OTFT)의 특성 및 안정성에 미치는 영향을 top gate 구조의 OTFT를 이용하여 연구하고자 한다"
                                    },
                                    {
                                        "doc_id": 'NEWS_environmental_pollution_C1',
                                        "document": "미세 플라스틱 환경오염 문제"
                                    }
                                ]
                    })
response = requests.post(f'http://127.0.0.1:{PORT}/predictions/{mar_name}', data=input_q)
pp.pprint(response.json())


######################################################################################
# 1. inference test (2)
######################################################################################
# print("\n")
# print("======== 1-(2) inference test - not valid categories ========")
# input_q = json.dumps({
#     "top_k": 3,
#     "batch_size": inference_bs,
#     "categories": ["dfdsf", "dsfjioq"],
#     "documents": [
#         {
#             "doc_id": 'query_0000',
#             "document": "설비전체가 아닌 유닛의 개발을 통해 유닛 솔루션 제공하는 사업 계획"
#         },
#         {
#             "doc_id": 'query_0001',
#             "document": "VR 기술로 뛰어난 몰입감의 가상현실 환경을 제공하고 VR장치들을 활용하여 가상의 재활 콘텐츠와 상호작용을 통해 재활 운동의 효과를 얻을 수 있는 VR시스템"
#         }
#     ]
# })
# response = requests.post(f'http://127.0.0.1:{PORT}/predictions/{mar_name}', data=input_q)
# pp.pprint(response.json())


######################################################################################
# 1. inference test (3)
######################################################################################
# print("\n")
# print("======== 1-(3) inference test - many and long========")
# input_q = json.dumps({
#     "top_k": 500,
#     "batch_size": inference_bs,
#     "categories": ["ntisProject", "ntisRpaper"],
#     "documents": [
#         {
#             "doc_id": 'query_0000',
#             "document": "설비전체가 아닌 유닛의 개발을 통해 유닛 솔루션 제공하는 사업 계획"
#         },
#         {
#             "doc_id": 'query_0001',
#             "document": "VR 기술로 뛰어난 몰입감의 가상현실 환경을 제공하고 VR장치들을 활용하여 가상의 재활 콘텐츠와 상호작용을 통해 재활 운동의 효과를 얻을 수 있는 VR시스템"
#         },
#         {
#             "doc_id": 'query_0002',
#             "document": "cnc 가공공정, 이동공정 등을 위한 협동 로봇의 그리퍼 설계 및 제조 반도체 제조용 기계 제조업 달리 분류되지 않는 반도체장비 반도체 장치; 다른 곳에 속하지 않는 전기적 고체 장치(측정에 반도체 장치를 사용 G01; 저항 일반 H01C; 자석, 인덕터, 변성기 H01F; 컨덴서 일반 H01G; 전해장치 H01G 9/00; 전지, 축전지 H01M; 도파관, 도파관형의 공진기 또는 선로 H01P; 전선접속기, 집전장치 H01R; 유도방출 소자 H01S; 전기기계적 공진기 H03H; 확성기, 마이크로폰, 축음기의 픽엎 또는 음향의 전기기계적 변환기 H04R; 전기적 광 자동화 로봇 자동화 협동로봇 자동배출 automation robot collaborative robot auto-dischargeable 52시간 노동환경 변화에 따른 협동로봇을 이용하여 자동화기술을 개발하고자 함. 로봇 그리퍼 설계 및 생산 조립공정, cnc 가공공정, 이동공정 등 노동집약적 공정에 투입하여 52시간에 제약없이 생산을 원하는 기업 로봇그리퍼"

#         },
#         {
#             "doc_id": 'query_0003',
#             "document": "스마트공장 솔루션 전자감지장치 제조업 위험감지/모니터링 장비 금속 분말의 가공; 금속분말로부터 물품의 제조; 금속분말의 제조(분말야금에 의한 합금의 제조 C22C: ); 금속 분말에 적용되는 특수 장치 또는 장비 모니터링 시스템 변화 감지 monitoring system transfer detect 제조, 방산, 원전 관련 IOT로 진동 감지 후 파형분석(AI,알고리즘)을 통해 고장 예측, 수요예측, 설비에 대해 진동파형분석을 통해 아날로그 신호를 디지털 신호로 감지해서 고장을 찾는 사전 대응하는 형태. - 데이터를 수집하는 장치 - 데이터를 송출하는 장치 - 무성통신장치로 진동 감지 모니터링 하는 장치 - 진동 감지 시스템을 기계 장비 혹은 공장내 설치하는 시스템 구축시 활용 예정 - 산학연 R&D ."

#         },
#         {
#             "doc_id": 'query_0004',
#             "document": "휴대용 생물입자 실시간 검출 배터리 Portable Biological particles Real-time detection Battery used"
#         },
#         {
#             "doc_id": 'query_0005',
#             "document": "차량용 투명 LED 플렉서블 디스플레이 및 그 조작(동작)제어 기술"
#         },
#         {
#             "doc_id": 'query_0006',
#             "document":  "- 무선 센서 네트워크의 가용 통신 전송 범위 : 28m 이상 - 스마트공장용 시스템과 연계 가능 - 스마트공장용 IoT에 적용 가능"
#         },
#         {
#             "doc_id": 'query_0007',
#             "document": "- 주요 예상 수요처 : 광고 플랫폼을 이용하는 전국 가맹점 - 아이디어에서 사업화까지 소요 일정 : 6개월 - 사업화 예상비용 등 : 2억원"
#         }
#     ]
# })
# response = requests.post(f'http://127.0.0.1:{PORT}/predictions/{mar_name}', data=input_q)
# pp.pprint(response.json())


# ######################################################################################
# # 2. indexing test (1)
# ######################################################################################
print("\n")
print("======== 2-(1). indexing test ========")
data = json.dumps({
    "mode": "indexing",
    "batch_size": indexing_bs,
    "config": {
        "ckpt_dir": ckpt_dir,
        "model_config": model_config,
        "ckpt_file": ckpt_file,
        "data_dir": data_dir,
        "emb_file": emb_file,
        "txt_file": txt_file
}
})
response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
pp.pprint(response.json())


# ######################################################################################
# # 2. indexing test(2)
# ######################################################################################
# print("\n")
# print("======== 2-(2). indexing test (deploy) ========")
# data = json.dumps({
#     "mode": "indexing_deploy",
#     "batch_size": indexing_bs,
#     "config": {
#         "ckpt_dir": ckpt_dir,
#         "model_config": model_config,
#         "ckpt_file": ckpt_file,
#         "data_dir": data_dir,
#         "emb_file": new_emb_file,
#         "txt_file": txt_file
# }
# })
# response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
# pp.pprint(response.json())

# #######################################################################################
# ## 3. add test
# #######################################################################################
# input
print("\n")
print("======== 3. add test ========")
data = json.dumps({
    "mode": "add",
    "batch_size": add_bs,
    "documents": [
        {
            "field_id": 'PT_ID0001_Y',
            "field_text": "사전훈련 언어모델에 관한 연구"},
        {
            "field_id": 'PK_ID0001_C2',
            "field_text": "최근 딥 러닝의 뛰어난 성능이 괄목할만한 성과를 얻으면서, 인공지능이 IT 분야의 중요 키워드로 떠오르고 있습니다. 자연어 처리는 기계에게 인간의 언어를 이해시킨다는 점에서 인공지능에서 가장 의미있는 연구 분야이면서도 아직도 정복되어야 할 산이 많은 분야입니다. 이 책에서는 자연어 처리에 필요한 전처리 방법, 딥 러닝 이전 주류로 사용되었던 통계 기반의 언어 모델, 그리고 자연어 처리의 비약적인 성능을 이루어낸 딥 러닝을 이용한 자연어 처리에 대한 전반적인 지식을 다룹니다. 이번 챕터는 자연어 처리 공부를 시작하기에 앞서 기본적인 셋팅 방법과 앞으로 공부하게 될 머신 러닝에 대한 전체적인 워크플로우에 대해서 다룹니다."
        },
        {
            "field_id": 'PK_ID0001_C3',
            'field_text': "SBERT를 이용한 토픽 모델인 BERTopic은 별도 논문은 나오지 않은 모델이지만, github에서 2k 이상의 스타를 받았을만큼 굉장히 주목받고 있는 토픽 모델입니다. 개발자는 BERTopic이 LDA를 대체할 수 있을만큼의 기술이라고 확신을 얻었다고 주장합니다. 여기서는 BERT 기반의 토픽 모델링 구현체인 BERTopic의 간단한 사용 방법에 대해서 다룹니다."
        },
        {
            "field_id": 'PK_ID0002_C5',
            'field_text': "SBERT를 이용한 토픽 모델인 BERTopic은 별도 논문은 나오지 않은 모델이지만, github에서 2k 이상의 스타를 받았을만큼 굉장히 주목받고 있는 토픽 모델입니다. 개발자는 BERTopic이 LDA를 대체할 수 있을만큼의 기술이라고 확신을 얻었다고 주장합니다. 여기서는 BERT 기반의 토픽 모델링 구현체인 BERTopic의 간단한 사용 방법에 대해서 다룹니다."
        },
        {
            "field_id": 'PK_ID0003_C6',
            'field_text': "SBERT를 이용한 토픽 모델인 BERTopic은 별도 논문은 나오지 않은 모델이지만, github에서 2k 이상의 스타를 받았을만큼 굉장히 주목받고 있는 토픽 모델입니다. 개발자는 BERTopic이 LDA를 대체할 수 있을만큼의 기술이라고 확신을 얻었다고 주장합니다. 여기서는 BERT 기반의 토픽 모델링 구현체인 BERTopic의 간단한 사용 방법에 대해서 다룹니다."
        }
    ]
})
response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
pp.pprint(response.json())


# #######################################################################################
# #  4. delete test
# #######################################################################################
# # input
# print("\n")
# print("======== 4. delete test ========")
# data = json.dumps({
#     "mode": "delete",
#     "batch_size": delete_bs,
#     "doc_ids": ['PT_ID0001', 'PK_ID0002', 'PK_ID0003']

# })

# response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
# pp.pprint(response.json())


# ######################################################################################
# # 5. add_file test
# ######################################################################################
# # input
# print("\n")
# print("======== 3. add_file test ========")
# data = json.dumps({
#     "mode": "add_file",
#     "batch_size": add_bs,
#     "add_file": add_file_dir
# })
# response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
# pp.pprint(response.json())

# #######################################################################################
# ## 6. delete_file test
# #######################################################################################
# # input
# print("\n")
# print("======== 4. delete_file test ========")
# data = json.dumps({
#     "mode": "delete_file", 
#     "batch_size": delete_bs,
#     "delete_file": delete_file_dir

# })

# response = requests.post(f'http://127.0.0.1:{PORT}/explanations/{mar_name}', data=data)
# pp.pprint(response.json())

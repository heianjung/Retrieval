import os
import pickle 

# 불러와서 내용 확인
# txt_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/sample_data/text.pickle" 
# emb_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/sample_data/emb.pickle" 

txt_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/empty_data/text.pickle" 
emb_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/empty_data/emb.pickle" 
with open(txt_path, 'rb') as f:
    field_texts = pickle.load(f)

with open(emb_path, 'rb') as f:
    field_embs = pickle.load(f)

print(f"field_texts : {field_texts}")
# print(f"field_embs : {field_embs}")


# # 저장
# txt_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/empty_data/text.pickle" 
# emb_path = "/home/wisenut/kibo_test/deployment/ais_data_dir/empty_data/emb.pickle" 

# with open(txt_path, 'wb') as file:
#     pickle.dump({}, file)

# with open(emb_path, 'wb') as file:
#     pickle.dump({}, file)

if __name__ == "__main__":
    print("a")
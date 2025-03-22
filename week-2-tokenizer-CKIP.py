import pandas as pd
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
from tqdm import tqdm
from urllib.parse import urlparse

# 啟用 tqdm 進度條功能
tqdm.pandas()

df = pd.read_csv('all-boards.csv')
# Initialize drivers
print("Initializing drivers ... WS")
# device=0 表示使用第一個 GPU，如果沒有 GPU，則可以設置為 -1 以使用 CPU）
ws_driver = CkipWordSegmenter(model="albert-base", device=-1)
# 初始化詞性標註和命名實體識別驅動程序：
print("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="albert-base", device=-1)
print("Initializing drivers ... NER")
ner_driver = CkipNerChunker(model="albert-base", device=-1)
print("Initializing drivers ... all done")
print()

def clean(sentence_ws, sentence_pos):
  short_with_pos = []
  short_sentence = []
#   Caa: 對等連接詞
#   Cab: 連接詞，如：等等
#   Cba: 連接詞，如：的話
#   Cbb: 關聯連接詞
#   P: 介詞
#   T: 語助詞
  stop_pos = set(['P', 'T', 'Caa', 'Cab', 'Cba', 'Cbb:'])
  for word_ws, word_pos in zip(sentence_ws, sentence_pos):
    # 只留名詞和動詞
    # is_N_or_V = word_pos.startswith("V") or word_pos.startswith("N")
    # 去掉名詞裡的某些詞性
    is_not_stop_pos = word_pos not in stop_pos
    # 只剩一個字的詞也不留
    is_not_one_charactor = not (len(word_ws) == 1)
    # 組成串列
    if is_not_stop_pos and is_not_one_charactor:
      short_with_pos.append(f"{word_ws}({word_pos})")
      short_sentence.append(f"{word_ws}")
  return (",".join(short_sentence), ",".join(short_with_pos))

# 提取版名
def extract_board(link):
    try:
        # 解析 URL，取得路徑部分
        path_parts = urlparse(link).path.split('/')
        if len(path_parts) > 2 and path_parts[1] == "bbs":
            return path_parts[2]  # 版名是第 3 個元素
    except:
        return None  # 如果解析失敗則回傳 None

def main():
    text = df["title"].dropna()
    ws = ws_driver(text.tolist())
    pos = pos_driver(ws)
    ner = ner_driver(text.tolist())
    # for sentence, sentence_ws, sentence_pos, sentence_ner in zip(text, ws, pos, ner):
    #     (short, res) = clean(sentence_ws, sentence_pos)
        # print("斷詞後：")
        # print(short)
        # print("斷詞後+詞性標注：")
        # print(res)
    results = [
        {
            "title": clean(sentence_ws, sentence_pos)[0],  # 只取斷詞後的結果
            "board": extract_board(df.loc[df["title"] == sentence, "link"].iloc[0])
        }
        for sentence, sentence_ws, sentence_pos in tqdm(zip(text, ws, pos), total=len(text))
    ]
    results_df = pd.DataFrame(results)
    results_df.to_csv("tokenized_titles.csv", index=False, encoding='utf-8-sig')

    print("處理完成，結果已保存到 tokenized_titles.csv")

if __name__ == "__main__":
    main()
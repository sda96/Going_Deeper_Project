import re
import matplotlib.pyplot
from tqdm import tqdm
from collections import defaultdict
import tensorflow as tf
import sentencepiece as spm
import gensim
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 텍스트 전처리 함수
def text_preprocessing(x):
    x = re.sub("[^a-zA-Zㄱ-ㅎ가-힣,?!\"\']+"," ", x)
    x = re.sub("[ ]+"," ", x)
    x = x.strip()
    return x    

# 공백만 있는 데이터를 제거
def remove_nan(dataset):
    dataset = dataset.replace(["",'', " ", ' '], np.nan)
    dataset = dataset.dropna()
    return dataset

# Mecab의 단어 빈도 사전 구축
def bin_dict(tokens):
    dic = defaultdict()
    for data in tokens:
        for word in data:
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 0

    dic = sorted(dic.items(), key = lambda x: x[1], reverse = True)
    dic = dict(dic)
    return dic


# Mecab의 단어-인덱스, 인덱스-단어 사전 구축
def word_index(dic, vocab_size):
    word_to_index = defaultdict()
    index_to_word = defaultdict()
    
    
    dic = list(dic.items())[:(vocab_size - 4)]
    dic = dict(dic)
    for ind, word in enumerate(dic):
        word_to_index[word] = ind + 4
        index_to_word[ind + 4] = word
    
    word_to_index["<pad>"] = 0
    index_to_word[0] = "<pad>"
    word_to_index["<unk>"] = 1
    index_to_word[1] = "<unk>"
    word_to_index["<sos>"] = 2
    index_to_word[2] = "<sos>"
    word_to_index["<eos>"] = 3
    index_to_word[3] = "<eos>"

    word_to_index = dict(word_to_index)
    index_to_word = dict(index_to_word)
    return word_to_index, index_to_word


#  Mecab 토큰화, 패딩 적용
def tokenization(corpus, dictionary, max_length = 0, padding = "post"):
    tokens = []
    for setence in tqdm(corpus):
        # 1은 unk 토큰의 인덱
        tmp = [dictionary[word] if word in dictionary else 1 for word in setence]
        
        if max_length < len(tmp):
            max_length = len(tmp)
        tokens += [tmp]

    # 패딩
    tokens = pad_sequences(tokens, maxlen = max_length, value = 0, padding = padding)
    
    return tokens


# 일반 판다스 데이터프레임을 텐서플로우 데이터셋으로 변환
def tensorflow_dataset(input_x, input_y, buffer_size, batch_size):
    tf_data = tf.data.Dataset.from_tensor_slices((input_x, input_y))
    tf_data = tf_data.shuffle(buffer_size)
    tf_data = tf_data.repeat()
    tf_data = tf_data.batch(batch_size)
    tf_data = tf_data.prefetch(buffer_size = -1)
    return tf_data 


# spm 모델 생성 및 단어장 생성
def SentencePiece(model_type, data, vocab_size, add, train_test, temp_file = None):
    if not temp_file:
        temp_file = f"./model/02_SentencePiece/{model_type}_{vocab_size}_pre{add}_{train_test}.tmp"
    
    with open(temp_file, 'w') as f:
        for row in data:
            f.write(str(row) + '\n')
    spm_input = f"""
    --input={temp_file} 
    --model_prefix={model_type}_{vocab_size}_{train_test}_spm 
    --vocab_size={vocab_size} 
    --model_type={model_type}
    --unk_id=0 
    --pad_id=1 
    --bos_id=2 
    --eos_id=3 
    --user_defined_symbols=<pad>,<bos>,<eos>
    """
    spm_input = re.sub("\n", "", spm_input)
    spm.SentencePieceTrainer.Train(spm_input)

    s = spm.SentencePieceProcessor()
    s.Load(f'{model_type}_{vocab_size}_{train_test}_spm.model')
    return s


# 학습과정 히스토리 시각화
def show_performance(history, title):
    loss, acc, val_loss, val_acc, lr = history.history.values()
    plt.plot(loss, label = "loss")
    plt.plot(acc, label = "acc")
    plt.plot(val_loss, label = "val_loss")
    plt.plot(val_acc, label = "val_acc")
    plt.title(title)
    plt.xlabel("epochs")
    plt.legend()
    plt.show()


# 사전학습된 임베딩 행렬 불러오기
def load_embedding_matrix(path, vocab_size, embedding_size, idx_word):
    word2vec = gensim.models.Word2Vec.load(path)
    embedding_matrix = np.random.rand(vocab_size, embedding_size)
    count = 0
    # embedding_matrix에 Word2Vec 워드 벡터를 단어 하나씩마다 차례차례 카피한다.
    for i in range(2,vocab_size):
        if idx_word[i] in word2vec:
            embedding_matrix[i] = word2vec[idx_word[i]]
            count += 1
    print(f"사전 학습에 사용 가능했던 단어 벡터 갯수 : {count}")
    return embedding_matrix


# Series 타입의 구조에서 원하는 word가 있는 인덱스를 반환하고 해당 문장을 출력
def find_word(word, sentences):
    find_indexes = []
    for idx, sentence in enumerate(sentences):
        if re.findall(word, sentence):
            find_indexes += [idx]
    return sentences[find_indexes]
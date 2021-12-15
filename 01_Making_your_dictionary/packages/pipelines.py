import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import re
import gensim
from konlpy.tag import Mecab
import sentencepiece as spm

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Constant

# 텍스트 전처리 함수
def text_prep(x):
    x = re.sub("[^ㄱ-ㅎ가-힣?!,.]+"," ", x)
    x = re.sub("[ ]+"," ", x)
    x = x.strip()
    return x    


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
    
    
    dic = list(dic.items())[:(vocab_size - 2)]
    dic = dict(dic)
    for ind, word in enumerate(dic):
        word_to_index[word] = ind + 2
        index_to_word[ind + 2] = word
    
    word_to_index["<pad>"] = 0
    index_to_word[0] = "<pad>"
    word_to_index["<unk>"] = 1
    index_to_word[1] = "<unk>"

    word_to_index = dict(word_to_index)
    index_to_word = dict(index_to_word)
    return word_to_index, index_to_word


#  Mecab 토큰화 적용
def tokenization(corpus, dictionary):
    tokens = []
    for setence in tqdm(corpus):
        # 1은 unk 토큰의 인덱
        tmp = [dictionary[word] if word in dictionary else 1 for word in setence]
        tokens += [tmp]
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
        for row in data["document"]:
            f.write(str(row) + '\n')
    spm_input = f"""
    --input={temp_file} 
    --model_prefix={model_type}_{vocab_size}_spm 
    --vocab_size={vocab_size} 
    --model_type={model_type}
    """
    spm_input = re.sub("\n", "", spm_input)
    spm.SentencePieceTrainer.Train(spm_input)

    s = spm.SentencePieceProcessor()
    s.Load(f'{model_type}_{vocab_size}_spm.model')
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



class DataPipeline():

    
    def __init__(self, path_to_file, train_test):
        self.path_to_file = path_to_file
        self.train_test = train_test
        # 데이터 불러오기
        self.data = pd.read_csv(path_to_file, sep = "\t")
        
        
    # 데이터 전처리 적용
    def preprocessing(self, add = False):
        cp_data = self.data.copy()
        self.add = add
        # 중복 데이터 제거
        cp_data = cp_data.drop_duplicates("document")
        cp_data = cp_data.reset_index()
        
        # 결측 데이터 제거
        cp_data = cp_data.replace(["",'', " ", ' '], np.nan)
        cp_data = cp_data.dropna()
        cp_data["len"] = cp_data["document"].apply(lambda x : len(x))
        if add:
            # 텍스트 전처리 함수 적용
            cp_data["document"] = cp_data["document"].apply(lambda x: text_prep(x))
            
            # IQR 기준 이상치 제거 훈련 데이터셋 적용
            data_Q3 = cp_data.describe().loc["75%", "len"]
            data_Q1 = cp_data.describe().loc["25%", "len"]
            data_IQR = data_Q3 - data_Q1
            data_upper = data_Q3 + data_IQR
            data_lower = data_IQR - data_Q1
            cp_data = cp_data.loc[(cp_data["len"] >= data_lower) & 
                                    (cp_data["len"] <= data_upper), :]            

            # 결측 데이터 제거
            cp_data = cp_data.replace(["",'', " ", ' '], np.nan)
            cp_data = cp_data.dropna()
            
            self.cp_data = cp_data
            return cp_data
        else:
            self.cp_data = cp_data
            return cp_data
        
    
    # 데이터 벡터화 적용 : Mecab, SentencePiece(BPE, Uni-gram)
    def vectorization(self, vec_model, vocab_size):
        
        # Mecab 형태소 분석
        if vec_model == 0: 
            # Mecab으로 형태소 단위로 분절
            mecab = Mecab()
            tqdm.pandas()
            tokens = self.cp_data["document"].progress_apply(lambda x: mecab.morphs(x))
            # 빈도 기반 단어 사전 구축
            num_dict = bin_dict(tokens)
            word_to_index_dict, index_to_word_dict = word_index(num_dict, vocab_size)
            tokens = tokenization(tokens, word_to_index_dict)
            # 패딩
            max_length = max(self.cp_data["len"])
            self.padd_token = pad_sequences(tokens, padding='post', maxlen = max_length, value = 0)
            return self.padd_token, index_to_word_dict, word_to_index_dict

        # SPM uni-gram
        elif vec_model == 1:
            model_type = "unigram"
            s = SentencePiece(model_type, self.cp_data, vocab_size, self.add, self.train_test)
            max_len = max(self.cp_data["len"])
            id_document = self.cp_data["document"].apply(lambda x : s.EncodeAsIds(str(x)))
            self.padd_token = pad_sequences(id_document, maxlen = max_len, padding = "post", value = 0)
            return self.padd_token, s

        # SPM bpe
        elif vec_model == 2:
            model_type = "bpe"
            s = SentencePiece(model_type, self.cp_data, vocab_size, self.add, self.train_test)
            max_len = max(self.cp_data["len"])
            id_document = self.cp_data["document"].apply(lambda x : s.EncodeAsIds(str(x)))
            self.padd_token = pad_sequences(id_document, maxlen = max_len, padding = "post", value = 0)
            return self.padd_token, s
    
    
    # 벡터화 적용된 데이터셋을 텐서플로우 데이터셋으로 변환
    def tensorflow_dataset(self, batch_size, split = None):
        x_data = self.padd_token
        y_data = self.cp_data["label"].to_numpy().reshape(-1, 1)
        # 훈련 데이터셋은 다시 검증 데이터셋으로 분할
        if split:
            x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                              test_size = 0.2,
                                                              random_state = 44)
            tr_buffer = len(x_train)
            tr_tf_data = tensorflow_dataset(x_train, y_train, tr_buffer, batch_size)
            val_buffer = len(x_val)
            val_tf_data = tensorflow_dataset(x_val, y_val, val_buffer, batch_size)
            return tr_tf_data, val_tf_data
        else:
            da_buffer = len(x_data)
            tf_data = tensorflow_dataset(x_data, y_data, da_buffer, batch_size)
            return tf_data


class NSMC_model(tf.keras.Model):
    
    def __init__(self, vocab_size, embedd_size, hidden_size, embedding_matrix = None):
        super(NSMC_model, self).__init__()
        # 사전학습 임베딩 벡터 존재시 사용
        if type(embedding_matrix) == type(None):
            self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                       output_dim = embedd_size)
        else:
            self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size,
                                                       output_dim = embedd_size,
                                                       trainable=True,
                                                       embeddings_initializer = Constant(embedding_matrix))        

        self.BiLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = hidden_size, return_sequences = True))
        self.BN = tf.keras.layers.BatchNormalization()
        self.dense = tf.keras.layers.Dense(16, activation = "relu")
        self.out = tf.keras.layers.Dense(1, activation = "sigmoid")
        
    def call(self, x):
        x = self.embedding(x)
        x = self.BiLSTM(x)
        x = self.BN(x)
        x = self.dense(x)
        x = self.out(x)
        return x


def ProjectPipelines(add, vectorization_type, word2vec, vocab_size):
    # 초기 환경 설정
    epochs = 5
    hidden_size = 32
    mecab = Mecab()
    batch_size = 128
    embedding_size = 200
    tr_path_to_file = "./nsmc_data/ratings_train.txt" # 훈련 데이터 디렉토리
    te_path_to_file = "./nsmc_data/ratings_test.txt" # 테스트 데이터 디렉토리

    
    # 형태소 분석기 (Mecab)
    if vectorization_type == 0:
        print("훈련 데이터 파이프라인 시작")
        # 훈련셋에 데이터 파이프라인 적용
        train = DataPipeline(tr_path_to_file, "train")
        pre_train = train.preprocessing(add = add)
        vector_train, idx_word_train, word_idx_train = train.vectorization(vectorization_type, vocab_size)
        tf_train, tf_val = train.tensorflow_dataset(batch_size, split = True)
        print("훈련 데이터 파이프라인 종료\n")
        
        # 테스트셋 불러오기
        print("테스트 데이터 파이프라인 시작")
        test = DataPipeline(te_path_to_file, "test")
        pre_test = test.preprocessing(add = add)

        # 훈련셋에서 만들어진 인덱스를 테스트셋에 적용
        tqdm.pandas()
        token_corpus_test = pre_test["document"].progress_apply(lambda x: mecab.morphs(x))
        idx_corpus_test = tokenization(token_corpus_test, word_idx_train)

        # 패딩
        max_length_test = max(pre_train["len"])
        padd_tokens_test = pad_sequences(idx_corpus_test, padding='post', maxlen = max_length_test, value = 0)

        # 테스트셋을 텐서플로우 데이터셋으로 변환
        x_test = padd_tokens_test
        y_test = pre_test["label"].to_numpy().reshape(-1,1)
        buffer_size_test = len(pre_test)
        tf_test = tensorflow_dataset(x_test, y_test, buffer_size_test, batch_size)
        print("테스트 데이터 파이프라인 종료")
        
    # SentencePiece (Uni-gram, BPE)
    elif vectorization_type == 1 or vectorization_type == 2:
        print("훈련 데이터 파이프라인 시작")
        # 훈련셋에 데이터 파이프라인 적용
        train = DataPipeline(tr_path_to_file, "train")
        pre_train = train.preprocessing(add = add)
        vector_train, spm = train.vectorization(vectorization_type, vocab_size)
        tf_train, tf_val = train.tensorflow_dataset(batch_size, split = True)
        print("훈련 데이터 파이프라인 종료\n")
        
        print("테스트 데이터 파이프라인 시작")
        # 테스트셋 불러오기
        test = DataPipeline(te_path_to_file, "test")
        pre_test = test.preprocessing(add = add)
        # 훈련셋에서 만들어진 인덱스를 테스트셋에 적용
        tqdm.pandas()
        token_corpus_test = pre_test["document"].progress_apply(lambda x : spm.EncodeAsIds(str(x)))
        max_length_test = max(pre_train["len"])
        padd_tokens_test = pad_sequences(token_corpus_test, maxlen = max_length_test, 
                                           padding = "post", value = 0)

        # 테스트셋을 텐서플로우 데이터셋으로 변환
        x_test = padd_tokens_test
        y_test = pre_test["label"].to_numpy().reshape(-1,1)
        buffer_size_test = len(pre_test)
        tf_test = tensorflow_dataset(x_test, y_test, buffer_size_test, batch_size)
        print("테스트 데이터 파이프라인 종료")

    # Mecab + SentencePiece (3이면 uni-gram, 4이면 bpe)
    elif vectorization_type == 3 or vectorization_type == 4:
        
        print("훈련 데이터 파이프라인 시작")
        # 훈련 데이터를 불러와서 전처리를 진행
        train = DataPipeline(tr_path_to_file, "train")
        pre_train = train.preprocessing(add = add)
        # Mecab으로 형태소 단위로 나눈 뒤 다시 str으로 합쳐줌
        tqdm.pandas()
        pre_train["document"] = pre_train["document"].progress_apply(lambda x: " ".join(mecab.morphs(x)))

        # 3이면 uni-gram, 4이면 bpe 모델 선택
        if vectorization_type == 3:
            model_type = "unigram"
        elif vectorization_type == 4:
            model_type = "bpe"
        spm = SentencePiece(model_type, pre_train, 
                              vocab_size, add,
                              temp_file = f"./model/02_SentencePiece/{model_type}_{vocab_size}_pre{add}_mecab_train.tmp",
                              train_test = "train")

        # 추가 전처리까지 된 코퍼스 unigram 타입으로 벡터화
        max_len = max(pre_train["len"])
        id_document = pre_train["document"].apply(lambda x : spm.EncodeAsIds(str(x)))
        padd_token = pad_sequences(id_document, maxlen = max_len, padding = "post")

        # 텐서플로우 데이터셋으로 변환
        x_data = padd_token
        y_data = pre_train["label"].to_numpy().reshape(-1, 1)
        x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, 
                                                                  test_size = 0.2,
                                                                  random_state = 44)
        tr_buffer = len(x_train)
        tf_train = tensorflow_dataset(x_train, y_train, tr_buffer, batch_size)
        val_buffer = len(x_val)
        tf_val = tensorflow_dataset(x_val, y_val, val_buffer, batch_size)
        print("훈련 데이터 파이프라인 종료\n")
        
        print("테스트 데이터 파이프라인 시작")
        # 테스트셋 불러오기
        test = DataPipeline(te_path_to_file, "test")
        pre_test = test.preprocessing(add = add)

        # 훈련셋에서 만들어진 인덱스를 테스트셋에 적용
        tqdm.pandas()
        token_corpus_test = pre_test["document"].progress_apply(lambda x : spm.EncodeAsIds(str(x)))
        max_length_test = max(pre_train["len"])
        padd_tokens_test = pad_sequences(token_corpus_test, maxlen = max_length_test, 
                                           padding = "post", value = 0)

        # 테스트셋을 텐서플로우 데이터셋으로 변환
        x_test = padd_tokens_test
        y_test = pre_test["label"].to_numpy().reshape(-1,1)
        buffer_size_test = len(pre_test)
        tf_test = tensorflow_dataset(x_test, y_test, buffer_size_test, batch_size)    
        print("테스트 데이터 파이프라인 종료\n")
    
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = 2)
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(patience = 2)
    # word2vec True면 사전학습된 임베딩 벡터로 전이학습
    if word2vec:
        # 사전학습된 임베딩 행렬을 불러옴
        path = './model/ko.bin'
        embedding_matrix = load_embedding_matrix(path, vocab_size, embedding_size, idx_word_train)
        
        # 서브클래스 모델 선언
        model = NSMC_model(vocab_size, embedding_size, hidden_size, embedding_matrix)

        # 모델 컴파일
        model.compile(optimizer="adam",
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"])

        # 모델 학습 시작
        train_buffer_size = int(len(pre_train) * 0.8)
        val_buffer_size = int(len(pre_train) * 0.2)
        history = model.fit(tf_train, 
                            steps_per_epoch = train_buffer_size // batch_size,
                            validation_data = tf_val,
                            validation_steps = val_buffer_size // batch_size,
                            epochs = epochs,
                            callbacks = [es, lr_reduce])
    # word2vec False이면 일반 BiLSTM 모델 학습
    else:
        # 서브클래스 모델 선언
        model = NSMC_model(vocab_size, embedding_size, hidden_size)

        # 모델 컴파일
        model.compile(optimizer="adam",
                      loss = "binary_crossentropy",
                      metrics = ["accuracy"])

        # 모델 학습 시작
        train_buffer_size = int(len(pre_train) * 0.8)
        val_buffer_size = int(len(pre_train) * 0.2)
        history = model.fit(tf_train, 
                            steps_per_epoch = train_buffer_size // batch_size,
                            validation_data = tf_val,
                            validation_steps = val_buffer_size // batch_size,
                            epochs = epochs,
                            callbacks = [es, lr_reduce])
        
        

    # 모델 학습 추이 시각화
    if vectorization_type == 0:
        title = f"Mecab / preprocessing {add} / word2vec {word2vec}"
    elif vectorization_type == 1:
        title = f"Unigram / preprocessing {add}"
    elif vectorization_type == 2:
        title = f"BPE / preprocessing {add}"
    elif vectorization_type == 3:
        title = f"Mecab + Uni-gram preprocessing / {add} model"
    elif vectorization_type == 4:
        title = f"Mecab + BPE preprocessing / {add} model"
    show_performance(history, title)

    # 테스트셋으로 모델 검증
    test_loss, test_acc = model.evaluate(tf_test, steps = buffer_size_test // batch_size)
    print(f"테스트셋 accuracy : {round(test_acc, 4) * 100}%")
    print(f"테스트셋 loss : {test_loss}")

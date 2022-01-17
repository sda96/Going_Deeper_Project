import re
from glob import glob
import matplotlib.pyplot
from tqdm import tqdm
from collections import defaultdict
import tensorflow as tf
import gensim
import numpy as np
import pandas as pd


def load_data(data_path:"int"):
    """
    args
        - data_path : 원소로 데이터의 디렉토리가 담긴 리스트를 입력받음
    desc
        - 데이터 파일명과 디렉토리를 인자로 받아서 판다스 데이터프레임 또는 리스트로 반환
    return
        - 만약 데이터 확장자가 .csv면 pandas.DataFrame
        - 만약 데이터 확장자가 .tsv면 pandas.DataFrame
        - 만약 없는 확장자이면 list
    """
    extend = data_path.split(".")[-1]
    
    if extend == "csv":        
        sentences = pd.read_csv(data_path)
        print(f"데이터 개수 : {sentences.shape[0]}")
        print(f"컬럼 개수 : {sentences.shape[1]}")
        print(f"컬럼명 : {sentences.columns}")
        return sentences
    
    with open(data_path, encoding = "utf-8") as f:        
        sentences = f.readlines()
        sentences = [re.sub("\n", "", sentence) for sentence in sentences]
        
        if extend == "tsv":
            sentences = [sentence.split("\t") for sentence in sentences]
            sentences = pd.DataFrame(sentences[1:], columns = sentences[0])
            print(f"데이터 개수 : {sentences.shape[0]}")
            print(f"컬럼 개수 : {sentences.shape[1]}")
            print(f"컬럼명 : {sentences.columns}")
            return sentences
        elif extend == "txt":
            sentences = [sentence.split(" ") for sentence in sentences]
            sentences = pd.DataFrame(sentences[1:], columns = sentences[0])
            print(f"데이터 개수 : {sentences.shape[0]}")
            print(f"컬럼 개수 : {sentences.shape[1]}")
            print(f"컬럼명 : {sentences.columns}")
            return sentences
    
        print(f"데이터 개수 : {len(sentences)}")
        return sentences


def load_data_all(path:"str"):
    """
    args
        - path : 데이터 파일의 위치를 나타내는 문자형 객체를 입력받음
    desc
        - 입력받은 파일의 위치에 있는 모든 파일을 불러오고 dict 타입으로 반환
    retrun
        - key는 파일명 value는 파일의 내용이 담긴 dict 타입을 반환
    """
    data_path = glob(path)
    data_dict = defaultdict()
    for path in data_path:
        name = path.split("/")[-1]
        data_dict[name] = load_data(path)
    return dict(data_dict)
    
    
def text_prep(x:"str"):
    """
    args
        - x : 전처리 시킬 문장을 입력받음
    desc
        - str 타입의 문장을 입력받아 전처리 후 str 타입으로 반환
    return
        - 전처리가 완료된 str 타입 반환
    """
    x = x.lower()
    x = re.sub("[^0-9a-zㄱ-ㅎ가-힣,?!\"\']+"," ", x)
    x = re.sub("[ ]+"," ", x)
    x = x.strip()
    return x    


def remove_nan(dataset:"pd.DataFrame"):
    """
    args
        - dataset : pandas.DataFrame을 입력으로 받음
    desc
        - 공백만 있는 데이터를 제거
    return
        - pandas.DataFrame
    """
    dataset = dataset.replace(["",'', " ", ' '], np.nan)
    dataset = dataset.dropna()
    return dataset


# 일반 판다스 데이터프레임을 텐서플로우 데이터셋으로 변환
def tensorflow_dataset(input_x, input_y, buffer_size, batch_size):
    tf_data = tf.data.Dataset.from_tensor_slices((input_x, input_y))
    tf_data = tf_data.shuffle(buffer_size)
    tf_data = tf_data.repeat()
    tf_data = tf_data.batch(batch_size)
    tf_data = tf_data.prefetch(buffer_size = -1)
    return tf_data 


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
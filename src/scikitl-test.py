# -*- coding: utf-8 -*-
import os
import MeCab
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

mecab = MeCab.Tagger('mecabrc')

# 記事データを管理してるディレクトリパス
DATA_PATH = 'text/'

# 辞書保存ファイル名
SAVE_FILE_NAME = 'dictionary.txt'

def tokenize(text):
    """
    形態素解析で単語を取り出す
    :param text:
    :return:
    """
    node = mecab.parseToNode(text)
    while node:
        if node.feature.split(',')[0] == '名詞':
            yield node.surface.lower()
        node = node.next

def get_words(contents):
    """
    記事郡のdictについて形態素解析を行いリストで返す
    :param contents:
    :return:
    """
    ret = []
    for k, content in contents.items():
        ret.append(get_words_main(content))

    return ret

def get_words_main(content):
    """
    １つの記事を形態素解析して返す
    :param content:
    :return:
    """
    return [token for token in tokenize(content)]

def dir_list():
    """
    記事を管理するディレトリ一覧を返す
    :return:
    """
    _dir_list = []
    for content in os.listdir(data_path()):
        if os.path.isdir(data_path(content)):
            _dir_list.append(content)

    return _dir_list

def file_list(directory):
    """
    指定したディレクトリ内から記事ファイル一覧を返す
    :param directory:
    :return:
    """
    filelist = []
    for file in os.listdir(data_path(directory)):
        if directory in file and os.path.isfile(data_path(directory) + file):
            filelist.append(file)

    return filelist

def data_path(directory=""):
    """
    記事データの相対パスで返す
    空文字の場合は記事データのディレクトリのみ返す
    :param directory:
    :return:
    """
    if len(directory) == 0:
        return DATA_PATH

    return DATA_PATH + directory + "/"

def read_data(filepath):
    """
    ファイルから記事を読み込み、不要な文字列を排除し返す
    :param filepath:
    :return:
    """
    with open(filepath, 'r') as r:
        text_contents = r.readlines()
    contents = [content.strip() for content in text_contents if len(content.strip()) != 0]

    return ''.join(contents)

def class_id(file):
    dir_lists = dir_list()
    dir_name = next(filter(lambda x: x in file, dir_lists), None)

    return dir_lists.index(dir_name)

def main():
    contents = {}
    data_train = []
    # 正解ラベル 0: 独女通信, 1:ITライフハック...
    label_train = []

    directories = dir_list()
    for directory in directories:
        files = file_list(directory)
        for file in files:
            content = read_data(data_path(directory) + file)
            contents[file] = content
            label_train.append(class_id(file))

    # ワードの重複を除いた、辞書リストの作成
    words = get_words(contents)
    wordbook = corpora.Dictionary(words)

    # ここは調整が必要
    # no_berow: 使われてる文章がno_berow個以下の単語無視
    # no_above: 使われてる文章の割合がno_above以上の場合無視
    # wordbook.filter_extremes(no_below=1, no_above=0.7)

    # 辞書リストを.txtに保存
    wordbook.save_as_text(SAVE_FILE_NAME)

    # 作った辞書ファイルをロードして(wordbook)辞書オブジェクト作る
    # wordbook = corpora.Dictionary.load_from_text(SAVE_FILE_NAME)

    # BoW (単語id, 出現回数)と表現される
    for w in words:
        vector = wordbook.doc2bow(w)
        # 特徴ベクトルの取得
        dense = list(matutils.corpus2dense([vector], num_terms=len(wordbook)).T[0])
        data_train.append(dense)

    # ランダムフォレストオブジェクト生成
    estimator = RandomForestClassifier()

    # 学習させる
    estimator.fit(data_train, label_train)

    # 予測
    # label_predict = estimator.predict(data_train)

    # 予測結果
    print(estimator.score(data_train, label_train))

    # 学習データと試験データに分ける
    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.5)

    # もう一度ランダムフォレストで検証する
    estimator2 = RandomForestClassifier()

    estimator2.fit(data_train_s, label_train_s)

    print(estimator2.score(data_train_s, label_train_s))

if __name__ == '__main__':
    main()

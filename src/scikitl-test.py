# -*- coding: utf-8 -*-
import os
import MeCab
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

mecab = MeCab.Tagger('mecabrc')

DATA_PATH = 'text/'

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
    dirList = []
    for list in os.listdir(data_path()):
        if os.path.isdir(data_path(list)):
            dirList.append(list)

    return dirList

def file_list(directory):
    """
    指定したディレクトリ内から記事ファイル一覧を返す
    :param directory:
    :return:
    """
    filelist = []
    for file in os.listdir(data_path(directory)):
        if os.path.isfile(data_path(directory) + file):
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
    r = open(filepath, 'r')
    text_contents = r.readlines()
    r.close()

    return [content.strip() for content in text_contents if len(content.strip()) != 0]

if __name__ == '__main__':
    # {}はマップ型といって連想配列
    words = get_words({
        'it-life-hack-001.txt': 'アナタはまだブラウザのブックマーク？　ブックマーク管理はライフリストがオススメ 最近ネットサーフィンをする際にもっぱら利用しているのが「ライフリスト」というサイトだ。この「ライフリスト」は、ひとことで言うと自分専用のブックマークサイトである。というよりブラウザのスタートページにするとブラウザのブックマーク管理が不要になる便利なサイトなのである。',
        'dokujo-tsushin-001.txt': 'たとえば、馴れ馴れしく近づいてくるチャラ男、クールを装って迫ってくるエロエロ既婚男性etc…に対し「下心、見え見え〜」と思ったことはないだろうか？ “下心”と一言で言うと、特に男性が女性のからだを目的に執拗に口説くなど、イヤらしい言葉に聞こえてしまう。実際、辞書で「下心」の意味を調べてみると、心の底で考えていること。かねて心に期すること、かねてのたくらみ。特に、わるだくみ。（広辞苑より）という意味があるのだから仕方がないのかもしれない。'
    })
    # ワードの重複を除いた、辞書リストの作成
    wordbook = corpora.Dictionary(words)

    # ここは調整が必要
    # no_berow: 使われてる文章がno_berow個以下の単語無視
    # no_above: 使われてる文章の割合がno_above以上の場合無視
    # wordbook.filter_extremes(no_below=1, no_above=0.7)

    # 辞書リストを.txtに保存
    wordbook.save_as_text('dictionary.txt')

    # 作った辞書ファイルをロードして(wordbook)辞書オブジェクト作る
    # wordbook = corpora.Dictionary.load_from_text('livedoordic.txt')

    print(wordbook.token2id)

    # BoW (単語id, 出現回数)と表現される
    dense_list = []
    for w in words:
        vector = wordbook.doc2bow(w)
        # 特徴ベクトルの取得
        dense = list(matutils.corpus2dense([vector], num_terms=len(wordbook)).T[0])

        dense_list.append(dense)

    print(dense_list)
    # 正解ラベル
    # 0: 'dokujo-tsushin' 1: 'it-life-hack' 2: 'kaden-channel' 3: 'livedoor-homme' 4:'movie-enter'
    # 5: 'peachy' 6: 'smax' 7: 'sports-watch' 8: 'topic-news'
    label_train = [1, 0] # 1:ITライフハック, 0: 独女通信

    estimator = RandomForestClassifier()

    # 学習させる
    estimator.fit(dense_list, label_train)

    # 予測
    label_predict = estimator.predict(dense_list)
    print(label_predict)

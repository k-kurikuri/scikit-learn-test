# -*- coding: utf-8 -*-
import MeCab
from gensim import corpora, matutils
from sklearn.ensemble import RandomForestClassifier

mecab = MeCab.Tagger('mecabrc')

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
    # []はリスト内包表記を表す文法で1ライナーで記述できる。以下のfor inと意味は同じだが処理も速いというベンチがある
    # arr = []
    # for token in tokenize(content):
    #     arr.append(token)
    # return arr
    return [token for token in tokenize(content)]

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
    wordbook.save_as_text('livedoordic.txt')

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
    label_train = [1, 0] # 1:ITライフハック, 0: 独女通信

    estimator = RandomForestClassifier()

    # 学習させる
    estimator.fit(dense_list, label_train)

    # 予測
    label_predict = estimator.predict(dense_list)
    print(label_predict)

#-*- encoding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')
import sys
import random
import glob
import math
import pickle
from collections import defaultdict
import re
import codecs
from datetime import datetime
import time
import copy
import numpy as np
import matplotlib.pylab as plt
import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
try:
    import cupy
except Exception as e:
    pass


class SentenceEncoderBiGRU(Chain):

    def __init__(self, emb_size, hidden_size, use_dropout=0.1, flag_gpu=True):
        """
        emb_size:入力される分散表現ベクトル次元数
        hidden_size:隠れ層次元数
        use_dropout:float値．どの程度ドロップアウト使うか
        """
        w = chainer.initializers.HeNormal()
        super(SentenceEncoderBiGRU, self).__init__(
            # word_embed=L.EmbedID(n_vocab, emb_size, ignore_label=-1),
            word_embed=L.Linear(emb_size, hidden_size, initialW=w),
            bi_gru=L.NStepBiGRU(n_layers=1, in_size=hidden_size,
                                out_size=hidden_size, dropout=use_dropout)
        )
        """
        n_layers:層数
        in_size:入力ベクトルの次元数
        out_size:出力ベクトルの次元数
        drop_out:dropout率
        """
        self.USE_DROPOUT = use_dropout
        self.USE_DROPOUT_keep = use_dropout
        self.FLAG_GPU = flag_gpu
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        if flag_gpu:
            self.ARR = cupy
            cupy.cuda.Device(0).use()
        else:
            self.ARR = np

    def __call__(self, x_data):
        """
        x_data:mecabなどで処理された系列データ
        """
        batchsize = len(x_data)
        # ここで初期化がゼロになっている（ランダムの方がいいか？）
        hx = None

        # print(self.ARR.shape(x_data))
        xs = []
        for x in x_data:
            if self.FLAG_GPU:
                x = cuda.to_gpu(x, device=0)
            x = self.word_embed(x)
            x = F.dropout(x, ratio=self.USE_DROPOUT)
            xs.append(x)

        # GRU
        # print(self.ARR.shape(xs))
        if len(xs)==0:
            print("xs is 0")
        if self.USE_DROPOUT != self.USE_DROPOUT_keep:
            #with chainer.using_config('train', False):
            _, ys = self.bi_gru(hx=hx, xs=xs)
        else:
            _, ys = self.bi_gru(hx=hx, xs=xs)

        # print(self.ARR.shape(ys))
        return ys

    def mode_change(self, train_or_test):
        if train_or_test == "train":
            self.USE_DROPOUT = self.USE_DROPOUT_keep
        if train_or_test == "test":
            self.USE_DROPOUT = 0.0


class Attention(Chain):

    def __init__(self, fnn_size, hidden_size, use_dropout=0.1, flag_gpu=1, flag_local=1):
        """
        Attentionのインスタンス化
        :param hidden_size: 隠れ層のサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        w = chainer.initializers.HeNormal()
        self.USE_DROPOUT = use_dropout
        self.USE_DROPOUT_keep = use_dropout
        # 隠れ層のサイズを記憶
        self.fnn_size = fnn_size
        self.hidden_size = hidden_size
        self.flag_local = flag_local
        # 決め打ち
        self.local_window = 10
        if flag_local == 0:
            super(Attention, self).__init__(
                # 順向き逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
                fbh=L.Linear(hidden_size, hidden_size, initialW=w),
                # FNN用ベクトルを隠れ層サイズのベクトルに変換する線形結合層
                nh=L.Linear(fnn_size, hidden_size, initialW=w),
                # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
                hw=L.Linear(hidden_size, 1, initialW=w)
            )
        else:
            super(Attention, self).__init__(
                # 順向き逆向きのEncoderの中間ベクトルを隠れ層サイズのベクトルに変換する線形結合層
                fbh=L.Linear(hidden_size, hidden_size, initialW=w),
                # FNN用ベクトルを隠れ層サイズのベクトルに変換する線形結合層
                nh=L.Linear(fnn_size, hidden_size, initialW=w),
                # 隠れ層サイズのベクトルをスカラーに変換するための線形結合層
                hw=L.Linear(hidden_size, 1, initialW=w),
                # local attention用の線形結合
                nt=L.Linear(fnn_size, hidden_size, initialW=w),
                tw=L.Linear(hidden_size, 1, initialW=w)
            )

        # GPUを使う場合はcupyを使わないときはnumpyを使う
        self.FLAG_GPU = flag_gpu
        if flag_gpu:
            self.ARR = cupy
            cupy.cuda.Device(0).use()
        else:
            self.ARR = np

    def __call__(self, fbs, ns):
        """
        Attentionの計算
        :param fbs: 順向き逆向きのEncoderの中間ベクトルが記録されたリスト
        (4, 1000)
        (3, 1000)
        (2, 1000)
        :param h: Decoderで出力された中間ベクトル
        :return: 順向きのEncoderの中間ベクトルの加重平均と逆向きのEncoderの中間ベクトルの加重平均
        """
        # ミニバッチのサイズを記憶
        #batch_size = ns.data.shape[0]
        # ウェイトを記録するためのリストの初期化
        ws = []
        # ウェイトの合計値を計算するための値を初期化
        # sum_w = Variable(self.ARR.zeros((batch_size, 1), dtype='float32'))
        # Encoderの中間ベクトルとDecoderの中間ベクトルを使ってウェイトの計算
        #start_time_x = time.time()
        for fb, n in zip(fbs, ns):
            # 順向きEncoderの中間ベクトル、逆向きEncoderの中間ベクトル、Decoderの中間ベクトルを使ってウェイトの計算
            # start_time = time.time()
            # print(fb.data.shape[0])
            # print(n.shape)
            # n_s = self.ARR.array(
            #     [n for _ in range(fb.data.shape[0])], dtype="float32")
            # print(n_s)
            n_s = self.ARR.tile(n, (fb.data.shape[0], 1))
            # print(n_s)
            # print(n_s.shape)
            # n_s = self.ARR.empty((0, self.fnn_size), dtype=np.float32)
            # for _ in range(fb.data.shape[0]):
            #     n_s = self.ARR.concatenate(
            #         [n_s, F.reshape(copy.deepcopy(n), (-1, self.fnn_size))], axis=0)
            # print(n_s.shape)
            # exit()
            # interval = float(time.time() - start_time)
            # print("n_s実行時間: {}sec".format(interval))
            # start_time = time.time()
            w = F.tanh(F.dropout(self.fbh(fb), ratio=self.USE_DROPOUT) +
                       F.dropout(self.nh(n_s), ratio=self.USE_DROPOUT))
            # w = F.tanh(F.dropout(self.fbh(fb), ratio=self.USE_DROPOUT) +
            # F.dropout(self.nh(self.ARR.full(fb.data.shape[0], n)),
            # ratio=self.USE_DROPOUT))
            # interval = float(time.time() - start_time)
            # print("tanh実行時間: {}sec".format(interval))
            # start_time = time.time()
            # softmax関数を使って正規化する
            w = F.exp(F.dropout(self.hw(w), ratio=self.USE_DROPOUT))
            # interval = float(time.time() - start_time)
            # print("exp実行時間: {}sec".format(interval))
            # 計算したウェイトを記録
            # print(self.ARR.sum(w))
            # sum_w += w
            ws.append(w / F.sum(w).data)
        #interval = float(time.time() - start_time_x)
        #print("tanh_exp実行時間: {}sec".format(interval))

        # 出力する加重平均ベクトルの初期化
        #att_fb = []
        #start_time_x = time.time()
        att_fb = self.ARR.empty(
            (0, self.hidden_size), dtype=np.float32)
        if self.flag_local == 0:
            for fb, w in zip(fbs, ws):
                # ウェイトの和が1になるように正規化
                # w /= sum_w
                # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
                # ここにローカルアテンション用の何かを入れる
                att_fb += F.reshape(F.matmul(fb, w),
                                    (-1, self.hidden_size))
        else:
            D = self.local_window

            for fb, w, n in zip(fbs, ws, ns):
                # ウェイトの和が1になるように正規化
                # w /= sum_w
                #start_time = time.time()
                w_local_input = fb.data.shape[
                    0] * F.sigmoid(F.dropout(self.tw(F.tanh(F.dropout(self.nt(F.reshape(n, (1, -1))), ratio=self.USE_DROPOUT))), ratio=self.USE_DROPOUT))
                # ここにローカルアテンション用の何かを入れる
                #interval = float(time.time() - start_time)
                #print("local_input実行時間: {}sec".format(interval))
                #start_time = time.time()
                w_local_output = self.ARR.array([self.ARR.exp(-(float(s + 1) - float(w_local_input.data)) ** 2 / ((
                    (D / 2)**2) * 2)) for s in range(fb.data.shape[0])], dtype='float32')
                # ウェイト * Encoderの中間ベクトルを出力するベクトルに足していく
                #interval = float(time.time() - start_time)
                #print("local_out_put実行時間: {}sec".format(interval))
                #start_time = time.time()
                w = w * F.reshape(w_local_output, (-1, 1))
                #interval = float(time.time() - start_time)
                #print("reshape実行時間: {}sec".format(interval))
                #start_time = time.time()
                # att_fb.append(self.ARR.sum(self.ARR.array(
                #     [fb_x.data * w_x.data for fb_x, w_x in zip(fb, w)], dtype='float32'), axis=0))
                # fb_w = F.reshape(self.ARR.sum(self.ARR.array(
                #     [fb_x.data * w_x.data for fb_x, w_x in zip(fb, w)], dtype='float32'), axis=0), (1, -1))
                # print(fb)
                # print(w)
                fb_w = F.reshape(F.matmul(F.transpose(w), fb), (1, -1))
                # print(fb_w)
                #interval = float(time.time() - start_time)
                #print("kakezan実行時間: {}sec".format(interval))
                #start_time = time.time()
                # if self.FLAG_GPU:
                #     print(type(fb_w))
                #     fb_w = cuda.to_gpu(fb_w, device=0)
                att_fb = F.concat((att_fb, fb_w), axis=0)
                #print("concatenate実行時間: {}sec".format(interval))
                #start_time = time.time()
        #interval = float(time.time() - start_time_x)
        #print("local実行時間: {}sec".format(interval))
        return att_fb

    def mode_change(self, train_or_test):
        if train_or_test == "train":
            self.USE_DROPOUT = self.USE_DROPOUT_keep
        if train_or_test == "test":
            self.USE_DROPOUT = 0.0


class FNN_Decoder(Chain):

    def __init__(self, input_size, hidden_size, num_of_middle_layer=4, use_dropout=0.1, flag_gpu=True):
        w = chainer.initializers.HeNormal()
        super(FNN_Decoder, self).__init__(
            l1=L.Linear(input_size, hidden_size, initialW=w),
            l2=L.Linear(hidden_size, hidden_size, initialW=w),
            l3=L.Linear(hidden_size, 2, initialW=w),
            bnorm=L.BatchNormalization(hidden_size))
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        if flag_gpu:
            self.ARR = cupy
            cupy.cuda.Device(0).use()
        else:
            self.ARR = np
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_of_middle_layer = num_of_middle_layer
        self.USE_DROPOUT = use_dropout
        self.USE_DROPOUT_keep = use_dropout

    def __call__(self, x_data, y_data):
        y_ = self._predict(x_data)
        # print(np.shape(y_))
        # print(y_data)
        return F.softmax_cross_entropy(y_, self.ARR.array(y_data))

    def _predict(self, x):
        h1 = F.dropout(F.relu(self.bnorm(self.l1(x))),
                       ratio=self.USE_DROPOUT)
        for x in range(self.num_of_middle_layer):
            h1 = F.dropout(F.relu(self.bnorm(self.l2(h1))),
                           ratio=self.USE_DROPOUT)
        h2 = self.l3(h1)
        return h2

    def get_predata(self, x):
        return self._predict(x)

    def mode_change(self, train_or_test):
        if train_or_test == "train":
            self.USE_DROPOUT = self.USE_DROPOUT_keep
        if train_or_test == "test":
            self.USE_DROPOUT = 0.0


class Att_Seq2TF(Chain):

    def __init__(self, emb_size, fnn_size, hidden_size, num_of_middle_layer, use_dropout, flag_gpu=True):
        """
        Seq2TF + Attentionのインスタンス化
        :param emb_size: 単語ベクトルのサイズ
        :param fnn_size: fnnに突っ込むベクトルのサイズ
        :param hidden_size: 隠れ層のサイズ
        :param num_of_middle_layer: 隠れ層の数
        :param flag_gpu: GPUを使うかどうか
        """
        super(Att_Seq2TF, self).__init__(
            # 順向き逆向きのEncoder
            fb_encoder=SentenceEncoderBiGRU(
                emb_size, hidden_size, use_dropout, flag_gpu),
            # Attention Model
            attention=Attention(fnn_size, hidden_size * \
                                2, use_dropout, flag_gpu),
            # Decoder
            decoder=FNN_Decoder(fnn_size + hidden_size * 2,
                                (fnn_size + hidden_size * 2) // 2, num_of_middle_layer, use_dropout, flag_gpu)
        )
        self.emb_size = emb_size
        self.fnn_size = fnn_size
        self.hidden_size = hidden_size
        self.num_of_middle_layer = num_of_middle_layer
        self.USE_DROPOUT = use_dropout
        self.USE_DROPOUT_keep = use_dropout

        # GPUを使うときはcupy、使わないときはnumpy
        if flag_gpu:
            self.ARR = cuda.cupy
            cupy.cuda.Device(0).use()
        else:
            self.ARR = np

        # 順向きのEncoderの中間ベクトル、逆向きのEncoderの中間ベクトルを保存するためのリストを初期化
        self.fbs = []

    def encode(self, words):
        """
        Encoderの計算
        :param words: 入力で使用する単語記録されたリスト(emb済み)
        :return:
        """
        #start_time = time.time()
        self.fbs = self.fb_encoder(words)
        #interval = int(time.time() - start_time)
        #print("enc実行時間: {}sec".format(interval))
        # enc.cleargrads()

    def decode(self, n, y):
        """
        Decoderの計算
        :param n: Decoderで入力するFNN用入力列
        :return: loss
        """
        # Attention Modelを使ってEncoderの中間層の加重平均を計算
        #start_time = time.time()
        att_fb = self.attention(self.fbs, n)
        #interval = int(time.time() - start_time)
        #print("att実行時間: {}sec".format(interval))
        # Decoderの中間ベクトル、順向きのAttention、逆向きのAttentionを使って
        # 次の中間ベクトル、内部メモリ、予測単語の計算
        # print(type(n))
        # print(type(att_fb))
        #start_time = time.time()
        loss = self.decoder(F.concat((n, att_fb), axis=1), y)
        #interval = int(time.time() - start_time)
        #print("dec実行時間: {}sec".format(interval))
        return loss

    def reset(self):
        """
        インスタンス変数を初期化する
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        # self.c = Variable(self.ARR.zeros(
        #     (self.batch_size, self.hidden_size), dtype='float32'))
        # self.h = Variable(self.ARR.zeros(
        #     (self.batch_size, self.hidden_size), dtype='float32'))
        # Encoderの中間ベクトルを記録するリストの初期化
        self.fbs = []
        # 勾配の初期化
        self.cleargrads()

    def get_predata(self, n):
        att_fb = self.attention(self.fbs, n)
        return self.decoder.get_predata(F.concat((n, att_fb), axis=1))

    def mode_change(self, train_or_test):
        if train_or_test == "train":
            self.USE_DROPOUT = self.USE_DROPOUT_keep
            self.fb_encoder.mode_change("train")
            self.attention.mode_change("train")
            self.decoder.mode_change("train")
        if train_or_test == "test":
            self.USE_DROPOUT = 0.0
            self.fb_encoder.mode_change("test")
            self.attention.mode_change("test")
            self.decoder.mode_change("test")


if __name__ == '__main__':
    n_vocab = 5000
    emb_size = 300
    hidden_size = 500
    use_dropout = 0.33
    enc = SentenceEncoderBiGRU(emb_size, hidden_size, use_dropout)
    x_data = [[0, 1, 2, 3], [4, 5, 6], [7, 8]]
    x_data = [np.array(x, dtype=np.int32) for x in x_data]
    vec_all = enc(x_data)
    for x in vec_all:
        print(x.data.shape)

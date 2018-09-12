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
try:
    import cupy
except Exception as e:
    pass
import matplotlib.pylab as plt
import chainer
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
from gru_model import *
from gru_vector import *
#from gru_vector_maker import *
from multiprocessing import Process
from multiprocessing import Queue


class TrainAndTest(object):
    """docstring for TrainAndTest"""

    def __init__(self, feature_type="rnn_att", use_dropout=0.1, num_of_middle_layer=1, flag_gpu=1, change_num=0):
        self.INPUT_PATH_KNP = '../../knp_dismantled_data/*.knp'
        #self.INPUT_PATH_KNP = './knp_dismantled_data/*.knp'
        #self.INPUT_PATH = '../../resultx*_data_alt1/{}.dat'
        self.INPUT_PATH = '../result_data_cfm_alt1_vector/{}.pickle'
        #self.INPUT_PATH = '../../all/all_test_divs_alt/{}.dat'
        #self.INPUT_PATH = './resultx*_data/{}.dat'
        #self.INPUT_PATH = './valid_data/{}.dat'
        if change_num == 0:
            self.OUTPUT_PATH = "../model_alt1_rnn_change_init/{}_type{}_dropout{}_dim{}_textcount{}_loss{:.13f}.hdf5"
        else:
            self.OUTPUT_PATH = "../model_alt1_rnn_change_init/{}" + \
                str(change_num) + \
                "_type{}_dropout{}_dim{}_textcount{}_loss{:.13f}.hdf5"
        #self.OUTPUT_PATH = "./model_rnn/{}_type{}_dropout{}_dim{}_textcount{}_loss{:.13f}.hdf5"
        #self.INPUT_PATH_TEST = "../../all/all_test_divs_alt/all_test_*.dat"
        self.INPUT_PATH_TEST = "../all_test_divs_alt_vector1/*.pickle"
        #self.INPUT_PATH_TEST = '../../resultx*_data_alt/*_PB*.dat'
        if change_num == 0:
            self.OUTPUT_PATH_TEST = "../predict_alt1_rnn_change_init/{}/{}_predict_{}.dat"
        else:
            self.OUTPUT_PATH_TEST = "../predict_alt1_rnn_change_init/{}" + \
                str(change_num) + "/{}_predict_{}.dat"

        self.FLAG_GPU = flag_gpu
        self.FEATURE_TYPE = feature_type

        # データ次元
        self.data_dimention = 496
        self.word_vec_dimention = 500
        self.data_dimention_added = self.data_dimention + self.word_vec_dimention * 6
        self.middle_data_dimention = self.data_dimention_added // 2
        self.case_list = ["ガ格", "ヲ格", "ニ格"]
        self.num_of_middle_layer = num_of_middle_layer

        self.EMBED_SIZE = self.word_vec_dimention
        self.FNN_SIZE = self.data_dimention_added
        self.HIDDEN_SIZE = self.word_vec_dimention // 4
        self.BATCH_SIZE = 256
        self.EPOCH_NUM = 100
        #self.EPOCH_NUM = 1
        self.EPOCH_TEXT = 5
        self.div_num = 1
        #self.div_num = 24
        #self.EPOCH_LIMIT = 50000
        self.EPOCH_LIMIT = 646763
        #self.EPOCH_LIMIT = 100
        self.USE_DROPOUT = use_dropout
        self.change_num = change_num

        # GPUのセット
        if self.FLAG_GPU:
            self.ARR = cupy
            cuda.get_device(0).use()
            # model.to_gpu(0)
        else:
            self.ARR = np

    def for_one_batch_training(self):
        loss_list = []
        text_count = 0
        model_list = glob.glob(
            "_".join(self.OUTPUT_PATH.format("model", self.FEATURE_TYPE, self.USE_DROPOUT, self.num_of_middle_layer, "*", 0).split("_")[:-1]))
        model = Att_Seq2TF(emb_size=self.EMBED_SIZE,
                           fnn_size=self.FNN_SIZE,
                           hidden_size=self.HIDDEN_SIZE,
                           num_of_middle_layer=self.num_of_middle_layer,
                           use_dropout=self.USE_DROPOUT,
                           flag_gpu=self.FLAG_GPU)
        if len(model_list) != 0:
            for model_cand in sorted(model_list, key=lambda x: int(x.split("_")[-2][9:])):
                loss_list.append(
                    float(model_cand[model_cand.find("loss") + 4:model_cand.rfind(".")]))
            serializers.load_hdf5(model_cand, model)
            text_count = int(model_cand.split("_")[-2][9:])
            print(model_cand)
            print(text_count)
            print(loss_list)
        if self.FLAG_GPU:
            model.to_gpu(0)
        model.reset()
        # print("d")
        opt = optimizers.Adam()
        # optimizer.use_cleargrads()
        opt.setup(model)
        opt.add_hook(optimizer.WeightDecay(0.0005))
        opt.add_hook(optimizer.GradientClipping(5))
        opt_list = glob.glob(
            "_".join(self.OUTPUT_PATH.format("opt", self.FEATURE_TYPE, self.USE_DROPOUT, self.num_of_middle_layer, "*", 0).split("_")[:-1]))
        if len(opt_list) != 0:
            opt_list = sorted(
                opt_list, key=lambda x: int(x.split("_")[-2][9:]))
            serializers.load_hdf5(opt_list[-1], opt)
            print(opt_list[-1])

        # rupe_of_trainging
        # train_losses = []
        # test_losses = []
        print("start...")
        start_time = time.time()
        # 学習開始
        q = Queue(100)
        q_valid = Queue(500)
        q_valid1 = Queue(500)
        minibatch_maker = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 0)
        p = Process(target=minibatch_maker.epoch_pickle, args=(q, ))
        p.start()
        # minibatch_maker1 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 1)
        # p1 = Process(target=minibatch_maker1.epoch_factory, args=(q, ))
        # p1.start()
        # minibatch_maker2 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 2)
        # p2 = Process(target=minibatch_maker2.epoch_factory, args=(q, ))
        # p2.start()
        # minibatch_maker3 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 3)
        # p3 = Process(target=minibatch_maker3.epoch_factory, args=(q, ))
        # p3.start()
        # minibatch_maker4 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 4)
        # p4 = Process(target=minibatch_maker4.epoch_factory, args=(q, ))
        # p4.start()
        # minibatch_maker5 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 5)
        # p5 = Process(target=minibatch_maker5.epoch_factory, args=(q, ))
        # p5.start()
        # minibatch_maker6 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 6)
        # p6 = Process(target=minibatch_maker6.epoch_factory, args=(q, ))
        # p6.start()
        # minibatch_maker7 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 7)
        # p7 = Process(target=minibatch_maker7.epoch_factory, args=(q, ))
        # p7.start()
        # minibatch_maker8 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 8)
        # p8 = Process(target=minibatch_maker8.epoch_factory, args=(q, ))
        # p8.start()
        # minibatch_maker9 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "train", text_count, 9)
        # p9 = Process(target=minibatch_maker9.epoch_factory, args=(q, ))
        # p9.start()
        #train_len = q.get()
        minibatch_maker_valid = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "valid", text_div=0)
        p_valid = Process(
            target=minibatch_maker_valid.epoch_pickle, args=(q_valid, ))
        p_valid.start()
        minibatch_maker_valid1 = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "valid", text_div=1)
        p_valid1 = Process(
            target=minibatch_maker_valid1.epoch_pickle, args=(q_valid1, ))
        p_valid1.start()
        valid_len = q_valid.get()
        valid_len1 = q_valid1.get()
        #valid_len1 = 0
        #print("altsvm" + str(train_len))
        print("altsvm" + str(valid_len))
        print("altsvm" + str(valid_len1))
        # p.terminate()
        # p_valid.terminate()
        # exit()
        waited_count = 0
        verb_data_count = 0
        pseudo_epoch_count = 0
        train_dict_keep = None
        while waited_count < 100 and (len(loss_list) <= 10 or min(loss_list[-10:]) != loss_list[-10]):
            if not q.empty():
                # print("something")
                text_count += self.EPOCH_TEXT
                try_count = 0
                # while try_count < 5:
                #     try:
                #         try_count += 1
                #         print(str(q.full()))
                enc_words, fnn_inputs, dec_scores = q.get()
                # if text_sentence_vec_dict != None:
                #train_dict_keep = text_sentence_vec_dict
                # except Exception as e:
                #     print("cant_get")
                #     print(e)
                # if len(x_train) > 0:
                #     print("can_get")
                #     break
                # sys.exit()
                N = len(dec_scores)
                verb_data_count += N
                if N != 0:
                    # training
                    start_time_train = time.time()
                    perm = np.random.permutation(N)
                    sum_loss = 0
                    # print("first_verb")
                    for i in range(0, N, self.BATCH_SIZE):
                        # print(i)
                        if self.FLAG_GPU:
                            enc_words_batch = []
                            for x in perm[i:i + self.BATCH_SIZE]:
                                enc_words_batch.append(enc_words[x])
                                # enc_words_batch.append(
                                #     train_dict_keep[enc_words[x][0]][enc_words[x][1]])
                            # enc_words_batch = cuda.to_gpu(
                            #    np.array(enc_words_batch), device=0)
                            fnn_inputs_batch = cuda.to_gpu(fnn_inputs[
                                perm[i:i + self.BATCH_SIZE]], device=0)
                            dec_scores_batch = cuda.to_gpu(dec_scores[
                                perm[i:i + self.BATCH_SIZE]], device=0)
                        else:
                            enc_words_batch = []
                            for x in perm[i:i + self.BATCH_SIZE]:
                                enc_words_batch.append(enc_words[x])
                                # enc_words_batch.append(
                                #     train_dict_keep[enc_words[x][0]][enc_words[x][1]])
                            fnn_inputs_batch = fnn_inputs[
                                perm[i:i + self.BATCH_SIZE]]
                            dec_scores_batch = dec_scores[
                                perm[i:i + self.BATCH_SIZE]]

                        # modelのリセット
                        model.reset()
                        # 順伝播
                        model.encode(enc_words_batch)
                        # デコーダーの計算
                        loss = model.decode(fnn_inputs_batch, dec_scores_batch)
                        # print(loss)
                        sum_loss += loss.data * len(dec_scores_batch)
                        loss.backward()
                        opt.update()
                    # print("first_verb_finished")
                    average_loss = sum_loss / N
                    # train_losses.append(average_loss)
                    interval = int(time.time() - start_time_train)
                    #print("train実行時間: {}sec, N: {}".format(interval,N))

                # test
                # loss = model(x_test, y_test)
                # test_losses.append(loss.data)

                # output learning process
                if text_count % 100 == 0:
                    print("text_count: {} train loss: {} verb_data_count: {} time: {}".format(
                        text_count, average_loss, verb_data_count, time.ctime()))
                if verb_data_count // self.EPOCH_LIMIT > pseudo_epoch_count:
                    pseudo_epoch_count += 1
                    # print(verb_data_count)
                    # print(pseudo_epoch_count)

                    total_loss = 0
                    total_count = 0
                    valid_dict_keep = None
                    model.mode_change("test")
                    #chainer.config.train = False
                    valid_count = 0
                    valid1_count = 0
                    while (valid_count + valid1_count) < (valid_len + valid_len1):
                        if valid_count < valid_len and not q_valid.empty():
                            enc_words, fnn_inputs, dec_scores = q_valid.get()
                            valid_count += 1
                        elif valid1_count < valid_len1 and not q_valid1.empty():
                            enc_words, fnn_inputs, dec_scores = q_valid1.get()
                            valid1_count += 1
                        else:
                            print("waiting valid " + str(valid_count) +
                                  " " + str(valid1_count))
                            time.sleep(10)
                            continue
                        # if text_sentence_vec_dict != None:
                        #valid_dict_keep = text_sentence_vec_dict
                        if len(dec_scores) == 0:
                            continue
                        N = len(dec_scores)
                        for i in range(0, N, self.BATCH_SIZE):
                            if self.FLAG_GPU:
                                enc_words_batch = []
                                for x in enc_words[i:i + self.BATCH_SIZE]:
                                    enc_words_batch.append(x)
                                    # enc_words_batch.append(
                                    #     valid_dict_keep[x[0]][x[1]])
                                # enc_words_batch = cuda.to_gpu(
                                #    enc_words_batch, device=0)
                                fnn_inputs_batch = cuda.to_gpu(
                                    fnn_inputs[i:i + self.BATCH_SIZE], device=0)
                                dec_scores_batch = cuda.to_gpu(
                                    dec_scores[i:i + self.BATCH_SIZE], device=0)
                            else:
                                enc_words_batch = []
                                for x in enc_words[i:i + self.BATCH_SIZE]:
                                    enc_words_batch.append(x)
                                    # enc_words_batch.append(
                                    #     valid_dict_keep[x[0]][x[1]])
                                # enc_words_batch = cuda.to_gpu(
                                #    enc_words_batch, device=0)
                                fnn_inputs_batch = fnn_inputs[
                                    i:i + self.BATCH_SIZE]
                                dec_scores_batch = dec_scores[
                                    i:i + self.BATCH_SIZE]
                            # modelのリセット
                            model.reset()
                            if len(enc_words_batch) == 0:
                                print(len(enc_words))
                                print(len(dec_scores_batch))
                                print(i)
                                exit()

                            with chainer.no_backprop_mode():
                                # 順伝播
                                model.encode(enc_words_batch)
                                # デコーダーの計算
                                loss_data = model.decode(
                                    fnn_inputs_batch, dec_scores_batch).data
                                if not self.ARR.isnan(loss_data):
                                    total_loss += loss_data * \
                                        len(dec_scores_batch)
                                    total_count += len(dec_scores_batch)
                                else:
                                    print(loss_data)

                    if total_count == 0:
                        print("skipped")
                        continue
                    valid_loss = float(total_loss / total_count)
                    model.mode_change("train")
                    #chainer.config.train = True
                    # print(valid_loss)
                    # print(total_loss)
                    # print(total_count)
                    print("valid_count: {} valid loss: {} time: {}".format(
                        verb_data_count // self.EPOCH_LIMIT, valid_loss, time.ctime()))
                    try:
                        # with open("test", mode="wb") as f:
                        #    pickle.dump("hui",f)
                        # with open(self.OUTPUT_PATH.format("opt", self.FEATURE_TYPE, str(self.USE_DROPOUT), str(self.num_of_middle_layer), str(verb_count // self.EPOCH_LIMIT), valid_loss), mode="wb") as f:
                        #    pickle.dump(opt,f)
                        # print("will_save")
                        # model_saved=model.copy()
                        # model_saved.to_cpu()
                        # fui=float(70)
                        serializers.save_hdf5(  # "/gs/hs0/tga-cl/yamashiro-s-aa/workspace/nn/fnn/model/model",model)
                            self.OUTPUT_PATH.format("model", self.FEATURE_TYPE, self.USE_DROPOUT, self.num_of_middle_layer, text_count, float(valid_loss)), model)
                        # print("model_saved")
                        serializers.save_hdf5(
                            self.OUTPUT_PATH.format("opt", self.FEATURE_TYPE, self.USE_DROPOUT, self.num_of_middle_layer, text_count, float(valid_loss)), opt)
                    except Exception as e:
                        raise e
                    # print("saved")
                    loss_list.append(valid_loss)

                    # q_valid.put((x_valid, y_valid))
                waited_count = 0
            else:
                print("waiting")
                time.sleep(10)
                print(str(text_count) + " " + str(q.qsize()))
                waited_count += 1

        print("end")
        p.terminate()
        # p1.terminate()
        # p2.terminate()
        # p3.terminate()
        # p4.terminate()
        # p5.terminate()
        # p6.terminate()
        # p7.terminate()
        # p8.terminate()
        # p9.terminate()
        p_valid.terminate()
        p_valid1.terminate()
        interval = int(time.time() - start_time)
        print("実行時間: {}sec, last pseudo_epoch: {}".format(
            interval, str(verb_data_count // self.EPOCH_LIMIT)))

    def for_one_batch_test(self):
        # テストをallでやる処理
        model = None
        model_list = glob.glob(
            "_".join(self.OUTPUT_PATH.format("model", self.FEATURE_TYPE, "*", "*", "*", 0).split("_")[:-1]))
        loss_min = 100
        loss_min_file_path = None
        for i in model_list:
            if float(i[i.find("loss") + 4:i.rfind(".")]) < loss_min:
                loss_min = float(i[i.find("loss") + 4:i.rfind(".")])
                loss_min_file_path = i
        print(loss_min_file_path)
        for y in loss_min_file_path.split("_"):
            if "dim" in y:
                self.num_of_middle_layer = int(y[3:])
            if "dropout" in y:
                self.USE_DROPOUT = float(y[7:])
        # print(self.num_of_middle_layer)
        # print(self.USE_DROPOUT)
        # exit()
        # model = MyChain(self.data_dimention_added, self.middle_data_dimention,
        # self.num_of_middle_layer, self.USE_DROPOUT, self.FLAG_GPU)
        model = Att_Seq2TF(emb_size=self.EMBED_SIZE,
                           fnn_size=self.FNN_SIZE,
                           hidden_size=self.HIDDEN_SIZE,
                           num_of_middle_layer=self.num_of_middle_layer,
                           use_dropout=self.USE_DROPOUT,
                           flag_gpu=self.FLAG_GPU)
        serializers.load_hdf5(loss_min_file_path, model)
        if self.FLAG_GPU:
            model.to_gpu(0)
        # self.EPOCH_NUM = 1
        q = Queue(100)
        self.div_num = 3
        minibatch_maker = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=0, change_num=self.change_num)
        p = Process(target=minibatch_maker.epoch_pickle_test, args=(q, ))
        p.start()
        minibatch_maker1 = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=1, change_num=self.change_num)
        p1 = Process(target=minibatch_maker1.epoch_pickle_test, args=(q, ))
        p1.start()
        minibatch_maker2 = MinibatchMaker(
            self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=2, change_num=self.change_num)
        p2 = Process(target=minibatch_maker2.epoch_pickle_test, args=(q, ))
        p2.start()
        # minibatch_maker3 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=3, div_t=self.div_t * 4 + self.FLAG_GPU)
        # p3 = Process(target=minibatch_maker3.epoch_factory_test, args=(q, ))
        # p3.start()
        # minibatch_maker4 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=4, div_t=self.div_t * 4 + self.FLAG_GPU)
        # p4 = Process(target=minibatch_maker4.epoch_factory_test, args=(q, ))
        # p4.start()
        # minibatch_maker5 = MinibatchMaker(
        #     self.FEATURE_TYPE, self.FLAG_GPU, "test", text_div=5, div_t=self.div_t * 4 + self.FLAG_GPU)
        # p5 = Process(target=minibatch_maker5.epoch_factory_test, args=(q, ))
        # p5.start()

        # print("altsvm" + str(test_len))
        # print("altsvm" + str(test_len1))
        # print("altsvm" + str(test_len2))

        waited_count = 0
        text_count = 0
        model.mode_change("test")
        text_num_keep = 0
        break_count = 0
        while waited_count < 500:
            if not q.empty():
                text_num_keep, enc_words, x_test = q.get()
                text_count += 1
            else:
                print("waiting test " + str(text_count))
                time.sleep(10)
                waited_count += 1
                continue
            if text_num_keep == -1:
                print(str(text_count) + ":" + str(self.FLAG_GPU))
                break_count += 1
                if break_count == self.div_num:
                    break
                continue
            N = len(x_test)
            if N == 0:
                continue
            f_new = codecs.open(
                self.OUTPUT_PATH_TEST.format(self.FEATURE_TYPE, loss_min_file_path.split("/")[-1], str(text_num_keep)), 'w', 'utf-8')

            for i in range(0, N, self.BATCH_SIZE):
                if self.FLAG_GPU > 0:
                    enc_words_batch = []
                    for x in enc_words[i:i + self.BATCH_SIZE]:
                        enc_words_batch.append(x)
                    x_test_batch = cuda.to_gpu(
                        x_test[i:i + self.BATCH_SIZE], device=0)
                else:
                    enc_words_batch = []
                    for x in enc_words[i:i + self.BATCH_SIZE]:
                        enc_words_batch.append(x)
                    x_test_batch = x_test[i:i + self.BATCH_SIZE]
                with chainer.no_backprop_mode():
                    # 順伝播
                    model.encode(enc_words_batch)
                    # デコーダーの計算
                    _, test_result = F.split_axis(
                        F.softmax(model.get_predata(x_test_batch), axis=1), 2, axis=1)
                    # if not self.ARR.isnan(loss_data):
                    #     total_loss += loss_data * \
                    #         len(dec_scores_batch)
                    #     total_count += len(dec_scores_batch)
                    # else:
                    #     print(loss_data)
                    # 1になる確率が出ているはず
                    # _, test_result = F.split_axis(
                    #     F.softmax(model.get_predata(x_test_batch), axis=1), 2, axis=1)
                # print(test_result.tolist())
                # print(type(test_result.tolist()))
                for result_x in test_result:
                    # print(np.asscalar(result_x).data)
                    # print(type(np.asscalar(result_x)))
                    f_new.write(str(float(result_x.data)) + "\n")
            # break
            # if text_count % 10 == 0:
            #     print("text_count: {} train loss: {}".format(
            #         text_count, average_loss))
            # print(text_count)
            waited_count = 0
            f_new.close()
        p.terminate()
        p1.terminate()
        p2.terminate()
        # p3.terminate()
        # p4.terminate()
        # p5.terminate()


if __name__ == '__main__':
    # minibatch_maker = MinibatchMaker(False)
    # enc_words, fnn_inputs, dec_scores, text_sentence_vec_dict = minibatch_maker.make_minibatch(
    #     ["knp_dismantled_data/00029_A_OY06_00168.knp"], 0)
    # print(np.shape(enc_words))
    # print(np.shape(fnn_inputs))
    # print(np.shape(dec_scores))
    # model = Att_Seq2TF(emb_size=500,
    #                    fnn_size=3496,
    #                    hidden_size=500,
    #                    batch_size=30,
    #                    use_dropout=0.1,
    #                    flag_gpu=False)

    # enc_words_list = []
    # for x in enc_words[:2]:
    #     enc_words_list.append(text_sentence_vec_dict[x[0]][x[1]])
    # # ここfor文で回さずに塊（バッチ）渡せば行列で計算できるようにしたい
    # model.reset()
    # model.encode(enc_words_list)
    # # デコーダーの計算
    # loss = model.decode(fnn_inputs[:2], dec_scores[:2])
    # print(loss)

    args = sys.argv
    feature_type = args[1]
    use_dropout = float(args[2])
    num_of_middle_layer = int(args[3])
    flag_gpu = int(args[4])
    change_num = int(args[5])
    trainer = TrainAndTest(feature_type, use_dropout,
                           num_of_middle_layer, flag_gpu, change_num)
    print(feature_type)
    print(use_dropout)
    print(num_of_middle_layer)
    print(flag_gpu)
    print(change_num)
    #trainer = TrainAndTest("rnn_att", 0.25, 1, 0)
    trainer.for_one_batch_training()
    # trainer.for_one_batch_test()
    # minibatch_maker = MinibatchMaker(
    #     feature_type, flag_gpu, "train", 0, num_of_middle_layer)
    # minibatch_maker.epoch_factory()
    # print("end:" + str(num_of_middle_layer))
    # 正解単語と予測単語を照らし合わせて損失を計算

    # print('開始: ', datetime.datetime.now())
    # try:
    #     train_and_test = TrainAndTest(False)
    #     train_and_test.train(0)
    # except:
    #     raise

    # print('終了: ', datetime.datetime.now())

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
import os.path
from gru_train_vec_maker import TrainAndTest
from gru_input import Input_for_gru


class MinibatchMaker(TrainAndTest):
    """docstring for MinibatchMaker"""

    def __init__(self, feature_type, flag_gpu, is_validation="train", text_count=0, text_div=0):
        super(MinibatchMaker, self).__init__(feature_type, flag_gpu=0)
        self.is_validation = is_validation
        self.vec_data_dict = {}
        self.text_count = text_count
        self.text_div = text_div
        self.genre_list = ["OY", "OC", "OW", "PB", "PN", "PM"]
        for genre in self.genre_list:
            vec_data_dict_sub = {}
            vec_data_dict_sub["text_name"] = None
            vec_data_dict_sub["vec_for_text_dic"] = {}
            vec_data_dict_sub["cf_vec_for_text_dic"] = {}
            self.vec_data_dict[genre] = vec_data_dict_sub

    def epoch_factory(self, q=None):
        #print(self.is_validation + " fetch")
        filepath = glob.glob(self.INPUT_PATH_KNP)
        filepath_part = sorted(self._fetch_data(filepath, self.is_validation))
        print(len(filepath_part))
        #print(self.is_validation + ":" + str(len(filepath_part)))
        # ここ120に固定できるように
        if self.is_validation == "valid":
            q.put(len(self._fetch_data(
                glob.glob(self.INPUT_PATH.format("*")), self.is_validation)))
        file_count = 0
        data_fetcher = Input_for_gru(self.FLAG_GPU, self.text_div)

        for epoch in range(self.EPOCH_NUM):
            # print(filepath_part)
            # koko
            already_pickle = glob.glob(
                "../all_test_divs_alt_vector1/*.pickle")
            if self.is_validation == "train" or self.is_validation == "test":
                # print(filepath_part)
                for file in filepath_part:
                    if self.FEATURE_TYPE == "rnn_att":
                        file_pack = []
                        data_fetcher = Input_for_gru(
                            self.FLAG_GPU, self.text_div)
                        data_fetcher(file)
                        # data_fetcher(
                        #     "../../knp_dismantled_data/00029_A_OY06_00168.knp")
                        #print("data fetched")
                        text_name_keeper = None
                        #print("x" + file)
                        print(list(data_fetcher.yougen_in_which_sentence.keys()))
                        for text_name in data_fetcher.yougen_in_which_sentence:
                            for row_num in data_fetcher.yougen_in_which_sentence[text_name]:
                                file_count += 1
                                if file_count < (self.text_count // self.div_num):
                                    continue
                                # koko
                                if "../all_test_divs_alt_vector1/" + text_name + "_" + row_num + ".pickle" in already_pickle:
                                    pass

                                file_pack.append((text_name, row_num))
                                # print("y")
                                if len(file_pack) >= self.EPOCH_TEXT:
                                    # print(len(file_pack))
                                    enc_words = []
                                    dec_scores = self.ARR.empty(
                                        (0, 1), dtype=np.float32)
                                    fnn_inputs = self.ARR.empty(
                                        (0, self.data_dimention_added), dtype=np.float32)
                                    for file_x in file_pack:
                                        # print("z")
                                        sentence_num = data_fetcher.yougen_in_which_sentence[
                                            file_x[0]][file_x[1]]
                                        # print("a")
                                        pickle_name = glob.glob(self.INPUT_PATH.format(
                                            file_x[0] + "_" + file_x[1]))[0]
                                        # if os.path.getsize(file) > 5000000:
                                        #     print(file)
                                        #     continue
                                        # print("b")
                                        data_list_x, data_list_y, train_info_dict = self._make_dataset_info(
                                            pickle_name)
                                        # print("c")
                                        cf_list = self._e_cf_list_fetcher(
                                            train_info_dict, text_name)
                                        # print(len(cf_list))
                                        # print("d")
                                        data_fetcher.cf_vec_fetcher(cf_list)
                                        # print("e")
                                        data_list_x, _ = self._vec_addeder(
                                            data_list_x, train_info_dict, data_fetcher)
                                        # 名前突っ込んどくので，ランダマイズの後からdata_fetcher.text_sentence_vec_dict[text_name][sentence_num]すること
                                        # print("f")
                                        for _ in range(len(train_info_dict)):
                                            enc_words.append(
                                                (file_x[0], sentence_num))
                                        # print(self.ARR.shape(fnn_inputs))
                                        # print(self.ARR.shape(data_list_x))
                                        # print("g")
                                        fnn_inputs = self.ARR.concatenate(
                                            [fnn_inputs, data_list_x], axis=0)
                                        # print(np.shape(data_list_y))
                                        # print(
                                        #     np.shape(data_list_y.reshape(-1, 1)))
                                        # print(np.shape(dec_scores))
                                        # print("h")
                                        dec_scores = self.ARR.concatenate(
                                            [dec_scores, data_list_y.reshape(-1, 1)], axis=0)
                                        # koko
                                        enc_words_batch = []
                                        for x in enc_words:
                                            enc_words_batch.append(
                                                data_fetcher.text_sentence_vec_dict[x[0]][x[1]])
                                        with open("../all_test_divs_alt_vector1/" + file_x[0] + "_" + file_x[1] + ".pickle", mode='wb') as f:
                                            pickle.dump((enc_words_batch, fnn_inputs, dec_scores.astype(
                                                self.ARR.int32).flatten()), f)
                                            print(file_x[0] +
                                                  "_" + file_x[1])
                                            # print(dec_scores.astype(
                                            #     self.ARR.int32).flatten())
                                    if len(dec_scores) == 0:
                                        print("kara")
                                        file_pack = []
                                        continue
                                    # if text_name != text_name_keeper:
                                    # koko
                                    # q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                                    #        data_fetcher.text_sentence_vec_dict))
                                    text_name_keeper = text_name
                                    # else:
                                    # q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                                    # None))
                                    file_pack = []
                                    # print(q.qsize())
            else:
                while 1:
                    for file in filepath_part:
                        if self.FEATURE_TYPE == "rnn_att":
                            data_fetcher = Input_for_gru(self.FLAG_GPU)
                            data_fetcher(file)
                            text_name_keeper = None
                            for text_name in data_fetcher.yougen_in_which_sentence:
                                for row_num in data_fetcher.yougen_in_which_sentence[text_name]:
                                    file_count += 1
                                    # ここで20制限かける
                                    enc_words = []
                                    dec_scores = self.ARR.empty(
                                        (0, 1), dtype=np.float32)
                                    fnn_inputs = self.ARR.empty(
                                        (0, self.data_dimention_added), dtype=np.float32)
                                    sentence_num = data_fetcher.yougen_in_which_sentence[
                                        text_name][row_num]
                                    pickle_name = glob.glob(self.INPUT_PATH.format(
                                        text_name + "_" + row_num))[0]

                                    data_list_x, data_list_y, train_info_dict = self._make_dataset_info(
                                        pickle_name)
                                    cf_list = self._e_cf_list_fetcher(
                                        train_info_dict, text_name)
                                    data_fetcher.cf_vec_fetcher(cf_list)
                                    data_list_x, _ = self._vec_addeder(
                                        data_list_x, train_info_dict, data_fetcher)
                                    # 名前突っ込んどくので，ランダマイズの後からdata_fetcher.text_sentence_vec_dict[text_name][sentence_num]すること
                                    for _ in range(len(train_info_dict)):
                                        enc_words.append(
                                            (text_name, sentence_num))
                                    # print(self.ARR.shape(fnn_inputs))
                                    # print(self.ARR.shape(data_list_x))
                                    fnn_inputs = self.ARR.concatenate(
                                        [fnn_inputs, data_list_x], axis=0)
                                    dec_scores = self.ARR.concatenate(
                                        [dec_scores, data_list_y.reshape(-1, 1)], axis=0)
                                    if len(dec_scores) == 0:
                                        print("kara")
                                        continue
                                    # if text_name != text_name_keeper:
                                    q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                                           data_fetcher.text_sentence_vec_dict))
                                    text_name_keeper = text_name
                                    # else:
                                    # q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                                    # None))

    def epoch_pickle(self, q=None):
        #print(self.is_validation + " fetch")
        filepath = glob.glob(self.INPUT_PATH.format("*"))
        filepath_part = sorted(self._fetch_data(filepath, self.is_validation))
        # print(len(filepath_part))
        #print(self.is_validation + ":" + str(len(filepath_part)))
        # ここ120に固定できるように
        if self.is_validation == "valid":
            q.put(len(self._fetch_data(
                glob.glob(self.INPUT_PATH.format("*")), self.is_validation)))
        file_count = 0
        #data_fetcher = Input_for_gru(self.FLAG_GPU, self.text_div)

        for epoch in range(self.EPOCH_NUM):
            # print(filepath_part)
            # koko
            #already_pickle = glob.glob("../all_test_divs_alt_vector/*.pickle")
            if self.is_validation == "train" or self.is_validation == "test":
                # print(filepath_part)
                for file in filepath_part:
                    if self.FEATURE_TYPE == "rnn_att":
                        file_pack = []
                        # data_fetcher = Input_for_gru(
                        #     self.FLAG_GPU, self.text_div)
                        # data_fetcher(file)
                        # data_fetcher(
                        #     "../../knp_dismantled_data/00029_A_OY06_00168.knp")
                        #print("data fetched")
                        text_name_keeper = None
                        #print("x" + file)
                        # print(list(data_fetcher.yougen_in_which_sentence.keys()))
                        # for text_name in data_fetcher.yougen_in_which_sentence:
                        # for row_num in
                        # data_fetcher.yougen_in_which_sentence[text_name]:
                        file_count += 1
                        if file_count < (self.text_count // self.div_num):
                            continue
                        # koko
                        # if "../all_test_divs_alt_vector/" + text_name + "_" + row_num + ".pickle" in already_pickle:
                        #     pass

                        file_pack.append(file)
                        # print("y")
                        if len(file_pack) >= self.EPOCH_TEXT:
                            #enc_words = []
                            dec_scores = self.ARR.empty(
                                (0, 1), dtype=np.float32)
                            fnn_inputs = self.ARR.empty(
                                (0, self.data_dimention_added), dtype=np.float32)
                            enc_words_pack = []
                            for file_x in file_pack:
                                # print("z")
                                # sentence_num = data_fetcher.yougen_in_which_sentence[
                                #     file_x[0]][file_x[1]]
                                # print("a")
                                # pickle_name = glob.glob(self.INPUT_PATH.format(
                                #     file_x[0] + "_" + file_x[1]))[0]
                                # if os.path.getsize(file) > 5000000:
                                #     print(file)
                                #     continue
                                # print("b")
                                # data_list_x, data_list_y, train_info_dict = self._make_dataset_info(
                                #     pickle_name)
                                # print("c")
                                # cf_list = self._e_cf_list_fetcher(
                                #     train_info_dict, text_name)
                                # print(len(cf_list))
                                # print("d")
                                # data_fetcher.cf_vec_fetcher(cf_list)
                                # print("e")
                                # data_list_x, _ = self._vec_addeder(
                                #     data_list_x, train_info_dict, data_fetcher)
                                # 名前突っ込んどくので，ランダマイズの後からdata_fetcher.text_sentence_vec_dict[text_name][sentence_num]すること
                                # print("f")
                                # for _ in range(len(train_info_dict)):
                                #     enc_words.append(
                                #         (file_x[0], sentence_num))
                                # print(self.ARR.shape(fnn_inputs))
                                # print(self.ARR.shape(data_list_x))
                                # print("g")
                                with open(file_x, mode='rb') as f:
                                    enc_words_batch, data_list_x, data_list_y = pickle.load(
                                        f)
                                enc_words_pack.extend(enc_words_batch)
                                fnn_inputs = self.ARR.concatenate(
                                    [fnn_inputs, data_list_x], axis=0)
                                # print(np.shape(data_list_y))
                                # print(
                                #     np.shape(data_list_y.reshape(-1, 1)))
                                # print(np.shape(dec_scores))
                                # print("h")
                                dec_scores = self.ARR.concatenate(
                                    [dec_scores, data_list_y.reshape(-1, 1)], axis=0)
                                # koko
                                # enc_words_batch = []
                                # for x in enc_words:
                                #     enc_words_batch.append(
                                #         data_fetcher.text_sentence_vec_dict[x[0]][x[1]])
                                # with open("../all_test_divs_alt_vector/" + file_x[0] + "_" + file_x[1] + ".pickle", mode='wb') as f:
                                #     pickle.dump((enc_words_batch, fnn_inputs, dec_scores.astype(
                                #         self.ARR.int32).flatten()), f)
                                #     print(file_x[0] + "_" + file_x[1])
                            if len(dec_scores) == 0:
                                print("kara")
                                file_pack = []
                                continue
                            # if text_name != text_name_keeper:
                            # koko
                            q.put((enc_words_pack, fnn_inputs, dec_scores.astype(
                                self.ARR.int32).flatten()))
                            text_name_keeper = text_name
                            # else:
                            # q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                            # None))
                            file_pack = []
                            # print(q.qsize())
            else:
                while 1:
                    for file in filepath_part:
                        if self.FEATURE_TYPE == "rnn_att":
                            # data_fetcher = Input_for_gru(self.FLAG_GPU)
                            # data_fetcher(file)
                            text_name_keeper = None
                            # for text_name in data_fetcher.yougen_in_which_sentence:
                            # for row_num in
                            # data_fetcher.yougen_in_which_sentence[text_name]:
                            file_count += 1
                            # ここで20制限かける
                            enc_words = []
                            dec_scores = self.ARR.empty(
                                (0, 1), dtype=np.float32)
                            fnn_inputs = self.ARR.empty(
                                (0, self.data_dimention_added), dtype=np.float32)
                            # sentence_num = data_fetcher.yougen_in_which_sentence[
                            #     text_name][row_num]
                            # pickle_name = glob.glob(self.INPUT_PATH.format(
                            #     text_name + "_" + row_num))[0]

                            # data_list_x, data_list_y, train_info_dict = self._make_dataset_info(
                            #     pickle_name)
                            # cf_list = self._e_cf_list_fetcher(
                            #     train_info_dict, text_name)
                            # data_fetcher.cf_vec_fetcher(cf_list)
                            # data_list_x, _ = self._vec_addeder(
                            #     data_list_x, train_info_dict, data_fetcher)
                            # 名前突っ込んどくので，ランダマイズの後からdata_fetcher.text_sentence_vec_dict[text_name][sentence_num]すること
                            # for _ in range(len(train_info_dict)):
                            #     enc_words.append(
                            #         (text_name, sentence_num))
                            # print(self.ARR.shape(fnn_inputs))
                            # print(self.ARR.shape(data_list_x))
                            with open(file_x, mode='rb') as f:
                                enc_words_batch, data_list_x, data_list_y = pickle.load(
                                    f)
                            enc_words.extend(enc_words_batch)
                            fnn_inputs = self.ARR.concatenate(
                                [fnn_inputs, data_list_x], axis=0)
                            dec_scores = self.ARR.concatenate(
                                [dec_scores, data_list_y.reshape(-1, 1)], axis=0)
                            if len(dec_scores) == 0:
                                print("kara")
                                continue
                            # if text_name != text_name_keeper:
                            q.put((enc_words, fnn_inputs, dec_scores.astype(
                                self.ARR.int32).flatten()))
                            text_name_keeper = text_name
                            # else:
                            # q.put((enc_words, fnn_inputs, dec_scores.astype(self.ARR.int32).flatten(),
                            # None))

    def _fetch_data(self, filepath, is_validation="train", is_BCCWJ=True):
        genre_train_div_num_dict = {
            #"OY": 283, "OC": 593, "OW": 42, "PB": 49, "PM": 51, "PN": 188}
            #"OY": 280, "OC": 580, "OW": 42, "PB": 49, "PM": 51, "PN": 187}
            "OY": 270, "OC": 567, "OW": 40, "PB": 48, "PM": 49, "PN": 180}
        genre_valid_div_num_dict = {
            "OY": 284, "OC": 597, "OW": 43, "PB": 50, "PM": 52, "PN": 189}
        #"OY": 1000, "OC": 597, "OW": 43, "PB": 50, "PM": 52, "PN": 189}
        train_set = set()
        valid_set = set()
        test_set = set()
        valid_dic = defaultdict(int)
        for file in filepath:
            if len(file.split("/")[-1]) > 10:
                file_num = int(file.split("/")[-1].split("_")[0])
                file_genre = file.split("/")[-1].split("_")[2][:2]
                #file_text_name = file[:file.rfind("_")] + "*.dat"
                file_text_name = file
                # koko
                if file_num < genre_train_div_num_dict[file_genre]:
                    if file_num % self.div_num == self.text_div:
                        train_set.add(file_text_name)
                elif file_num < genre_valid_div_num_dict[file_genre]:
                    # if valid_dic[file_genre] < 20:
                    # ここは1にする（1KNPだけ入る）
                    # if file_num % 2 == self.text_div:
                    valid_set.add(file_text_name)
                    valid_dic[file_genre] += 1
                else:
                    if file_num % self.div_num == self.text_div:
                        test_set.add(file_text_name)
        print(valid_dic)
        if is_validation == "valid":
            return list(valid_set)
        elif is_validation == "train":
            return list(train_set)
        else:
            return list(test_set)

    def make_minibatch(self, minibatch):
        # enc_wordsの作成
        data_fetcher = Input_for_gru(self.FLAG_GPU)
        data_fetcher(minibatch)
        enc_words = []
        dec_scores = []
        fnn_inputs = self.ARR.empty(
            (0, self.data_dimention_added), dtype=np.float32)
        for text_name in data_fetcher.yougen_in_which_sentence:
            for row_num in data_fetcher.yougen_in_which_sentence[text_name]:
                sentence_num = data_fetcher.yougen_in_which_sentence[
                    text_name][row_num]
                pickle_name = "resultx*_data/" + \
                    text_name + "_" + row_num + ".dat"
                data_list_x, data_list_y, train_info_dict = self._make_dataset_info(
                    pickle_name)
                data_list_x, _ = self._vec_addeder(
                    data_list_x, train_info_dict, data_fetcher)
                # 名前突っ込んどくので，ランダマイズの後からdata_fetcher.text_sentence_vec_dict[text_name][sentence_num]すること
                for _ in range(len(train_info_dict)):
                    enc_words.append((text_name, sentence_num))
                # print(self.ARR.shape(fnn_inputs))
                # print(self.ARR.shape(data_list_x))
                fnn_inputs = self.ARR.concatenate(
                    [fnn_inputs, data_list_x], axis=0)
                dec_scores.extend(data_list_y)
                # print(text_name)
                # print(row_num)
                break

        return enc_words, fnn_inputs, dec_scores, data_fetcher.text_sentence_vec_dict

    def _make_dataset_info(self, pickle_name):
        #print('fetch zero_feature data')
        text1 = pickle_name
        count = 0
        train_info_dict = {}
        train_info_dict_sub = {}
        # data_list_x = self.ARR.empty(
        #     (0, self.data_dimention), dtype=np.float32)
        # data_list_x_sub = self.ARR.array([], dtype=np.float32)
        data_list_x = []
        data_list_x_sub = []
        data_list_y = self.ARR.empty((0, 1), dtype=np.float32)
        # print(data_list_x.dtype)
        ori_num_keep = -1
        text_name_keep = None
        with codecs.open(text1, 'r', 'utf-8') as f:
            for row1 in f:
                words1 = row1.rstrip("\n").split()
                #data_list_x_sub = self.ARR.array([], dtype=np.float32)
                data_list_x_sub = []
                if len(words1) > 1:
                    if row1[0] == "#":
                        if len(words1[1]) == 18:
                            train_info_dict_sub["text_name"] = words1[1]
                            text_name_keep = words1[1]
                        elif words1[1] == "用言:":
                            train_info_dict_sub["p_num"] = int(
                                words1[2].split(":")[0])
                            train_info_dict_sub["p"] = words1[2].split(":")[1]
                            train_info_dict_sub["cf"] = words1[4]
                            if words1[6].split(":")[0] != "None":
                                train_info_dict_sub["ガ格_num"] = int(
                                    words1[6].split(":")[0])
                            else:
                                train_info_dict_sub["ガ格_num"] = words1[6].split(":")[
                                    0]
                            train_info_dict_sub["ガ格"] = words1[6].split(":")[1]
                            if words1[8].split(":")[0] != "None":
                                train_info_dict_sub["ヲ格_num"] = int(
                                    words1[8].split(":")[0])
                            else:
                                train_info_dict_sub["ヲ格_num"] = words1[8].split(":")[
                                    0]
                            train_info_dict_sub["ヲ格"] = words1[8].split(":")[1]
                            if words1[10].split(":")[0] != "None":
                                train_info_dict_sub["ニ格_num"] = int(
                                    words1[10].split(":")[0])
                            else:
                                train_info_dict_sub["ニ格_num"] = words1[10].split(":")[
                                    0]
                            train_info_dict_sub["ニ格"] = words1[10].split(":")[
                                1]
                            if len(train_info_dict_sub) > 0:
                                train_info_dict[count] = train_info_dict_sub
                                text_name_keeper = train_info_dict_sub[
                                    "text_name"]
                                train_info_dict_sub = {}
                                train_info_dict_sub[
                                    "text_name"] = text_name_keeper
                                count += 1
                    if row1[0] != "#":
                        count_tag = 1
                        for word_num1, word1 in enumerate(words1):
                            if word_num1 == 0 and word1 == "0":
                                data_list_y = self.ARR.append(
                                    data_list_y, int(0))
                            elif word_num1 == 0 and (word1 == "1" or word1 == "2"):
                                data_list_y = self.ARR.append(
                                    data_list_y, int(1))
                            elif word_num1 == 1 and word1[:4] == "qid:":
                                if ori_num_keep != int(word1[4:]):
                                    ori_num_keep = int(word1[4:])
                                    # if len(data_list_x) > 0:
                                    #     data_list_x.append(data_list_x)
                                    #     data_list_x = self.ARR.empty(
                                    #         (0, self.data_dimention), dtype='float32')
                                    #     data_list_y.append(
                                    #         data_list_y[:-1])
                                    #     data_list_y = self.ARR.array(
                                    #         [data_list_y[-1:]], dtype='float32')

                            elif word_num1 > 1:
                                tag_num = int(word1.split(":")[0])
                                if tag_num == count_tag:
                                    # data_list_x_sub = self.ARR.append(
                                    # data_list_x_sub,
                                    # float(word1.split(":")[1]))
                                    data_list_x_sub.append(
                                        float(word1.split(":")[1]))
                                    count_tag += 1
                                else:
                                    while tag_num > count_tag:
                                        # data_list_x_sub = self.ARR.append(
                                        #     data_list_x_sub, float(0))
                                        data_list_x_sub.append(float(0))
                                        count_tag += 1
                                    # data_list_x_sub = self.ARR.append(
                                    # data_list_x_sub,
                                    # float(word1.split(":")[1]))
                                    data_list_x_sub.append(
                                        float(word1.split(":")[1]))
                                    count_tag += 1

                        while count_tag <= self.data_dimention:
                            # data_list_x_sub = self.ARR.append(
                            #     data_list_x_sub, float(0))
                            data_list_x_sub.append(float(0))
                            count_tag += 1
                if len(data_list_x_sub) > 0:
                    # print(len(data_list_x_sub))
                    # data_list_x = self.ARR.concatenate(
                    #     (data_list_x, data_list_x_sub.reshape(1, -1)), axis=0)
                    data_list_x.append(data_list_x_sub)
                # debag_from
                # if count > 1000:
                #     break
                # debag_to
            # if len(data_list_x_sub) > 0:
            #     data_list_x.append(data_list_x_sub)
            #     data_list_x_sub = self.ARR.empty(
            #         (0, self.data_dimention), float)
            #     data_list_y.append(data_list_y_sub)
            #     data_list_y_sub = self.ARR.array([])
        data_list_x_np = self.ARR.empty(
            (0, self.data_dimention), dtype=np.float32)
        if len(data_list_x) > 0:
            for data_list_x_sub_ in data_list_x:
                data_list_x_np = self.ARR.concatenate(
                    (data_list_x_np, self.ARR.array(data_list_x_sub_, dtype=np.float32).reshape(1, -1)), axis=0)

        return data_list_x_np.astype(self.ARR.float32), data_list_y.astype(self.ARR.int32), train_info_dict

    def _e_cf_list_fetcher(self, info_dict, text_name):
        # e_set = set()
        cf_set = set()
        for num in info_dict:
            if info_dict[num]["text_name"] == text_name:
                cf_set.add(info_dict[num]["cf"])
                # for case in self.case_list:
                #     e_set.add(info_dict[num][case])
                # now_word = info_dict[num]["p"].split("+")[0].replace("~テ形", "")
                # if "?" in now_word:
                #     now_word = now_word.split("?")[0].split("/")[1]
                # else:
                #     now_word = now_word.split("/")[0]
                # e_set.add(now_word)
        return list(cf_set)

    def _take_word_vec(self, info_dic, data_fetcher):
        text_name = info_dic["text_name"]

        vec_dict = {}
        unknown_words = {}
        for now_case in self.case_list:
            now_word = info_dic[now_case]
            if now_word != "None" and now_word in data_fetcher.vec_for_text_dic:
                vec_dict[now_case] = data_fetcher.vec_for_text_dic[now_word]
            else:
                vec_dict[now_case] = self.ARR.zeros(
                    self.word_vec_dimention, dtype=np.float32)

        now_cf = info_dic["cf"]
        for now_case in self.case_list:
            if (now_cf, now_case) not in data_fetcher.cf_vec_for_text_dic:
                data_fetcher.cf_vec_for_text_dic_maker(now_cf)
                vec_dict[
                    "cf_" + now_case] = data_fetcher.cf_vec_for_text_dic[(now_cf, now_case)]
            else:
                vec_dict[
                    "cf_" + now_case] = data_fetcher.cf_vec_for_text_dic[(now_cf, now_case)]

        return vec_dict

    def _vec_addeder(self, data_list_x, info_dict, data_fetcher):
        new_x_train = self.ARR.empty(
            (0, self.data_dimention_added), dtype=np.float32)
        #new_x_train_sub = self.ARR.array([], dtype=np.float32)
        new_x_train_sub = []
        count_for_info = 0
        # print(self.ARR.shape(data_list_x))
        for one_x_train in data_list_x:
            #new_x_train_sub = one_x_train.copy()
            new_x_train_sub = list(one_x_train.copy())
            word_vec_dict = self._take_word_vec(
                info_dict[count_for_info], data_fetcher)
            count_for_info += 1
            for now_case in self.case_list:
                # print(self.ARR.shape(new_x_train_sub))
                # print(self.ARR.shape(word_vec_dict[now_case]))
                # print(a)
                # new_x_train_sub = self.ARR.concatenate(
                #     (new_x_train_sub, word_vec_dict[now_case]), axis=0)
                new_x_train_sub.extend(word_vec_dict[now_case])
            for now_case in self.case_list:
                # new_x_train_sub = self.ARR.concatenate(
                #     (new_x_train_sub, word_vec_dict["cf_" + now_case]), axis=0)
                new_x_train_sub.extend(word_vec_dict["cf_" + now_case])
            try:
                new_x_train = self.ARR.concatenate(
                    (new_x_train, self.ARR.array(new_x_train_sub, dtype=np.float32).reshape(1, -1)), axis=0)
            except Exception as e:
                print(e)
                print(len(new_x_train_sub), word_vec_dict)
                print()
                sys.exit(info_dict[count_for_info - 1])

        return new_x_train.astype(self.ARR.float32), count_for_info

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

import sqlite3
#from gru_train_vec_maker import TrainAndTest
from gru_train import TrainAndTest


class Input_for_gru(TrainAndTest):
    """三文前までの文章を取ってきてベクトル化する"""

    def __init__(self, flag_gpu, text_div=0):
        super(Input_for_gru, self).__init__(flag_gpu=0)
        # GPUを使う場合はcupyを使わないときはnumpyを使う
        if flag_gpu:
            self.ARR = cupy
        else:
            self.ARR = np
        self.connector_for_input = sqlite3.connect(
            "../../cf_J_w2v_mini_alt" + str(text_div % 16) + ".db", timeout=100.0)
        self.cursor_for_input = self.connector_for_input.cursor()
        self.text_div = text_div

    def __call__(self, text_name_list):
        if not isinstance(text_name_list, list):
            text_name_list = [text_name_list]
        pickle_names = []
        self.text_all_dict = {}
        for text_name_of_knp in text_name_list:
            text_name_of_all = text_name_of_knp.split("/")[-1][:-4]
            pickle_names.extend(self._read_pickles(text_name_of_all))
            with codecs.open(text_name_of_knp, 'r', 'utf-8') as f:
                self.text_all_dict[
                    text_name_of_all] = self._words_list_maker(f.read())
        words_set = set()
        for _text_name, text_all in self.text_all_dict.items():
            words_set |= set([word[1] for word in text_all])
        self.vec_for_text_dic = self._vec_for_text_dic_maker(words_set)
        self.cf_vec_for_text_dic = {}
        # print(self.vec_for_text_dic.keys())
        # print(self.text_all_dict)
        # print(pickle_names)
        self.yougen_in_which_sentence = defaultdict(dict)
        self.text_sentence_vec_dict = defaultdict(dict)
        for pickle_name in pickle_names:
            text_name_of_all = "_".join(
                pickle_name.split("/")[-1][:-4].split("_")[:-1])
            yougen_row_num = pickle_name.split("/")[-1][:-4].split("_")[-1]
            # print(text_name_of_all)
            # print(yougen_row_num)
            self._previous_sentences_fetcher(
                text_name_of_all, yougen_row_num)
        # print(type(self.text_sentence_vec_dict['00005_A_PB56_00002'][8]))
        # print(self.text_sentence_vec_dict['00005_A_PB56_00002'][8])
        # print(self.yougen_in_which_sentence)

    def _read_pickles(self, knp_name):
        text_fnames = glob.glob(self.INPUT_PATH.format(knp_name + "_*"))
        return (text_fnames)

    def _words_list_maker(self, text_raw):
        """
        text_raw:KNPで解析済みの文章
        各行の一個目を取って，区切り文字列辞書（鍵は行番号）を返す．
        """
        words_list = []
        rows = text_raw.split("\n")
        for row_num, row in enumerate(rows):
            words = row.split(" ")
            if len(words) > 0 and words[0] not in ["#", "+", "*"]:
                #"EOS"を含んでいる．三行前までを判断するため．学習（RNN）には入力として使わない
                words_list.append((row_num, words[0]))
        return words_list

    def _vec_for_text_dic_maker(self, words_set):
        """
        対象文書中の単語すべてのベクトルをSQLから取ってきて，辞書にして返す．
        未知語は入れてないのでゼロベクトルにでもして対応すること
        """
        e_list_for_vec = list(words_set)
        e_list_for_vec_copy = e_list_for_vec[:]
        vec_for_text_dic = {}
        result_pre = []
        if len(e_list_for_vec) > 0:
            for e_list_for_vec_x in range(int(len(e_list_for_vec) / 900) + 1):
                e_list_for_vec_copy = e_list_for_vec[
                    900 * e_list_for_vec_x:min(900 * (e_list_for_vec_x + 1), len(e_list_for_vec))]
                sql1 = u"SELECT word,vec FROM word_vec_BCCWJ WHERE "
                for x_num, result_x in enumerate(e_list_for_vec_copy):
                    if x_num == 0:
                        sql1 += u"word='" + result_x + "'"
                    else:
                        sql1 += u"OR word='" + result_x + "'"
                for _ in range(10):
                    try:
                        self.cursor_for_input.execute(sql1)
                        break
                    except Exception as e:
                        print(e)
                        print(sql1)
                        print("u")
                        self.connector_for_input = sqlite3.connect(
                            "../../cf_J_w2v_mini_alt" + str(self.text_div) + ".db", timeout=100.0)
                        self.cursor_for_input = self.connector_for_input.cursor()
                list_cursor = self.cursor_for_input.fetchall()
                result_pre.extend(list_cursor)
            for result_pre_y in result_pre:
                vec_for_text_dic[result_pre_y[0]] = self.ARR.fromstring(result_pre_y[
                    1], sep="\n", dtype='float32')
        return vec_for_text_dic

    def cf_vec_for_text_dic_maker(self, now_cf):
        # 一個ずつ取ってくるので効率悪い．出来るだけcf_vec_fetcherを使うこと
        print("tukawaretayo")
        for now_case in self.case_list:
            sql1 = u"SELECT id,mean_vec FROM id_mean_vec_" + \
                now_case + " WHERE "
            sql1 += u"id='" + now_cf + "'"
            for _ in range(10):
                try:
                    self.cursor_for_input.execute(sql1)
                    break
                except Exception as e:
                    print(e)
                    print(unknown_words)
                    print(sql1)
                    print("u")
                    self.connector_for_input = sqlite3.connect(
                        "../../cf_J_w2v_mini_alt" + str(self.text_div) + ".db", timeout=100.0)
                    self.cursor_for_input = self.connector_for_input.cursor()
            list_cursor = self.cursor_for_input.fetchall()
            if list_cursor[0][1] != "NULL":
                self.cf_vec_for_text_dic[(now_cf, now_case)] = np.fromstring(
                    list_cursor[0][1], sep="\n")
            else:
                self.cf_vec_for_text_dic[
                    (now_cf, now_case)] = np.zeros(self.word_vec_dimention)

    def cf_vec_fetcher(self, result_cf_list_init):
        result_cf_list = list(set(
            result_cf_list_init) - set([cf_case[0] for cf_case in self.cf_vec_for_text_dic.keys()]))
        for case_name in self.case_list:
            # 初期化ミス
            result_pre2 = []
            result_pre3 = []
            if len(result_cf_list) > 0:
                result_cf_list_copy = result_cf_list[:]
                for result_cf_list_x in range(int(len(result_cf_list) / 900) + 1):
                    result_cf_list_copy = result_cf_list[
                        900 * result_cf_list_x:min(900 * (result_cf_list_x + 1), len(result_cf_list))]
                    sql2 = u"SELECT id, mean_vec FROM id_mean_vec_" + case_name + " WHERE "
                    for x_num_1, result_x in enumerate(result_cf_list_copy):
                        if x_num_1 == 0:
                            sql2 += u"id='" + result_x + "'"
                        else:
                            sql2 += u" OR id='" + result_x + "'"
                    for _ in range(10):
                        try:
                            self.cursor_for_input.execute(sql2)
                            break
                        except Exception as e:
                            print(e)
                            print(sql2)
                            print("u")
                            self.connector_for_input = sqlite3.connect(
                                "../../cf_J_w2v_mini_alt" + str(self.text_div) + ".db", timeout=100.0)
                            self.cursor_for_input = self.connector_for_input.cursor()
                    list_cursor = self.cursor_for_input.fetchall()
                    result_pre2.extend(list_cursor)
                for result_pre_z in result_pre2:
                    if result_pre_z[1] != "NULL":
                        self.cf_vec_for_text_dic[(result_pre_z[0], case_name)] = np.fromstring(result_pre_z[
                            1], sep="\n")
                    else:
                        self.cf_vec_for_text_dic[
                            (result_pre_z[0], case_name)] = np.zeros(500)

    def _previous_sentences_fetcher(self, text_name_of_all, yougen_row_num, how_many_sentences=3):
        """
        self.text_sentence_vec_dictに，yougen_row_numに対応する三文前ベクトルリストを格納する
        """
        eos_keep = []
        flag = 0
        for how_many_words, text_word in enumerate(self.text_all_dict[text_name_of_all]):
            if text_word[1] == "EOS":
                eos_keep.append((how_many_words, text_word))
                if flag == 1:
                    break
            if text_word[0] > int(yougen_row_num):
                flag = 1
                text_word[0]
        now_words_list = None
        self.yougen_in_which_sentence[text_name_of_all][
            yougen_row_num] = len(eos_keep)
        if len(eos_keep) in self.text_sentence_vec_dict[text_name_of_all]:
            return
        if len(eos_keep) > how_many_sentences + 1:
            now_words_list = self.text_all_dict[text_name_of_all][
                eos_keep[-how_many_sentences - 2][0] + 1:eos_keep[-1][0]]
        else:
            now_words_list = self.text_all_dict[
                text_name_of_all][:eos_keep[-1][0]]
        #print([x[1] for x in now_words_list if x[1] != "EOS"])
        self.text_sentence_vec_dict[text_name_of_all][len(eos_keep)] = self._sentence_vec_list(
            [x[1] for x in now_words_list if x[1] != "EOS"])

    def _sentence_vec_list(self, word_list):
        """
        与えられた単語リストをw2vベクトルリストにして返す
        """
        sentence_vec = []
        # print(word_list)
        for word in word_list:
            if word in self.vec_for_text_dic:
                sentence_vec.append(self.vec_for_text_dic[word])
            else:
                sentence_vec.append(self.ARR.zeros(500, dtype='float32'))

        return self.ARR.array(sentence_vec, dtype='float32')


if __name__ == '__main__':
    gru = Input_for_gru(0)
    gru(["knp_dismantled_data/00005_A_PB56_00002.knp",
         "knp_dismantled_data/00001_A_OC01_00001.knp"])

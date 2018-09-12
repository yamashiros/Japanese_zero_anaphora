#-*- encoding: utf-8 -*-
from __future__ import print_function
#import nltk
import sys
import random
import glob
import math
import pickle
from collections import defaultdict
import re
import codecs
import subprocess
import sqlite3
from datetime import datetime
from using_cf2_methods_hill_climbing import *
#from gensim.models import word2vec
import numpy as np
#import zenhan
from scipy.spatial import distance

SMOOTHING = 0.1
case_list = ["ガ格", "ヲ格", "ニ格"]
try:
    with open('words_dict.pickle', 'rb') as f:
        words_dict = pickle.load(f)
except IOError as e:
    print(e)
    raise e
try:
    with open('juman_dict.pickle', 'rb') as f:
        juman_dict = pickle.load(f)
except IOError as e:
    print(e)
    raise e
try:
    with open('named_entity_dict.pickle', 'rb') as f:
        named_entity_dict = pickle.load(f)
except IOError as e:
    print(e)
    raise e
# try:
#     with open('w2v_dict.pickle', 'rb') as f:
#         w2v_dict = pickle.load(f)
# except IOError as e:
#     print(e)


def dict_len(dic):
    all_len = 0
    for x in dic.items():
        all_len += x[1]
    return all_len
# dict_len(juman_dict)はカテゴリが付かない単語を無視しているのでsum_of_jumanはよくない
sum_of_word = dict_len(words_dict)  # 1172119.0
sum_of_juman = dict_len(juman_dict)  # 350199.00000000064
sum_of_named_entity = dict_len(named_entity_dict)  # 42628.0

# 学習済みモデルのロード
#model = word2vec.Word2Vec.load("sample.model")


class Extract_features(object):
    """docstring for Extract_features"""

    def __init__(self, text_name, num):
        with codecs.open(text_name, 'r', 'utf-8') as f:
            self.x = f.read()  # KNPを見てるとする
        self.connector = sqlite3.connect(
            "cf_J_w2v_mini_alt" + str(num % 16) + ".db", timeout=100.0)
        self.cursor = self.connector.cursor()
        self.num = num

    def close_all(self):
        self.cursor.close()
        self.connector.close()

    def set_lists(self, phrase_lists):
        self.phrase_list = phrase_lists[0]
        self.phrase_case_list = phrase_lists[1]

    def set_lists1(self, phrase_lists, row_list_cf_case_list):
        self.row_list_e = phrase_lists[0]
        self.row_list_cf = phrase_lists[1]
        self.row_list_cf_case_list = row_list_cf_case_list

    def set_info(self, cf, case, e, row_num, row_num_e):
        self.cf = cf
        self.case = case
        self.e = e
        self.row_num = row_num  # KNPの何行目
        self.row_num_e = row_num_e  # KNPの何行目
        self.e_normalize = None
        self.take_categories(row_num_e)

    def set_text_dics(self, named_entity_for_text_dic, vec_for_text_dic):
        self.named_entity_for_text_dic = named_entity_for_text_dic
        self.vec_for_text_dic = vec_for_text_dic
        row_list_e = self.row_list_e
        e_list_for_vec = []
        for row_e_x in row_list_e:
            for row_e_y in row_e_x:
                e_list_for_vec.append(row_e_y[2])
        result_pre = []
        if len(e_list_for_vec) > 0:
            # print(e_list_for_vec)
            e_list_for_vec_copy = e_list_for_vec[:]
            for e_list_for_vec_x in range(int(len(e_list_for_vec) / 900) + 1):
                e_list_for_vec_copy = e_list_for_vec[
                    900 * e_list_for_vec_x:min(900 * (e_list_for_vec_x + 1), len(e_list_for_vec))]
                sql1 = u"SELECT word,vec FROM word_vec_BCCWJ WHERE "
                for x_num, result_x in enumerate(e_list_for_vec_copy):
                    if x_num == 0:
                        sql1 += u"word='" + result_x + "'"
                    else:
                        sql1 += u"OR word='" + result_x + "'"
                try:
                    self.cursor.execute(sql1)
                except Exception as e:
                    print(e)
                    print(sql1)
                    print("u")
                    raise e
                list_cursor = self.cursor.fetchall()
                result_pre.extend(list_cursor)
            for result_pre_y in result_pre:
                self.vec_for_text_dic[result_pre_y[0]] = np.fromstring(result_pre_y[
                                                                       1], sep="\n")
            # print(self.vec_for_text_dic.keys())

    def set_result(self, result, cf_dic, headword_id_dic, mean_vec_dic, all_mean_vec_dic):
        result_cf_list = []
        result_cf_all_set = set()
        for result_x in result:
            result_cf_list.append(result_x)
            result_cf_all_set.add(result_x.split(":")[0])
        result_cf_all_list = list(result_cf_all_set)
        self.cf_dic = cf_dic
        self.headword_id_dic = headword_id_dic
        self.mean_vec_dic = mean_vec_dic
        self.all_mean_vec_dic = all_mean_vec_dic
        result_pre = []
        result_pre1 = []
        result_pre2 = []
        result_pre3 = []
        for case_name in case_list:
            # 初期化ミス
            result_pre = []
            result_pre2 = []
            result_pre3 = []
            if len(result_cf_list) > 0:
                result_cf_list_copy = result_cf_list[:]
                for result_cf_list_x in range(int(len(result_cf_list) / 900) + 1):
                    result_cf_list_copy = result_cf_list[
                        900 * result_cf_list_x:min(900 * (result_cf_list_x + 1), len(result_cf_list))]
                    sql = u"SELECT id, word, occurrence FROM " + case_name + " WHERE "
                    sql2 = u"SELECT id, mean_vec FROM id_mean_vec_" + case_name + " WHERE "
                    for x_num_1, result_x in enumerate(result_cf_list_copy):
                        if x_num_1 == 0:
                            sql += u"id='" + result_x + "'"
                            sql2 += u"id='" + result_x + "'"
                        else:
                            sql += u" OR id='" + result_x + "'"
                            sql2 += u" OR id='" + result_x + "'"
                    try:
                        self.cursor.execute(sql)
                    except Exception as e:
                        print(e)
                        print(sql)
                        print("u")
                        raise e
                    list_cursor = self.cursor.fetchall()
                    result_pre.extend(list_cursor)

                    try:
                        self.cursor.execute(sql2)
                    except Exception as e:
                        print(e)
                        print(sql2)
                        print("u")
                        raise e
                    list_cursor = self.cursor.fetchall()
                    result_pre2.extend(list_cursor)
                for result_pre_x in result_pre:
                    self.cf_dic[case_name][
                        result_pre_x[0]].append(result_pre_x)
                for result_pre_z in result_pre2:
                    if result_pre_z[1] != "NULL":
                        self.mean_vec_dic[case_name][result_pre_z[0]] = np.fromstring(result_pre_z[
                            1], sep="\n")
                    else:
                        self.mean_vec_dic[case_name][
                            result_pre_z[0]] = np.zeros(500)
            if len(result_cf_all_list) > 0:
                result_cf_all_list_copy = result_cf_all_list[:]
                for result_cf_list_x in range(int(len(result_cf_all_list) / 900) + 1):
                    result_cf_all_list_copy = result_cf_all_list[
                        900 * result_cf_list_x:min(900 * (result_cf_list_x + 1), len(result_cf_all_list))]
                    sql3 = u"SELECT id, mean_vec FROM id_all_mean_vec_" + case_name + " WHERE "
                    for x_num, result_x in enumerate(result_cf_all_list_copy):
                        if x_num == 0:
                            sql3 += u"id='" + result_x + "'"
                        else:
                            sql3 += u" OR id='" + result_x + "'"
                    try:
                        self.cursor.execute(sql3)
                    except Exception as e:
                        print(e)
                        print(sql3)
                        print("u")
                        raise e
                    list_cursor = self.cursor.fetchall()
                    result_pre3.extend(list_cursor)
                for result_pre_u in result_pre3:
                    if result_pre_u[1] != "NULL":
                        # if case_name == "ヲ格" and result_pre_u[0] == "分かれる/わかれる":
                        #     print(result_pre_u[1])
                        #     print(len(result_pre3))
                        #     exit()

                        self.all_mean_vec_dic[case_name][result_pre_u[0]] = np.fromstring(result_pre_u[
                            1], sep="\n")
                    else:
                        self.all_mean_vec_dic[case_name][
                            result_pre_u[0]] = np.zeros(500)
        # print(self.cf_dic)
        if len(result_cf_list) > 0:
            result_cf_list_copy = result_cf_list[:]
            for result_cf_list_x in range(int(len(result_cf_list) / 900) + 1):
                result_cf_list_copy = result_cf_list[
                    900 * result_cf_list_x:min(900 * (result_cf_list_x + 1), len(result_cf_list))]
                sql1 = u"SELECT id"
                for case_name in case_list:
                    sql1 += ", " + case_name + u"_size"
                sql1 += u" FROM headword_id WHERE "
                for x_num, result_x in enumerate(result_cf_list_copy):
                    if x_num == 0:
                        sql1 += u"id='" + result_x + "'"
                    else:
                        sql1 += u"OR id='" + result_x + "'"
                try:
                    self.cursor.execute(sql1)
                except Exception as e:
                    print(e)
                    print(sql1)
                    print("u")
                    raise e
                list_cursor = self.cursor.fetchall()
                result_pre1.extend(list_cursor)
            for result_pre_y in result_pre1:
                for case_num, case_name in enumerate(case_list):
                    self.headword_id_dic[case_name][
                        result_pre_y[0]].append([result_pre_y[0], result_pre_y[1 + case_num]])

    def take_categories(self, row_num_e):
        modified_e_list = []
        basic_form_e_list = []
        juman_e_list = []
        named_entity_e_list = []

        rows = self.x.split("\n")
        pre_words = ""

        for row in rows[:row_num_e]:
            words = row.split()
            if len(words) > 0:
                if words[0] == "*":
                    pre_words = words[2]
        words = rows[row_num_e].split()
        if len(words) > 0:
            text = ""
            if not words[0] in ["EOS"]:
                modified_e_list.append(
                    "".join(words[0:1]) + "/" + "".join(words[1:2]))
                basic_form_e_list.append("".join(words[2:3]))
            for word in words:
                if "<カテゴリ" in word:
                    flag = 0
                    juman_e_list_new = []
                    for i in word[word.find("<カテゴリ") + 6:]:
                        if i == ">":
                            flag = 1
                        if flag == 0:
                            juman_e_list.append(i)
                        elif flag == 1:
                            break
            if "<正規化代表表記:" in pre_words:
                flag = 0
                text_list_new1 = []
                for i in pre_words[pre_words.find("<正規化代表表記:") + 9:]:
                    # print(i)
                    if i == ">":
                        flag = 1
                    if flag == 0:
                        text_list_new1.append(i)
                    elif flag == 1:
                        break
                text = "".join(text_list_new1)
            if len(text) > 0:
                self.e_normalize = text
            else:
                self.e_normalize = "+".join(modified_e_list)

        self.basic_form_e = "+".join(basic_form_e_list)
        self.modified_e = "+".join(modified_e_list)
        self.juman_e = set()
        juman_candidate_list = "".join(juman_e_list).split(";")
        for juman_x_candidate in juman_candidate_list:
            if juman_x_candidate != "":
                self.juman_e.add(juman_x_candidate)
        # BIO部分の切り取り、SQLから取り出すものも
        # B-LOCATION:I-LOCATION:O
        # 抽象物:組織・団体;数量、:数量:人工物-その他
        # words_dictなどの作り直し？
        # self.basic_form_eに直すこと
        # 毎回作るのバカらしいので，外側に辞書を作っておく
        self.named_entity_e = set()
        if self.basic_form_e not in set(self.named_entity_for_text_dic.keys()):
            named_entity_e_list = named_entity(
                self.basic_form_e, self.num).split(":")
            for named_entity_candidate in named_entity_e_list:
                if named_entity_candidate != "O" and named_entity_candidate != "":
                    self.named_entity_e.add(named_entity_candidate[2:])
            self.named_entity_for_text_dic[
                self.basic_form_e] |= self.named_entity_e
        else:
            self.named_entity_e |= self.named_entity_for_text_dic[
                self.basic_form_e]

        self.vec_e = np.zeros(500)
        if self.basic_form_e in self.vec_for_text_dic:
            self.vec_e = self.vec_for_text_dic[
                self.basic_form_e]

        # if self.basic_form_e in set(w2v_dict.keys()):
        #     self.vec_e = w2v_dict[self.basic_form_e]

    def set_cf_p_dict(self, cf_p_dict, cf_candidate):
        cf_left = cf_candidate.split(":")[0]
        cf_p_dict_keys = cf_p_dict.keys()
        if cf_candidate in cf_p_dict_keys:
            cf_all_count = 0
            cf_c_count = 0
            for cf_dict_x, cf_num in cf_p_dict.items():
                if cf_left in cf_dict_x:
                    cf_all_count += cf_num
                if cf_candidate == cf_dict_x:
                    cf_c_count = cf_num
            if cf_c_count != 0:
                return math.log2(float(cf_c_count) / float(cf_all_count))
            else:
                print(cf_c_count)
                print(cf_candidate)
                print(cf_dict_x)
                exit()
        else:
            sql = u"SELECT headword,id,component_size FROM headword_id WHERE headword='" + cf_left + "'"
            try:
                self.cursor.execute(sql)
            except Exception as e:
                print(e)
                print("headword")
                raise e
            result = self.cursor.fetchall()
            cf_c_count = 0
            cf_all_count = 0
            for result_x in result:
                cf_p_dict[result_x[1]] = float(result_x[2])
                if cf_left in result_x[1]:
                    cf_all_count += float(result_x[2])
                if cf_candidate == result_x[1]:
                    cf_c_count = float(result_x[2])
            if cf_c_count != 0:
                return math.log2(float(cf_c_count) / float(cf_all_count))

            else:
                print(cf_c_count)
                print(cf_candidate)
                print(result)
                print(cf_left)
                exit()

        # eがcfのcに対応付く確率
    def feature1(self):
        # inf避けで0.01でスムージング
        self.cf_c_size = 0.0
        self.feature_1 = float("-inf")
        self.feature_2 = float("-inf")
        self.feature_3 = float("-inf")
        self.f1_inf_flag = 1
        self.f2_inf_flag = 1
        self.f3_inf_flag = 1
        # feature2と結合?
        # try:
        #     # sql = u"SELECT id, word, occurrence FROM "+self.case+" WHERE id='"+self.cf+"' AND word='"+self.modified_e+"'"
        #     # sql = u"SELECT id, word, occurrence FROM " + \
        #     # sizeをoccurrenceに書き換えてください
        #     sql = u"SELECT id, word, occurrence FROM " + \
        #         self.case + " WHERE id='" + self.cf + "'"
        #     self.cursor.execute(sql)
        # except Exception as e:
        #     print(e)
        #     print("u")

        #     return 0
        # result = self.cursor.fetchall()
        # 一度に取っておけば早いかと思ったが，likeとか使うとめちゃくちゃ遅い
        # print(self.cf_dic)
        # print(self.headword_id_dic)
        # exit()
        result = self.cf_dic[self.case][self.cf]
        result1 = self.headword_id_dic[self.case][self.cf]
        # id_cf = self.cf.split(":")[0]
        # print(id_cf)
        # result = []
        # result_pre = []
        # if id_cf not in cf_dic_keys:
        #     self.cf_dic[self.case] = {}
        #     try:
        #         # sql = u"SELECT id, word, occurrence FROM "+self.case+" WHERE id='"+self.cf+"' AND word='"+self.modified_e+"'"
        #         # sql = u"SELECT id, word, occurrence FROM " + \
        #         # sizeをoccurrenceに書き換えてください
        #         sql = u"SELECT id, word, occurrence FROM " + \
        #             self.case + " WHERE id='" + self.cf + "'"
        #         self.cursor.execute(sql)
        #     except Exception as e:
        #         print(e)
        #         print("u")

        #         return 0
        #     result_pre = self.cursor.fetchall()
        #     self.cf_dic[self.case][id_cf] = result_pre
        #     print(result_pre)
        # else:
        #     result_pre = self.cf_dic[self.case][id_cf]

        # for result_pre_x in result_pre:
        #     if result_pre_x[0] == self.cf:
        #         result.append(result_pre_x)

        # try:
        #     sql = u"SELECT id, " + self.case + \
        #         "_size FROM headword_id WHERE id='" + self.cf + "'"
        #     self.cursor.execute(sql)
        # except Exception as e:
        #     print(e)
        #     print("z")

        # result1 = self.cursor.fetchall()

        # print(result)
        # print(result1)
        # ここもう少し良くするべき（追加的に改良した）
        # 候補、結果の中身ともに+？で割って比較か？
        #「短期/たんき+決戦/けっせん」はなくて「決戦/けっせん」のみ
        #「初期/しょき+値/あたい?値/ち」これを「初期値、初期ち、初期あたい、しょき値、しょきち、しょきあたい」に候補分けしている。len(predict_list)について三つ以上はCF中の単語にはない
        result_hit = []
        result_dict = {}
        for sentence in result:
            result_dict[sentence[1]] = sentence[2]
            predict_list = []
            predict_set = set()
            predict_sub_list = []
            predict_sub_list1 = []
            predict_sub_list = sentence[1].replace(
                'a', '').replace('v', '').split("+")
            for splited_predict in predict_sub_list:
                predict_sub_list1 = splited_predict.split("?")
                predict_list.append(predict_sub_list1)
                predict_sub_list1 = []
            # print(predict_list)
            if len(predict_list) == 1:
                for predict in predict_list[0]:
                    predict_set.add(predict[:predict.find("/")])
            if len(predict_list) == 2:
                for predict in predict_list[0]:
                    for predict1 in predict_list[1]:
                        predict_set.add(predict[:predict.find(
                            "/")] + predict1[:predict1.find("/")])
            if len(predict_list) == 3:
                for predict in predict_list[0]:
                    for predict1 in predict_list[1]:
                        for predict2 in predict_list[2]:
                            predict_set.add(predict[:predict.find(
                                "/")] + predict1[:predict1.find("/")] + predict2[:predict2.find("/")])

            predict_x_list = sentence[1].replace(
                'a', '').replace('v', '').split("+")
            predict_x_sub_set = set()
            x_e_normalize = self.e_normalize
            predict_x_e_list = x_e_normalize.replace(
                'a', '').replace('v', '').split("+")
            predict_x_e_sub_set = set()

            for splited_x_predict in predict_x_list:
                predict_x_sub_set |= set(splited_x_predict.split("?"))
            for splited_x_e_predict in predict_x_e_list:
                predict_x_e_sub_set |= set(splited_x_e_predict.split("?"))
            # test
            # if sentence[1] == "決戦/けっせん+場/じょう?場/ば":
            #     print(predict_x_list)
            #     print(predict_x_sub_set)
            #     print(predict_x_e_sub_set)
            # test
            # ここのfor文あっても一回目で抜けてそう
            for predict_sen in predict_set:
                if self.e == predict_sen or self.e_normalize == sentence[1] or len(predict_x_sub_set & predict_x_e_sub_set) > 0:
                    result_hit.append(sentence)
                    # test
                    # print(len(predict_x_sub_set & predict_x_e_sub_set))
                    # print(predict_x_sub_set)
                    # print(predict_x_e_sub_set)
                    # test
                    break
        # print(result_hit)
        # result_hitの値全部足すか？初期と値が別に当たったら両方。初期値と偏差値もあたるではある
        # 基本的に上の方が高いが保証はない。のでマックス取ってみた
        max_score = 0.0
        max_result = None
        for result_x in result_hit:
            if result_x[2] > max_score:
                max_result = result_x
                max_score = result_x[2]
        # test
        # print(max_result)
        # test

        if len(result_hit) > 0 and max_result[2] != 0 and result1 != None and len(result1) != 0 and float(result1[0][1]) > 0:
            self.cf_c_size = float(result1[0][1])
            self.feature_1 = math.log2(
                float(max_result[2]) / float(result1[0][1]))
            self.f1_inf_flag = 0

        elif result1 != None and len(result1) != 0 and float(result1[0][1]) > 0:
            self.cf_c_size = float(result1[0][1])
            self.feature_1 = math.log2(
                float(SMOOTHING) / float(result1[0][1]))

        if len(result_dict) > 0:
            # Expression tree is too large (maximum depth 1000)対策
            result_dict_keys_ori = list(result_dict.keys())
            result2 = []
            for result_dict_x in range(int(len(result_dict_keys_ori) / 900) + 1):
                result_dict_keys = result_dict_keys_ori[
                    900 * result_dict_x:min(900 * (result_dict_x + 1), len(result_dict_keys_ori))]
                # if int(len(result_dict_keys_ori) / 900) > 1:
                # print(len(result_dict_keys))
                sql = ""
                for id_num, id_name in enumerate(result_dict_keys):
                    if id_num == 0:
                        sql += u"SELECT word, JUMAN, named_entity, domain FROM word_JUMAN WHERE word='" + \
                            id_name + "'"
                    else:
                        sql += u"OR word='" + id_name + "'"
                sql += ";"
                try:
                    self.cursor.execute(sql)
                except Exception as e:
                    print(e)
                    print("juman")
                    raise e
                list_cursor = self.cursor.fetchall()
                result2.extend(list_cursor)
            # test
            # print(result2)
            # print(self.cf_c_size)
            # print(result_dict)
            # test
            result_juman = 0.0
            result_named_entity = 0.0
            # result_domain = 0.0

            for sentence in result2:
                result_juman_set = set()
                result_named_entity_set = set()
                if sentence[1] != "NULL":
                    for result_juman_candidate in sentence[1].split(":"):
                        for juman_x_candidate in result_juman_candidate.split(";"):
                            if juman_x_candidate != "":
                                result_juman_set.add(juman_x_candidate)
                    for result_juman_x in result_juman_set:
                        if result_juman_x in self.juman_e:
                            result_juman += float(result_dict[sentence[0]]) / (
                                len(result_juman_set) * len(self.juman_e))
                if sentence[2] != "NULL":
                    for result_named_entity_candidate in sentence[2].split(":"):
                        if result_named_entity_candidate != "O":
                            result_named_entity_set.add(
                                result_named_entity_candidate[2:])
                    for result_named_entity_x in result_named_entity_set:
                        if result_named_entity_x in self.named_entity_e:
                            result_named_entity += float(result_dict[sentence[0]]) / (
                                len(result_named_entity_set) * len(self.named_entity_e))

            # test
            # print(result_named_entity)
            # print(self.named_entity_e)
            # test
            if result_juman != 0 and self.cf_c_size != 0:
                self.feature_2 = math.log2(
                    float(result_juman) / self.cf_c_size)
                self.f2_inf_flag = 0
            elif self.cf_c_size != 0:
                self.feature_2 = math.log2(
                    float(SMOOTHING) / self.cf_c_size)
            else:
                self.feature_2 = float("-inf")
            if result_named_entity != 0 and self.cf_c_size != 0:
                self.feature_3 = math.log2(
                    float(result_named_entity) / self.cf_c_size)
                self.f3_inf_flag = 0
            elif self.cf_c_size != 0:
                self.feature_3 = math.log2(
                    float(SMOOTHING) / self.cf_c_size)
            else:
                self.feature_3 = float("-inf")

    # NA作るときのみ使用
    def calcurate_cf_c_size(self):
        self.cf_c_size = 0.0

        # try:
        #     sql = u"SELECT id, " + self.case + \
        #         "_size FROM headword_id WHERE id='" + self.cf + "'"
        #     self.cursor.execute(sql)
        # except Exception as e:
        #     print(e)
        #     print("y")

        #     return 0
        #result1 = self.cursor.fetchall()
        result1 = self.headword_id_dic[self.case][self.cf]

        if result1 != None and len(result1) != 0:
            self.cf_c_size = float(result1[0][1])
        else:
            return 0

    # eの持つJUMANカテゴリがcfのcに対応付く確率
    # eの持つ固有表現がcfのcに対応付く確率
    # feature1に統合しますので使わないでください
    def feature2(self):
        pass
        # self.feature_2 = float("-inf")

        # try:
        #     # sql = u"SELECT id, word, size, JUMAN FROM "+self.case+" WHERE
        #     # id='"+self.cf+"' AND word='"+self.e+"' AND
        #     # JUMAN='"+self.juman_e+"'"
        #     sql = u"SELECT id, word, occurrence FROM " + \
        #         self.case + " WHERE id='" + self.cf + "'"
        #     self.cursor.execute(sql)
        # except Exception as e:
        #     print(e)
        #     print("x")
        #     return 0
        # result = self.cursor.fetchall()
        # # print(result)
        # result_int = 0
        # for sentence in result:
        #     if sentence[3] == self.juman_e:
        #         result_int += int(sentence[2])
        # if result_int != 0 and self.cf_c_size != 0:
        #     self.feature_2 = math.log2(float(result_int) / self.cf_c_size)
        # else:
        #     self.feature_2 = float("-inf")
    # eの出現確率
    # eの持つカテゴリの出現確率
    # eの固有表現の出現確率

    def feature4(self):
        self.feature_4 = float("-inf")
        self.feature_5 = float("-inf")
        self.feature_6 = float("-inf")
        self.f4_inf_flag = 1
        self.f5_inf_flag = 1
        self.f6_inf_flag = 1
        if not words_dict[self.basic_form_e] == 0:
            self.feature_4 = math.log2(
                words_dict[self.basic_form_e] / sum_of_word)
            self.f4_inf_flag = 0
        else:
            self.feature_4 = math.log2(
                float(SMOOTHING) / sum_of_word)

        # 注意sum_of_word
        total_feature_5 = 0.0
        for juman_x in self.juman_e:
            total_feature_5 += juman_dict[juman_x] / \
                (sum_of_word * len(self.juman_e))

        if total_feature_5 != 0.0:
            self.feature_5 = math.log2(total_feature_5)
            self.f5_inf_flag = 0
        else:
            self.feature_5 = math.log2(
                float(SMOOTHING) / sum_of_word)

        # 注意sum_of_word
        total_feature_6 = 0.0
        for named_entity_x in self.named_entity_e:
            total_feature_6 += named_entity_dict[named_entity_x] / \
                (sum_of_word * len(self.named_entity_e))

        if total_feature_6 != 0.0:
            self.feature_6 = math.log2(total_feature_6)
            self.f6_inf_flag = 0
        else:
            self.feature_6 = math.log2(
                float(SMOOTHING) / sum_of_word)
        # test
        # print(self.feature_6)
        # print(self.named_entity_e)
        # exit()
        # test

    # eとcfのcの間の自己相互情報量
    # eの持つJUMANカテゴリとcfのcの間の自己相互情報量
    # eの持つ固有表現とcfのcの間の自己相互情報量
    # 上記３つの自己相互情報量のうち最大のものの値

    def feature7(self):
        self.feature_7 = float("-inf")
        self.feature_8 = float("-inf")
        self.feature_9 = float("-inf")
        self.feature_10 = float("-inf")
        self.f7_inf_flag = 1
        self.f8_inf_flag = 1
        self.f9_inf_flag = 1
        if self.feature_1 != float("-inf") and self.feature_4 != float("-inf"):
            self.feature_7 = math.log2(
                math.pow(2, self.feature_1) / math.pow(2, self.feature_4))
            self.f7_inf_flag = 0
        elif self.feature_4 != float("-inf"):
            self.feature_7 = math.log2(
                float(SMOOTHING) / math.pow(2, self.feature_4))
        else:
            self.feature_7 = float("-inf")
        if self.feature_2 != float("-inf") and self.feature_5 != float("-inf"):
            self.feature_8 = math.log2(
                math.pow(2, self.feature_2) / math.pow(2, self.feature_5))
            self.f8_inf_flag = 0
        elif self.feature_5 != float("-inf"):
            self.feature_8 = math.log2(
                float(SMOOTHING) / math.pow(2, self.feature_5))
        else:
            self.feature_8 = float("-inf")
        if self.feature_3 != float("-inf"):
            self.feature_9 = math.log2(
                math.pow(2, self.feature_3) / math.pow(2, self.feature_6))
            self.f9_inf_flag = 0
        elif self.feature_6 != float("-inf"):
            self.feature_9 = math.log2(
                float(SMOOTHING) / math.pow(2, self.feature_6))
        else:
            self.feature_9 = float("-inf")
        self.feature_10 = max(self.feature_7, self.feature_8, self.feature_9)
        # self.feature_10=max(self.feature_7,self.feature_8)
    # cfのcが何らかの要素で対応付けられる確率
    # cfが持つ用例のうちcの用例の割合
    # feature_11xはfeature_11の確率を1から引いた物（のlog）

    def feature11(self):
        self.feature_11 = float("-inf")
        self.feature_11x = float(0)
        self.feature_12 = float("-inf")
        self.f11_inf_flag = 1
        self.f11x_inf_flag = 1
        self.f12_inf_flag = 1
        self.feature_hissu = 0
        try:
            sql = u"SELECT * FROM headword_id WHERE id='" + self.cf + "'"
            self.cursor.execute(sql)
        except Exception as e:
            print(e)
            raise e
        result1 = self.cursor.fetchall()
        if len(result1) != 0:
            # print(result1)
            self.cf_size = float(max(result1[0][3:]))
            self.cf_full_size = float(result1[0][2])
            if self.cf_c_size != 0 and self.cf_c_size != self.cf_size:
                self.feature_11 = math.log2(self.cf_c_size / self.cf_size)
                self.feature_11x = math.log2(
                    1.0 - float(self.cf_c_size / self.cf_size))
                self.feature_12 = math.log2(self.cf_c_size / self.cf_full_size)
                self.f11_inf_flag = 0
                self.f11x_inf_flag = 0
                self.f12_inf_flag = 0
            elif self.cf_c_size != 0:
                self.feature_11 = math.log2(self.cf_c_size / self.cf_size)
                self.feature_11x = math.log2(
                    1.0 - float(self.cf_c_size / (self.cf_size + SMOOTHING)))
                self.feature_12 = math.log2(self.cf_c_size / self.cf_full_size)
                self.f11_inf_flag = 0
                self.f12_inf_flag = 0
                self.feature_hissu = 1
            elif self.cf_size != 0 and self.cf_full_size != 0:
                self.feature_11 = math.log2(SMOOTHING / self.cf_size)
                self.feature_11x = math.log2(
                    1.0 - float(SMOOTHING / self.cf_size))
                self.feature_12 = math.log2(SMOOTHING / self.cf_full_size)
            else:
                self.feature_11 = float("-inf")
                self.feature_11x = float(0)
                self.feature_12 = float("-inf")

        else:
            self.feature_11 = float("-inf")
            self.feature_11x = float(0)
            self.feature_12 = float("-inf")

    # cfのcが直前格かどうか
    # cfのcが必須格かどうか

    # 用言pの持つモダリティ
    # 用言pの持つ敬語表現
    # 用言pの持つ時制
    # 用言pが可能表現かどうか
    # 用言pの係り受けタイプ（連用，連帯，文末）
    # 用言pが胴体述語か様態述語か # 動態，状態

    def feature13(self):

        text = ""
        text1 = ""
        text2 = ""
        flag_for_possible = 0
        text3 = ""
        text4 = ""
        self.feature_13 = text
        self.feature_14 = text1
        self.feature_15 = text2
        self.feature_16 = flag_for_possible
        self.feature_17 = text3
        self.feature_18 = text4
        rows = self.x.split("\n")
        for x, row in enumerate(rows):
            words = row.split()
            if len(words) > 0:
                if words[0] in ["#", "+", "*", "EOS"]:
                    keep_row_num = x
                elif x == self.row_num:
                    for word in rows[keep_row_num].split():
                        if "<モダリティ" in word:
                            flag = 0
                            text_list_new = []
                            for i in word[word.find("<モダリティ") + 7:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new.append(i)
                                elif flag == 1:
                                    pass
                            text = "".join(text_list_new)
                        if "<敬語:" in word:
                            flag = 0
                            text_list_new1 = []
                            for i in word[word.find("<敬語:") + 4:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new1.append(i)
                                elif flag == 1:
                                    pass
                            text1 = "".join(text_list_new1)
                        if "<時制-" in word:
                            flag = 0
                            text_list_new2 = []
                            for i in word[word.find("<時制-") + 4:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new2.append(i)
                                elif flag == 1:
                                    pass
                            text2 = "".join(text_list_new2)
                        if "<可能表現>" in word:
                            flag_for_possible = 1
                        if "<係:" in word:
                            flag = 0
                            text_list_new3 = []
                            for i in word[word.find("<係:") + 3:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new3.append(i)
                                elif flag == 1:
                                    pass
                            text3 = "".join(text_list_new3)
                        if "述語>" in word:
                            flag = 0
                            text_list_new4 = []
                            for i in word[word.find("述語>") - 2:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new4.append(i)
                                elif flag == 1:
                                    pass
                            text4 = "".join(text_list_new4)
        self.feature_13 = text
        self.feature_14 = text1
        self.feature_15 = text2
        self.feature_16 = flag_for_possible
        self.feature_17 = text3
        self.feature_18 = text4

    # eが持つ固有表現の種類
    def feature19(self):
        self.feature_19 = ":".join(list(self.named_entity_e))

    # eが入力文tで言及された回数
    # eが入力文tで用言pより前方で言及された回数
    # eが入力文tで用言pより後方で言及された回数
    # eが対象の文で助詞「は」を伴なって言及されたか
    # eが何文前で言及されたか
    # eが先頭の文で言及されたか
    # eが先頭の文で助詞「は」を伴なって言及されたか
    # eが先頭の文で助詞「が」を伴なって言及されたか
    # eが先頭の文の文頭で言及されたか
    # eが先頭の文の文頭で助詞「は」を伴なって言及されたか
    # eが先頭の文の文末で言及されたか
    # eが先頭の文の文末で判定詞を伴なって言及されたか
    # eが先頭の文の文末で体言止めで言及されたか
    # eの権限性スコアが1以上あるか

    def feature20(self):
        word_all = 0
        word_before = 0
        word_after = 0
        flag_predict_ha = 0
        flag_ha = 0
        # Itopic用。『は』が同一文でない、二回出てきたうちの述語に近い方でない場合を弾くためなのでこれが1ならItopicは0
        # flag_ha_second = 0
        # 間違い。解析対象先行詞（e）のみじゃなく、同言及全般調べるのでカテゴリー側のメソッド内に移行

        flag_predict_ga = 0
        flag_ga = 0
        flag_predict_wo = 0
        flag_wo = 0
        flag_predict_doku = 0
        flag_doku = 0
        flag_top_sentence = 0
        flag_top_row = 0
        flag_tail_row = 0
        flag_hanteishi = 0
        flag_taigendome = 0
        candidates_list = []
        kengen = 0
        eos_predict = 0
        eos_freq = 0
        keep_row_num = 0
        keep_row_num_1 = 1
        keep_row_num_2 = 2

        self.feature_20 = word_all
        self.feature_21 = word_before
        self.feature_22 = word_after
        self.feature_23 = flag_ha
        self.feature_24 = eos_freq
        self.feature_25 = flag_top_sentence
        self.feature_26 = flag_top_sentence * flag_ha
        self.feature_27 = flag_top_sentence * flag_ga
        self.feature_28 = flag_top_sentence * flag_top_row
        self.feature_29 = flag_top_sentence * flag_top_row * flag_ha
        self.feature_30 = flag_top_sentence * flag_tail_row
        self.feature_31 = flag_top_sentence * flag_tail_row * flag_hanteishi
        self.feature_32 = flag_top_sentence * flag_tail_row * flag_taigendome
        self.feature_33 = kengen
        # self.Itopic = flag_ha

        rows = self.x.split("\n")
        for x, row in enumerate(rows):
            # self.row_num_eが今解析対象にしている先行詞の行数なので、xがこれより先の行数になったら数え始めるためのフラグ
            if flag_predict_ha == 0 and self.row_num_e < x:
                flag_predict_ha = 1
            if flag_predict_ga == 0 and self.row_num_e < x:
                flag_predict_ga = 1
            if flag_predict_wo == 0 and self.row_num_e < x:
                flag_predict_wo = 1
            if flag_predict_doku == 0 and self.row_num_e < x:
                flag_predict_doku = 1
            if eos_predict == 0 and self.row_num_e < x:
                eos_predict = 1
            # self.row_numが今解析対象にしている述語の行数なので、xがこれより先の行数になったら数えやめるためのフラグ
            if eos_predict == 1 and self.row_num < x:
                eos_predict = 2
            words = row.split()
            if len(words) > 0:
                if not words[0] in ["#", "+", "*", "EOS"]:
                    if words[2] == self.basic_form_e:
                        if x < self.row_num:
                            word_before += 1
                        elif x > self.row_num:
                            word_after += 1
                    if flag_predict_ha == 1 and words[0] == "は" and words[3] == "助詞":
                        flag_ha = 1
                    # if flag_ha == 1 and eos_predict == 1 and words[0] == "は" and words[3] == "助詞":
                    #     flag_ha_second = 1
                    if flag_predict_ga == 1 and words[0] == "が" and words[3] == "助詞":
                        flag_ga = 1
                    if flag_predict_wo == 1 and words[0] == "を" and words[3] == "助詞":
                        flag_wo = 1
                    if flag_predict_doku == 1 and words[0] == "、" and words[6] == "読点":
                        flag_doku = 1
                    if words[4] == "名詞":
                        if x > row_num_e:
                            candidates_list += words[3]
                    if x == self.row_num_e:
                        for word in rows[keep_row_num].split():
                            if "<文頭>" in word:
                                flag_top_row = 1
                            if "<文末>" in word:
                                flag_tail_row = 1
                            if "<判定詞>" in word:
                                flag_hanteishi = 1
                            if "<体言止>" in word:
                                flag_taigendome = 1
                        for word in rows[keep_row_num_1].split():
                            if "S-ID:1" in word:
                                flag_top_sentence = 1
                        # for word in rows[keep_row_num_2].split():
                        # 	if "<格関係" in word:
                        # 		pass
                        # 	#直接の係り受けを取ってくる処理
                elif eos_predict == 1 and words[0] == "EOS":
                    eos_freq += 1
                elif flag_predict_ha == 1:
                    flag_predict_ha = 2
                elif flag_predict_ga == 1:
                    flag_predict_ga = 2
                elif flag_predict_wo == 1:
                    flag_predict_wo = 2
                elif flag_predict_doku == 1:
                    flag_predict_doku = 2
                elif words[0] in ["#"]:
                    keep_row_num_1 = x
                elif words[0] in ["+"] and x > self.row_num_e:
                    keep_row_num_2 = x
                else:
                    keep_row_num = x
        word_all = word_after + word_after
        if((2.0 * (flag_tail_row or flag_ha) + 1.0 * (flag_doku or flag_ga or flag_wo)) * (0.5 if eos_freq > 0 else 1.0) > 1):
            kengen = 1
        else:
            kengen = 0
        self.feature_20 = word_all
        self.feature_21 = word_before
        self.feature_22 = word_after
        self.feature_23 = flag_ha
        self.feature_24 = eos_freq
        self.feature_25 = flag_top_sentence
        self.feature_26 = flag_top_sentence * flag_ha
        self.feature_27 = flag_top_sentence * flag_ga
        self.feature_28 = flag_top_sentence * flag_top_row
        self.feature_29 = flag_top_sentence * flag_top_row * flag_ha
        self.feature_30 = flag_top_sentence * flag_tail_row
        self.feature_31 = flag_top_sentence * flag_tail_row * flag_hanteishi
        self.feature_32 = flag_top_sentence * flag_tail_row * flag_taigendome
        self.feature_33 = kengen
        # if eos_freq == 0 and flag_ha_second == 0:
        #     self.Itopic = flag_ha

    # Itopic:	副助詞「は」をともなって出現
    # 85個の素性のうち論文中の記載で「Itopic:副助詞「は」をともなって出現」は
    # 他の素性と異なり、selfの場合にのみ使用される素性で、正確には「解析対象
    # の述語と同一文、かつ、対象述語より前に出てきた「は」の中で直近のものを
    # 伴う言及」である場合のみ1となります。

    #;It-self:   副助詞「は」で終わり、かつ、述語を超えて係り先を持つ文節の主辞
    # IP-self:	解析対象述語の係り先 (Parent)
    # IC-self:	解析対象の述語に係る (Child)
    # IGP-self:	解析対象述語の係り先の係り先 (Grand-Parent)
    # IGC-self:	係り先が解析対象の述語に係る (Grand-Child)

    #;former-self:   解析対象述語と並列関係(前方)
    #;latter-self:   解析対象述語と並列関係(後方)
    # IB-self:	上記以外で同一文の前方に出現
    # IA-self:	上記以外で同一文の後方に出現

    #;It-ga-ov:  副助詞「は」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のガ格 (非省略)
    #;It-ga-om:  副助詞「は」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のガ格 (省略)
    #;It-wo-ov:  副助詞「を」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のヲ格 (非省略)
    #;It-wo-om:  副助詞「を」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のヲ格 (省略)
    #;It-ni-ov:  副助詞「に」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のニ格 (非省略)
    #;It-ni-om:  副助詞「に」で終わり、かつ、述語を超えて係り先を持つ文節の主辞のニ格 (省略)
    # IP-ga-ov:	解析対象述語の係り先述語のガ格 (非省略)
    # IP-ga-om:	解析対象述語の係り先述語のガ格 (省略)
    # IP-wo-ov:	解析対象述語の係り先述語のヲ格 (非省略)
    # IP-wo-om:	解析対象述語の係り先述語のヲ格 (省略)
    # IP-ni-ov:	解析対象述語の係り先述語のニ格 (非省略)
    # IP-ni-om:	解析対象述語の係り先述語のニ格 (省略)
    # IC-ga-ov:	解析対象の述語に係る述語のガ格 (非省略)
    # IC-ga-om:	解析対象の述語に係る述語のガ格 (省略)
    # IC-wo-ov:	解析対象の述語に係る述語のヲ格 (非省略)
    # IC-wo-om:	解析対象の述語に係る述語のヲ格 (省略)
    # IC-ni-ov:	解析対象の述語に係る述語のニ格 (非省略)
    # IC-ni-om:	解析対象の述語に係る述語のニ格 (省略)
    # IGP-ga-ov:	解析対象述語の係り先の係り先述語のガ格 (非省略)
    # IGP-ga-om:	解析対象述語の係り先の係り先述語のガ格 (省略)
    # IGP-wo-ov:	解析対象述語の係り先の係り先述語のヲ格 (非省略)
    # IGP-wo-om:	解析対象述語の係り先の係り先述語のヲ格 (省略)
    # IGP-ni-ov:	解析対象述語の係り先の係り先述語のニ格 (非省略)
    # IGP-ni-om:	解析対象述語の係り先の係り先述語のニ格 (省略)
    # IGC-ga-ov:	係り先が解析対象の述語に係る述語のガ格 (非省略)
    # IGC-ga-om:	係り先が解析対象の述語に係る述語のガ格 (省略)
    # IGC-wo-ov:	係り先が解析対象の述語に係る述語のヲ格 (非省略)
    # IGC-wo-om:	係り先が解析対象の述語に係る述語のヲ格 (省略)
    # IGC-ni-ov:	係り先が解析対象の述語に係る述語のニ格 (非省略)
    # IGC-ni-om:	係り先が解析対象の述語に係る述語のニ格 (省略)
    # ;former-ga-ov:  解析対象述語と並列関係(前方)述語のガ格 (非省略)
    # ;former-ga-om:  解析対象述語と並列関係(前方)述語のガ格 (省略)
    # ;former-wo-ov:  解析対象述語と並列関係(前方)述語のヲ格 (非省略)
    # ;former-wo-om:  解析対象述語と並列関係(前方)述語のヲ格 (省略)
    # ;former-ni-ov:  解析対象述語と並列関係(前方)述語のニ格 (非省略)
    # ;former-ni-om:  解析対象述語と並列関係(前方)述語のニ格 (省略)
    # ;latter-ga-ov:  解析対象述語と並列関係(後方)述語のガ格 (非省略)
    # ;latter-ga-om:  解析対象述語と並列関係(後方)述語のガ格 (省略)
    # ;latter-wo-ov:  解析対象述語と並列関係(後方)述語のヲ格 (非省略)
    # ;latter-wo-om:  解析対象述語と並列関係(後方)述語のヲ格 (省略)
    # ;latter-ni-ov:  解析対象述語と並列関係(後方)述語のニ格 (非省略)
    # ;latter-ni-om:  解析対象述語と並列関係(後方)述語のニ格 (省略)
    # ;IB-ga-ov:  上記以外で同一文の前方に出現する述語のガ格 (非省略)
    # ;IB-ga-om:  上記以外で同一文の前方に出現する述語のガ格 (省略)
    # ;IB-wo-ov:  上記以外で同一文の前方に出現する述語のヲ格 (非省略)
    # ;IB-wo-om:  上記以外で同一文の前方に出現する述語のヲ格 (省略)
    # ;IB-ni-ov:  上記以外で同一文の前方に出現する述語のニ格 (非省略)
    # ;IB-ni-om:  上記以外で同一文の前方に出現する述語のニ格 (省略)
    # ;IA-ga-ov:  上記以外で同一文の後方に出現する述語のガ格 (非省略)
    # ;IA-ga-om:  上記以外で同一文の後方に出現する述語のガ格 (省略)
    # ;IA-wo-ov:  上記以外で同一文の後方に出現する述語のヲ格 (非省略)
    # ;IA-wo-om:  上記以外で同一文の後方に出現する述語のヲ格 (省略)
    # ;IA-ni-ov:  上記以外で同一文の後方に出現する述語のニ格 (非省略)
    # ;IA-ni-om:  上記以外で同一文の後方に出現する述語のニ格 (省略)

    # B1:	直前の文に出現
    # B1-ga-ov:	直前の文にガ格として出現 (非省略)
    # B1-ga-om:	直前の文にガ格として出現 (省略)
    # B1-wo-ov:	直前の文にヲ格として出現 (非省略)
    # B1-wo-om:	直前の文にヲ格として出現 (省略)
    # B1-ni-ov:	直前の文にニ格として出現 (非省略)
    # B1-ni-om:	直前の文にニ格として出現 (省略)
    # B2:	2文前に出現
    # B2-ga-ov:	2文前にガ格として出現 (非省略)
    # B2-ga-om:	2文前にガ格として出現 (省略)
    # B2-wo-ov:	2文前にヲ格として出現 (非省略)
    # B2-wo-om:	2文前にヲ格として出現 (省略)
    # B2-ni-ov:	2文前にニ格として出現 (非省略)
    # B2-ni-om:	2文前にニ格として出現 (省略)
    # B3:	3文以上前に出現
    # B3-ga-ov:	3文以上前にガ格として出現 (非省略)
    # B3-ga-om:	3文以上前にガ格として出現 (省略)
    # B3-wo-ov:	3文以上前にヲ格として出現 (非省略)
    # B3-wo-om:	3文以上前にヲ格として出現 (省略)
    # B3-ni-ov:	3文以上前にニ格として出現 (非省略)
    # B3-ni-om:	3文以上前にニ格として出現 (省略)
    def position_category(self, row_num_position, row_num_e_position, e_position_list):
        # position系のタプルが持っているのは二重リスト内のリスト番号
        self.Itopic = 0
        e_position_list_intra = []
        # 使わないかも？e_position_listを破壊しないために一応
        e_row_set_intra = set()
        # e_position_list_intra:解析対象述語と同一文内のe同形単語のself.phrase_list二重リスト内のリスト番号
        # e_row_set_intra:解析対象述語と同一文内のe同形単語の*の行番号
        for e_position in e_position_list:
            if row_num_position[0] == e_position[0]:
                e_position_list_intra.append(e_position)
                e_row_set_intra.add(
                    self.phrase_list[e_position[0]][e_position[1]][0])
        # print(e_position_list_intra)

        rows = self.x.split("\n")

        flag_ha_row = 0
        flag_ha_row_pre = 0
        ha_list = []
        yougen_list = []
        top_phrase_row = int(self.phrase_list[row_num_position[0]][0][0])
        for x, row in enumerate(rows):
            words = row.split()
            if len(words) > 0 and x >= top_phrase_row:
                if not words[0] in ["#", "+", "*", "EOS"]:
                    if words[0] == "は" and words[5] == "副助詞":
                        # test
                        # print(flag_ha_row_pre)
                        # print(e_row_set_intra)
                        # test
                        if self.row_num >= x and str(flag_ha_row_pre) in e_row_set_intra:
                            flag_ha_row = flag_ha_row_pre
                        elif self.row_num >= x:
                            flag_ha_row = 0
                        ha_list.append(str(flag_ha_row_pre))
                elif words[0] == "*":
                    flag_ha_row_pre = x
                elif words[0] == "EOS":
                    break
                if words[0] == "*" and words[2].find("<用言:") != -1:
                    yougen_list.append(x)
        # test
        # print(flag_ha_row)
        # test
        if flag_ha_row != 0:
            self.Itopic = 1
            for e_position in e_position_list_intra:
                if self.phrase_list[e_position[0]][e_position[1]][0] == flag_ha_row:
                    e_position_list_intra.remove(e_position)
        self.It_self = 0
        It_positions = []
        # It-self:   副助詞「は」で終わり、かつ、述語を超えて係り先を持つ文節の主辞
        # のために係り先までの間に用言がないかを調べている
        # 係り先が解析対象であった場合は0になって数えないはず
        phrase_list_intra = self.phrase_list[row_num_position[0]][:]
        for list_num, intra_position in enumerate(phrase_list_intra):
            # print(intra_position)
            # print((row_num_position[0], list_num))
            # print(e_position_list_intra)
            if intra_position[0] in ha_list:
                # print(list_num)
                # print(row_num_position[0])
                ha_e_phrase_info = self.phrase_list[
                    row_num_position[0]][list_num]
                ha_e_row = int(ha_e_phrase_info[0])
                ha_kakari_row = int(self.phrase_list[
                    row_num_position[0]][int(ha_e_phrase_info[1][0:-1])][0])
                for yougen_num in yougen_list:
                    if (ha_e_row - yougen_num) * (ha_kakari_row - yougen_num) < 0:
                        if (row_num_position[0], list_num) in e_position_list_intra:
                            e_position_list_intra.remove(
                                (row_num_position[0], list_num))
                            # print(yougen_num)
                            # print(ha_e_row)
                            # print(ha_kakari_row)
                            self.It_self = 1
                        else:
                            It_positions.append(
                                (row_num_position[0], list_num))

        self.IP_self = 0
        self.IC_self = 0
        self.IGP_self = 0
        self.IGC_self = 0
        self.former_self = 0
        self.latter_self = 0
        IC_positions = []
        IGC_positions = []
        IGP_position = None
        IP_position = None
        former_positions = []
        latter_position = None
        #["D", "A"]なら係り受け、["P", "I"]なら並列
        # ただし、孫、祖父はこの差をつけずに取っている
        # print(self.phrase_list[row_num_position[0]][row_num_position[1]])
        if self.phrase_list[row_num_position[0]][row_num_position[1]][1][-1:] in ["D", "A"]:
            IP_position = (row_num_position[0], int(self.phrase_list[
                           row_num_position[0]][row_num_position[1]][1][0:-1]))
        else:
            latter_position = (row_num_position[0], int(self.phrase_list[
                row_num_position[0]][row_num_position[1]][1][0:-1]))
        for row_y, row in enumerate(self.phrase_list[row_num_position[0]]):
            if int(row[1][0:-1]) == row_num_position[1]:
                if row[1][-1:] in ["D", "A"]:
                    IC_positions.append((row_num_position[0], row_y))
                else:
                    former_positions.append((row_num_position[0], row_y))
                    # print(former_positions)
        if IP_position != None and IP_position[1] != -1:
            IGP_position = (row_num_position[0], int(self.phrase_list[
                            IP_position[0]][IP_position[1]][1][0:-1]))
        else:
            IGP_position = None
        for IC_position in IC_positions:
            if IC_position[1] != -1:
                for row_y, row in enumerate(self.phrase_list[IC_position[0]]):
                    if int(row[1][0:-1]) == IC_position[1]:
                        IGC_positions.append((IC_position[0], row_y))

        if IP_position != None and IP_position in e_position_list_intra:
            self.IP_self = 1
            e_position_list_intra.remove(IP_position)
            # 済e_position_listからリムーブ、文内であることを確認
        for IC_position in IC_positions:
            if IC_position in e_position_list_intra:
                self.IC_self = 1
                e_position_list_intra.remove(IC_position)

        if IGP_position in e_position_list_intra:
            self.IGP_self = 1
            e_position_list_intra.remove(IGP_position)

        for IGC_position in IGC_positions:
            if IGC_position in e_position_list_intra:
                self.IGC_self = 1
                e_position_list_intra.remove(IGC_position)
        if latter_position != None and latter_position in e_position_list_intra:
            self.latter_self = 1
            e_position_list_intra.remove(latter_position)
        # print(e_position_list_intra)
        for former_position in former_positions:
            if former_position in e_position_list_intra:
                self.former_self = 1
                e_position_list_intra.remove(former_position)
        # 済IB、IAは文内
        self.IB_self = 0
        self.IA_self = 0
        self.B1 = 0
        self.B2 = 0
        self.B3 = 0
        for e_position in e_position_list:
            if row_num_position[0] == e_position[0] and row_num_position[1] > e_position[1]:
                self.IB_self = 1
            if row_num_position[0] == e_position[0] and row_num_position[1] < e_position[1]:
                self.IA_self = 1
            if row_num_position[0] == e_position[0] + 1:
                self.B1 = 1
            if row_num_position[0] == e_position[0] + 2:
                self.B2 = 1
            if row_num_position[0] == e_position[0] + 3:
                self.B3 = 1
        self.It_ga_ov = 0
        self.It_ga_om = 0
        self.It_wo_ov = 0
        self.It_wo_om = 0
        self.It_ni_ov = 0
        self.It_ni_om = 0
        for It_position in It_positions:
            if It_position[1] != -1:
                It_case_dict = self.phrase_case_list[
                    It_position[0]][It_position[1]]

                for case_name in case_list:
                    It_case_set = It_case_dict[case_name]
                    for x in It_case_set:
                        It_case_list = x.split("/")
                        if len(It_case_list) > 2 and It_case_list[2] == self.basic_form_e and It_case_list[1] != "O":
                            if case_name == "ガ格":
                                self.It_ga_ov = 1
                            if case_name == "ヲ格":
                                self.It_wo_ov = 1
                            if case_name == "ニ格":
                                self.It_ni_ov = 1
                        if len(It_case_list) > 2 and It_case_list[2] == self.basic_form_e and It_case_list[1] == "O":
                            if case_name == "ガ格":
                                self.It_ga_om = 1
                            if case_name == "ヲ格":
                                self.It_wo_om = 1
                            if case_name == "ニ格":
                                self.It_ni_om = 1

        self.IP_ga_ov = 0
        self.IP_ga_om = 0
        self.IP_wo_ov = 0
        self.IP_wo_om = 0
        self.IP_ni_ov = 0
        self.IP_ni_om = 0
        if IP_position != None and IP_position[1] != -1:
            IP_case_dict = self.phrase_case_list[
                IP_position[0]][IP_position[1]]
            for case_name in case_list:
                IP_case_set = IP_case_dict[case_name]
                for x in IP_case_set:
                    IP_case_list = x.split("/")
                    if len(IP_case_list) > 2 and IP_case_list[2] == self.basic_form_e and IP_case_list[1] != "O":
                        if case_name == "ガ格":
                            self.IP_ga_ov = 1
                        if case_name == "ヲ格":
                            self.IP_wo_ov = 1
                        if case_name == "ニ格":
                            self.IP_ni_ov = 1
                    if len(IP_case_list) > 2 and IP_case_list[2] == self.basic_form_e and IP_case_list[1] == "O":
                        if case_name == "ガ格":
                            self.IP_ga_om = 1
                        if case_name == "ヲ格":
                            self.IP_wo_om = 1
                        if case_name == "ニ格":
                            self.IP_ni_om = 1
        self.IC_ga_ov = 0
        self.IC_ga_om = 0
        self.IC_wo_ov = 0
        self.IC_wo_om = 0
        self.IC_ni_ov = 0
        self.IC_ni_om = 0
        for IC_position in IC_positions:
            if IC_position[1] != -1:
                IC_case_dict = self.phrase_case_list[
                    IC_position[0]][IC_position[1]]

                for case_name in case_list:
                    IC_case_set = IC_case_dict[case_name]
                    for x in IC_case_set:
                        IC_case_list = x.split("/")
                        if len(IC_case_list) > 2 and IC_case_list[2] == self.basic_form_e and IC_case_list[1] != "O":
                            if case_name == "ガ格":
                                self.IC_ga_ov = 1
                            if case_name == "ヲ格":
                                self.IC_wo_ov = 1
                            if case_name == "ニ格":
                                self.IC_ni_ov = 1
                        if len(IC_case_list) > 2 and IC_case_list[2] == self.basic_form_e and IC_case_list[1] == "O":
                            if case_name == "ガ格":
                                self.IC_ga_om = 1
                            if case_name == "ヲ格":
                                self.IC_wo_om = 1
                            if case_name == "ニ格":
                                self.IC_ni_om = 1
        self.IGP_ga_ov = 0
        self.IGP_ga_om = 0
        self.IGP_wo_ov = 0
        self.IGP_wo_om = 0
        self.IGP_ni_ov = 0
        self.IGP_ni_om = 0
        if IGP_position != None and IGP_position[1] != -1:
            IGP_case_dict = self.phrase_case_list[
                IGP_position[0]][IGP_position[1]]
            for case_name in case_list:
                IGP_case_set = IGP_case_dict[case_name]
                for x in IGP_case_set:
                    IGP_case_list = x.split("/")
                    if len(IGP_case_list) > 2 and IGP_case_list[2] == self.basic_form_e and IGP_case_list[1] != "O":
                        if case_name == "ガ格":
                            self.IGP_ga_ov = 1
                        if case_name == "ヲ格":
                            self.IGP_wo_ov = 1
                        if case_name == "ニ格":
                            self.IGP_ni_ov = 1
                    if len(IGP_case_list) > 2 and IGP_case_list[2] == self.basic_form_e and IGP_case_list[1] == "O":
                        if case_name == "ガ格":
                            self.IGP_ga_om = 1
                        if case_name == "ヲ格":
                            self.IGP_wo_om = 1
                        if case_name == "ニ格":
                            self.IGP_ni_om = 1
        self.IGC_ga_ov = 0
        self.IGC_ga_om = 0
        self.IGC_wo_ov = 0
        self.IGC_wo_om = 0
        self.IGC_ni_ov = 0
        self.IGC_ni_om = 0
        for IGC_position in IGC_positions:
            if IGC_position[1] != -1:
                IGC_case_dict = self.phrase_case_list[
                    IGC_position[0]][IGC_position[1]]
                for case_name in case_list:
                    IGC_case_set = IGC_case_dict[case_name]
                    for x in IGC_case_set:
                        IGC_case_list = x.split("/")
                        if len(IGC_case_list) > 2 and IGC_case_list[2] == self.basic_form_e and IGC_case_list[1] != "O":
                            if case_name == "ガ格":
                                self.IGC_ga_ov = 1
                            if case_name == "ヲ格":
                                self.IGC_wo_ov = 1
                            if case_name == "ニ格":
                                self.IGC_ni_ov = 1
                        if len(IGC_case_list) > 2 and IGC_case_list[2] == self.basic_form_e and IGC_case_list[1] == "O":
                            if case_name == "ガ格":
                                self.IGC_ga_om = 1
                            if case_name == "ヲ格":
                                self.IGC_wo_om = 1
                            if case_name == "ニ格":
                                self.IGC_ni_om = 1
        self.latter_ga_ov = 0
        self.latter_ga_om = 0
        self.latter_wo_ov = 0
        self.latter_wo_om = 0
        self.latter_ni_ov = 0
        self.latter_ni_om = 0
        if latter_position != None and latter_position[1] != -1:
            latter_case_dict = self.phrase_case_list[
                latter_position[0]][latter_position[1]]
            for case_name in case_list:
                latter_case_set = latter_case_dict[case_name]
                for x in latter_case_set:
                    latter_case_list = x.split("/")
                    if len(latter_case_list) > 2 and latter_case_list[2] == self.basic_form_e and latter_case_list[1] != "O":
                        if case_name == "ガ格":
                            self.latter_ga_ov = 1
                        if case_name == "ヲ格":
                            self.latter_wo_ov = 1
                        if case_name == "ニ格":
                            self.latter_ni_ov = 1
                    if len(latter_case_list) > 2 and latter_case_list[2] == self.basic_form_e and latter_case_list[1] == "O":
                        if case_name == "ガ格":
                            self.latter_ga_om = 1
                        if case_name == "ヲ格":
                            self.latter_wo_om = 1
                        if case_name == "ニ格":
                            self.latter_ni_om = 1
        self.former_ga_ov = 0
        self.former_ga_om = 0
        self.former_wo_ov = 0
        self.former_wo_om = 0
        self.former_ni_ov = 0
        self.former_ni_om = 0
        for former_position in former_positions:
            if former_position[1] != -1:
                former_case_dict = self.phrase_case_list[
                    former_position[0]][former_position[1]]

                for case_name in case_list:
                    former_case_set = former_case_dict[case_name]
                    for x in former_case_set:
                        former_case_list = x.split("/")
                        if len(former_case_list) > 2 and former_case_list[2] == self.basic_form_e and former_case_list[1] != "O":
                            if case_name == "ガ格":
                                self.former_ga_ov = 1
                            if case_name == "ヲ格":
                                self.former_wo_ov = 1
                            if case_name == "ニ格":
                                self.former_ni_ov = 1
                        if len(former_case_list) > 2 and former_case_list[2] == self.basic_form_e and former_case_list[1] == "O":
                            if case_name == "ガ格":
                                self.former_ga_om = 1
                            if case_name == "ヲ格":
                                self.former_wo_om = 1
                            if case_name == "ニ格":
                                self.former_ni_om = 1
        self.IB_ga_ov = 0
        self.IB_ga_om = 0
        self.IB_wo_ov = 0
        self.IB_wo_om = 0
        self.IB_ni_ov = 0
        self.IB_ni_om = 0
        IB_case_lists = self.phrase_case_list[
            row_num_position[0]][:row_num_position[1]]
        if len(IB_case_lists) > 0:
            for IB_case_dict in IB_case_lists:
                for case_name in case_list:
                    IB_case_set = IB_case_dict[case_name]
                    for x in IB_case_set:
                        IB_case_list = x.split("/")
                        if len(IB_case_list) > 2 and IB_case_list[2] == self.basic_form_e and IB_case_list[1] != "O":
                            if case_name == "ガ格":
                                self.IB_ga_ov = 1
                            if case_name == "ヲ格":
                                self.IB_wo_ov = 1
                            if case_name == "ニ格":
                                self.IB_ni_ov = 1
                        if len(IB_case_list) > 2 and IB_case_list[2] == self.basic_form_e and IB_case_list[1] == "O":
                            if case_name == "ガ格":
                                self.IB_ga_om = 1
                            if case_name == "ヲ格":
                                self.IB_wo_om = 1
                            if case_name == "ニ格":
                                self.IB_ni_om = 1
        self.IA_ga_ov = 0
        self.IA_ga_om = 0
        self.IA_wo_ov = 0
        self.IA_wo_om = 0
        self.IA_ni_ov = 0
        self.IA_ni_om = 0
        if len(self.phrase_case_list[row_num_position[0]]) > row_num_position[1] + 1:
            IA_case_lists = self.phrase_case_list[
                row_num_position[0]][row_num_position[1] + 1:]
            if len(IA_case_lists) > 0:
                for IA_case_dict in IA_case_lists:
                    for case_name in case_list:
                        IA_case_set = IA_case_dict[case_name]
                        for x in IA_case_set:
                            IA_case_list = x.split("/")
                            if len(IA_case_list) > 2 and IA_case_list[2] == self.basic_form_e and IA_case_list[1] != "O":
                                if case_name == "ガ格":
                                    self.IA_ga_ov = 1
                                if case_name == "ヲ格":
                                    self.IA_wo_ov = 1
                                if case_name == "ニ格":
                                    self.IA_ni_ov = 1
                            if len(IA_case_list) > 2 and IA_case_list[2] == self.basic_form_e and IA_case_list[1] == "O":
                                if case_name == "ガ格":
                                    self.IA_ga_om = 1
                                if case_name == "ヲ格":
                                    self.IA_wo_om = 1
                                if case_name == "ニ格":
                                    self.IA_ni_om = 1
        self.B1_ga_ov = 0
        self.B1_ga_om = 0
        self.B1_wo_ov = 0
        self.B1_wo_om = 0
        self.B1_ni_ov = 0
        self.B1_ni_om = 0
        if row_num_position[0] > 0:
            B1_case_lists = self.phrase_case_list[row_num_position[0] - 1]
            if len(B1_case_lists) > 0:
                for B1_case_dict in B1_case_lists:
                    for case_name in case_list:
                        B1_case_set = B1_case_dict[case_name]
                        for x in B1_case_set:
                            B1_case_list = x.split("/")
                            if len(B1_case_list) > 2 and B1_case_list[2] == self.basic_form_e and B1_case_list[1] != "O":
                                if case_name == "ガ格":
                                    self.B1_ga_ov = 1
                                if case_name == "ヲ格":
                                    self.B1_wo_ov = 1
                                if case_name == "ニ格":
                                    self.B1_ni_ov = 1
                            if len(B1_case_list) > 2 and B1_case_list[2] == self.basic_form_e and B1_case_list[1] == "O":
                                if case_name == "ガ格":
                                    self.B1_ga_om = 1
                                if case_name == "ヲ格":
                                    self.B1_wo_om = 1
                                if case_name == "ニ格":
                                    self.B1_ni_om = 1

        self.B2_ga_ov = 0
        self.B2_ga_om = 0
        self.B2_wo_ov = 0
        self.B2_wo_om = 0
        self.B2_ni_ov = 0
        self.B2_ni_om = 0
        if row_num_position[0] > 1:
            B2_case_lists = self.phrase_case_list[row_num_position[0] - 2]
            if len(B2_case_lists) > 0:

                for B2_case_dict in B2_case_lists:
                    for case_name in case_list:
                        B2_case_set = B2_case_dict[case_name]
                        for x in B2_case_set:
                            B2_case_list = x.split("/")
                            if len(B2_case_list) > 2 and B2_case_list[2] == self.basic_form_e and B2_case_list[1] != "O":
                                if case_name == "ガ格":
                                    self.B2_ga_ov = 1
                                if case_name == "ヲ格":
                                    self.B2_wo_ov = 1
                                if case_name == "ニ格":
                                    self.B2_ni_ov = 1
                            if len(B2_case_list) > 2 and B2_case_list[2] == self.basic_form_e and B2_case_list[1] == "O":
                                if case_name == "ガ格":
                                    self.B2_ga_om = 1
                                if case_name == "ヲ格":
                                    self.B2_wo_om = 1
                                if case_name == "ニ格":
                                    self.B2_ni_om = 1

        self.B3_ga_ov = 0
        self.B3_ga_om = 0
        self.B3_wo_ov = 0
        self.B3_wo_om = 0
        self.B3_ni_ov = 0
        self.B3_ni_om = 0
        if row_num_position[0] > 2:
            B3_case_lists = []
            for x in self.phrase_case_list[:row_num_position[0] - 3]:
                B3_case_lists.extend(x)
            if len(B3_case_lists) > 0:

                for B3_case_dict in B3_case_lists:
                    for case_name in case_list:
                        B3_case_set = B3_case_dict[case_name]
                        for x in B3_case_set:
                            B3_case_list = x.split("/")
                            if len(B3_case_list) > 2 and B3_case_list[2] == self.basic_form_e and B3_case_list[1] != "O":
                                if case_name == "ガ格":
                                    self.B3_ga_ov = 1
                                if case_name == "ヲ格":
                                    self.B3_wo_ov = 1
                                if case_name == "ニ格":
                                    self.B3_ni_ov = 1
                            if len(B3_case_list) > 2 and B3_case_list[2] == self.basic_form_e and B3_case_list[1] == "O":
                                if case_name == "ガ格":
                                    self.B3_ga_om = 1
                                if case_name == "ヲ格":
                                    self.B3_wo_om = 1
                                if case_name == "ニ格":
                                    self.B3_ni_om = 1

    def w2v_features(self):
        self.mean_vec = self.mean_vec_dic[self.case][self.cf]
        if self.cf.split(":")[0] in self.all_mean_vec_dic[self.case]:
            self.all_mean_vec = self.all_mean_vec_dic[
                self.case][self.cf.split(":")[0]]
        else:
            self.all_mean_vec = np.zeros(500)
        # print(self.vec_e)
        if any(self.vec_e != np.zeros(500)) and any(self.mean_vec != np.zeros(500)):
            self.w2v_feature_mean_vec = 1.0 - \
                distance.cosine(self.vec_e, self.mean_vec)
        else:
            self.w2v_feature_mean_vec = 0.0
        if any(self.vec_e != np.zeros(500)) and any(self.all_mean_vec != np.zeros(500)):
            self.w2v_feature_all_mean_vec = 1.0 - \
                distance.cosine(self.vec_e, self.all_mean_vec)
        else:
            self.w2v_feature_all_mean_vec = 0.0


if __name__ == "__main__":
    x = Extract_features(
        # text_name="knp_dismantled_data/00621_B_OC06_02189.knp", num=0)
        text_name="knp_dismantled_data/00003_A_PM24_00003.knp", num=9)
    prepare_for_text(x)
    named_entity_for_text_dic = defaultdict(set)
    vec_for_text_dic = {}
    x.set_text_dics(named_entity_for_text_dic, vec_for_text_dic)
    # for x_row in x.row_list_cf:
    #     for y_row in x_row:
    #         pass
    cf_dic = {}
    headword_id_dic = {}
    mean_vec_dic = {}
    all_mean_vec_dic = {}
    for case_name in case_list:
        cf_dic[case_name] = defaultdict(list)
        headword_id_dic[case_name] = defaultdict(list)
        mean_vec_dic[case_name] = {}
        all_mean_vec_dic[case_name] = {}
    x.set_result(["分かる/わかる:動1", "分かる/わかる:動2"],
                 cf_dic, headword_id_dic, mean_vec_dic, all_mean_vec_dic)
    print(x.headword_id_dic)
    # print(x.mean_vec_dic)
    # print(x.all_mean_vec_dic)
    # x.set_info(cf="成る/なる:動1", case="ニ格", e="扇形", row_num=193, row_num_e=180)
    x.set_info(cf="分かる/わかる:動1", case="ガ格", e="こと", row_num=263, row_num_e=256)
    #x.set_info(cf="分かる/わかる:動1", case="ガ格", e="円", row_num=180, row_num_e=189)
    # x.set_info(cf="苦しむ/くるしむ:動1", case="ガ格",e="私", row_num=128, row_num_e=120)
    # x.set_info(cf="苦しむ/くるしむ:動1", case="ニ格", e="理解", row_num=128, row_num_e=124)
    # x.set_info(cf="向ける/むける:動2", case="ニ格", e="決戦", row_num=128,
    # row_num_e=137)
    e_list = search_e(x.x, x.basic_form_e, x.e_normalize)
    e_position_list = []
    if len(e_list) > 1:
        for e_num in e_list:
            e_position_list.append(
                search_phrase_list(x.phrase_list, int(e_num)))
    # e_position_listは自分(e)以外の同じ語句が存在する場合のx.phrase_list上の位置インデックスを返している
    # print(x.basic_form_e)
    # print(x.modified_e)
    # print(x.juman_e)
    # print(x.phrase_list)
    # print(x.phrase_case_list)
    # print(x.row_list_e)
    print(x.row_list_cf)
    print(x.row_list_cf_case_list)
    exit()
    x.feature1()
    # x.feature2()
    # x.feature4()
    # x.feature5()
    # x.feature7()
    # x.feature11()
    # print(x.feature_12)
    print(x.feature_1)
    print(x.feature_2)
    print(x.feature_3)
    # x.feature13()
    # x.feature20()
    x.position_category(search_phrase_list(x.phrase_list, x.row_num),
                        search_phrase_list(x.phrase_list, x.row_num_e), e_position_list)
    # x.position_category(search_phrase_list(x.phrase_list, x.row_num), search_phrase_list(x.phrase_list, x.row_num_e), [(4, 0), (4, 1), (4, 2)])
    # print(search_phrase_list(x.phrase_list, 128))
    # print(search_phrase_list(x.phrase_list, 124))
    print([x.Itopic, x.It_self, x.IP_self, x.IC_self, x.IGP_self, x.IGC_self, x.former_self, x.latter_self, "intra",
           x.IB_self, x.IA_self, x.B1, x.B2, x.B3, "inter",
           x.It_ga_ov, x.It_ga_om, x.It_wo_ov, x.It_wo_om, x.It_ni_ov, x.It_ni_om, "It",
           x.IP_ga_ov, x.IP_ga_om, x.IP_wo_ov, x.IP_wo_om, x.IP_ni_ov, x.IP_ni_om, "IP",
           x.IC_ga_ov, x.IC_ga_om, x.IC_wo_ov, x.IC_wo_om, x.IC_ni_ov, x.IC_ni_om, "IC",
           x.IGP_ga_ov, x.IGP_ga_om, x.IGP_wo_ov, x.IGP_wo_om, x.IGP_ni_ov, x.IGP_ni_om, "IGP",
           x.IGC_ga_ov, x.IGC_ga_om, x.IGC_wo_ov, x.IGC_wo_om, x.IGC_ni_ov, x.IGC_ni_om, "IGC",
           x.former_ga_ov, x.former_ga_om, x.former_wo_ov, x.former_wo_om, x.former_ni_ov, x.former_ni_om, "former",
           x.latter_ga_ov, x.latter_ga_om, x.latter_wo_ov, x.latter_wo_om, x.latter_ni_ov, x.latter_ni_om, "latter",
           x.IA_ga_ov, x.IA_ga_om, x.IA_wo_ov, x.IA_wo_om, x.IA_ni_ov, x.IA_ni_om, "IA",
           x.IB_ga_ov, x.IB_ga_om, x.IB_wo_ov, x.IB_wo_om, x.IB_ni_ov, x.IB_ni_om, "IB",
           x.B1_ga_ov, x.B1_ga_om, x.B1_wo_ov, x.B1_wo_om, x.B1_ni_ov, x.B1_ni_om, "B1",
           x.B2_ga_ov, x.B2_ga_om, x.B2_wo_ov, x.B2_wo_om, x.B2_ni_ov, x.B2_ni_om, "B2",
           x.B3_ga_ov, x.B3_ga_om, x.B3_wo_ov, x.B3_wo_om, x.B3_ni_ov, x.B3_ni_om, "B3"])
    # print(x.feature_20)
    # print(x.feature_21)
    # print(x.feature_22)
    # print(x.feature_23)
    # print(x.feature_24)
    # print(x.feature_25)
    # print(x.feature_26)
    # print(x.feature_27)
    # print(x.feature_28)
    # print(x.feature_29)
    # print(x.feature_30)
    # print(x.feature_31)
    # print(x.feature_32)

    # y=Extract_features(cf="馬鹿だ/ばかだ:形0", text_name="knp_text_for_using_cf2.knp")
    # y.set_lists(make_phrase_list("knp_text_for_using_cf2.knp"))
    # y.set_info(case="ガ格" , e="君", row_num=12, row_num_e=3)
    # print(y.basic_form_e)
    # print(y.modified_e)
    # print(y.juman_e)
    # #y.feature1()
    # #y.feature2()
    # #y.feature4()
    # #y.feature5()
    # #y.feature7()
    # #y.feature11()
    # #y.feature13()
    # y.feature20()
    # print(y.feature_20)
    # print(y.feature_21)
    # print(y.feature_22)
    # print(y.feature_23)
    # print(y.feature_24)
    # print(y.feature_25)
    # print(y.feature_26)
    # print(y.feature_27)
    # print(y.feature_28)
    # print(y.feature_29)
    # print(y.feature_30)
    # print(y.feature_31)
    # print(y.feature_32)
    # print(y.feature_33)

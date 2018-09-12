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
import heapq
from datetime import datetime
import copy
from test_using_cf2_hill_climbing import *

case_list_init = ["ガ格", "ヲ格", "ニ格"]
with open("id_list_dic.pickle", mode="rb") as f:
    id_list_dic = pickle.load(f)
id_list = list(id_list_dic.keys())
# with open("KNP_info_dict_row_num_fixed1.pickle", mode="rb") as f:
KNP_info_dict_row_num = None
# 今見ている文章の*の行番号リスト．komejirusi_list_maker()で初期化
komejirusi_list = None
with open("hyousou_dic.pickle", mode="rb") as f:
    hyousou_dic = pickle.load(f)
hyousou_dic_case = defaultdict(float)
for case, hyousou_case_dic in hyousou_dic.items():
    for hyousou_case, hyousou_count in hyousou_case_dic.items():
        if hyousou_case != "None":
            hyousou_dic_case[hyousou_case] += hyousou_count


def id_list_search(id_head_list):
    # print(id_head_list)
    id_head_list_copy = id_head_list[:]
    id_head_in_set_dic = defaultdict(set)
    for id_head in id_list:
        for id_head_x in id_head_list_copy:
            if len(id_head_list_copy) == 1:
                # BCCWJ中の出現が漢字表記で曖昧性がない時
                if id_head_x == id_head + ":":
                    id_head_in_set_dic[id_head_x] |= id_list_dic[id_head]
            else:
                # BCCWJ中の出現がひらがな表記で曖昧性がある時
                if id_head_x + ":" in id_head + ":" or id_head_x + "?" in id_head + ":":
                    id_head_in_set_dic[id_head_x] |= id_list_dic[id_head]
    # 最後ので初期化
    # &=で共通集合のみ取り出す
    id_list_keep = id_head_in_set_dic[id_head_x]
    for id_item in id_head_in_set_dic.items():
        id_list_keep &= id_item[1]

    # print(id_head_in_set)
    return sorted(list(id_list_keep))


def id_list_search1(id_head_list):
    # print(id_head_list)
    id_head_list_copy = id_head_list[:]
    id_head_in_set_dic = defaultdict(set)
    for id_head in id_list:
        for id_head_x in id_head_list_copy:
            if id_head_x == id_head[:len(id_head_x)]:
                id_head_in_set_dic[id_head_x] |= id_list_dic[id_head]
    # 最後ので初期化
    # &=で共通集合のみ取り出す
    id_list_keep = id_head_in_set_dic[id_head_x]
    for id_item in id_head_in_set_dic.items():
        id_list_keep &= id_item[1]
    # print(id_head_in_set)
    return sorted(list(id_list_keep))


# 使わない
"""
def id_list_search_old(id_head_list):
    # print(id_head_list)
    id_head_list_copy = id_head_list[:]
    id_head_in_set_dic = set()
    for id_head in id_list:
        for id_head_x in id_head_list_copy:
            if id_head_x + ":" in id_head + ":" or id_head_x + "?" in id_head + ":":
                id_head_in_set_dic |= id_list_dic[id_head]
    # 最後ので初期化
    # id_list_keep = id_head_in_set_dic[id_head_x]
    # for id_item in id_head_in_set_dic.items():
    #     id_list_keep |= id_item[1]

    # print(id_head_in_set)
    return id_head_in_set_dic


def id_list_search1_old(id_head_list):
    # print(id_head_list)
    id_head_list_copy = id_head_list[:]
    id_head_in_set = set()
    for id_head in id_list:
        for id_head_x in id_head_list_copy:
            if id_head_x == id_head[:len(id_head_x)]:
                id_head_in_set |= id_list_dic[id_head]
    # print(id_head_in_set)
    return id_head_in_set
"""


def read_pickles(num, NTC_flag):
    if NTC_flag == 0:
        text_fnames = glob.glob('resultx' + str(num) + '/*.pickle')
    elif NTC_flag == 1:
        text_fnames = glob.glob('resulty' + str(num) + '/*.pickle')
    return (text_fnames)


def is_p_exist(text_name, yougen_candidate):
    a_yougen_row_num_dic = KNP_info_dict_row_num[text_name]
    a_yougen_kome_dic = defaultdict(set)
    for x in a_yougen_row_num_dic:
        a_yougen_kome_dic[komejirusi_devide(text_name, x)].add(x)
    y_row_kome_set = a_yougen_kome_dic[
        komejirusi_devide(text_name, str(yougen_candidate))]
    a_row_num_dic = None
    # 同じ※内にあるBCCWJの情報のうち、格情報がある行を選んでa_row_num_dicに入れている
    for a_yougen_kome in y_row_kome_set:
        for case_name in case_list_init:
            if jap_to_eng(case_name) in a_yougen_row_num_dic[a_yougen_kome].keys():
                a_row_num_dic = a_yougen_row_num_dic[a_yougen_kome]
                break
    if a_row_num_dic == None:
        return -1
    return 1


def komejirusi_list_maker(full_text):
    list1 = []
    global komejirusi_list

    rows = full_text.split("\n")
    for row_num, row in enumerate(rows):
        words = row.split()
        if len(words) > 0:
            if words[0] == "*":
                list1.append(str(row_num))
    komejirusi_list = list1


def komejirusi_devide(text, num):
    k_num_list = komejirusi_list

    for x_num, x in enumerate(k_num_list):
        if int(x) > int(num):
            return k_num_list[x_num - 1]
    return k_num_list[x_num]


def jap_to_eng(jap_case):
    # 注意、BCCWJようなので"o"
    if jap_case == "ガ格":
        return "ga"
    if jap_case == "ヲ格":
        return "o"
    if jap_case == "ニ格":
        return "ni"


# hyousou_counter: 表層格の比率のlogを返す
def hyousou_counter(case, e_candidate):
    if hyousou_dic[case][e_candidate[4]] != 0 and hyousou_dic_case[case] != 0:
        return math.log2(hyousou_dic[case][e_candidate[4]] / hyousou_dic_case[case])
    else:
        return math.log2(1 / hyousou_dic_case[case])


# score_returner: cfとeとcaseの組合せから，候補の擬似的なスコアを返す
def score_returner(score_vec, case_candidate, e_candidate, cf_p_dict, cf_candidate, x):
    # ここ距離的な素性も入れたほうがいいかも:score_vec[23]
    #hyousou_counter(case_candidate[:2], e_candidate)
    # 本当はcf_pスコア1/3するべき？しないであってる．Noneの含まれない格数で割るから
    returned_score = score_vec[0] + score_vec[10] + score_vec[-2] + \
        x.set_cf_p_dict(cf_p_dict, cf_candidate) - 0.5 * score_vec[23]
    # if cf_candidate == "絞る/しぼる~テ形+ある/ある:動1" and case_candidate == "ガ格" and
    # returned_score == float("-inf"):
    # print(cf_candidate, e_candidate, case_candidate)
    #     print(score_vec[0])
    #     print(score_vec[10])
    #     print(hyousou_counter(case_candidate[:2], e_candidate))
    #     print(x.set_cf_p_dict(cf_p_dict, cf_candidate))
    # if e_candidate[0] == "127":
    #     print(score_vec[23])
    #     print(score_vec[57])
    #     print(score_vec[58])
    #     exit()
    return returned_score


# 使わない
"""
def reduce_cf_candidate(vec_cf_dic, x, cf_p_dict):
    # e_cand,score
    score_vec_dic = defaultdict(list)
    cf_candidates = sorted(list(vec_cf_dic.keys()))

    # print(len(cf_candidates))
    for cf_candidate in cf_candidates:
        if cf_candidate == "removed_case":
            continue
        vec_e_dic = vec_cf_dic[cf_candidate]
        e_candidates = sorted(list(vec_e_dic.keys()),
                              key=lambda e_data: e_data[0])
        # print(e_candidates)

        for e_candidate in e_candidates:
            vec_case_dic = vec_e_dic[e_candidate]
            case_candidates = sorted(list(vec_case_dic.keys()))
            # relate_cases = vec_e_dic['None'].keys()
            for case_candidate in case_candidates:
                score_vec = vec_case_dic[case_candidate]
                if case_candidate[-2:] != "NA":
                    # try:
                    score = score_returner(
                        score_vec, case_candidate, e_candidate, cf_p_dict, cf_candidate, x)
                    # なんか足す

                    # ここで-inf値取る場合はcfに入る要素が0なので，弾くのは正しい
                    # except Exception as e:
                    #     print(score_vec)
                    #     print(case_candidates)
                    #     raise e

                    heapq.heappush(
                        score_vec_dic[case_candidate], (score, cf_candidate, e_candidate))
    nth_leargest_dic = n_best_cf_candidates(score_vec_dic)

    vec_cf_dic_alt = {}
    for cf_candidate in cf_candidates:
        if cf_candidate == "removed_case":
            vec_cf_dic_alt[cf_candidate] = vec_cf_dic[cf_candidate]
            continue
        vec_e_dic = vec_cf_dic[cf_candidate]
        vec_e_dic_alt = {}
        for e_candidate in e_candidates:
            vec_case_dic = vec_e_dic[e_candidate]
            case_candidates = sorted(list(vec_case_dic.keys()))
            # relate_cases = vec_e_dic['None'].keys()
            vec_case_dic_alt = {}
            for case_candidate in case_candidates:
                score_vec = vec_case_dic[case_candidate]
                if case_candidate[-2:] != "NA":
                    if (cf_candidate, e_candidate) in nth_leargest_dic[case_candidate]:
                        vec_case_dic_alt[case_candidate] = vec_case_dic[
                            case_candidate]
                else:
                    vec_case_dic_alt[case_candidate] = vec_case_dic[
                        case_candidate]
            vec_e_dic_alt[e_candidate] = vec_case_dic_alt
        vec_cf_dic_alt[cf_candidate] = vec_e_dic_alt
    return vec_cf_dic_alt


def n_best_cf_candidates(score_vec_dic):
    n = 100
    cases = sorted(list(score_vec_dic.keys()))
    nth_leargest_dic = defaultdict(list)
    for case in cases:
        nth_leargest_dic[case].extend(
            [(x[1], x[2]) for x in heapq.nlargest(n, score_vec_dic[case]) if x[0] != float("-inf")])
        # print(score_vec_dic[case])
        # print(heapq.nlargest(n, score_vec_dic[case]))
        # print(nth_leargest_dic[case])
        # sys.exit()
    return nth_leargest_dic


def reduce_e_candidate(vec_e_dic):
    # e_cand,score
    score_vec_dic = defaultdict(list)
    e_candidates = sorted(list(vec_e_dic.keys()), key=lambda x: x[0])
    print(e_candidates)

    for e_candidate in e_candidates:
        vec_case_dic = vec_e_dic[e_candidate]
        case_candidates = sorted(list(vec_case_dic.keys()))
        # relate_cases = vec_e_dic['None'].keys()
        for case_candidate in case_candidates:
            score_vec = vec_case_dic[case_candidate]
            if case_candidate[-2:] != "NA":
                try:
                    score = score_vec[0] + score_vec[10]
                except Exception as e:
                    print(score_vec)
                    print(case_candidates)
                    raise e

                heapq.heappush(score_vec_dic[case_candidate], score)
    nth_leargest_dic = n_best_e_candidates(score_vec_dic)

    vec_e_dic_alt = {}
    for e_candidate in e_candidates:
        vec_case_dic = vec_e_dic[e_candidate]
        case_candidates = sorted(list(vec_case_dic.keys()))
        # relate_cases = vec_e_dic['None'].keys()
        vec_case_dic_alt = {}
        for case_candidate in case_candidates:
            score_vec = vec_case_dic[case_candidate]
            if case_candidate[-2:] != "NA":
                score = score_vec[0] + score_vec[10]
                if score < nth_leargest_dic[case_candidate]:
                    pass
                else:
                    vec_case_dic_alt[case_candidate] = vec_case_dic[
                        case_candidate]
            else:
                vec_case_dic_alt[case_candidate] = vec_case_dic[case_candidate]
        vec_e_dic_alt[e_candidate] = vec_case_dic_alt
    return vec_e_dic_alt


def n_best_e_candidates(score_vec_dic):
    n = 5
    cases = sorted(list(score_vec_dic.keys()))
    nth_leargest_dic = defaultdict(float)
    for case in cases:
        nth_leargest_dic[case] = heapq.nlargest(n, score_vec_dic[case])[-1]
        print(score_vec_dic[case])
        print(heapq.nlargest(n, score_vec_dic[case]))
        sys.exit()
    return nth_leargest_dic
"""


# reject_aux: BCCWJとKNPでは使役，受身の取扱が違うので”aux”タグを見たら弾いてる
def reject_aux(text_name, a_yougen_row_num_dic, a_yougen_kome_dic):
    reject_aux_kome_list = []
    for row_num, info_dic in a_yougen_row_num_dic.items():
        for info_type, info in info_dic.items():
            if "aux" in info:
                reject_aux_kome_list.append(
                    komejirusi_devide(text_name, row_num))
    # print(reject_aux_kome_list)
    # print(a_yougen_kome_dic)
    reject_aux_list = []
    for x in reject_aux_kome_list:
        reject_aux_list.extend(a_yougen_kome_dic[x])
    return reject_aux_list


# take_answer_yougen_row_num_dic: 指定されたテキスト（BCCWJ）の付与情報を取ってくる
def take_answer_yougen_row_num_dic(text_name):
    a_yougen_row_num_dic = KNP_info_dict_row_num[text_name]
    a_yougen_kome_dic = defaultdict(list)
    for x in a_yougen_row_num_dic:
        a_yougen_kome_dic[komejirusi_devide(text_name, x)].append(x)
    return a_yougen_row_num_dic, a_yougen_kome_dic


def relate_answer_kome_reader(text_name, e_candidate, info_set, komejirusi_devided_dic):
    e_candidate[0]
    e_kome = ""
    if e_candidate[0] not in komejirusi_devided_dic:
        e_kome = komejirusi_devide(
            text_name, str(e_candidate[0]))
        komejirusi_devided_dic[
            e_candidate[0]] = e_kome
    else:
        e_kome = komejirusi_devided_dic[
            e_candidate[0]]
    kome_info_set = set()
    for info in info_set:
        info_kome = ""
        # print(info)
        if info not in komejirusi_devided_dic:
            # print(info_set)
            info_kome = komejirusi_devide(
                text_name, str(info))
            komejirusi_devided_dic[
                info] = info_kome
        else:
            info_kome = komejirusi_devided_dic[info]
        kome_info_set.add(info_kome)
    if e_kome in kome_info_set:
        return e_candidate
    else:
        return -1


# relate_info_extracter: 係り受け関係をBCCWJから取ってくる
def relate_info_extracter(text_name, a_yougen_row_num_dic, a_yougen_kome_dic, komejirusi_devided_dic, yougen_candidate, case_name_now):
    # is_e_answer()から作った
    # print(komejirusi_devided_dic[yougen_candidate[0]])

    yougen_kome = ""
    if yougen_candidate[0] not in komejirusi_devided_dic:
        yougen_kome = komejirusi_devide(
            text_name, str(yougen_candidate[0]))
        komejirusi_devided_dic[
            yougen_candidate[0]] = yougen_kome
    else:
        yougen_kome = komejirusi_devided_dic[
            yougen_candidate[0]]
    y_row_kome_set = sorted(list(a_yougen_kome_dic[yougen_kome]), key=int)
    a_row_num_dic = None
    # 同じ※内にあるBCCWJの情報のうち、格情報がある行を選んでa_row_num_dicに入れている
    break_flag = 0
    for a_yougen_kome in y_row_kome_set:
        for case_name in case_list_init:
            if jap_to_eng(case_name) in a_yougen_row_num_dic[a_yougen_kome]:
                a_row_num_dic = a_yougen_row_num_dic[a_yougen_kome]
                break_flag = 1
        if break_flag == 1:
            break
    if a_row_num_dic == None:
        return -1, -1
    # print(a_yougen_row_num_dic)
    # print(a_row_num_dic)
    # print(e_candidate)
    #a_row_num_dic = {"ga": {"exo2", "373"}}
    # 辞書を作ってkomejirusi_devideを呼ばない
    if case_name_now != "all":
        #'ga/ni'はanswer_BCCWJで解消 #ga/niは使役などでひっくり返ってるやつなので，gaかつniみたいな変な処理してるけど無視されてるはず
        returned_list = []
        answer_info_list = []
        eng_case_name_now = jap_to_eng(case_name_now)
        if eng_case_name_now in a_row_num_dic and eng_case_name_now + "_dep" in a_row_num_dic:
            if "dep" in a_row_num_dic[eng_case_name_now + "_dep"]:
                returned_list.extend(a_row_num_dic[eng_case_name_now])
                # print(info_x)
                # print(coreference_dic)
                # # print(len(coreference_dic[info_x]))
                # print(a_row_num_dic)
                # print(a_yougen_row_num_dic)
                # exit()
            elif "zero" in a_row_num_dic[eng_case_name_now + "_dep"]:
                answer_info_list.extend(a_row_num_dic[eng_case_name_now])
        elif eng_case_name_now in a_row_num_dic:
            if len(a_row_num_dic[eng_case_name_now] & set('exo1', 'exo2', 'exog')) > 0:
                answer_info_list.extend(a_row_num_dic[eng_case_name_now])
        else:
            answer_info_list.append("Not_fill")

        return returned_list, answer_info_list
    else:
        returned_case_dict = defaultdict(list)
        answer_info_dict = defaultdict(list)
        for case_init in case_list_init:
            eng_case_name_now = jap_to_eng(case_init)
            if eng_case_name_now in a_row_num_dic and eng_case_name_now + "_dep" in a_row_num_dic:
                if "dep" in a_row_num_dic[eng_case_name_now + "_dep"]:
                    returned_case_dict[case_init].extend(
                        a_row_num_dic[eng_case_name_now])
                elif "zero" in a_row_num_dic[eng_case_name_now + "_dep"]:
                    answer_info_dict[case_init].extend(
                        a_row_num_dic[eng_case_name_now])
            elif eng_case_name_now in a_row_num_dic:
                if len(a_row_num_dic[eng_case_name_now] & set(['exo1', 'exo2', 'exog'])) > 0:
                    answer_info_dict[case_init].extend(
                        a_row_num_dic[eng_case_name_now])
            else:
                answer_info_dict[case_init].append("Not_fill")
        return returned_case_dict, answer_info_dict


# coreference_dic_maker: 共参照関係を取ってくる
def coreference_dic_maker(a_yougen_row_num_dic):
    coreference_dic = {}
    for a_yougen_row_num in a_yougen_row_num_dic:
        a_yougen_row_num_info = a_yougen_row_num_dic[a_yougen_row_num]
        if "eq" in a_yougen_row_num_info:
            for row_num_info in a_yougen_row_num_info["eq"]:
                coreference_dic[row_num_info] = a_yougen_row_num_info["eq"]
    return coreference_dic


# coreference_reducer:
# coreference_dicを使って「これ」と「ライセンス」なんかを統一．紛らわしい候補（"名詞形態指示詞"）を削除
def coreference_reducer(e_set, coreference_dic):
    e_set_returned = copy.copy(e_set)
    coreference_list = []
    for e_candidate in e_set:
        if e_candidate[0] in coreference_dic:
            coreference_set_sub = set()
            coreference_set_sub.add(e_candidate)
            for e_candidate_co in e_set:
                if e_candidate_co[0] in coreference_dic[e_candidate[0]]:
                    coreference_set_sub.add(e_candidate_co)
            coreference_list.append(coreference_set_sub)
    for coreference_e_set in coreference_list:
        # 共参照ごとにまとめたとき，その集合中に"名詞"と"名詞形態指示詞"が同居してたら"名詞形態指示詞"を削除
        sijisi_flag = 0
        meisi_flag = 0
        sijisi_list = []
        meisi_list = []
        for e_candidate_x in coreference_e_set:
            if e_candidate_x[5] == "名詞形態指示詞":
                sijisi_flag = 1
                sijisi_list.append(e_candidate_x)
            elif e_candidate_x[5] == "名詞":
                meisi_flag = 1
                meisi_list.append(e_candidate_x)
        # print(coreference_list)
        # print(sijisi_list)
        meisi_list.sort(key=lambda e_co: len(e_co[2]), reverse=True)
        if sijisi_flag == 1 and meisi_flag == 1:
            for removed_e in sijisi_list:
                if removed_e in e_set_returned:
                    # 共参照関係にある名詞のうち最も長いもので名詞形態指示詞を置き換える
                    alter_meisi = (removed_e[0], meisi_list[0][1], meisi_list[0][
                                   2], meisi_list[0][3], removed_e[4], removed_e)

                    e_set_returned.add(alter_meisi)
                    e_set_returned.remove(removed_e)
    # print(e_set_returned)
    return e_set_returned


# feature_vec_maker: 素性ベクトルを作る
def feature_vec_maker(x, cf_candidate, case_candidate, e_candidate, y_row, feature34_dict, vec_all_dic):
    vec_keep = []
    feature_34 = feature34_dict[case_candidate]
    if (cf_candidate, e_candidate, case_candidate) in vec_all_dic:
        return vec_all_dic[(cf_candidate, e_candidate, case_candidate)][1][:]
    case_keep = None
    for case_init in case_list_init:
        if (cf_candidate, e_candidate, case_init) in vec_all_dic:
            case_keep = case_init
            break
    if case_keep != None:
        x.set_info(cf=cf_candidate, case=case_candidate, e=e_candidate[
                   1], row_num=int(y_row[0]), row_num_e=int(e_candidate[0]))
        x.feature1()
        if x.feature_1 == float("-inf"):
            # if cf_candidate == "絞る/しぼる~テ形+ある/ある:動1" and case_candidate == "ガ格":
            #     print("here")
            #     exit()
            return 1
        x.feature7()
        x.feature11()
        x.w2v_features()

        vec_keep = vec_all_dic[(cf_candidate, e_candidate, case_keep)][1][:]
        # ここはJUMAN、固有表現も生きてるなら入れるところ
        vec_keep[0] = x.feature_1
        vec_keep[1] = x.feature_2
        vec_keep[2] = x.feature_3
        vec_keep[6] = x.feature_7
        vec_keep[7] = x.feature_8
        vec_keep[8] = x.feature_9
        vec_keep[9] = x.feature_10
        vec_keep[10] = x.feature_11
        vec_keep[-2] = x.w2v_feature_mean_vec
        vec_keep[-1] = x.w2v_feature_all_mean_vec
    else:
        # case_flagが立っていない（一番目の格）なら格に関係ない素性も計算しないといけない
        try:
            x.set_info(cf=cf_candidate, case=case_candidate, e=e_candidate[
                1], row_num=int(y_row[0]), row_num_e=int(e_candidate[0]))
        except Exception as e:
            print(cf_candidate)
            print(case_candidate)
            print(e_candidate)
            raise e

        # (x.set_info)xに解析させる候補などを指定する
        e_list = search_e(
            x.x, x.basic_form_e, x.e_normalize)
        #　(search_e)今見ているeの基本形と同じ形の単語を前の文中に探す。e_listは行番号のリスト。文脈素性に使う
        e_position_list = []
        if len(e_list) > 1:
            for e_num in e_list:
                e_position_list.append(
                    search_phrase_list(x.phrase_list, int(e_num)))
                # search_phrase_listは(x.phrase_list)の中からこの行番号がどの間に挟まるか調べる
                # phrase_listは"*"で始まる行の行番号と係り先番号のタプル

        x.feature1()
        if x.feature_1 == float("-inf"):
            # if cf_candidate == "絞る/しぼる~テ形+ある/ある:動1" and case_candidate == "ガ格":
            #     print("here1")
            #     exit()
            return 1
        x.feature4()
        x.feature7()
        x.feature11()
        x.feature13()
        x.feature19()
        x.feature20()
        x.position_category(search_phrase_list(x.phrase_list, int(
            y_row[0])), search_phrase_list(x.phrase_list, int(e_candidate[0])), e_position_list)
        x.w2v_features()

        vec_keep = [x.feature_1, x.feature_2, x.feature_3, x.feature_4, x.feature_5, x.feature_6,
                    x.feature_7, x.feature_8, x.feature_9, x.feature_10, x.feature_11, x.feature_12,
                    x.feature_13, x.feature_14, x.feature_15, x.feature_16, x.feature_17, x.feature_18,
                    x.feature_19, x.feature_20, x.feature_21, x.feature_22, x.feature_23, x.feature_24,
                    x.feature_25, x.feature_26, x.feature_27, x.feature_28, x.feature_29, x.feature_30,
                    x.feature_31, x.feature_32, x.feature_33, feature_34, x.feature_hissu,
                    x.f1_inf_flag, x.f2_inf_flag, x.f3_inf_flag, x.f4_inf_flag, x.f5_inf_flag, x.f6_inf_flag,
                    x.f7_inf_flag, x.f8_inf_flag, x.f9_inf_flag, x.f11_inf_flag, x.f12_inf_flag,
                    x.Itopic, x.It_self, x.IP_self, x.IC_self, x.IGP_self, x.IGC_self, x.former_self, x.latter_self,
                    x.IB_self, x.IA_self, x.B1, x.B2, x.B3,
                    x.It_ga_ov, x.It_ga_om, x.It_wo_ov, x.It_wo_om, x.It_ni_ov, x.It_ni_om,
                    x.IP_ga_ov, x.IP_ga_om, x.IP_wo_ov, x.IP_wo_om, x.IP_ni_ov, x.IP_ni_om,
                    x.IC_ga_ov, x.IC_ga_om, x.IC_wo_ov, x.IC_wo_om, x.IC_ni_ov, x.IC_ni_om,
                    x.IGP_ga_ov, x.IGP_ga_om, x.IGP_wo_ov, x.IGP_wo_om, x.IGP_ni_ov, x.IGP_ni_om,
                    x.IGC_ga_ov, x.IGC_ga_om, x.IGC_wo_ov, x.IGC_wo_om, x.IGC_ni_ov, x.IGC_ni_om,
                    x.former_ga_ov, x.former_ga_om, x.former_wo_ov, x.former_wo_om, x.former_ni_ov, x.former_ni_om,
                    x.latter_ga_ov, x.latter_ga_om, x.latter_wo_ov, x.latter_wo_om, x.latter_ni_ov, x.latter_ni_om,
                    x.IA_ga_ov, x.IA_ga_om, x.IA_wo_ov, x.IA_wo_om, x.IA_ni_ov, x.IA_ni_om,
                    x.IB_ga_ov, x.IB_ga_om, x.IB_wo_ov, x.IB_wo_om, x.IB_ni_ov, x.IB_ni_om,
                    x.B1_ga_ov, x.B1_ga_om, x.B1_wo_ov, x.B1_wo_om, x.B1_ni_ov, x.B1_ni_om,
                    x.B2_ga_ov, x.B2_ga_om, x.B2_wo_ov, x.B2_wo_om, x.B2_ni_ov, x.B2_ni_om,
                    x.B3_ga_ov, x.B3_ga_om, x.B3_wo_ov, x.B3_wo_om, x.B3_ni_ov, x.B3_ni_om,
                    x.w2v_feature_mean_vec, x.w2v_feature_all_mean_vec]

    return vec_keep
    # print(vec_case_dic.keys())


def fetch_initial_e(x, sorted_e_list, result, relate_dict):
    # sorted_e_list回す前にコピーして係り受け関係に入ってるeを抜くべきだがあえて取り除いてない
    # あくまで初期値であり，例えばガ格とヲ格に同じeが入っていてもfetch_better_eで弾けば良いから
    n = 5
    predicate_set = set()
    all_mean_vec = None
    vec_e = None
    initial_dict = {}
    distance_dict = {}
    for cf_candidate in result:
        predicate_set.add(cf_candidate.split(":")[0])
    for predicate_candidate in predicate_set:
        initial_dict[predicate_candidate] = defaultdict(list)
    for case_init in case_list_init:
        for e_candidate in relate_dict[case_init]:
            for predicate_candidate in predicate_set:
                initial_dict[predicate_candidate][
                    case_init].append(e_candidate)
    kakari_flag = 0
    for predicate_candidate in predicate_set:
        distance_dict[predicate_candidate] = {}
        for case_init in case_list_init:
            kakari_flag = 0
            distance_dict[predicate_candidate][
                case_init] = (float("-inf"), None)
            if len(initial_dict[predicate_candidate][case_init]) > 0:
                # 係り受け関係のある格である場合
                kakari_flag = 1
            if predicate_candidate not in x.all_mean_vec_dic[case_init]:
                continue
            all_mean_vec = x.all_mean_vec_dic[
                case_init][predicate_candidate]
            # if case_init == "ヲ格":
            #     print(all_mean_vec)
            #     print(x.all_mean_vec_dic[case_init][predicate_candidate])
            #     print(predicate_candidate)
            #     exit()
            if kakari_flag == 0:
                for e_candidate in sorted_e_list:
                    if e_candidate[2] in x.vec_for_text_dic:
                        vec_e = x.vec_for_text_dic[e_candidate[2]]
                        if any(vec_e != np.zeros(500)) and any(all_mean_vec != np.zeros(500)):
                            distance_score = 1.0 - \
                                distance.cosine(vec_e, all_mean_vec)
                            # print(e_candidate)
                            # print(distance_score)
                            if distance_score > distance_dict[predicate_candidate][case_init][0]:
                                distance_dict[predicate_candidate][
                                    case_init] = (distance_score, e_candidate)
            else:
                for e_candidate in initial_dict[predicate_candidate][case_init]:
                    if e_candidate[2] in x.vec_for_text_dic:
                        vec_e = x.vec_for_text_dic[e_candidate[2]]
                        if any(vec_e != np.zeros(500)) and any(all_mean_vec != np.zeros(500)):
                            distance_score = 1.0 - \
                                distance.cosine(vec_e, all_mean_vec)
                            # print(e_candidate)
                            # print(distance_score)
                            if distance_score > distance_dict[predicate_candidate][case_init][0]:
                                distance_dict[predicate_candidate][
                                    case_init] = (distance_score, e_candidate)
    nlargest_list = []
    for predicate_candidate in predicate_set:
        case_score = 0.0
        case_count = 0
        for case_init in case_list_init:
            if distance_dict[predicate_candidate][case_init][0] != float("-inf"):
                case_score += distance_dict[predicate_candidate][case_init][0]
                case_count += 1
        if case_count != 0:
            heapq.heappush(nlargest_list, (case_score /
                                           case_count, predicate_candidate))
        # else:
        #     print(predicate_candidate)
    nlargest_list_after = [x[1] for x in heapq.nlargest(n, nlargest_list) if x[
        0] != float("-inf")]
    initial_dict_alt = {}
    for predicate_candidate in predicate_set:
        if predicate_candidate in nlargest_list_after:
            initial_dict_alt[predicate_candidate] = defaultdict(list)
    for predicate_candidate in predicate_set:
        for case_init in case_list_init:
            if predicate_candidate in nlargest_list_after:
                if distance_dict[predicate_candidate][case_init][0] != float("-inf"):
                    initial_dict_alt[predicate_candidate][case_init].append(
                        distance_dict[predicate_candidate][case_init][1])
    # print(list(x.vec_for_text_dic.keys()))
    # print(distance_dict)
    # print(initial_dict)
    # print(initial_dict_alt)
    # print(nlargest_list)
    # print(nlargest_list_after)
    # exit()
    # たぶん5best
    return initial_dict_alt


def fetch_better_e(x, sorted_e_list, better_cf_dict, relate_dict, vec_all_dic, y_row, feature34_dict, cf_p_dict):
    # sorted_e_list回す前にコピーして係り受け関係に入ってるeを抜いている
    n = 3
    better_e_dict = {}
    e_score_dict = {}
    e_score_dict_nlargest = {}
    sorted_e_list_alt = sorted_e_list.copy()
    vec_all_dic_alt = {}
    for predicate_candidate in better_cf_dict:
        better_e_dict[predicate_candidate] = defaultdict(list)
    for predicate_candidate in better_cf_dict:
        for case_init in case_list_init:
            for e_candidate in relate_dict[case_init]:
                better_e_dict[predicate_candidate][
                    case_init].append(e_candidate)
                if e_candidate in sorted_e_list_alt:
                    sorted_e_list_alt.remove(e_candidate)
    for predicate_candidate in better_cf_dict:
        e_score_dict[predicate_candidate] = {}
        e_score_dict_nlargest[predicate_candidate] = defaultdict(list)
        cf_candidate = better_cf_dict[predicate_candidate]
        for case_init in case_list_init:
            e_score_dict[predicate_candidate][
                case_init] = (float("-inf"), None)
            if len(better_e_dict[predicate_candidate][case_init]) > 0:
                continue
            for e_candidate in sorted_e_list_alt:
                if (cf_candidate, e_candidate, case_init) not in vec_all_dic:
                    vec_keep = feature_vec_maker(x, cf_candidate, case_init,
                                                 e_candidate, y_row, feature34_dict, vec_all_dic)
                    if vec_keep == 1:
                        continue
                    vec_score = score_returner(
                        vec_keep, case_init, e_candidate, cf_p_dict, cf_candidate, x)
                    vec_all_dic_alt[(cf_candidate, e_candidate, case_init)] = (
                        vec_score, vec_keep)
                else:
                    vec_score, _ = vec_all_dic[
                        (cf_candidate, e_candidate, case_init)]
                if vec_score > e_score_dict[predicate_candidate][case_init][0]:
                    e_score_dict[predicate_candidate][
                        case_init] = (vec_score, e_candidate)
                heapq.heappush(e_score_dict_nlargest[predicate_candidate][
                               case_init], (vec_score, e_candidate))
    for predicate_candidate in better_cf_dict:
        for case_init in case_list_init:
            if e_score_dict[predicate_candidate][case_init][0] != float("-inf"):
                better_e_dict[predicate_candidate][case_init].append(e_score_dict[
                    predicate_candidate][case_init][1])
    # print(list(x.vec_for_text_dic.keys()))
    # print(e_score_dict)
    # print(better_e_dict)
    # exit()
    # print(e_score_dict_nlargest['見る/みる+こと/こと+可能だ/かのうだ'])

    for vec_all_dic_alt_key in vec_all_dic_alt:
        # print(e_score_dict_nlargest[vec_all_dic_alt_key[
        #       0].split(":")[0]][vec_all_dic_alt_key[2]])
        # exit()
        case_nbest = [x[1] for x in heapq.nlargest(
            n, e_score_dict_nlargest[vec_all_dic_alt_key[0].split(":")[0]][vec_all_dic_alt_key[2]]) if x[0] != float("-inf")]
        # print(vec_all_dic_alt_key[2])
        # print(case_nbest)

        if vec_all_dic_alt_key[1] in case_nbest:
            vec_all_dic[vec_all_dic_alt_key] = vec_all_dic_alt[
                vec_all_dic_alt_key]
    # たぶん3best
    # exit()
    return better_e_dict


def fetch_better_cf(x, initial_e_dict, result, y_row, feature34_dict,  cf_p_dict, vec_all_dic):
    # (cf_candidate,e_candidate,case_init):(score,vec)
    vec_all_dic_max_keep = {}
    for predicate_candidate in initial_e_dict:
        for cf_candidate in result:
            if predicate_candidate == cf_candidate.split(":")[0]:
                for case_init in case_list_init:
                    # 係り受けがある時，e_candidateが複数の場合がある
                    for e_candidate in initial_e_dict[predicate_candidate][case_init]:
                        if (cf_candidate, e_candidate, case_init) not in vec_all_dic:
                            vec_keep = feature_vec_maker(x, cf_candidate, case_init,
                                                         e_candidate, y_row, feature34_dict, vec_all_dic)
                            if vec_keep == 1:
                                # if cf_candidate == "絞る/しぼる~テ形+ある/ある:動1" and case_init == "ガ格":
                                #     print("peko")
                                continue
                            vec_score = score_returner(
                                vec_keep, case_init, e_candidate, cf_p_dict, cf_candidate, x)
                            if vec_score == float("-inf"):
                                # if cf_candidate == "絞る/しぼる~テ形+ある/ある:動1" and case_init == "ガ格":
                                #     print("poko")
                                continue
                            vec_all_dic[(cf_candidate, e_candidate, case_init)] = (
                                vec_score, vec_keep)
                        else:
                            vec_score, _ = vec_all_dic[
                                (cf_candidate, e_candidate, case_init)]
                        if (cf_candidate, case_init) in vec_all_dic_max_keep and vec_score > vec_all_dic_max_keep[(cf_candidate, case_init)][0]:
                            vec_all_dic_max_keep[(cf_candidate, case_init)] = (
                                vec_score, e_candidate)
                        elif (cf_candidate, case_init) not in vec_all_dic_max_keep:
                            vec_all_dic_max_keep[(cf_candidate, case_init)] = (
                                vec_score, e_candidate)

    better_cf_score_dict = {}
    for predicate_candidate in initial_e_dict:
        better_cf_score_dict[predicate_candidate] = defaultdict(float)
        for cf_candidate in result:
            if predicate_candidate == cf_candidate.split(":")[0]:
                case_count = 0
                for case_init in case_list_init:
                    # ここでkey error．feature_vec_maker()が-infが出た時continueしてるため．
                    # cfがヲ格を取らない場合など．格を多く取るcfが有利なのは変なので取った格の数で割るべき
                    if (cf_candidate, case_init) in vec_all_dic_max_keep and vec_all_dic_max_keep[(cf_candidate, case_init)][0] != float("-inf"):
                        better_cf_score_dict[predicate_candidate][cf_candidate] += vec_all_dic_max_keep[
                            (cf_candidate, case_init)][0]
                        case_count += 1
                if case_count != 0:
                    better_cf_score_dict[predicate_candidate][
                        cf_candidate] /= case_count
                # else:
                    # print(cf_candidate)
                    # print(vec_all_dic_max_keep[(cf_candidate, "ガ格")][0])
                    # exit()
    # print(better_cf_score_dict)
    # exit()
    better_cf_dict = {}
    for predicate_candidate in initial_e_dict:
        better_cf_dict[predicate_candidate] = max(
            better_cf_score_dict[predicate_candidate].items(), key=lambda x: x[1])[0]

    return better_cf_dict


# vec_dic_constructor: vec_all_dicを階層構造のdictに組み替える
def vec_dic_constructor(x, vec_all_dic, removed_case):
    vec_all_dic_keys = list(vec_all_dic.keys())
    cf_candidates = [all_key[0] for all_key in vec_all_dic_keys]
    e_candidates = [all_key[1] for all_key in vec_all_dic_keys]
    vec_cf_dic = {}
    vec_e_dic = {}
    vec_case_dic = {}
    for cf_candidate in cf_candidates:
        vec_e_dic = {}
        for e_candidate in e_candidates:
            vec_case_dic = {}
            for case_init in case_list_init:
                if (cf_candidate, e_candidate, case_init) in vec_all_dic:
                    vec_case_dic[case_init + "_A"] = vec_all_dic[
                        (cf_candidate, e_candidate, case_init)][1][:]
            if len(vec_case_dic) > 0:
                vec_e_dic[e_candidate] = copy.deepcopy(vec_case_dic)
        vec_case_dic = {}
        # これ入れ忘れてツラかった
        for case_candidate in case_list:

            x.cf = cf_candidate
            x.case = case_candidate
            x.calcurate_cf_c_size()
            x.feature11()
            vec_case_dic[case_candidate +
                         "_NA"] = [x.feature_11x, x.feature_12, x.feature_hissu, x.f11x_inf_flag, x.f12_inf_flag]
        for case_candidate in removed_case:
            vec_case_dic[case_candidate + "_NA"] = [0] * 5
        vec_e_dic["None"] = copy.deepcopy(vec_case_dic)
        # ボツ:ここにPASScore計算してe候補を減らす処理
        # vec_new_e_dic = reduce_e_candidate(vec_e_dic)
        vec_cf_dic[cf_candidate] = copy.deepcopy(vec_e_dic)

    return vec_cf_dic


# fetch_cf_candidate: idとの照合を行ってcf_candidateを集める処理
def fetch_cf_candidate(id_head):
    # id_headの?の処理#左から取りましょう
    # cf候補の取得．「うっとり/うっとり+する/する~テ形+しまう/しまう:動4」なら
    # +~テ形に置き換えて、id_head_listが取り出されたあと、id_head_xなどではずす処理をすることで下記解決済み
    # 済「うっとり/うっとり+する/する~テ形+しまう/しまう:*」で調べて，次に「うっとり/うっとり+する/する~テ形*」で調べる
    # 済 DBのなかでは「努める/つとめる~テ形+いらっしゃる/いらっしゃる?務める/つとめる~テ形+いらっしゃる/いらっしゃる?勤める/つとめる~テ形+いらっしゃる/いらっしゃる」
    # 済 なので~テ形の処理がよくないことになっている「舞う/まう?待つ/まつ~テ形+いる/いる+られる/られる+ない/ない:%」が「舞う/まう+いる/いる+られる/られる+ない/ない」探してる
    # 済 DBのなかの「舞う/まう~テ形+いる/いる?待つ/まつ~テ形+いる/いる」が取れていない
    # 済 using_cf2_method.pyの239行目側で先に「+~テ形」にしてしまうのは怖い
    predict_sub_list_ori = id_head.replace('~テ形', '+~テ形').split("+")
    predict_sub_list_len = len(predict_sub_list_ori)
    head_reduce = 0
    # head_reduceは頭から削っていくカウント「御」とか削りたい
    # test
    # predict_sub_list = "有る/ある~テ形".replace(
    #     '~テ形', '+~テ形').split("+")
    # test
    result = []
    # sqlのlikeが死ぬほど遅いので，setに問い合わせる方式
    while(predict_sub_list_len - head_reduce > 0):
        predict_sub_list = predict_sub_list_ori[head_reduce:]
        predict_list = []
        predict_sub_head_list = []
        id_head_list = []

        #(0-0)?(0-1)+(1-0)+(2-0)?(2-1)
        # predict_sub_list=[(0-0)?(0-1),(1-0),(2-0)?(2-1)
        # print(predict_sub_list)
        predict_sub_head_list = predict_sub_list[0].split("?")
        # predict_sub_head_list=[(0-0),(0-1)]
        # predict_list=[(0-0),(1-0),(2-0)]
        for splited_predict in predict_sub_list:
            predict_list.append(splited_predict.split("?")[0])
        for head in predict_sub_head_list:
            test_list = predict_list[1:]
            test_list.insert(0, head)
            id_head_list.append(
                "+".join(test_list).replace('+~テ形', '~テ形'))
        # id_head_list=[(0-0)+(1-0)+(2-0),(0-1)+(1-0)+(2-0)]
        # 左端の表記ゆれだけ残して、以降右の表記ゆれは考慮しない構え.(2-1)がなくなっている
        id_head_in_set = id_list_search(id_head_list)
        # print(id_head_in_set)
        # exit()
        if len(id_head_in_set) > 0:
            result.extend(id_head_in_set)
            break
        id_head_in_set = id_list_search1(id_head_list)
        if len(id_head_in_set) > 0:
            result.extend(id_head_in_set)
            break
        if len(predict_sub_list) > 1:
            count = 0
            while 1:
                count += 1
                id_head_list_pre = []
                for id_head_x in id_head_list:
                    id_head_list_pre.append(
                        "+".join(id_head_x.replace('~テ形', '+~テ形').split("+")[:(-1 * count)]).replace('+~テ形', '~テ形'))
                id_head_in_set = id_list_search(id_head_list_pre)
                if len(id_head_in_set) > 0:
                    result.extend(id_head_in_set)
                    break
                if len(predict_sub_list) - count == 1:
                    break
        if len(result) > 0:
            break
        if len(predict_sub_list) > 1:
            count = 0
            while 1:
                count += 1
                id_head_list_pre = []
                for id_head_x in id_head_list:
                    id_head_list_pre.append(
                        "+".join(id_head_x.replace('~テ形', '+~テ形').split("+")[:(-1 * count)]).replace('+~テ形', '~テ形'))
                id_head_in_set = id_list_search1(id_head_list_pre)
                if len(id_head_in_set) > 0:
                    result.extend(id_head_in_set)
                    break
                if len(predict_sub_list) - count == 1:
                    break
        if len(result) > 0:
            break
        else:
            head_reduce += 1
    # test
    # print(result)
    # test
    if len(result) == 0:
        print("error can't find sql : " + id_head)
        return 1
    else:
        # 集めたresultのうち，最短の候補のみを集める
        result_len_dic = defaultdict(set)
        for result_x in result:
            result_len_dic[
                len(result_x.split("?")[0].replace('~テ形', '+~テ形').split("+"))].add(result_x)
        result = sorted(list(
            result_len_dic[min(list(result_len_dic.keys()))]))
        # result_len_dic1 = defaultdict(set)
        # for result_y in result:
        #     result_len_dic1[
        #         len(result_y.split("?"))].add(result_y)
        # # print(result_y.split("?"))
        # result = list(
        #     result_len_dic1[min(list(result_len_dic1.keys()))])
    # test
    # print(result)
    # exit()
    # test
    return result


def nearest_e_fetcher(sorted_e_list, y_row):
    threshold = 5
    new_e_list = []
    for e_candidate in sorted_e_list:
        new_e_list.append(
            (e_candidate, abs(int(e_candidate[0]) - int(y_row[0]))))
    return sorted([x[0] for x in sorted(new_e_list, key=lambda y:y[1]) if x[1] != 0][:threshold], key=lambda y: int(y[0]), reverse=True)


def cand_reduction(x, sorted_e_list, result, relate_dict, vec_all_dic, y_row, feature34_dict, cf_p_dict, removed_case):
    sorted_e_list_alt = nearest_e_fetcher(sorted_e_list, y_row)
    sorted_e_list_alt_keep = sorted_e_list_alt[:]
    case_list_alt = case_list_init[:]
    vec_all_dic_alt = {}
    # print(relate_dict)
    # print(removed_case)
    for cf_candidate in result:
        vec_all_dic_alt[cf_candidate] = {}
        for case_init in case_list_init:
            vec_all_dic_alt[cf_candidate][case_init] = {}
            if case_init in relate_dict:
                for e_candidate in relate_dict[case_init]:
                    if e_candidate in sorted_e_list_alt:
                        sorted_e_list_alt.remove(e_candidate)
                        if case_init in case_list_alt:
                            case_list_alt.remove(case_init)
                    if (cf_candidate, e_candidate, case_init) not in vec_all_dic:
                        vec_keep = feature_vec_maker(x, cf_candidate, case_init,
                                                     e_candidate, y_row, feature34_dict, vec_all_dic)
                        if vec_keep == 1:
                            continue
                        vec_score = score_returner(
                            vec_keep, case_init, e_candidate, cf_p_dict, cf_candidate, x)
                        vec_all_dic_alt[cf_candidate][case_init][
                            e_candidate] = vec_score
                        vec_all_dic[(cf_candidate, e_candidate, case_init)] = (
                            vec_score, vec_keep)
    # print(case_list_alt)
    for cf_candidate in result:
        for case_init in case_list_alt:
            for e_candidate in sorted_e_list_alt:
                if (cf_candidate, e_candidate, case_init) not in vec_all_dic:
                    vec_keep = feature_vec_maker(x, cf_candidate, case_init,
                                                 e_candidate, y_row, feature34_dict, vec_all_dic)
                    if vec_keep == 1:
                        # print(case_init)
                        continue
                    vec_score = score_returner(
                        vec_keep, case_init, e_candidate, cf_p_dict, cf_candidate, x)
                    vec_all_dic_alt[cf_candidate][case_init][
                        e_candidate] = vec_score
                    vec_all_dic[(cf_candidate, e_candidate, case_init)] = (
                        vec_score, vec_keep)

    for cf_candidate in result:
        for case_candidate in case_list_alt:
            x.cf = cf_candidate
            x.case = case_candidate
            x.calcurate_cf_c_size()
            x.feature11()
            vec_all_dic_alt[cf_candidate][case_candidate][
                "None"] = x.feature_11x

        for case_candidate in removed_case:
            vec_all_dic_alt[cf_candidate][case_candidate][
                "None"] = 0
    return sorted_e_list_alt_keep, vec_all_dic_alt


def cf_duplication(vec_all_dic_alt, sorted_e_list_alt, relate_dict, vec_all_dic):
    vec_cf_duplication = defaultdict(list)
    for cf_candidate in vec_all_dic_alt:
        e_candidate_dict = vec_all_dic_alt[cf_candidate]
        for ga_candidate in e_candidate_dict["ガ格"]:
            for wo_candidate in e_candidate_dict["ヲ格"]:
                for ni_candidate in e_candidate_dict["ニ格"]:
                    if "ガ格" in relate_dict and ga_candidate not in relate_dict["ガ格"]:
                        pass
                    elif "ヲ格" in relate_dict and wo_candidate not in relate_dict["ヲ格"]:
                        pass
                    elif "ニ格" in relate_dict and ni_candidate not in relate_dict["ニ格"]:
                        pass
                    elif (ga_candidate == ni_candidate and ga_candidate != "None") or (ni_candidate == wo_candidate and ni_candidate != "None") or (wo_candidate == ga_candidate and wo_candidate != "None"):
                        pass
                    score = e_candidate_dict["ガ格"][ga_candidate] + e_candidate_dict[
                        "ヲ格"][wo_candidate] + e_candidate_dict["ニ格"][ni_candidate]
                    # print("x")
                    vec_cf_duplication[
                        (ga_candidate, wo_candidate, ni_candidate)].append((cf_candidate, score))
    # print(vec_cf_duplication)
    max_vec_cf_duplication = {}
    for e_candidate_set in vec_cf_duplication:
        for cf_score in vec_cf_duplication[e_candidate_set]:
            if e_candidate_set in max_vec_cf_duplication:
                if max_vec_cf_duplication[e_candidate_set][1] < cf_score[1]:
                    max_vec_cf_duplication[
                        e_candidate_set] = cf_score
            else:
                max_vec_cf_duplication[
                    e_candidate_set] = cf_score
    # print(max_vec_cf_duplication)

    p_vec_all_dic_key = []
    for e_candidate_set in max_vec_cf_duplication:
        for case_init in case_list_init:
            if case_init == "ガ格":
                p_vec_all_dic_key.append((max_vec_cf_duplication[e_candidate_set][
                                         0], e_candidate_set[0], case_init))
            if case_init == "ヲ格":
                p_vec_all_dic_key.append((max_vec_cf_duplication[e_candidate_set][
                                         0], e_candidate_set[1], case_init))
            if case_init == "ニ格":
                p_vec_all_dic_key.append((max_vec_cf_duplication[e_candidate_set][
                                         0], e_candidate_set[2], case_init))
    vec_all_dic_removed = {}
    for x_vec_all_dic_key in vec_all_dic:
        if x_vec_all_dic_key in p_vec_all_dic_key:
            vec_all_dic_removed[x_vec_all_dic_key] = vec_all_dic[
                x_vec_all_dic_key]
    return vec_all_dic_removed


def cf_factory(text_name_of_knp, num, NTC_flag=0):
    genre_num = num // 8
    # ここ16で割るべき
    db_num = num % 16
    connector_for_main = sqlite3.connect(
        "cf_J_w2v_mini_alt" + str(db_num) + ".db", timeout=100.0)
    global KNP_info_dict_row_num
    cursor_for_main = connector_for_main.cursor()

    if NTC_flag == 0:
        with open("KNP_info_dict_row_num_fixed1.pickle", mode="rb") as f:
            KNP_info_dict_row_num = pickle.load(f)
    elif NTC_flag == 1:
        with open("KNP_info_dict_row_num_fixed1_NTC.pickle", mode="rb") as f:
            KNP_info_dict_row_num = pickle.load(f)
    # reconnect(num)
    pickle_names = read_pickles(genre_num, NTC_flag)
    exist_pickles = []
    if NTC_flag == 0:
        for y in pickle_names:
            if len(y.split("/")[1]) > 18:
                exist_pickles.append(y.split("/")[1][:18])
    elif NTC_flag == 1:
        for y in pickle_names:
            if len(y.split("/")[1]) > 11:
                exist_pickles.append(y.split("/")[1][:11])
    text_name_of_all = text_name_of_knp.split("/")[1][:-4]
    exist_row_num_set = set()
    text_fnames = []
    if text_name_of_all in exist_pickles:
        if NTC_flag == 0:
            text_fnames = glob.glob('resultx' + str(genre_num) +
                                    '/' + text_name_of_all + '*.pickle')
        elif NTC_flag == 1:
            text_fnames = glob.glob('resulty' + str(genre_num) +
                                    '/' + text_name_of_all + '*.pickle')
        for text_fname in text_fnames:
            if NTC_flag == 0:
                if len(text_fname.split("/")[1].split("_")) < 5:
                    cursor_for_main.close()
                    connector_for_main.close()
                    # test
                    # print(text_name_of_all)
                    # test
                    return 0
                else:
                    exist_row_num_set.add(
                        text_fname.split("/")[1].split("_")[4][:-7])
            elif NTC_flag == 1:
                if len(text_fname.split("/")[1]) < 11:
                    cursor_for_main.close()
                    connector_for_main.close()
                    # test
                    # print(text_name_of_all)
                    # test
                    return 0
                else:
                    exist_row_num_set.add(
                        text_fname.split("/")[1].split("_")[1][:-7])

    # f=codecs.open("resultx"+str(num)+"/"+text_name_of_all+".txt",'w','utf-8')
    # test
    # print(exist_row_num_set)
    # exit()
    # test
    # check
    check_dict = {}

    x = Extract_features(text_name=text_name_of_knp, num=num)
    prepare_for_text(x)
    named_entity_for_text_dic = defaultdict(set)
    vec_for_text_dic = {}

    x.set_text_dics(named_entity_for_text_dic, vec_for_text_dic)
    komejirusi_list_maker(x.x)

    a_yougen_row_num_dic, a_yougen_kome_dic = take_answer_yougen_row_num_dic(
        text_name_of_all)
    reject_aux_list = reject_aux(
        text_name_of_all, a_yougen_row_num_dic, a_yougen_kome_dic)
    # aux:受身，使役で反転してるもの．を弾いている
    # 『表示させる』が取れていない．米印部分で一致をみるべき．auxが入ってる米印の中にある動詞を弾く（今は米印の一個後ろの動詞行『表示』とタグの付いている行『さ』『せる』の一致を取っている）（00052_C_PM15_00178）
    # print(x.row_list_cf)
    # print(reject_aux_list)
    # exit()

    # id,eqのある行番号を入れると，それとeq関係にある行番号のリストを返す
    coreference_dic = coreference_dic_maker(a_yougen_row_num_dic)

    # x.set_vec_for_text_dic(vec_for_text_dic)
    # exo_list=["None","exo1","exo2","exog"]
    # exo_list = [(-1, "None")]  # 使ってない
    e_set = set()
    e_set_alt = set()
    sorted_e_list = list()
    # todo:疑似スコアに距離素性を追加．各素性を10000例くらいでSVM回して係数決めてしまう？

    vec_yougen_dic = {}
    vec_cf_dic = {}
    vec_e_dic = {}
    vec_case_dic = {}

    # 行番号を入れるとその行を含む※の行番号を返す
    komejirusi_devided_dic = {}

    for x_num, x_row in enumerate(x.row_list_cf):
        # x.row_list_cfは二重リストになっていて、文章単位で用言の位置を区切っている。
        # x.row_list_cf=[[('35', 'なる/なる+ない/ない')], [('46',
        # '失点/しってん+する/する~テ形+いる/いる'), ('55', '言う/いう'), ('86',
        # '壊す/こわす~テ形+いる/いる+ます/ます')], [('104', '生ける/いける?行ける/いける'), ('108',
        # '思う/おもう')], [('128', '苦しむ/くるしむ+ます/ます')], [('154',
        # '舞う/まう?待つ/まつ~テ形+いる/いる+られる/られる+ない/ない'), ('183', 'よる/よる~テ形'), ('209',
        # '思う/おもう+ます/ます')]]
        now_e_set = set(x.row_list_e[x_num])
        e_set |= now_e_set
        # 今から見る予定の用言と同じ文中にあるe(名詞)をe_setに追加している。
        if x_num > 3:
            e_set -= set(x.row_list_e[x_num - 4])
        # e_setは破壊しない
        e_set_alt = coreference_reducer(e_set, coreference_dic)

        # 近い文から見ている
        sorted_e_list = sorted(
            list(e_set_alt), key=lambda x: int(x[0]), reverse=True)

        # for now_e_candidate in now_e_set:
        #     if now_e_candidate[0] not in komejirusi_devided_dic:
        #         e_kome = komejirusi_devide(
        #             text_name_of_all, str(now_e_candidate[0]))
        #         komejirusi_devided_dic[now_e_candidate[0]] = e_kome

        # a_yougen_kome_dic[e_kome]
        # print(now_e_set)
        # print(coreference_dic)
        for y_num, y_row in enumerate(x_row):

            # if not y_row[0] == '8694':
            #     continue
            # check
            # if y_row[0] in exist_row_num_set:
            #     continue
            if y_row[0] in reject_aux_list:
                continue
            # print(y_row[0])
            if is_p_exist(text_name_of_all, y_row[0]) == -1:
                continue
            # else:
            #     print(exist_row_num_set)
            #     print(text_name_of_knp)
            #     print(y_row[0])
            # if not y_row[0] == '25':
            #     continue
            # print(sorted_e_list)
            # exit()

            case_list = ["ガ格", "ヲ格", "ニ格"]
            case_list_init = ["ガ格", "ヲ格", "ニ格"]
            removed_case = []

            id_head = y_row[1]
            print(id_head + ":" + str(num))
            result = fetch_cf_candidate(id_head)
            if result == 1:
                continue

            # 初期化
            cf_dic = {}
            headword_id_dic = {}
            mean_vec_dic = {}
            all_mean_vec_dic = {}
            for case_name in case_list_init:
                cf_dic[case_name] = defaultdict(list)
                headword_id_dic[case_name] = defaultdict(list)
                mean_vec_dic[case_name] = {}
                all_mean_vec_dic[case_name] = {}

            x.set_result(result, cf_dic, headword_id_dic,
                         mean_vec_dic, all_mean_vec_dic)

            # BCCWJ見て係り受け関係，ゼロ照応の答えを取ってくる．
            relate_dict = defaultdict(set)
            answer_dict = defaultdict(set)
            feature34_dict = defaultdict(int)
            relate_info_dict = None
            answer_info_dict = None
            relate_info_dict, answer_info_dict = relate_info_extracter(text_name_of_all, a_yougen_row_num_dic, a_yougen_kome_dic,
                                                                       komejirusi_devided_dic, y_row, "all")
            # print(a_yougen_row_num_dic["888"])
            # print(relate_info_dict)
            # print(answer_info_dict)
            # exit()
            for e_candidate in sorted_e_list:
                for case_init in case_list_init:
                    if relate_info_dict[case_init] != -1:
                        if len(set(relate_info_dict[case_init]) - set(['exo1', 'exo2', 'exog'])) > 0:
                            # ここ米印を元にidと同居する名詞を代わりに取ってくる処理
                            row_nums = list(
                                set(relate_info_dict[case_init]) - set(['exo1', 'exo2', 'exog']))
                            kome_returned = relate_answer_kome_reader(
                                text_name_of_all, e_candidate, row_nums, komejirusi_devided_dic)
                            if kome_returned != -1:
                                relate_dict[case_init].add(kome_returned)
                                if case_init in case_list:
                                    case_list.remove(case_init)
                        # ここ（answer_info_dictとくらべて）else文ないのはexo類と単語が並列して係り受け関係にあるとアノテーションされている時，
                        # 実際の係り受け関係にあるのは単語のみで，タグとしてのexo類はzero照応であるのでここではanswer側に突っ込んでる．．
                        if len(set(relate_info_dict[case_init]) & set(['exo1', 'exo2', 'exog'])) > 0:
                            # 本当はanswer_info_dict[case_init]がexoとかの時，strにしたかったけどsetになってるので．
                            answer_dict[case_init] |= (set(relate_info_dict[case_init]) & set([
                                'exo1', 'exo2', 'exog']))

                    else:
                        print("error:relate_info_dict")
                        print(text_name_of_all)
                        print(y_row)
                        print(case_init)
                    if answer_info_dict[case_init] != -1:
                        if len(set(answer_info_dict[case_init]) - set(['exo1', 'exo2', 'exog', "Not_fill"])) > 0:
                            # ここ米印を元にidと同居する名詞を代わりに取ってくる処理
                            row_nums = list(
                                set(answer_info_dict[case_init]) - set(['exo1', 'exo2', 'exog', "Not_fill"]))
                            kome_returned = relate_answer_kome_reader(
                                text_name_of_all, e_candidate, row_nums, komejirusi_devided_dic)
                            if kome_returned != -1:
                                answer_dict[case_init].add(kome_returned)
                        if len(set(answer_info_dict[case_init]) & set(['exo1', 'exo2', 'exog', "Not_fill"])) > 0:
                            # 本当はanswer_info_dict[case_init]がexoとかの時，strにしたかったけどsetになってるので．
                            answer_dict[case_init] |= (set(answer_info_dict[case_init]) & set([
                                'exo1', 'exo2', 'exog', "Not_fill"]))
                    else:
                        print("error:answer_info_dict")
                        print(text_name_of_all)
                        print(y_row)
                        print(case_init)
            for case_init in case_list_init:
                if len(relate_info_dict[case_init]) != 0:
                    feature34_dict[case_init] = 1
            # print(relate_info_dict)
            # print(relate_dict)
            # print(answer_info_dict)
            # print(answer_dict)
            # print(case_list)
            # print(relate_dict)
            # print(sorted_e_list)
            # exit()
            # 直接の係り受けで埋まっている格(removed_case)
            for case in case_list_init:
                if case not in case_list:
                    removed_case.append(case)
            # check

            check_dict[text_name_of_all + '_' + y_row[0]
                       ] = (len(sorted_e_list) + 1)**len(case_list) - len(case_list) * len(sorted_e_list)
            continue
            vec_cf_dic = {}
            # ここから本来のhill_climbing
            # cf_p_dict = {}
            # score_vec_dic = defaultdict(list)
            # # all_mean_vecにもとづきeを固定（->cfを動かす，固定->eを動かす，固定）以下候補数が増えなくなるまで繰り返す
            # initial_e_dict = fetch_initial_e(
            #     x, sorted_e_list, result, relate_dict)
            # # print(initial_e_dict)
            # vec_all_dic = {}
            # better_cf_dict = fetch_better_cf(
            #     x, initial_e_dict, result, y_row, feature34_dict,  cf_p_dict, vec_all_dic)
            # vec_all_dic_len_keep = 0
            # while vec_all_dic_len_keep != len(vec_all_dic):
            #     # print(len(vec_all_dic))
            #     # print(better_cf_dict)
            #     vec_all_dic_len_keep = len(vec_all_dic)
            #     better_e_dict = fetch_better_e(
            #         x, sorted_e_list, better_cf_dict, relate_dict, vec_all_dic, y_row, feature34_dict, cf_p_dict)
            #     # print(better_e_dict)
            #     better_cf_dict = fetch_better_cf(
            #         x, better_e_dict, result, y_row, feature34_dict, cf_p_dict, vec_all_dic)
            # # print(list(vec_all_dic.keys()))
            # ここまで本来のhill_climbing
            # ここからbaseline
            # cf_p_dict = {}
            # vec_all_dic = {}
            # sorted_e_list_alt, vec_all_dic_alt = cand_reduction(x, sorted_e_list, result, relate_dict,
            #                                                     vec_all_dic, y_row, feature34_dict, cf_p_dict, removed_case)
            # print(list(vec_all_dic.keys()))
            # print(vec_all_dic_alt)
            # vec_all_dic = cf_duplication(
            #     vec_all_dic_alt, sorted_e_list_alt, relate_dict, vec_all_dic)
            # ここまでbaseline
            # ここからuni_baseline
            cf_p_dict = {}
            vec_all_dic = {}
            sorted_e_list_alt, vec_all_dic_alt = cand_reduction(x, sorted_e_list, result, relate_dict,
                                                                vec_all_dic, y_row, feature34_dict, cf_p_dict, removed_case)
            print(list(vec_all_dic.keys()))
            print(vec_all_dic_alt)
            vec_all_dic = cf_duplication(
                vec_all_dic_alt, sorted_e_list_alt, relate_dict, vec_all_dic)
            # ここまでuni_baseline
            print(list(vec_all_dic.keys()))
            exit()

            vec_cf_dic = vec_dic_constructor(
                x, vec_all_dic, removed_case)

            vec_cf_dic["answer"] = answer_dict
            vec_cf_dic["answer_info_dict"] = answer_info_dict
            vec_cf_dic["relate_dict"] = relate_dict
            vec_cf_dic["relate_info_dict"] = relate_info_dict
            vec_cf_dic["removed_case"] = removed_case[:]
            vec_yougen_dic = {}
            vec_yougen_dic[y_row] = vec_cf_dic
            try:
                if NTC_flag == 0:
                    with open("resultx" + str(genre_num) + "/" + text_name_of_all + '_' + y_row[0] + '.pickle', mode='wb') as f:
                        pickle.dump(vec_yougen_dic, f)
                elif NTC_flag == 1:
                    with open("resulty" + str(genre_num) + "/" + text_name_of_all + '_' + y_row[0] + '.pickle', mode='wb') as f:
                        pickle.dump(vec_yougen_dic, f)
            except Exception as e:
                print(e)
                print("error_pickle:" + text_name_of_all +
                      ":" + y_row[0] + ":" + str(num))
                raise e
            # vec_new_cf_dic = reduce_cf_candidate(vec_cf_dic, x, cf_p_dict)
            vec_yougen_dic = {}

    # f.close()
    # try:
    #     with open("resultx" + str(num) + "/" + text_name_of_all + '.pickle', mode='wb') as f:
    #         pickle.dump(vec_yougen_dic, f)
    # except Exception as e:
    #     print(e)
    #     print("error_pickle:" + text_name_of_all + ":" + str(num))
    x.close_all()
    cursor_for_main.close()
    connector_for_main.close()
    with open("check_dicts/check_dict" + str(num) + '.pickle', mode='wb') as f:
        pickle.dump(check_dict, f)


if __name__ == "__main__":
    #cf_factory("knp_dismantled_original/9501ED-0000.knp", 0, 1)
    cf_factory("knp_dismantled_data/00052_C_PM15_00178.knp", 0)
    # x = Extract_features(
    #     text_name="knp_dismantled_data/00621_B_OC06_02189.knp")
    # prepare_for_text(x)

    # for x_row in x.row_list_cf:
    #     for y_row in x_row:
    #         pass

    # x.set_info(cf="苦しむ/くるしむ:動1", case="ニ格", e="理解", row_num=128, row_num_e=124)
    # e_list = search_e(x.x, x.basic_form_e, x.e_normalize)
    # e_position_list = []
    # if len(e_list) > 1:
    #     for e_num in e_list:
    #         e_position_list.append(
    #             search_phrase_list(x.phrase_list, int(e_num)))
    # print(x.basic_form_e)
    # print(x.modified_e)
    # print(x.juman_e)
    # # print(x.phrase_list)
    # # print(x.phrase_case_list)
    # # print(x.row_list_e)
    # print(x.row_list_cf)
    # # print(x.row_list_cf_case_list)

    # x.feature1()
    # x.feature2()
    # x.feature4()
    # x.feature5()
    # x.feature7()
    # x.feature11()
    # print(x.feature_12)
    # x.feature13()
    # x.feature20()
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
    # x.position_category(search_phrase_list(x.phrase_list,128),search_phrase_list(x.phrase_list,124),e_position_list)

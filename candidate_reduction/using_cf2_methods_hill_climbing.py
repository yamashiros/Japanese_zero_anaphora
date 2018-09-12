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


def joint_case_lists(case_dic0, case_dic1):
    # *の内側に二つ以上+行があった時、phrase_case_listが+の数だけ取れるので、先にある情報を優先しつつ、多い方を取っている
    # phrase_case_listの中身['ガ/U/-/-/-/-', 'ヲ/U/-/-/-/-','ニ/C/久保田/7/0/1']
    # set辞書にかえましたので純粋なOR統合になっています
    # phrase_case_listの中身defaultdict(<class 'set'>, {'ニ格': {'/C/久保田/7/0/1'},
    # 'ヲ格': {'/U/-/-/-/-'}, 'ガ格': {'/U/-/-/-/-'}})
    case_list = ["ガ格", "ヲ格", "ニ格"]
    new_case_dic = defaultdict(set)

    for case_name in case_list:
        new_case_dic[case_name] |= case_dic0[case_name]
        new_case_dic[case_name] |= case_dic1[case_name]
    return new_case_dic


def make_phrase_list(x):
    #phrase_listの中身('1', '1D')
    # phrase_case_listの中身['ガ/U/-/-/-/-', 'ヲ/U/-/-/-/-', 'ニ/C/久保田/7/0/1']
    # set辞書にかえました
    # phrase_case_listの中身defaultdict(<class 'set'>, {'ニ格': {'/C/久保田/7/0/1'},
    # 'ヲ格': {'/U/-/-/-/-'}, 'ガ格': {'/U/-/-/-/-'}})
    # どちらも二重リスト（三重リスト）になっていて、*の区切りと対応している
    phrase_list = []
    phrase_case_list = []
    phrase_list_sub = []
    phrase_case_list_sub = []
    # x=codecs.open(text_name,'r','utf-8').read()	#KNPを見てるとする
    rows = x.split("\n")
    case_list_flag = 0
    for x, row in enumerate(rows):
        words = row.split()
        if len(words) > 0:
            if words[0] == "*":
                # print(str(x))
                phrase_list_sub.append((str(x), words[1]))
                case_list_flag = 0
            if words[0] == "+":
                if case_list_flag == 0:
                    case_list_flag = 1
                    phrase_case_list_sub.append(search_case_list(words[2]))
                if case_list_flag == 1:
                    taken_phrase_case = phrase_case_list_sub[-1]
                    new_phrase_case = search_case_list(words[2])
                    phrase_case_list_sub = phrase_case_list_sub[0:-1]
                    phrase_case_list_sub.append(joint_case_lists(
                        taken_phrase_case, new_phrase_case))
            if words[0] == "EOS":
                phrase_list.append(phrase_list_sub)
                phrase_case_list.append(phrase_case_list_sub)
                phrase_list_sub = []
                phrase_case_list_sub = []
    return phrase_list, phrase_case_list


def make_case_list(row_list_cf, phrase_list, phrase_case_list):
    row_list_cf_case_list = []
    row_list_cf_case_list_sub = []
    for x_row in row_list_cf:
        for y_row in x_row:
            case_position = search_phrase_list(phrase_list, int(y_row[0]))
            row_list_cf_case_list_sub.append(
                phrase_case_list[case_position[0]][case_position[1]])
        row_list_cf_case_list.append(row_list_cf_case_list_sub)
        row_list_cf_case_list_sub = []
    return row_list_cf_case_list


def search_phrase_list(phrase_list, row_num):
    # print(row_num)
    last_x = 0
    last_y = 0
    for x_num, x in enumerate(phrase_list):
        for y_num, y in enumerate(x):
            if int(y[0]) > row_num:
                return (last_x, last_y)
            last_y = y_num
            last_x = x_num
    return (last_x, last_y)


def search_case_list(words):
    # set辞書にかえました
    # phrase_case_listの中身defaultdict(<class 'set'>, {'ニ格': {'/C/久保田/7/0/1'},
    # 'ヲ格': {'/U/-/-/-/-'}, 'ガ格': {'/U/-/-/-/-'}})
    case_dic = defaultdict(set)
    ga_list = []
    wo_list = []
    ni_list = []
    ga_num = -1
    ga_num_keep = 0
    wo_num = -1
    wo_num_keep = 0
    ni_num = -1
    ni_num_keep = 0
    while 1:
        ga_num_keep += ga_num + 1
        ga_num = words[ga_num_keep:].find("ガ/")
        if ga_num != -1:
            ga_num += 1
            ga_list = []
            for x in words[ga_num + ga_num_keep:]:
                if x not in [";", ">"]:
                    ga_list += x
                else:
                    break
            case_dic["ガ格"].add("".join(ga_list))
        else:
            break
    while 1:
        wo_num_keep += wo_num + 1
        wo_num = words[wo_num_keep:].find("ヲ/")
        if wo_num != -1:
            wo_num += 1
            wo_list = []
            for x in words[wo_num + wo_num_keep:]:
                if x not in [";", ">"]:
                    wo_list += x
                else:
                    break
            case_dic["ヲ格"].add("".join(wo_list))
        else:
            break
    while 1:
        ni_num_keep += ni_num + 1
        ni_num = words[ni_num_keep:].find("ニ/")
        if ni_num != -1:
            ni_num += 1
            ni_list = []
            for x in words[ni_num + ni_num_keep:]:
                if x not in [";", ">"]:
                    ni_list += x
                else:
                    break
            case_dic["ニ格"].add("".join(ni_list))
        else:
            break
    return case_dic


def search_e(x, e, e_normalize):
    row_list = []
    rows = x.split("\n")
    for y, row in enumerate(rows):
        words = row.split()
        if len(words) > 0:
            if not words[0] in ["#", "+", "*", "EOS"]:

                text = ""
                text_candidate = ""
                if "<正規化代表表記:" in words:
                    flag = 0
                    text_list_new1 = []
                    for i in words[words.find("<正規化代表表記:") + 9:]:
                        if i == ">":
                            flag = 1
                        if flag == 0:
                            text_list_new1.append(i)
                        elif flag == 1:
                            pass
                            text = "".join(text_list_new1)
                    fragments = text.split("+")
                    for fragment in fragments:
                        text_candidate += fragment.split(
                            "?")[0].split("/")[0]
                    # print(text_candidate)
                if words[2] == e or text_candidate == e_normalize:
                    row_list.append(str(y))
    return row_list


def search_e_candidates(x):
    row_list_e = []
    row_list_cf = []
    row_list_e_sub = []
    row_list_cf_sub = []
    cf_text_list = []
    rows = x.split("\n")
    pre_words = ""

    pre_phrase = ""
    pre_phrase_num = -1
    pre_phrase_num_record = -1
    yougen_num = 0
    tegata_flag = 0

    cf_text = ""
    for y, row in enumerate(rows):
        words = row.split()
        if len(words) > 0:
            if words[0] == "*":
                pre_words = words[2]
            if words[0] == "+":
                pre_phrase = words[2]
                pre_phrase_num = y
                if cf_text_list != []:
                    cf_text = "+".join(cf_text_list)
                    row_list_cf_sub.append((str(yougen_num), cf_text))
                    cf_text_list = []
            elif not words[0] in ["#", "+", "*", "EOS"]:
                if words[3] == "名詞" or words[5] == "名詞形態指示詞":
                    noun_type = None
                    if words[3] == "名詞":
                        noun_type = "名詞"
                    elif words[5] == "名詞形態指示詞":
                        noun_type = "名詞形態指示詞"
                    text = ""
                    text_candidate = ""
                    text1 = ""
                    text2 = ""
                    word = pre_words
                    if "<正規化代表表記:" in word:
                        flag = 0
                        text_list_new1 = []
                        for i in word[word.find("<正規化代表表記:") + 9:]:
                            # print(i)
                            if i == ">":
                                flag = 1
                            if flag == 0:
                                text_list_new1.append(i)
                            elif flag == 1:
                                pass
                        text = "".join(text_list_new1)
                    fragments = text.split("+")
                    for fragment in fragments:
                        text_candidate += fragment.split("?")[0].split("/")[0]
                    if "<係:" in word:
                        flag = 0
                        text_list_new1 = []
                        for i in word[word.find("<係:") + 3:]:
                            # print(i)
                            if i == ">":
                                flag = 1
                            if flag == 0:
                                text_list_new1.append(i)
                            elif flag == 1:
                                pass
                        text2 = "".join(text_list_new1)
                    # 逆だったので直した<正規化代表表記:後/あと?後/のち>
                    # 正規化代表表記(<正規化代表表記:二/に+線/せん>)は二線が二と線に分けられるのを二線と二線の二回数えている
                    # これをそのまま取れば（/とか処理せずに）格フレームから取った<数詞>+個なんかも処理せずにイケる気はする．いや逆にこれがあるから一度分けて組み合わせてをやるのか？
                    # 四つめにtextで入れてみたが、使ってない。一応
                    # 五つめに表層格で入れてみる
                    # 六つめに名詞タイプを入れてみる（共参照関係内部で優劣をつけるため）

                    if "<時間>" in pre_phrase:
                        row_list_e_sub.append(
                            (str(y), "<時間>", words[2], text, text2, noun_type))
                    elif "<補文>" in pre_phrase:
                        row_list_e_sub.append(
                            (str(y), "<補文>", words[2], text, text2, noun_type))
                    elif "<数量>" in pre_phrase:
                        word = pre_words
                        if "<カウンタ:" in word:
                            flag = 0
                            text_list_new1 = []
                            for i in word[word.find("<カウンタ:") + 6:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new1.append(i)
                                elif flag == 1:
                                    pass
                            text1 = "".join(text_list_new1)
                            row_list_e_sub.append(
                                (str(y), "<数量>" + text1, words[2], text, text2, noun_type))
                        else:
                            row_list_e_sub.append(
                                (str(y), "<数量>", words[2], text, text2, noun_type))
                    elif "<SM-主体>" in pre_phrase:
                        row_list_e_sub.append(
                            (str(y), "<主体準>", words[2], text, text2, noun_type))
                        # 二つ足してみる #あまり意味ない気がする．e_normalize==textでもSQL引いてるから．
                        row_list_e_sub.append(
                            (str(y), text_candidate, words[2], text, text2, noun_type))
                    else:
                        row_list_e_sub.append(
                            (str(y), text_candidate, words[2], text, text2, noun_type))
                tegata_flag = 0
                # or words[3]=="形容詞" or words[3]=="判定詞":
                if pre_phrase.find("<用言:動>") != -1:
                    text2 = ""
                    if pre_phrase_num_record != pre_phrase_num:
                        # row_list_cf_sub.append((str(y),"".join(words[0:1])+"/"+"".join(words[1:2])+"/"+"".join(words[2:3])))
                        yougen_num = y
                        pre_phrase_num_record = pre_phrase_num
                    for word in words:
                        if "<正規化代表表記:" in word:
                            flag = 0
                            text_list_new1 = []
                            for i in word[word.find("<正規化代表表記:") + 9:]:
                                # print(i)
                                if i == ">":
                                    flag = 1
                                if flag == 0:
                                    text_list_new1.append(i)
                                elif flag == 1:
                                    pass
                            text2 = "".join(text_list_new1)
                        if "タ系連用テ形" == word:
                            tegata_flag = 1
                        # if "タ系連用テ形複合辞" in word:
                        #     tegata_flag = 0

                    if text2 != "":
                        if tegata_flag == 1:
                            text2 += "~テ形"
                        cf_text_list.append(text2)

            if words[0] == "EOS":
                if cf_text_list != []:
                    cf_text = "+".join(cf_text_list)
                    row_list_cf_sub.append((str(yougen_num), cf_text))
                    cf_text_list = []
                row_list_e.append(row_list_e_sub)
                row_list_cf.append(row_list_cf_sub)
                row_list_e_sub = []
                row_list_cf_sub = []
    return row_list_e, row_list_cf


def prepare_for_text(x):
    e_candidates_lists = search_e_candidates(x.x)
    x.set_lists(make_phrase_list(x.x))
    row_list_cf_case_list = make_case_list(
        e_candidates_lists[1], x.phrase_list, x.phrase_case_list)
    x.set_lists1(e_candidates_lists, row_list_cf_case_list)


def juman(text, num):

    cmd = "juman < gomi/test_juman" + \
        str(num) + ".txt > gomi/test_juman" + str(num) + ".jmn"
    output = subprocess.check_output(cmd, shell=True)
    f = codecs.open("gomi/test_juman" + str(num) + ".jmn", 'r', 'utf-8')
    x = f.read()
    f.close()
    x_list = []
    z_list = []
    # print(x)
    for word in x.split():
        if "カテゴリ:" in word:
            x_list.append(word[5:].replace('\"', ""))
        if "ドメイン:" in word:
            z_list.append(word[5:].replace('\"', ""))
    return (":".join(x_list), ":".join(z_list))


def named_entity(text, num):

    f = codecs.open("gomi/test_juman" + str(num) + ".txt", 'w', 'utf-8')
    f.write(text)
    f.close()
    cmd = "cabocha -f1 -n1 < gomi/test_juman" + \
        str(num) + ".txt > gomi/test_cabocha" + str(num) + ".cabocha"
    output = subprocess.check_output(cmd, shell=True)
    f = codecs.open("gomi/test_cabocha" + str(num) + ".cabocha", 'r', 'utf-8')
    y = f.read()
    f.close()
    y_list = []
    for row in y.split("\n"):
        words = row.split()
        if len(words) > 0:
            if not words[0] in ["#", "+", "*", "EOS"]:
                y_list.extend(words[-1:])

    return ":".join(y_list)


# if __name__ == '__main__':
#     x = search_case_list(
#         "ガ/N/人材/14/0/19;ガ/N/教育/16/0/19;ヲ/C/手法/10/0/19;ヲ/C/教育/12/0/19")
#     print(x)

# encoding:utf-8
import numpy as np
import math
import jieba
import re
from gensim.models.word2vec import Word2Vec
import os

##################
# consider all factor
###################

Train = True
Test = False
Lambda = 0.001
Gamma = 0.001
alpha = 2
K = 10
aaa = 1
bbb = 1
ccc = 1
ddd = 1
train_path = "../../data/weibo/newtrain.txt"
test_path = "../../data/weibo/newtest.txt"
word2vec_model_path = '../../data/model/weibo_train.model'
result_dir = 'result' + str (Lambda) + '_' + str (Gamma) + '_' + str (alpha) + '_' + str (aaa) + '_' + str (
    bbb) + '_' + str (ccc)

model = Word2Vec.load (word2vec_model_path)  # load word2vec模型
print ('loaded word2vec model')

if not (os.path.exists (result_dir)):
    os.makedirs (result_dir)


def buildWordVector(text, size):
    vec = np.zeros (size).reshape ((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word].reshape ((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def Load(path_data):  # user_ids, gender_ids, time_ids, emoji_ids
    time_ids = {}
    users_ids = {}
    gender_ids = {}
    emoji_ids = {}
    print ('making dictionary')
    file = open (path_data, 'r', encoding='utf-8').read ()

    def make_dir(dic, key):
        num = 0
        for line in file.split ('\n')[:-1]:
            if line.split (' ')[key] not in dic:
                dic.setdefault (line.split (' ')[key], num)
                num = num + 1
            else:
                continue
        return dic

    users_ids = make_dir (users_ids, 0)
    gender_ids = make_dir (gender_ids, 1)
    time_ids = make_dir (time_ids, 2)
    p1 = r"(\[[a-z_A-Z0-9]*\])"
    num = 0
    for line in file.split ('\n')[:-1]:
        tags = re.findall (p1, line)
        for tag in tags:
            if tag not in emoji_ids:
                emoji_ids.setdefault (tag, num)
                num += 1
            else:
                continue

    return users_ids, gender_ids, time_ids, emoji_ids


def symbiosis(data_path, emoji_ids):
    emoji_symbiosis = []
    file = open (data_path, 'r', encoding='utf-8')
    for i in range (len (emoji_ids)):
        emoji_symbiosis.append (np.zeros (len (emoji_ids)))
    for line in file.read ().split ('\n')[:-1]:
        p1 = r"(\[[a-z_A-Z0-9]*\])"
        emojis = re.findall (p1, line)
        if emojis != []:
            l = len (emojis)
            for emo1 in emojis:
                for emo2 in emojis:
                    if emo1 != emo2:
                        emoji_symbiosis[emoji_ids[emo1]][emoji_ids[emo2]] += 1
                        emoji_symbiosis[emoji_ids[emo2]][emoji_ids[emo1]] += 1
    for i in range (len (emoji_symbiosis)):
        if np.sum (emoji_symbiosis[i]) == 0:
            emoji_symbiosis[i] = np.zeros (len (emoji_ids))
        else:
            emoji_symbiosis[i] = emoji_symbiosis[i] / np.sum (emoji_symbiosis[i])
    return emoji_symbiosis


def Score_function(user_id_u, gender_id_g, Time_id_t, emoji_id_c, User_latent_vectors, Gender_latent_vectors,
                   Time_latent_vectors, Context_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                   Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    f = aaa * np.dot (Emoji_latent_vectors_1[emoji_id_c], User_latent_vectors[user_id_u]) + bbb * np.dot (
        Emoji_latent_vectors_2[emoji_id_c], Gender_latent_vectors[gender_id_g]) + ccc * np.dot (
        Emoji_latent_vectors_3[emoji_id_c], Time_latent_vectors[Time_id_t]) + ddd * np.dot (
        Emoji_latent_vectors_4[emoji_id_c], Context_vectors)
    return f


def Sigma(user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors, Gender_latent_vectors,
          Time_latent_vectors, Context_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
          Emoji_latent_vectors_4):
    x = Score_function (user_id_u, gender_id_g, time_id_t, emoji_id_e_p, User_latent_vectors, Gender_latent_vectors,
                        Time_latent_vectors, Context_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                        Emoji_latent_vectors_3, Emoji_latent_vectors_4) - Score_function (user_id_u, gender_id_g,
                                                                                          time_id_t, emoji_id_e_n,
                                                                                          User_latent_vectors,
                                                                                          Gender_latent_vectors,
                                                                                          Time_latent_vectors,
                                                                                          Context_vectors,
                                                                                          Emoji_latent_vectors_1,
                                                                                          Emoji_latent_vectors_2,
                                                                                          Emoji_latent_vectors_3,
                                                                                          Emoji_latent_vectors_4)
    sigma = 1 / (1 + math.exp (-x))
    return sigma


def Derivative_U_u(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                   Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                   Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    subtraction = np.array (Emoji_latent_vectors_1[emoji_id_e_p]) - np.array (Emoji_latent_vectors_1[emoji_id_e_n])
    derivative = - delta * subtraction + Lambda * np.array (User_latent_vectors[user_id_u])
    return derivative


def Derivative_G_g(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                   Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                   Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    subtraction = np.array (Emoji_latent_vectors_2[emoji_id_e_p]) - np.array (Emoji_latent_vectors_2[emoji_id_e_n])
    derivative = - delta * subtraction + Lambda * np.array (Gender_latent_vectors[gender_id_g])
    return derivative


def Derivative_T_t(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                   Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                   Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    subtraction = np.array (Emoji_latent_vectors_3[emoji_id_e_p]) - np.array (Emoji_latent_vectors_3[emoji_id_e_n])
    derivative = - delta * subtraction + Lambda * np.array (Time_latent_vectors[time_id_t])
    return derivative


def Derivative_G_g(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                   Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                   Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    subtraction = np.array (Emoji_latent_vectors_2[emoji_id_e_p]) - np.array (Emoji_latent_vectors_2[emoji_id_e_n])
    derivative = - delta * subtraction + Lambda * np.array (Gender_latent_vectors[gender_id_g])
    return derivative


def Derivative_Emoji1_ep(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = - delta * np.array (User_latent_vectors[user_id_u]) + Lambda * np.array (
        Emoji_latent_vectors_1[emoji_id_e_p])
    return derivative


def Derivative_Emoji2_ep(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = - delta * np.array (Gender_latent_vectors[gender_id_g]) + Lambda * np.array (
        Emoji_latent_vectors_2[emoji_id_e_p])
    return derivative


def Derivative_Emoji3_ep(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = - delta * np.array (Time_latent_vectors[time_id_t]) + Lambda * np.array (
        Emoji_latent_vectors_3[emoji_id_e_p])
    return derivative


def Derivative_Emoji4_ep(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Context_vector, Emoji_latent_vectors_1,
                         Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = - delta * np.array (Context_vector) + Lambda * np.array (Emoji_latent_vectors_4[emoji_id_e_p])
    return derivative


def Derivative_Emoji1_en(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = delta * np.array (User_latent_vectors[user_id_u]) + Lambda * np.array (
        Emoji_latent_vectors_1[emoji_id_e_n])
    return derivative


def Derivative_Emoji2_en(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = delta * np.array (Gender_latent_vectors[gender_id_g]) + Lambda * np.array (
        Emoji_latent_vectors_2[emoji_id_e_n])
    return derivative


def Derivative_Emoji3_en(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                         Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = delta * np.array (Time_latent_vectors[time_id_t]) + Lambda * np.array (
        Emoji_latent_vectors_3[emoji_id_e_n])
    return derivative


def Derivative_Emoji4_en(delta, user_id_u, gender_id_g, time_id_t, emoji_id_e_p, emoji_id_e_n, User_latent_vectors,
                         Gender_latent_vectors, Time_latent_vectors, Context_vector, Emoji_latent_vectors_1,
                         Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    derivative = delta * np.array (Context_vector) + Lambda * np.array (Emoji_latent_vectors_4[emoji_id_e_n])
    return derivative


def Iteration(delta, u, g, t, e_p, e_n, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors, Context_vector,
              Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    Derivative_User_latent_vectors_U_u = Derivative_U_u (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                         Gender_latent_vectors, Time_latent_vectors,
                                                         Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                         Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Gender_latent_vectors_G_g = Derivative_G_g (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                           Gender_latent_vectors, Time_latent_vectors,
                                                           Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                           Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Time_latent_vectors_T_t = Derivative_T_t (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                         Gender_latent_vectors, Time_latent_vectors,
                                                         Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                         Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_1_ep = Derivative_Emoji1_ep (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_2_ep = Derivative_Emoji2_ep (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_3_ep = Derivative_Emoji3_ep (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_4_ep = Derivative_Emoji4_ep (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Context_vector, Emoji_latent_vectors_1,
                                                                 Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                                                                 Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_1_en = Derivative_Emoji1_en (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_2_en = Derivative_Emoji2_en (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_3_en = Derivative_Emoji3_en (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                                                 Emoji_latent_vectors_3, Emoji_latent_vectors_4)

    Derivative_Emoji_latent_vectors_4_en = Derivative_Emoji4_en (delta, u, g, t, e_p, e_n, User_latent_vectors,
                                                                 Gender_latent_vectors, Time_latent_vectors,
                                                                 Context_vector, Emoji_latent_vectors_1,
                                                                 Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                                                                 Emoji_latent_vectors_4)

    return Derivative_User_latent_vectors_U_u, Derivative_Gender_latent_vectors_G_g, Derivative_Time_latent_vectors_T_t, Derivative_Emoji_latent_vectors_1_ep, Derivative_Emoji_latent_vectors_2_ep, Derivative_Emoji_latent_vectors_3_ep, Derivative_Emoji_latent_vectors_4_ep, Derivative_Emoji_latent_vectors_1_en, Derivative_Emoji_latent_vectors_2_en, Derivative_Emoji_latent_vectors_3_en, Derivative_Emoji_latent_vectors_4_en


def Training_new(users_ids, gender_ids, time_ids, emoji_ids, User_latent_vectors, Gender_latent_vectors,
                 Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                 Emoji_latent_vectors_4, train_data):
    file = open (train_data, 'r', encoding='utf-8')
    for line in file.read ().split ('\n')[:-1]:
        u = users_ids[line.split (' ')[0]]
        g = gender_ids[line.split (' ')[1]]
        t = time_ids[line.split (' ')[2]]
        p1 = r"(\[[a-z_A-Z0-9]*\])"
        emojis = re.findall (p1, line)
        # line=line.encode('utf-8','strict')
        pattern = re.compile ('[\u4e00-\u9fa5]')
        words = pattern.findall (line)
        str_words = ''.join (words)
        seg_list = jieba.cut (str_words, cut_all=False)
        Context_vector = buildWordVector (seg_list, K)[0]
        if emojis != []:
            for emoji in emojis:
                e_p = emoji_ids[emoji]
                for emo in emoji_ids:
                    if emo not in emojis:
                        e_n = emoji_ids[emo]
                        delta = 1 - Sigma (u, g, t, e_p, e_n, User_latent_vectors, Gender_latent_vectors,
                                           Time_latent_vectors, Context_vector, Emoji_latent_vectors_1,
                                           Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4)
                        Derivative_User_latent_vectors_U_u, Derivative_Gender_latent_vectors_G_g, Derivative_Time_latent_vectors_T_t, Derivative_Emoji_latent_vectors_1_ep, Derivative_Emoji_latent_vectors_2_ep, Derivative_Emoji_latent_vectors_3_ep, Derivative_Emoji_latent_vectors_4_ep, Derivative_Emoji_latent_vectors_1_en, Derivative_Emoji_latent_vectors_2_en, Derivative_Emoji_latent_vectors_3_en, Derivative_Emoji_latent_vectors_4_en = Iteration (
                            delta, u, g, t, e_p, e_n, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors,
                            Context_vector, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                            Emoji_latent_vectors_4)
                        User_latent_vectors[u] = [max (0, x) for x in
                                                  User_latent_vectors[u] - Gamma * Derivative_User_latent_vectors_U_u]
                        Time_latent_vectors[t] = [max (0, x) for x in
                                                  Time_latent_vectors[t] - Gamma * Derivative_Time_latent_vectors_T_t]
                        Gender_latent_vectors[g] = [max (0, x) for x in Gender_latent_vectors[
                            g] - Gamma * Derivative_Gender_latent_vectors_G_g]
                        Emoji_latent_vectors_1[e_p] = [max (0, x) for x in Emoji_latent_vectors_1[
                            e_p] - Gamma * Derivative_Emoji_latent_vectors_1_ep]
                        Emoji_latent_vectors_2[e_p] = [max (0, x) for x in Emoji_latent_vectors_2[
                            e_p] - Gamma * Derivative_Emoji_latent_vectors_2_ep]
                        Emoji_latent_vectors_3[e_p] = [max (0, x) for x in Emoji_latent_vectors_3[
                            e_p] - Gamma * Derivative_Emoji_latent_vectors_3_ep]
                        Emoji_latent_vectors_4[e_p] = [max (0, x) for x in Emoji_latent_vectors_4[
                            e_p] - Gamma * Derivative_Emoji_latent_vectors_4_ep]
                        Emoji_latent_vectors_1[e_n] = [max (0, x) for x in Emoji_latent_vectors_1[
                            e_n] - Gamma * Derivative_Emoji_latent_vectors_1_en]
                        Emoji_latent_vectors_2[e_n] = [max (0, x) for x in Emoji_latent_vectors_2[
                            e_n] - Gamma * Derivative_Emoji_latent_vectors_2_en]
                        Emoji_latent_vectors_3[e_n] = [max (0, x) for x in Emoji_latent_vectors_3[
                            e_n] - Gamma * Derivative_Emoji_latent_vectors_3_en]
                        Emoji_latent_vectors_4[e_n] = [max (0, x) for x in Emoji_latent_vectors_4[
                            e_n] - Gamma * Derivative_Emoji_latent_vectors_4_en]

    return User_latent_vectors, Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4


def Training(users_ids, gender_ids, time_ids, emoji_ids, User_latent_vectors, Gender_latent_vectors,
             Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
             Emoji_latent_vectors_4, train_data):
    file = open (train_data, 'r', encoding='utf-8')
    Loss = 0.0
    for line in file.read ().split ('\n')[:-1]:
        u = users_ids[line.split (' ')[0]]
        g = gender_ids[line.split (' ')[1]]
        t = time_ids[line.split (' ')[2]]
        p1 = r"(\[[a-z_A-Z0-9]*\])"
        emojis = re.findall (p1, line)
        # line = line.decode('utf-8', 'strict')  #python2
        pattern = re.compile (r'[\u4e00-\u9fa5]')
        words = pattern.findall (line)
        str_words = ''.join (words)
        seg_list = jieba.cut (str_words, cut_all=False)
        Context_vector = buildWordVector (seg_list, K)[0]
        for emoji in emojis:
            e_p = emoji_ids[emoji]
            for i in range (5):
                k = np.random.randint (len (emoji_ids))
                if k != e_p:
                    e_n = k
                    sigma = Sigma (u, g, t, e_p, e_n, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors,
                                   Context_vector, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                                   Emoji_latent_vectors_3, Emoji_latent_vectors_4)
                    delta = 1 - sigma
                    Loss = Loss - math.log (sigma)
                    Derivative_User_latent_vectors_U_u, Derivative_Gender_latent_vectors_G_g, Derivative_Time_latent_vectors_T_t, Derivative_Emoji_latent_vectors_1_ep, Derivative_Emoji_latent_vectors_2_ep, Derivative_Emoji_latent_vectors_3_ep, Derivative_Emoji_latent_vectors_4_ep, Derivative_Emoji_latent_vectors_1_en, Derivative_Emoji_latent_vectors_2_en, Derivative_Emoji_latent_vectors_3_en, Derivative_Emoji_latent_vectors_4_en = Iteration (
                        delta, u, g, t, e_p, e_n, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors,
                        Context_vector, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                        Emoji_latent_vectors_4)

                    User_latent_vectors[u] = [x for x in
                                              User_latent_vectors[u] - Gamma * Derivative_User_latent_vectors_U_u]
                    Time_latent_vectors[t] = [x for x in
                                              Time_latent_vectors[t] - Gamma * Derivative_Time_latent_vectors_T_t]
                    Gender_latent_vectors[g] = [x for x in
                                                Gender_latent_vectors[g] - Gamma * Derivative_Gender_latent_vectors_G_g]
                    Emoji_latent_vectors_1[e_p] = [x for x in Emoji_latent_vectors_1[
                        e_p] - Gamma * Derivative_Emoji_latent_vectors_1_ep]
                    Emoji_latent_vectors_2[e_p] = [x for x in Emoji_latent_vectors_2[
                        e_p] - Gamma * Derivative_Emoji_latent_vectors_2_ep]
                    Emoji_latent_vectors_3[e_p] = [x for x in Emoji_latent_vectors_3[
                        e_p] - Gamma * Derivative_Emoji_latent_vectors_3_ep]
                    Emoji_latent_vectors_4[e_p] = [x for x in Emoji_latent_vectors_4[
                        e_p] - Gamma * Derivative_Emoji_latent_vectors_4_ep]
                    Emoji_latent_vectors_1[e_n] = [x for x in Emoji_latent_vectors_1[
                        e_n] - Gamma * Derivative_Emoji_latent_vectors_1_en]
                    Emoji_latent_vectors_2[e_n] = [x for x in Emoji_latent_vectors_2[
                        e_n] - Gamma * Derivative_Emoji_latent_vectors_2_en]
                    Emoji_latent_vectors_3[e_n] = [x for x in Emoji_latent_vectors_3[
                        e_n] - Gamma * Derivative_Emoji_latent_vectors_3_en]
                    Emoji_latent_vectors_4[e_n] = [x for x in Emoji_latent_vectors_4[
                        e_n] - Gamma * Derivative_Emoji_latent_vectors_4_en]
    # matrix=np.concatenate((Emoji_latent_vectors_1,Emoji_latent_vectors_2,Emoji_latent_vectors_3,Emoji_latent_vectors_4,User_latent_vectors,Time_latent_vectors,Gender_latent_vectors),axis=0)
    # reg_loss=np.linalg.norm(matrix,ord='fro')

    # print("loss1:",Loss)
    # print("reg_loss",reg_loss*Lambda)
    return User_latent_vectors, Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4


def Precision(ranks):
    True_5 = 0
    True_10 = 0
    for r in ranks:
        for i in r:
            if i < 6:
                True_5 = True_5 + 1
            if i < 11:
                True_10 = True_10 + 1
    precision_5 = True_5 / (5.0 * len (ranks))
    precision_10 = True_10 / (10.0 * len (ranks))
    return precision_5, precision_10


def Recall(ranks):
    True_5 = 0
    True_10 = 0
    for r in ranks:
        for i in r:
            if i < 6:
                True_5 = True_5 + 1
            if i < 11:
                True_10 = True_10 + 1
    gt_num = 0
    for x in ranks:
        for y in x:
            gt_num += 1

    recall_5 = True_5 / float (gt_num)
    recall_10 = True_10 / float (gt_num)
    return recall_5, recall_10


def Ranks(test_path, users_ids, gender_ids, time_ids, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors,
          Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4):
    file = open (test_path, 'r', encoding='utf-8').read ()
    num = 0
    ranks = []
    rks = []
    for line in file.split ('\n')[:-1]:
        try:
            rk = []
            u = users_ids[line.split (' ')[0]]
            g = gender_ids[line.split (' ')[1]]
            t = time_ids[line.split (' ')[2]]
            p1 = r"(\[.*?\])"
            emojis = re.findall (p1, line)
            # line = line.encode('utf-8', 'strict')
            pattern = re.compile ('[\u4e00-\u9fa5]')
            words = pattern.findall (line)
            str_words = ''.join (words)
            seg_list = jieba.cut (str_words, cut_all=False)
            Context_vector = buildWordVector (seg_list, K)[0]
            scores = []
            if emojis != []:
                for emo in emoji_ids:
                    e = emoji_ids[emo]
                    scores.append ([Score_function (u, g, t, e, User_latent_vectors, Gender_latent_vectors,
                                                    Time_latent_vectors, Context_vector, Emoji_latent_vectors_1,
                                                    Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                                                    Emoji_latent_vectors_4), emo])
                scores.sort ()
                scores.reverse ()
                rank = []
                for i in range (len (scores)):
                    if scores[i][1] in emojis:
                        rank.append (i + 1)
                ranks.append (rank)

                rk.append ([u, line.split (' ')[1], line.split (' ')[2]])
                rk.append (scores)
            rks.append ((rk))
        except:
            continue

    return ranks, rks


print ('reading in data...')
users_ids, gender_ids, time_ids, emoji_ids = Load (train_path)
symbiosis = symbiosis (train_path, emoji_ids)

print ('initializing latent vectors...')
np.random.seed (1234)
if Train:
    User_latent_vectors = np.random.rand (len (users_ids), K)
    Gender_latent_vectors = np.random.rand (len (gender_ids), K)
    Time_latent_vectors = np.random.rand (len (time_ids), K)
    Emoji_latent_vectors_1 = np.random.rand (len (emoji_ids), K)
    Emoji_latent_vectors_2 = np.random.rand (len (emoji_ids), K)
    Emoji_latent_vectors_3 = np.random.rand (len (emoji_ids), K)
    Emoji_latent_vectors_4 = np.random.rand (len (emoji_ids), K)

if Test:
    User_latent_vectors = np.load (result_dir + '/User_latent_vectors.npy')
    Gender_latent_vectors = np.load(result_dir + '/Gender_latent_vectors.npy')
    Time_latent_vectors = np.load (result_dir + '/Time_latent_vectors.npy')
    Emoji_latent_vectors_1 = np.load (result_dir + '/Emoji_latent_vectors_1.npy')
    Emoji_latent_vectors_2 = np.load (result_dir + '/Emoji_latent_vectors_2.npy')
    Emoji_latent_vectors_3 = np.load (result_dir + '/Emoji_latent_vectors_3.npy')
    Emoji_latent_vectors_4 = np.load (result_dir + '/Emoji_latent_vectors_4.npy')


def newEmoji_latent_vectors(Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                            Emoji_latent_vectors_4):
    for e in range (len (emoji_ids)):
        sum11 = np.zeros (K)
        sum12 = np.zeros (K)
        sum13 = np.zeros (K)
        sum14 = np.zeros (K)
        for i in range (len (emoji_ids)):
            if i != e:
                sum11 = sum11 + symbiosis[e][i] * Emoji_latent_vectors_1[i]
                sum12 = sum12 + symbiosis[e][i] * Emoji_latent_vectors_2[i]
                sum13 = sum13 + symbiosis[e][i] * Emoji_latent_vectors_3[i]
                sum14 = sum14 + symbiosis[e][i] * Emoji_latent_vectors_4[i]
        sum21 = np.zeros (K)
        sum22 = np.zeros (K)
        sum23 = np.zeros (K)
        sum24 = np.zeros (K)
        for j in range (len (emoji_ids)):
            if j != e:
                sum31 = np.zeros (K)
                sum32 = np.zeros (K)
                sum33 = np.zeros (K)
                sum34 = np.zeros (K)
                for k in range (len (emoji_ids)):
                    sum31 = sum31 + symbiosis[j][k] * Emoji_latent_vectors_1[k]
                    sum32 = sum32 + symbiosis[j][k] * Emoji_latent_vectors_2[k]
                    sum33 = sum33 + symbiosis[j][k] * Emoji_latent_vectors_3[k]
                    sum34 = sum34 + symbiosis[j][k] * Emoji_latent_vectors_4[k]

                sum21 = sum21 + Emoji_latent_vectors_1[j] - sum31
                sum22 = sum22 + Emoji_latent_vectors_2[j] - sum32
                sum23 = sum23 + Emoji_latent_vectors_3[j] - sum33
                sum24 = sum24 + Emoji_latent_vectors_4[j] - sum34

        delta_E1 = alpha * (Emoji_latent_vectors_1[e] - sum11 - sum21 * symbiosis[j][e])
        delta_E2 = alpha * (Emoji_latent_vectors_2[e] - sum12 - sum22 * symbiosis[j][e])
        delta_E3 = alpha * (Emoji_latent_vectors_3[e] - sum13 - sum23 * symbiosis[j][e])
        delta_E4 = alpha * (Emoji_latent_vectors_4[e] - sum14 - sum24 * symbiosis[j][e])

        Emoji_latent_vectors_1[e] = [x for x in Emoji_latent_vectors_1[e] - Gamma * delta_E1]
        Emoji_latent_vectors_2[e] = [x for x in Emoji_latent_vectors_2[e] - Gamma * delta_E2]
        Emoji_latent_vectors_3[e] = [x for x in Emoji_latent_vectors_3[e] - Gamma * delta_E3]
        Emoji_latent_vectors_4[e] = [x for x in Emoji_latent_vectors_4[e] - Gamma * delta_E4]
    return Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4


def ndcg(ranks, k):
    # ranks 是真实emoji 在推荐的emoji 里的排序
    # k topk
    log = [1 / math.log2 (x + 2) for x in range (k)]
    result = []
    for gt in ranks:
        res = np.zeros (k)
        for t in gt:
            if t <= k:
                res[t - 1] = 1

        if np.sum (res) == 0:
            ndcg1 = 0
        else:
            ndcg1 = (np.dot (np.array (res), log / np.dot (-np.sort (-res), log)))
        result.append (ndcg1)
    return np.sum (np.array (result)) / len (result)


iteration = 0
Pre_Precision_5 = 0
Pre_Precision_10 = 0
Pre_Recall_5 = 0
Pre_Recall_10 = 0

file = open (result_dir + '/result.txt', 'a+')
file.write ('lambda: ' + str (Lambda))
file.write ('\t')
file.write ('alpha: ' + str (alpha))
file.write ('\n')

max_p, max_r, max_f1 = 0, 0, 0
if Train:
    while (iteration < 15):
        print ('training...' + str (iteration))
        User_latent_vectors, Gender_latent_vectors, Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4 = Training (
            users_ids, gender_ids, time_ids, emoji_ids, User_latent_vectors, Gender_latent_vectors, Time_latent_vectors,
            Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4, train_path)
        Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4 = newEmoji_latent_vectors (
            Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3, Emoji_latent_vectors_4)

        if iteration % 1 == 0:
            print ('ranking...')
            ranks, rk = Ranks (test_path, users_ids, gender_ids, time_ids, User_latent_vectors, Gender_latent_vectors,
                               Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2,
                               Emoji_latent_vectors_3,
                               Emoji_latent_vectors_4)
            print ('evaluating...')
            Precision_5, Precision_10 = Precision (ranks)
            Recall_5, Recall_10 = Recall (ranks)
            print (str (iteration) + ":  " + str (Precision_5) + " " + str (Precision_10) + "   " + str (
                Recall_5) + " " + str (Recall_10))
            Pre_Precision_5 = Precision_5
            Pre_Precision_10 = Precision_10
            Pre_Recall_5 = Recall_5
            Pre_Recall_10 = Recall_10
            f1_5 = (Recall_5 + Precision_5) / 2
            f1_10 = (Recall_5 + Precision_5) / 2

            file.write ('iter:%d' % iteration)
            iteration = iteration + 1
            file.write ('\n')
            print ('precision:%.5f recall: %.5f f1-score:%.5f' % (Precision_5, Recall_5, f1_5))
            file.write ('precision:' + str (Precision_5) + ' recall: ' + str (Recall_5) + ' f1-score' + str (f1_10))
            file.write ('\n')
            ndcg_5 = ndcg (ranks, 5)
            ndcg_10 = ndcg (ranks, 10)
            print ('ndcg 5:', ndcg_5)
            file.write ('ndcg5: ')
            file.write (str (ndcg_5))
            file.write ('\n')
            print ('ndcg 10:', ndcg_10)
            file.write ('ndcg10:')
            file.write (str (ndcg_10))
            file.write ('\n')

            if Precision_5 > max_p and Recall_5 > max_r and f1_5 > max_f1:
                max_p, max_r, f1_5 = Precision_5, Recall_5, f1_5
                np.save (result_dir + '/User_latent_vectors.npy', User_latent_vectors)
                np.save (result_dir + '/Time_latent_vectors.npy', Time_latent_vectors)
                np.save (result_dir + '/Emoji_latent_vectors_1.npy', Emoji_latent_vectors_1)
                np.save (result_dir + '/Emoji_latent_vectors_2.npy', Emoji_latent_vectors_2)
                np.save (result_dir + '/Emoji_latent_vectors_3.npy', Emoji_latent_vectors_3)

        iteration = iteration + 1

if Test:
    print ('ranking...')
    ranks, rk = Ranks (test_path, users_ids, gender_ids, time_ids, User_latent_vectors, Gender_latent_vectors,
                       Time_latent_vectors, Emoji_latent_vectors_1, Emoji_latent_vectors_2, Emoji_latent_vectors_3,
                       Emoji_latent_vectors_4)
    print ('evaluating...')
    Precision_5, Precision_10 = Precision (ranks)
    Recall_5, Recall_10 = Recall (ranks)
    print (str (iteration) + ":  " + str (Precision_5) + " " + str (Precision_10) + "   " + str (
        Recall_5) + " " + str (Recall_10))
    Pre_Precision_5 = Precision_5
    Pre_Precision_10 = Precision_10
    Pre_Recall_5 = Recall_5
    Pre_Recall_10 = Recall_10
    f1_5 = (Recall_5 + Precision_5) / 2
    f1_10 = (Recall_5 + Precision_5) / 2

    print ('precision:%.5f recall: %.5f f1-score:%.5f' % (Precision_5, Recall_5, f1_5))

    ndcg_5 = ndcg (ranks, 5)
    ndcg_10 = ndcg (ranks, 10)
    print ('ndcg 5:', ndcg_5)
    print ('ndcg 10:', ndcg_10)

file.close ()
print ("Finished!")

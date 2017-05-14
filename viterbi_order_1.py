
#!/usr/bin/python
import os, sys
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(PROJECT_DIR)
from database import connect
import nltk
import difflib
import pandas as pd
import re
pd.options.mode.chained_assignment = None  # default='warn'
import nltk
from numpy import array, ones, zeros, multiply
import numpy as np
import csv
import cPickle as pickle
from itertools import izip

def treatmentLabelNoisyChannel(label):
    return label.apply(lambda x : x.replace('\\n', '') if x is not None else '').str.replace(r"\S*\d\S*", "").str.replace(r"\(.*\)", "").str.replace("[^a-zA-Z]", " ").\
        str.replace("(s)( |$)", " ").str.replace(r"\s\s+", " ").apply(lambda x: x.rstrip()).apply(lambda x: x.lstrip()).\
        apply(lambda x : re.sub(' +',' ', x))
def make_relation(result, texte):
    counts = np.zeros((26, 26))
    for i in range(len(result)):
        for j in range(len(result[i])):
            counts[result[i][j], texte[i][j]] += 1
    relation = np.argmax(counts, axis = 1)
    return relation

#  transform index prediction into letters
def etat_to_letters(result, relation):
    res_final = []
    for mot in result:
        mot_res = []
        for letter in mot:
            mot_res = mot_res + [hmm.omega_X[relation[int(letter)]]]
        res_final.append(mot_res)
    return res_final

def make_text_corr(typos):
    texte_corr = []
    for mot in typos:
        mot_corr = []
        for letter in mot:
            letter1 , letter2 = letter
            mot_corr.append(letter2)
        texte_corr.append(mot_corr)
    return texte_corr


class HMM_unsupervised:
    def __init__(self, nb_state, observation_list, state_list=None, A=None, B=None, pi=None):
        print ("HMM creating with: ")
        self.N = nb_state  # number of states
        self.M = len(observation_list)  # number of possible emissions
        print (str(self.N) + " states")
        print (str(self.M) + " observations")
        self.omega_X = observation_list
        if A is None:
            self.A = np.random.uniform(0, 1, (self.N, self.N))
            self.A = self.A / np.sum(self.A, axis=0).reshape((1, self.N))
        else:
            self.A = A
        if B is None:
            self.B = np.random.uniform(0, 1, (self.M, self.N))
            self.B = self.B / np.sum(self.B, axis=0).reshape((1, self.N))
        else:
            self.B = B
        if pi is None:
            self.pi = np.random.uniform(0, 1, self.N)
            self.pi = self.pi / np.sum(self.pi)
        else:
            self.pi = pi
        if state_list is None:
            self.omega_Y = np.zeros(self.N)
        else:
            self.omega_Y = state_list
        self.smoothing_obs = 15.0
        self.make_indexes()

    def make_indexes(self):
        """Creates the reverse table that maps states/observations names
        to their index in the probabilities array"""
        self.Y_index = {}
        for i in range(self.N):
            self.Y_index[self.omega_Y[i]] = i
        self.X_index = {}
        for i in range(self.M):
            self.X_index[self.omega_X[i]] = i

    def observation_estimation(self, pair_counts):
        """ Build the observation distribution:
            observation_proba is the observation probablility matrix
            [b_ki],  b_ki = Pr(X_t=v_k|Y_t=q_i)"""
        # fill with counts
        for pair in pair_counts:
            wrd = pair[0]
            tag = pair[1]
            cpt = pair_counts[pair]
            k = 0  # for <unk>
            if wrd in self.X_index:
                k = self.X_index[wrd]
            i = self.Y_index[tag]
            self.B[k, i] = cpt
        # normalize
        self.B = self.B + self.smoothing_obs
        self.B = self.B / self.B.sum(axis=0).reshape(1, self.N)

    def transition_estimation(self, trans_counts):
        """ Build the transition distribution:
            transition_proba is the transition matrix with :
            [a_ij] a[i,j] = Pr(Y_(t+1)=q_i|Y_t=q_j)
        """
        # fill with counts
        for pair in trans_counts:
            i = self.Y_index[pair[1]]
            j = self.Y_index[pair[0]]
            self.A[i, j] = trans_counts[pair]
            # normalize
        self.A = self.A / self.A.sum(axis=0).reshape(1, self.N)

    def init_estimation(self, init_counts):
        """Build the init. distribution"""
        # fill with counts
        for tag in init_counts:
            i = self.Y_index[tag]
            self.pi[i] = init_counts[tag]
        # normalize
        self.pi = self.pi / sum(self.pi)

    def supervised_training(self, pair_counts, trans_counts, init_counts):
        """ Train the HMM's parameters. This function wraps everything"""
        self.observation_estimation(pair_counts)
        self.transition_estimation(trans_counts)
        self.init_estimation(init_counts)

    def Viterbi(self, indexes):

        delta = self.pi * self.B[indexes[0]]
        phi = np.zeros((len(indexes), self.N))
        for count in range(1, len(indexes)):
            delta_inter = np.zeros(self.N)
            for i in range(self.N):
                inter = [delta[j] * self.A[i][j] * self.B[indexes[count]][i] for j in range(self.N)]
                delta_inter[i] = np.max(inter)
                phi[count][i] = np.argmax(inter)
            delta = delta_inter
        N_ind = len(indexes)
        etat = np.zeros(N_ind)
        etat[N_ind - 1] = np.argmax(delta)
        for i in range(1, N_ind):
            etat[N_ind - 1 - i] = phi[N_ind - i][etat[N_ind - i]]
        return etat

    def reestimation(self, data):
        eps = 0.0005
        pi = np.zeros(self.N)
        B = np.zeros((self.M, self.N))
        A = np.zeros((self.N, self.N))
        nb_mots = len(data)
        for i in range(nb_mots):
            indexes = [self.X_index[letter] for letter in data[i]]
            etat = self.Viterbi(indexes)
            # print(etat)
            pi[etat[0]] += 1
            for j in range(len(etat)):
                B[indexes[j], etat[j]] += 1
                if j > 0:
                    A[etat[j], etat[j - 1]] += 1
        pi = pi + eps
        B = B + eps
        A = A + eps
        self.pi = pi / np.sum(pi)
        self.B = B / np.sum(B, axis=0).reshape(1, self.N)
        self.A = A / np.sum(A, axis=0).reshape(1, self.N)

    def Viterbi_EM(self, data, n_iter):
        for i in range(n_iter):
            etat = self.reestimation(data)

    def predict(self, data):
        result = []
        texte = []
        nb_mots = len(data)
        for i in range(nb_mots):
            indexes = [self.X_index[letter] for letter in data[i]]
            etat = self.Viterbi(indexes)
            result.append(etat)
            texte.append(indexes)
        return result, texte

char_word_list = pickle.load(open("/home/ahmed/internship/testbouygues.pkl", "rb"))
# observation list
c_words = dict()
for mot in char_word_list:
    for letter in mot:
        c_words[letter] = 1
hmm = HMM_unsupervised(len(c_words.keys()), c_words.keys())
# training viterbi em
hmm.Viterbi_EM(char_word_list, 10000)
# prediction
result, texte = hmm.predict(char_word_list)

relation = make_relation(result, texte)
res_final = etat_to_letters(result, relation)

result_final2= [''.join(characters) for characters in res_final ]
char_word_list2 = [''.join(characters) for characters in char_word_list]
with open('/home/ahmed/internship/try.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(izip(result_final2, char_word_list2))

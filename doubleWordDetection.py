import os, sys
from database import connect
import nltk
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(PROJECT_DIR)

def main():
    pass

# Initialize database configuration from command line, if provided
connect.init()
con = connect.make_connection()
cur = con.cursor()
query_remove_breaks = "update classifier.invoice_line set label = regexp_replace(label, E'[\\n\\r]+', ' ', 'g' );"
cur.execute(query_remove_breaks)
query_add_space_after_backslash_b_right = "update classifier.invoice_line set label = regexp_replace(label, '\\n([^ ])', '\\n \1');"
cur.execute(query_add_space_after_backslash_b_right)
query_add_space_after_backslash_b_left = "update classifier.invoice_line set label = regexp_replace(label, '([^ ])\\n', '\\n \1');"
cur.execute(query_add_space_after_backslash_b_left)
con.commit()

query_train = 'select id, file, lower(unaccent(label)) as label, label as label_raw from classifier.invoice_line  ;'
# Load data
data_train = connect.load_data(query_train)
data_train['id'] = data_train.index
data_train.label = data_train.label.fillna('').apply(lambda x : x.replace('\\n', '')).apply(lambda x : x.replace('\n', ''))\
    .str.replace(r"\s\s+", " ").apply(lambda x: x.rstrip()).apply(lambda x: x.lstrip())
print data_train.shape
data_train['word'] = data_train.label.apply(lambda x : x.split(' '))
word_list = [item for sublist in data_train['word'].tolist() for item in sublist]
word_count = nltk.FreqDist(word_list)
data_train['bigram'] = data_train.label.apply(lambda x: list(nltk.bigrams(x.split(' '))))
bigram_list = [item for sublist in data_train.bigram.tolist() for item in sublist]
bigram_count = nltk.FreqDist(bigram_list)

dict_word = {}
unique_word = set(word_list)
for k in unique_word :
    dict_word[k] = []
for idx, row in data_train.iterrows():
    bb = row['word']
    for elem in bb:
        dict_word[elem].append(idx)

dict_bigram = {}
unique_bigram = set(bigram_list)
for k in unique_bigram :
    dict_bigram[k] = []
for idx, row in data_train.iterrows():
    bb = row['bigram']
    for elem in bb:
        dict_bigram[elem].append(idx)

double_word=[]
count_double_word = []
update_bigram = []
count_bigram_update =[]
first_if = 0
second_if = 0
compt=0
for bigram in bigram_count.keys():
    count_bigram = bigram_count.get(bigram)
    concat_bigram = list(bigram)[0] + list(bigram)[1]
    concat_bigram = concat_bigram.replace("'", "''")
    if concat_bigram.isdigit():
        continue
    count_word = word_count.get(concat_bigram)
    if count_word is not None:
        if count_word > count_bigram:
            incorrect = list(bigram)[0] + ' ' + list(bigram)[1]
            incorrect = incorrect.replace("'", "''")
            idx_update = str(tuple(dict_bigram.get(bigram)))
            if len(dict_bigram.get(bigram))==1:
                idx_update = idx_update.replace(',', '')
            print(1)
            query = "update classifier.invoice_line set doubleword=label, label = replace(lower(unaccent(label)), '{}', '{}')  where id in {}  ; ".format( incorrect, concat_bigram, idx_update)
            print(query)
            first_if = first_if + 1
            cur.execute(query)
            compt = compt + 1
            print compt
        if count_word < count_bigram:
            correct = list(bigram)[0] + ' ' + list(bigram)[1]
            correct = correct.replace("'", "''")
            idx_update = str(tuple(dict_word.get(concat_bigram)))
            if len(dict_word.get(concat_bigram)) == 1:
                idx_update = idx_update.replace(',', '')
            print(2)
            query = "update classifier.invoice_line set doubleword=label, label = replace(lower(unaccent(label)), '{}', '{}' )  where id in {} ; ".format(concat_bigram, correct, idx_update)
            print(query)
            second_if = second_if + 1
            cur.execute(query)
            compt = compt + 1
            print compt
        double_word.append(concat_bigram)
        count_double_word.append(count_word)
        update_bigram.append(bigram)
        count_bigram_update.append(count_bigram)
con.commit()
con.close()

errorDF = pd.DataFrame(data = [double_word, count_double_word, update_bigram, count_bigram_update]).T
errorDF.columns = ['double_word', 'count_double_word', 'bigram', 'count_bigram_update']
errorDF[errorDF.count_double_word != errorDF.count_bigram_update].to_csv('/home/Desktop/errorDF1.csv', sep=';')

pd.DataFrame(data = [double_word, count_double_word, update_bigram, count_bigram_update]).T.to_csv('/home/Desktop/errorDF11.csv', sep=';')

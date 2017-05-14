import os, sys
from database import connect
import nltk
import difflib
import pandas as pd
import re

pd.options.mode.chained_assignment = None  # default='warn'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(PROJECT_DIR)

def main():
    pass

def treatmentLabelNoisyChannel(label):
    return label.apply(lambda x : x.replace('\\n', '') if x is not None else '').str.replace(r"\S*\d\S*", "").str.replace(r"\(.*\)", "").str.replace("[^a-zA-Z]", " ").\
        str.replace("(s)( |$)", " ").str.replace(r"\s\s+", " ").apply(lambda x: x.rstrip()).apply(lambda x: x.lstrip()).\
        apply(lambda x : re.sub(' +',' ', x))

def probaXn1givenXn(Xn1, Xn, bigram_list):
    count_Xn = 0
    count_Xn1_Xn = 0
    for w1, w2 in bigram_list:
        if w1 == Xn and w2 == Xn1:
            count_Xn1_Xn += 1
        if w1 == Xn :
            count_Xn += 1
    proba_Xn1_Xn = count_Xn1_Xn / float(count_Xn)
    return proba_Xn1_Xn, count_Xn1_Xn, count_Xn

def detectError(x, bigram_list):
    error = {}
    replace = {}
    n_x = len(x)
    for idx in range(1, n_x):
        similars = difflib.get_close_matches( str(x[idx]), word_list_unique, n=5, cutoff=0.8)
        proba = []
        word = []
        countXN=[]
        countXN1 = []
        for similar in similars:
            word.append(similar)
            proba_Xn1_Xn, count_Xn1, count_Xn = probaXn1givenXn(similar, x[idx-1], bigram_list)
            proba.append(proba_Xn1_Xn)
            countXN.append(count_Xn)
            countXN1.append(count_Xn1)

        vector= pd.DataFrame([proba,countXN1, countXN]).T
        vector.index = word
        vector.columns = ['proba', 'countingXN1', 'countingXN']
        if str(vector.proba.argmax()) != x[idx]:
            error[x[idx]] = [vector.proba.argmax(), vector.proba.max(), vector.loc[x[idx]]['proba'], vector.loc[x[idx]]['countingXN1'],  vector.loc[x[idx]]['countingXN']]
            replace[x[idx]] = [vector.proba.argmax()]
    return error, replace

def applyDetectError(data):
    error_vect = []
    replace_vect = []
    index = []
    i= 0
    print( "shape data train without duplicated : " + str(data.shape))
    for idx, row in data.label.iteritems():
        print str(i) + ' ' + str(idx) + ' ' + str(row)
        i += 1
        if len(row.split(' ')) < 2 :
            continue
        index.append(idx)
        error, replace = detectError(row.split(' '), bigram_list)
        if error != {}:
            print(idx + '    ' + str(error) + '   ' + str(replace))
        error_vect.append(error)
        replace_vect.append(replace)
    return index, error_vect, replace_vect

# Initialize database configuration from command line, if provided
connect.init()
query_train = 'select id, file, lower(unaccent(label)) as label, label as label_raw from classifier.invoice_line  ;'
# Load data
data_train = connect.load_data(query_train)
data_train.label = treatmentLabelNoisyChannel(data_train.label)
print data_train.shape
data_train_wth_duplicated = data_train.drop_duplicates(subset='label')
word_list_raw = data_train.label.str.split(' ').tolist()
word_list = [item for sublist in word_list_raw for item in sublist]
word_count = nltk.FreqDist(word_list)
data_train['bigram'] = data_train.label.apply(lambda x: list(nltk.bigrams(x.split(' '))))
bigram_list = [item for sublist in data_train.bigram.tolist() for item in sublist]
n = len(word_list)
num_bigram = len(bigram_list)
word_list_unique = list(set(word_list))
index, error_vect, replace_vect = applyDetectError(data_train_wth_duplicated)
data_train_wth_duplicated.loc[index, 'error'] = error_vect
data_train_wth_duplicated.loc[index, 'replace_error'] = replace_vect
data_train_wth_duplicated = data_train_wth_duplicated[~((data_train_wth_duplicated.error.isnull()) | (data_train_wth_duplicated.error == {}))]
data_train_wth_duplicated['id'] = data_train_wth_duplicated.index
idx_to_remove = []
for idx, row in data_train_wth_duplicated.iterrows():
    keys = row.error.keys()
    if len(keys) >1:
        for key in keys:
            temp_df = pd.DataFrame([[row.file, row.label, row.label_raw, dict({key: row.error.get(key)}), dict({key: row.replace_error.get(key)}), row.id  ]], columns = ['file', 'label', 'label_raw', 'error', 'replace_error', 'id'])
            data_train_wth_duplicated = data_train_wth_duplicated.append(temp_df, ignore_index=True)
        idx_to_remove.append(data_train_wth_duplicated[data_train_wth_duplicated['id'] == row.id].index[0])

data_train_wth_duplicated = data_train_wth_duplicated.drop(idx_to_remove)
data_train_wth_duplicated['diff_proba'] = data_train_wth_duplicated.error.apply(lambda x: x.get(x.keys()[0])[1] - x.get(x.keys()[0])[2])
data_train_wth_duplicated['occurence'] = data_train_wth_duplicated.error.apply(lambda x: x.get(x.keys()[0])[1]*x.get(x.keys()[0])[4])
data_train_wth_duplicated['occurence2'] = data_train_wth_duplicated.error.apply(lambda x: x.get(x.keys()[0])[3])
threshold_proba = 0.05
threshold_n = 10
data_train_wth_duplicated.to_csv('/home/faris/Desktop/data_train_wth_duplicated.csv')
data_write = data_train_wth_duplicated[(data_train_wth_duplicated['diff_proba'] > threshold_proba) & (data_train_wth_duplicated['occurence2'] < threshold_n)]
data_write.to_csv('/home/faris/Desktop/data_write.csv')

# Write into DB
con = connect.make_connection()
cur = con.cursor()
for idx, row in data_write.iterrows():
    id_similar = '(' + str(data_train[data_train['label' ] == row['label']].index.tolist())[1:-1] + ')'
    update_invoice_label_noisychanel = "update classifier.invoice_line set label_noisychanel = label where label_noisychanel is null and id in {} ;".format(id_similar)
    cur.execute(update_invoice_label_noisychanel)
    key = row.replace_error.keys()[0]
    value = row.replace_error.get(key)[0]
    try:
        substring = row['label_raw'].split(' ')
        postion_key = substring.index(key)
        concat_key_wordbefore = substring[postion_key -1] + key
        if value == concat_key_wordbefore:
            update_invoice = "update classifier.invoice_line set label = regexp_replace(label, '{}', '{}', 'gi') where id in {} ;".format(substring[postion_key -1] + ' ' + key , concat_key_wordbefore, id_similar)
            cur.execute(update_invoice)
            continue
    except Exception :
        pass
    update_invoice = "update classifier.invoice_line set label = regexp_replace(unaccent(lower(label)), '{}', '{}', 'gi') where id in {} ;".format(key , value, id_similar)
    print (update_invoice)
    cur.execute(update_invoice)
con.commit()
con.close()


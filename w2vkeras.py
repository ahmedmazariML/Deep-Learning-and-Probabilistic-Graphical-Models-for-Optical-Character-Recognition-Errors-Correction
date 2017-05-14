#!/usr/bin/python
import sys
import os

import pandas as pd
import psycopg2
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Merge, Convolution1D, MaxPooling1D
from keras.models import Sequential, Model
from database import connect
from helpers.features_helpers import designationTreated
from helpers.w2v_helpers import get_word2vec

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
# -----------------------------------
# Parameters
# -----------------------------------
# --- Model hyperparameters
embedding_dim = 100
sequence_lenght = 56
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150

# --- Training parameters
batch_size = 32
num_epochs = 100
val_split = 0.1

# --- word2vec parameters
context = 10


# -----------------------------------
# Data preparation
# -----------------------------------
# --- Load data from database
try:
    cnx = connect.do()
    cur = cnx.cursor()
    query_train = 'select i.id, cla.invoice_line as cla_invoice, date, ' \
                  'lower(unaccent(label)) || \' \' || lower(unaccent(parent1)) || \' \' || lower(unaccent(parent2)) as concat,   ' \
                  'lower(unaccent(label)) as label, lower(unaccent(parent1)) as parent1, lower(unaccent(parent2)) as parent2, provider, "group", asked, answered, manual, name ' \
                  'from classifier.invoice_line i ' \
                  'left join classifier.question q on q.invoice_line = i.id ' \
                  'left join classifier.classification cla on cla.invoice_line = i.id  ' \
                  'left join classifier.category cat on cat.id = cla.category ;'
    query_test = 'select i.id, cla.invoice_line as cla_invoice, date,' \
                 'lower(unaccent(label)) || \' \' || lower(unaccent(parent1)) || \' \' || lower(unaccent(parent2)) as concat, ' \
                 'lower(unaccent(label)) as label, lower(unaccent(parent1)) as parent1, lower(unaccent(parent2)) as parent2 , provider ' \
                 'from classifier.invoice_line i ' \
                 'left join classifier.question q on q.invoice_line = i.id ' \
                 'left join classifier.classification cla on cla.invoice_line = i.id  ' \
                 'left join classifier.category cat on cat.id = cla.category ' \
                 'where (name is null or name != \'NT\') ' \
                 'group by i.id, cla_invoice, date, label, parent1, parent2 ;'
    cur.execute(query_train)
    rows_train = cur.fetchall()
    colnames_train = [desc[0] for desc in cur.description]

    cur.execute(query_test)
    rows_test = cur.fetchall()
    colnames_test = [desc[0] for desc in cur.description]

    #cnx.close()
except psycopg2.Error as e:
    print "Error {}".format(e)
    sys.exit(1)

data_train = pd.DataFrame(rows_train, columns=colnames_train).set_index('id')
data_test = pd.DataFrame(rows_test, columns=colnames_test).set_index('id')

# --- Change type and clean the fields
data_train['date'] = pd.to_datetime(data_train['date'])

colnames = ['label', 'parent1', 'parent2', 'provider']
for col in colnames:
    data_train[col] = designationTreated(data_train[col])
    data_test[col] = designationTreated(data_test[col])


data_train['provider'] = data_train['provider'].str.replace(" ", "")
data_test['provider'] = data_test['provider'].str.replace(" ", "")

# --- Load Word2Vec pre trained model
embedding_weights = get_word2vec()

# -----------------------------------
# Features extraction
# -----------------------------------



# -----------------------------------
# Building model
# -----------------------------------

graph_in = Input(shape=(sequence_lenght,embedding_dim))
convs = []

for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters, filter_length=fsz, border_mode='valid', activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)

# --- main sequential model
model = Sequential()
model.add(Dropout(dropout_prob[0], input_shape=(sequence_lenght, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# -----------------------------------
# Fitting and prediction
# -----------------------------------
"""

fit_category = GaussianNB()
fit_category.fit(x_train, y_train)
y_pred_train = fit_category.predict(x_train)

print "shape x train : " + str(x_train.shape)
print accuracy_score(y_train, y_pred_train)
print cross_validation.cross_val_score(fit_category, x_train, y_train, cv=5, n_jobs=-1)

y_proba_test = pd.DataFrame(fit_category.predict_proba(x_test), index=x_test.index)

nb_show = 3
index_to_write = list(x_test.index)
cla_index = data_test[~data_test['cla_invoice'].isnull()]['cla_invoice'].tolist()
threshold = 0.8
i=0
for idx in index_to_write:
    proba = y_proba_test.loc[idx].sort_values(ascending=False).iloc[0:nb_show]
    proba_max = proba.iloc[0]
    asked = False
    if proba_max < threshold:
        asked = True
    if idx in cla_index:
        i = i + 1
        cla_id_delete = "select cla.id from classifier.classification cla join classifier.category c on c.id = cla.category where type='category' and invoice_line =  '" + idx + "'"
        delete_classification = "delete from classifier.classification where id in ( " + cla_id_delete + " ) ;"
        cur.execute(delete_classification)
    for p_temp in proba.iteritems():
        name = p_temp[0]
        cat_id = "(select id from classifier.category where type='category' and name = '" + name + "') "
        print cat_id
        classification_id = "select id from classifier.classification where invoice_line = '" + idx + "'; "
        insert_classification="insert into classifier.classification ( category, confidence, invoice_line, prediction ) values ( " + cat_id + " , " + str(p_temp[1]) + ", '" + idx + "', true );"
        print insert_classification
        update_question = "update classifier.question set asked = " + str(asked) + " where invoice_line = '" + idx +"';"
        cur.execute(update_question)
        cur.execute(insert_classification)

"""
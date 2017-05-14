import csv
import pandas as pd
import numpy as np
from  itertools import chain
import ast
df = pd.read_csv('/home/ahmed/internship/cnn_ocr/scale_train/positions.csv',index_col=0)
df1 = df.applymap(lambda x: [y for y in x if len(y) > 0])
df1 = df1[df1.applymap(len).ne(0).all(axis=1)]


df1 = df.replace(['\[\],','\[\[\]\]', ''],['','', np.nan], regex=True)
df1 = df1.dropna()
df.positionlrtb=df.positionlrtb.apply(ast.literal_eval)
df.positionlrtb=df.positionlrtb.apply(lambda x: [y for y in x if len(y) > 0])
df1 = pd.DataFrame({
        "from_line": np.repeat(df.index.values, df.positionlrtb.str.len()),
        "b": list(chain.from_iterable(df.positionlrtb)),
        "c" : np.repeat(df['page_number'].values, df.positionlrtb.str.len()),
        "d" : np.repeat(df['words'].values,df.positionlrtb.str.len())
})


df1['all_chars_in_same_row'] = df1['b'].apply(lambda x: tuple([y for y in x if isinstance(y, str)]))
df1['page_number']=df1['c']
df1['words']=df1['d']
df1 = df1.set_index(['from_line','all_chars_in_same_row','page_number','words'])
df1 = pd.DataFrame(df1.b.values.tolist(), index=df1.index)
df1.columns = [df1.columns % 5, df1.columns // 5]
df1 = df1.stack().reset_index(level=4, drop=True)
cols = ['char','left','top','right','bottom']
df1.columns = cols
df1[cols[1:]] = df1[cols[1:]].astype(int)
df1 = df1.reset_index()
df1['all_chars_in_same_row'] = df1['all_chars_in_same_row'].apply(list)
print(df1.head(5))
df1.to_csv("/home/ahmed/internship/cnn_ocr/scale_train/char_position.csv")

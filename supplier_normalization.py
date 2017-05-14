from database import connect
import difflib
from pylev import levenschtein
import pandas as pd

'''
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
sys.path.append(PROJECT_DIR)
# Initialize database configuration from command line, if provided
parser = argparse.ArgumentParser(description='Run supplier normalization.')
#parser.add_argument('--conf', dest='conf', action='store', help='Path to the server configuration.yaml file')
#args = parser.parse_args()
'''

connect.init()
con = connect.make_connection()
cur = con.cursor()

query_update = "update classifier.invoice_line set provider = regexp_replace(regexp_replace(lower(unaccent(provider)), '[^a-zA-Z\d\s:]', '', 'g'), '\s+', ' ', 'g') ; "
cur.execute(query_update)
query_select = "select provider  as provider_lower, count(*) from classifier.invoice_line where provider not in ('', ' ') group by provider_lower order by provider_lower ;"
cur.execute(query_select)
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
data_train = pd.DataFrame(rows, columns=colnames)
data_train['replacing'] = None
print data_train.shape

i = 0
lev_max= 1
levenstein_distance=[]
for idx, row in data_train.iterrows():
    provider_current = row['provider_lower']
    count_current = row['count']
    data_train_current = data_train[data_train['provider_lower'] != row['provider_lower']]['provider_lower']
    closest_word = difflib.get_close_matches(provider_current , data_train_current, n=1, cutoff=0.6)
    if len(closest_word) > 0:
        match = closest_word[0]
        distance = levenschtein(provider_current,match )
        if distance <= lev_max:
            count_closest = data_train[data_train['provider_lower']== match]['count'].item()
            if count_closest > count_current:
                #print "replace " + provider_current +  " by " + match + " ( " + str(count_current) + " , " + str(count_closest) + ")"
                idx = data_train[data_train['provider_lower'] == provider_current].index
                data_train.loc[idx, 'replacing'] = match
                i = i+1
            if count_closest == count_current:
                i = i+1
                idx = data_train[data_train['provider_lower'] == provider_current].index
                data_train.loc[idx, 'replacing'] = match
                idx = data_train[data_train['provider_lower'] == match].index
                data_train.loc[idx, 'replacing'] = match

print i

data_write = data_train[~data_train['replacing'].isnull()]
for idx, row in data_write.iterrows():
    supplier_new = row['replacing']
    supplier_old = row['provider_lower']
    update_provider = "update classifier.invoice_line set provider = '{}' where provider = '{}' ;".format(supplier_new, supplier_old)
    cur.execute(update_provider)
cur.execute("update classifier.invoice_line SET provider=initcap(provider);")
con.commit()
con.close()

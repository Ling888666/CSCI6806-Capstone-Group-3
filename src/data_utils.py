import pandas as pd
import sqlite3
import os
import re
import numpy as np
from datetime import datetime
from dateutil import parser
from functools import reduce

def parse_date_string(date_string):
    try:
        return parser.parse(date_string)
    except (ValueError, OverflowError, TypeError) as e:
        # print(f"Error parsing date string '{date_string}': {e}")
        return None

def parse_number_string(number_string):
    try:
        return pd.to_numeric(number_string, errors='raise')
    except Exception as e:
        # print(f"Error parsing number string '{number_string}': {e}")
        return None

def parse_word_string(text):
    if (parse_date_string(text) is None) and (parse_number_string(text) is None):
        return text
    else:
        return None
    
def parse_data_string(text):
    # a_date = parse_date_string(text)
    # a_number = None
    # if a_date is None:
    #     a_number = parse_number_string(text)
    a_number = parse_number_string(text)
    a_date = None
    if a_number is None:
        a_date = parse_date_string(text)
    a_ref = None
    if isinstance(text,str):
        da_reg = r"\(ID\#:(\d+)\)"
        ref_res = re.search(da_reg, text)
        if (ref_res is not None):
            a_ref = parse_number_string(ref_res[1])
    a_word = None
    if (a_date is None) and (a_number is None) and (a_ref is None):
        a_word = text
    a_null = None
    if (a_date is None) and (a_number is None) and (a_ref is None) and (a_word is None):
        a_null = True
    # print(f"text={text}, a_ref={a_ref}")
    return a_date, a_number, a_word, a_null, a_ref

def datetime_to_int_timestamp(dt):
    if isinstance(dt, datetime):
        try:
            int_timestamp = int(dt.timestamp())
            return int_timestamp
        except Exception:
            return None
    else:
        # raise TypeError("Provided argument is not a datetime object")
        return None

def datetime_to_int_timestamp_ms(dt):
    if isinstance(dt, datetime):
        try:
            int_timestamp_ms = int(dt.timestamp() * 1000)
            return int_timestamp_ms
        except Exception:
            return None
    else:
        # raise TypeError("Provided argument is not a datetime object")
        return None

def split_columns(df, easy=False, splitter='-', type_suffix=" Type", id_suffix=" ID"):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided argument is not a pandas DataFrame")
    
    columns_to_drop = []

    for column in df.columns:

        if df[column].apply(lambda x: isinstance(x, str) and splitter in x and (easy or x.split(splitter, 1)[1].isdigit())).all():

            split_df = df[column].str.split(splitter, expand=True)
            split_df.columns = [column + type_suffix, column + id_suffix]

            split_df[column + id_suffix] = split_df[column + id_suffix].astype(int)

            df = pd.concat([df, split_df], axis=1)
            columns_to_drop.append(column)
    
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df

def fill_none(data,none_value=-1):
    if data is None:
        return none_value
    return data

def fill_date(date, col_meta):
    date_val = datetime_to_int_timestamp(date)
    if date_val is None:
        return -1
    # col_meta['value_count']+=1
    col_meta['value_min'] = min(col_meta['value_min'],date_val)
    col_meta['value_max'] = max(col_meta['value_max'],date_val)
    return date_val

def fill_number(number, col_meta):
    if number is None:
        return -1
    # col_meta['value_count']+=1
    col_meta['value_min'] = min(col_meta['value_min'],number)
    col_meta['value_max'] = max(col_meta['value_max'],number)
    return number

def fill_word_id(word,col_id,cursor,conn,col_meta):
    if word is None:
        return -1
    try:
        cursor.execute(f'''SELECT id FROM col_{col_id}_dict WHERE word=?''',(word,))
        rows = cursor.fetchall()
        if len(rows)>0:
            col_meta['word_count']+=1
            # print(f"{word}={rows}")
            return rows[0][0]
        else:
            cursor.execute(f'''INSERT INTO col_{col_id}_dict (word) VALUES (?)''', (word,))
            rowid = cursor.lastrowid
            col_meta['dict_count']+=1
            conn.commit()
            col_meta['word_count']+=1
            return rowid
    except sqlite3.OperationalError as e:
        print(e)
        return -1

def fill_is_null(data,col_meta):
    if data == True:
        col_meta['null_count']+=1
        return 1
    return 0

def fill_ref(data, da_df, id_column, value_name, word_name, col_meta):
    if data is None:
        # print("fill_ref none")
        return -1,-1
    row_view = da_df.loc[da_df[id_column] == data]
    if (len(row_view)<=0):
        return -1,-1
    da_row = da_df.loc[da_df[id_column] == data].iloc[0]
    # print(da_row)
    ref_value = da_row[value_name]
    ref_word = da_row[word_name]
    # print(f"value_name={value_name}, word_name={word_name}, ref_value={ref_value}, ref_word={ref_word}")
    if (ref_value>=0 or ref_word>=0):
        col_meta['ref_count']+=1
    return ref_value, ref_word

def fill_row_value(row,col_names,col_meta):
    for col_name in col_names:
        col_val = row[col_name]
        if (col_val is not None) and (col_val >= 0):
            col_meta['value_count']+=1
            return col_val
    return -1

def fill_row_ref(row,ref_map,col_meta):
    src_value =  row[ref_map['value_src']] if 'value_src' in ref_map else -1
    dst_value = row[ref_map['value_dst']] if 'value_dst' in ref_map else -1
    if ((dst_value is None or dst_value<0) and src_value>=0):
        col_meta['value_count']+=1
        dst_value = src_value
    src_word = row[ref_map['word_src']] if 'word_src' in ref_map else -1
    dst_word = row[ref_map['word_dst']] if 'word_dst' in ref_map else -1
    if ((dst_word is None or dst_word<0) and src_word>=0):
        col_meta['word_count']+=1
        dst_word = src_word
    # print(ref_map)
    # print(f"src_value={src_value},src_word={src_word}")
    # print(f"dst_value={dst_value},dst_word={dst_word}")
    return dst_value,dst_word

cnt = 0

def row_insert(row,col_id,name_map,cursor,conn):
    global cnt
    # CREATE TABLE IF NOT EXISTS col_{col_id}_value (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     row_id INTEGER,
    #     is_null INTEGER NOT NULL DEFAULT 0,
    #     value REAL DEFAULT NULL,
    #     word_id INTEGER DEFAULT NULL,
    #     FOREIGN KEY(word_id) REFERENCES col_{col_id}_dict(id)
    # )
    row_id = int(row[name_map['row_id']])
    is_null = 1 if ('is_null' in name_map) and (row[name_map['is_null']]>0) else 0
    value = float(row[name_map['value']]) if 'value' in name_map else None
    word_id = int(row[name_map['word_id']]) if 'word_id' in name_map else None
    cursor.execute(f'''INSERT INTO col_{col_id}_value (row_id,is_null,value,word_id) VALUES (?,?,?,?)''', (row_id,is_null,value,word_id,))

    # cursor.execute(f'''SELECT * FROM col_{col_id}_value WHERE row_id=? LIMIT 1''', (row_id,))
    # id,row_id,is_null,value,word_id = cursor.fetchall()[0]
    # print(f"col_id={col_id},row_id={row_id},value={value}")
    cnt+=1
    if (cnt>1000):
        cnt=0
        conn.commit()

def update_col_meta(col_id,col_meta,cursor,conn):
    # CREATE TABLE IF NOT EXISTS column_meta (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     name TEXT UNIQUE NOT NULL,
    #     null_count INTEGER DEFAULT 0,
    #     value_count INTEGER DEFAULT 0,
    #     value_min REAL DEFAULT 1e308,
    #     value_max REAL DEFAULT -1e308,
    #     ref_count INTEGER DEFAULT 0,
    #     word_count INTEGER DEFAULT 0,
    #     dict_count INTEGER DEFAULT 0
    # )
    null_count= int(col_meta['null_count'])
    value_count= int(col_meta['value_count'])
    value_min= float(col_meta['value_min'])
    value_max= float(col_meta['value_max'])
    ref_count= int(col_meta['ref_count'])
    word_count= int(col_meta['word_count'])
    dict_count= int(col_meta['dict_count'])
    cursor.execute(f'''UPDATE column_meta SET null_count=?, value_count=?,
        value_min=?, value_max=?, ref_count=?, word_count=?, dict_count=? WHERE id=?''',
        (null_count,value_count,value_min,value_max,ref_count,word_count,dict_count,col_id))
    conn.commit()

def df_norm(d):
    d_norm = (d-d.min())/(d.max()-d.min())
    return d_norm

def df_uint_scale(d):
    d_uint=df_norm(d) * 0xFFFFFFFF
    d_uint.round(0)
    return d_uint

def simple_hash(x):
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = ((x >> 16) ^ x) * 0x45d9f3b
    x = (x >> 16) ^ x
    return x

def simple_unhash(x):
    x = ((x >> 16) ^ x) * 0x119de1f3
    x = ((x >> 16) ^ x) * 0x119de1f3
    x = (x >> 16) ^ x
    return x

def hash_combine(lhs,rhs):
    #  lhs = simple_hash(lhs)
    #  rhs = simple_hash(rhs)
    # lhs = lhs.int() if isinstance(lhs,pd.Series) else lhs
    # rhs = rhs.int() if isinstance(rhs,pd.Series) else rhs
    # print(lhs)
    # print(rhs)
    lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2)
    lhs &= 0xFFFFFFFF
    # # lhs = lhs + simple_hash(rhs)
    # lhs = simple_hash(lhs) + rhs
    # lhs &= 0xFFFFFFFF
    return lhs

def add_combine(x,y):
    return x + y

# debug_cnt =0

def combine_row_hash(row,col_names,thresh):
    global debug_cnt
    slist = [row[x].astype(int) for x in col_names]
    # out = reduce(hash_combine, slist[1:], simple_hash(slist[0]))
    out = reduce(hash_combine, slist[1:], slist[0])
    # debug_cnt+=1
    # if (debug_cnt<100):
    #     # print(f"out={out},len={len(col_names)},blocker={blocker}")
    #     print(f"out={out},frac={out/0xFFFFFFFF}")
    # return 1 if (out/0xFFFFFFFF)<=thresh else 0
    return out

def combine_row_add(row,col_names,thresh):
    global debug_cnt
    slist = [row[x].astype(int)/0xFFFFFFFF for x in col_names]
    out = reduce(add_combine, slist[1:], slist[0])
    # debug_cnt+=1
    # if (debug_cnt<100):
    #     print(f"out={out}")
    return out

class SqlLiteDataFrame:
    def __init__(self, do_laundry, csv_input, column_names=None, id_column=None, name_column=None, easy_split=True):
        self.do_laundry = do_laundry
        self.conn = None
        if do_laundry:
            if isinstance(csv_input, str):
                self.csv_path = csv_input
            else:
                self.csv_path = "./data/input.csv"
            self.db_path = os.path.splitext(self.csv_path)[0] + '.sqlite'
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.id_column = id_column or 'ID'
            self.name_column = name_column
            self.column_data = {}  # format：column_name -> (col_id, has_datetime, ...)
            
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            if isinstance(csv_input, pd.DataFrame):
                self.df = csv_input
            else:
                if column_names:
                    self.df = pd.read_csv(csv_input, names=column_names)
                else:
                    self.df = pd.read_csv(csv_input)

            self.target_name = None
            self.easy_split = easy_split
            self.df = split_columns(self.df,easy_split)

            # print(self.df)
            print(self.df.head())

            self.out_df = pd.DataFrame(self.df[self.id_column])
            
            self.initialize_database()
        else:
            self.id_column = id_column or 'ID'
            self.name_column = name_column
            self.target_name = None
            self.easy_split = easy_split
            if isinstance(csv_input, str):
                self.csv_path = csv_input
                self.out_df = pd.read_csv(csv_input)
                print(self.out_df.head())
            else:
                raise Exception("Logic Branch Not Implemented!")

    def initialize_database(self):

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS column_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                null_count INTEGER DEFAULT 0,
                value_count INTEGER DEFAULT 0,
                value_min REAL DEFAULT 1e308,
                value_max REAL DEFAULT -1e308,
                ref_count INTEGER DEFAULT 0,
                word_count INTEGER DEFAULT 0,
                dict_count INTEGER DEFAULT 0
            )
        ''')

        for col_name in self.df.columns:
            if col_name == self.id_column:
                self.df[self.id_column] = self.df[self.id_column].apply(parse_number_string)
                self.out_df[self.id_column] = self.df[self.id_column]
                continue  # Skip the ID column if it's provided
            
            self.cursor.execute('INSERT INTO column_meta (name) VALUES (?)', (col_name,))
            col_id = self.cursor.lastrowid
            
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS col_{col_id}_dict (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL
                )
            ''')
            self.cursor.execute(f'''CREATE UNIQUE INDEX col_{col_id}_dict_text ON col_{col_id}_dict(word)''')
            self.cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS col_{col_id}_value (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    row_id INTEGER,
                    is_null INTEGER NOT NULL DEFAULT 0,
                    value REAL DEFAULT NULL,
                    word_id INTEGER DEFAULT NULL,
                    FOREIGN KEY(word_id) REFERENCES col_{col_id}_dict(id)
                )
            ''')
            self.cursor.execute(f'''CREATE UNIQUE INDEX col_{col_id}_value_row ON col_{col_id}_value(row_id)''')
            self.conn.commit()

            col_name_date = col_name+" as Date"
            col_name_number = col_name+" as Number"
            col_name_word = col_name+" as Word"
            col_name_null = col_name+" as Null"
            col_name_ref = col_name+" as Ref"
            col_name_ref_value = col_name+" as Ref Value"
            col_name_ref_word = col_name+" as Ref Word"
            col_name_value = col_name+" Value"

            self.out_df[col_name_date], \
                self.out_df[col_name_number], \
                self.out_df[col_name_word], \
                self.out_df[col_name_null], \
                self.out_df[col_name_ref] =  zip(*self.df[col_name].apply(parse_data_string))

            has_datetime = self.out_df[col_name_date].notna().any()
            has_numeric = self.out_df[col_name_number].notna().any()
            has_word = self.out_df[col_name_word].notna().any()
            has_null = self.out_df[col_name_null].notna().any()
            has_reference = self.out_df[col_name_ref].notna().any()

            col_meta = {
                'col_id': col_id,
                'name': col_name,
                'has_datetime': has_datetime,
                'has_numeric': has_numeric,
                'has_word': has_word,
                'has_null': has_null,
                'has_reference': has_reference,
                'null_count': 0,
                'value_count': 0,
                'value_min': 1e308,
                'value_max': -1e308,
                'ref_count': 0,
                'word_count': 0,
                'dict_count': 0,
            }

            # print(col_meta['name'],col_meta['has_reference'])

            self.column_data[col_name] = col_meta

            if has_datetime:
                self.out_df[col_name_date] = self.out_df[col_name_date].apply(fill_date,args=(col_meta,))

            if has_numeric:
                self.out_df[col_name_number] = self.out_df[col_name_number].apply(fill_number,args=(col_meta,))

            if has_word:
                self.out_df[col_name_word] = self.out_df[col_name_word].apply(fill_word_id,args=(col_id,self.cursor,self.conn,col_meta,))
            
            if has_null:
                self.out_df[col_name_null] = self.out_df[col_name_null].apply(fill_is_null,args=(col_meta,))

            if has_datetime or has_numeric:
                self.out_df[col_name_value] = self.out_df.apply(fill_row_value,axis=1,args=([col_name_date,col_name_number],col_meta,))

            # print(self.out_df.head())

            if has_reference:
                # print(col_meta)
                # print(self.out_df)
                self.out_df[col_name_ref_value],self.out_df[col_name_ref_word] = zip(*self.out_df[col_name_ref].apply(fill_ref,args=(self.out_df,self.id_column,col_name_value,col_name_word, col_meta,)))
                ref_map={
                    'value_src':col_name_ref_value,
                    'value_dst':col_name_value,
                    'word_src':col_name_ref_word,
                    'word_dst':col_name_word
                }
                # self.out_df.apply(fill_row_ref,axis=1,args=(ref_map,col_meta,))
                # self.out_df.apply(fill_row_ref,args=(ref_map,col_meta,))

                self.out_df[col_name_value],self.out_df[col_name_word] = zip(*self.out_df.apply(fill_row_ref,axis=1,args=(ref_map,col_meta,)))

                # print(self.out_df[col_name_ref])
            # else:
            #     self.out_df.drop(columns=[col_name_ref], inplace=True)

            name_map = {
                'row_id':self.id_column
            }

            if col_meta['null_count'] == 0:
                self.out_df.drop(columns=[col_name_null], inplace=True)
            else:
                name_map['is_null']=col_name_null
            
            if col_meta['value_count'] == 0:
                self.out_df.drop(columns=[col_name_date,col_name_number,col_name_value], inplace=True, errors='ignore')
            else:
                self.out_df.drop(columns=[col_name_date,col_name_number], inplace=True)
                name_map['value']=col_name_value
            
            if col_meta['word_count'] == 0:
                self.out_df.drop(columns=[col_name_word], inplace=True)
            else:
                name_map['word_id']=col_name_word

            if has_reference:
                self.out_df.drop(columns=[col_name_ref,col_name_ref_value,col_name_ref_word], inplace=True)
            else:
                self.out_df.drop(columns=[col_name_ref], inplace=True)

            print(col_meta)
            self.out_df.apply(row_insert,axis=1,args=(col_id,name_map,self.cursor,self.conn,))

            update_col_meta(col_id,col_meta,self.cursor,self.conn)

        # Commit the changes
        self.conn.commit()

    def generate_target(self,target_name="IsDefective", used_ratio=0.5, thresh=0.5, gen_state=None,gen_weight=None, gen_source_columns = None):

        # use_hash = True
        target = None
        tmp=[]
        tmp_weight = None
        if gen_weight is None:
            tmp_weight = [113, 112]
        else:
            tmp_weight = gen_weight

        tar_src = None
        if (gen_source_columns is None):
            # tar_src = self.output_int.sample(frac=used_ratio,axis=1)
            tar_src = self.output_int.sample(frac=used_ratio,axis=1,random_state=gen_state)
        else:
            tar_src = self.output_int[gen_source_columns]
        src_columns = tar_src.columns.tolist()
        self.src_columns = src_columns
        print(self.src_columns)
        for x in range(2):
            tmp_tar = None
            match x:
                case 0:
                    tmp_tar = tar_src.apply(combine_row_hash,axis=1,args=(src_columns,thresh))
                    tmp_tar = tmp_tar.apply(lambda x: 1 if (x/0xFFFFFFFF)<=thresh else 0)
                case 1:
                    tmp_tar = tar_src.apply(combine_row_add,axis=1,args=(src_columns,thresh))
                    compare_val = tmp_tar.mean()
                    tmp_tar = tmp_tar.apply(lambda x: 1 if x<=compare_val else 0)
                case _:
                    continue
            tmp.append(tmp_tar)
        
        target = (tmp_weight[0]*tmp[0]) + (tmp_weight[1]*tmp[1])
        compare_val = target.mean()
        target = target.apply(lambda x: 1 if x<=compare_val else 0)

        self.target_name = target_name
        self.target = target
        print(self.target.head())
        print(f"Target Mean = {target.mean()}")
        print(self.target.tail())

    def to_csv(self):
        target_path = (os.path.splitext(self.csv_path)[0] + '.out.csv') if self.do_laundry else self.csv_path
        out = self.output
        if (self.target_name is not None):
            out[self.target_name] = self.target
        print(out.head())
        out.to_csv(target_path,index=False)

    def to_int_csv(self):
        target_path = (os.path.splitext(self.csv_path)[0] + '.int.csv') if self.do_laundry else self.csv_path
        out = self.output_int
        if (self.target_name is not None):
            out[self.target_name] = self.target
        print(out.head())
        out.to_csv(target_path,index=False)

    def to_normalized_csv(self):
        target_path = (os.path.splitext(self.csv_path)[0] + '.normalized.csv') if self.do_laundry else self.csv_path
        out = self.output_normalized
        if (self.target_name is not None):
            out[self.target_name] = self.target
        print(out.head())
        out.to_csv(target_path,index=False)
    
    @property
    def num_input_columns(self):
        # do not count the id column
        return (len(self.out_df.columns)-1)
    @property
    def output(self):
        return self.out_df.drop([self.id_column], axis=1,errors='ignore')
    @property
    def output_int(self):
        return df_uint_scale(self.output)
    @property
    def output_normalized(self):
        return df_norm(self.output)
                
    def __del__(self):
        if self.conn is not None:
            self.conn.close()

def test():
    a_df = pd.DataFrame({
        'ID': [1, 2 ,3, 4, 5],
        '_Split_': ["TypeA-10", "TypeB-20","TypeC-30","TypeB-40","TypeA-50"],
        '_Date_': ["10/20/2000", "10-21-2000","22/Oct/2000","23-Oct-2000","24 Oct 2000"],
        '_Ref_': ["(ID#:4)", "(ID#:5)","3","4","5"]
        })
    sqlLiteDf = SqlLiteDataFrame(True,a_df)
    sqlLiteDf.generate_target(used_ratio=1.0,thresh=0.3)
    sqlLiteDf.to_csv()
    sqlLiteDf.to_int_csv()
    sqlLiteDf.to_normalized_csv()

# test()
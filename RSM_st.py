import re
import streamlit as st
import time
import pandas as pd
import sys
import os
import pdfplumber
import numpy as np
st.set_page_config(layout="wide",page_title="RSM US",page_icon='🖖')
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

padding = 3
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)




global upload_file
sys.path.append(os.getcwd())

@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(header=None,index=False,encoding='utf-8',errors='ignore')
def convert_df2(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False,encoding='utf-8',errors='ignore')
sys.path.append(os.getcwd())

def get_respective_fields(header,ending_key,ending_column,words_list):
    
    headers=list(filter(lambda x:header in x['text'].lower().rstrip(),words_list))
    headers =  [i for i in headers if header == i['text'].lower().rstrip() ]
    # st.write(headers)
    
    
    if len(headers)==0:
        return 
    ending_keys=list(filter(lambda x:ending_key in x['text'].lower(),words_list))
    ending_keys =  [i for i in ending_keys if ending_key == i['text'].lower().rstrip().rstrip("∆")]
    # st.write(ending_keys)
    
    if len(ending_keys)>1:
        ending_k_min=[i['bottom'] for i in ending_keys]
        endkeymin = np.min(ending_k_min)
        ending_keys= [i for i in ending_keys if i['bottom']==endkeymin ]
        
 
    ending_columns=list(filter(lambda x:ending_column in x['text'].lower(),words_list))

    # st.write(ending_columns)
    x_head=headers[0]
    
    ending_pts=[i for i in ending_columns if i['x0']>x_head['x0']]
    # st.write(ending_pts)
    ending_x_min=[i['x0'] for i in ending_pts]
    # st.write(ending_x_min)
    ending_x_min=sorted(list(set(ending_x_min)))[0]
    min_x0=ending_x_min
#     print(min_x0)
    pt=None
    for point in ending_pts:
        if point['x0']==min_x0:
            if pt is None:
                pt=point
            elif pt['top']>point['top']:
                pt=point
    header= headers[0]
    ending_keys=ending_keys[0]
    bbox=(header['x0'],header['top'],pt['x1'],ending_keys['bottom'])
#     words_list_filtered=[word for word in words_list if word['x0']>=bbox[0] and word['x1']<=bbox[2] and word['top']>=bbox[1] and word['bottom']<=bbox[3]]
    return header,ending_keys,pt,bbox,




def extract(header,ending_key,ending_column):

    table_settings={
    "vertical_strategy": "text", 
    "horizontal_strategy": "text",
     "keep_blank_chars": True,}
    # box= []
    page = []
    for i in page_cnt:
        pdf = pdfplumber.open(upload_file)
        text = pdf.pages[i].extract_words(use_text_flow=True,keep_blank_chars=True)
        p0 = pdf.pages[i]

        out = get_respective_fields(he, endk,endc,text)
        if out == None:
            pass
       
        else:
            # st.write(i+1)

#             page.append(i)
            # box.append(out[-1])
            box2 = out[-1]
            # st.write(box2)
            
            image =p0.crop(box2)
            te = image.extract_tables(table_settings)
            # st.write(te)
            te[0][0][0]=te[0][0][0]+str(" --> page")+str(i+1)
            # st.write(te)
            page.append(te)
  


            
    return page




def extract_tru(header,ending_key,ending_column,lengths):

    table_settings={
    "vertical_strategy": "text", 
    "horizontal_strategy": "text",
     "keep_blank_chars": True,}
    # box= []
    page = []
    for i in page_cnt2:
        pdf = pdfplumber.open(upload_file)
        text = pdf.pages[i].extract_words(use_text_flow=True,keep_blank_chars=True)
        p0 = pdf.pages[i]

        out = get_respective_fields(header,ending_key,ending_column,text)
        if out == None:
            pass
       
        else:
            # st.write(i+1)

#             page.append(i)
            # box.append(out[-1])
            box2 = out[-1]
            truncate = list(box2)
            truncate[2] = truncate[2]+lengths
            box2 = tuple(truncate)
            
            image =p0.crop(box2)
            te = image.extract_tables(table_settings)
            # st.write(te)
            te[0][0][0]=te[0][0][0]+str(" --> page")+str(i+1)
            # st.write(te)
            page.append(te)
  


            
    return page

def dataframe(data):
    ls2 = []
    for i in data:
        for j in i:
            ls2.append(pd.DataFrame(j))
            
            # st.write(ls2)
            final_df = pd.concat(ls2,ignore_index=True)

            final_df.index = list(range(1, len(final_df) + 1))


    return final_df    




def multi():
    
    s=[]
    global he
    global endk
    global endc
    # fname = upload_file.name.rstrip("pdf").rstrip("PDF").rstrip(".")+str("_output")+str(".xlsx")

    writer = pd.ExcelWriter(fname,engine='xlsxwriter')
    for i in range(len(rowss)):
        he, endk,endc = rowss[i]

        s.append(dataframe(extract(he, endk,endc)))
        dfs = s[i]
        # csv = convert_df(dfs)
        st.write(dfs)
        st.download_button(label="Download data as CSV",data=convert_df(s[i]),file_name='pdf_output.csv' ,mime='text/csv',)

        s[i].to_excel(writer,sheet_name=he,header=None,index=False)

    return writer.save()



def multi_tru(lengths):
    
    # sx=[]
    global he1
    # global endk
    # global endc
    # fname = upload_file.name.rstrip("pdf").rstrip("PDF").rstrip(".")+str("_output")+str(".xlsx")

    # writer = pd.ExcelWriter(fname,engine='xlsxwriter')
    # st.write(zx)
    # for i in range(len(zx)):
    he1, endk1,endc1 = zx

    # sx.append(dataframe(extract_tru(he1, endk1,endc1,lengths)))
    gre = dataframe(extract_tru(he1, endk1,endc1,lengths))
    # dfs1 = sx
    # st.write(type(dfs1))
    # csv = convert_df(dfs)
    # st.write(dfs1)
    st.write(gre)
    st.download_button(label="Download data as CSV",data=convert_df(gre),file_name='pdf_output.csv' ,mime='text/csv',)

    # sx[i].to_excel(writer,sheet_name=he1,header=None,index=False)
    # writer.save()

    return 

def cleanse_key(key):
    if type(key) == str:
        key = key.lower()
        key = key.replace('-', ' ')
        key = key.replace('/', ' ')
        replace_chars = ['&', ',', '(', ')', '_', '�']
        for i in replace_chars:
            key = key.replace(i, '')
    return key


def get_points(my_list):
    # output will be x0,y0,x1,y1
    if type(my_list) != list:
        my_list = [my_list]
    x0 = my_list[0]['x0']
    y0 = my_list[0]['y0']
    x1 = my_list[-1]['x1']
    y1 = my_list[-1]['y1']
    return x0, y0, x1, y1


def find_lexical_sequence(indexes, w_indexes):
    if len(indexes) == 1:
        return w_indexes
    out = []
    s = indexes[0]
    for idx, i in enumerate(indexes[1:]):
        if s - i == -1:
            out.append(w_indexes[idx + 1])
        else:
            out = []
        s = i
    if len(out) > 0:
        out = [out[0] - 1] + out
    return out


def find_lexical_sequence_1(indexes, w_indexes, value_check=False):
    if len(indexes) == 0:
        return []
    check = len(set(indexes))
    out = []
    buffer = []
    s = indexes[0]
    for idx, i in enumerate(indexes[1:]):
        if s - i == -1:
            buffer.append(w_indexes[idx + 1])
        else:
            if len(buffer) > 0:
                out.append([w_indexes[w_indexes.index(buffer[0]) - 1]] + buffer)
            buffer = []
        s = i
    if len(buffer) > 0 and buffer not in out:
        out.append([w_indexes[w_indexes.index(buffer[0]) - 1]] + buffer)
    main_out = []
    for l in out:
        if len(l) == check:
            if not value_check:
                if max(l) - min(l) <= len(l) - 1:
                    main_out.append(l)
            else:
                if max(l) - min(l) <= 10:
                    main_out.append(l)

    if value_check and len(sorted(set(indexes))) == 1 and len(main_out) == 0:
        for i in w_indexes:
            main_out.append([i, i])
    return main_out


def clean_row_points(row_points, word_list, word):
    final_row_points = []
    row_points_key = list(row_points.keys())[0]
    ignore_words = ['and']
    for row in row_points[row_points_key]:
        word_found = word_list[row[-1][0]:row[-1][-1] + 1]
        w_split = [i for i in word.split(" ") if len(i) > 0]
        matched = 0
        unmatched = 0
        for word_x in cleanse_key(" ".join(word_found)).split():
            word_x = word_x.strip()
            if word_x in ignore_words or len(word) == 0:
                continue
            elif word_x in word:
                matched += 1
            else:
                unmatched += 1
        if matched == len(w_split) and unmatched == 0:
            final_row_points.append(row)
    row_points[row_points_key] = final_row_points
    return row_points


def look_for_key_vals(word_list, word, words_with_location):
    word_split = [i for i in word.split(" ") if len(i) > 0]
    indexes = []
    w_indexes = []
    w_c = word
    for word_idx, word in enumerate(word_list):
        if word in word_split:
            w_indexes.append(word_idx)
            indexes.append(word_split.index(word))
        else:
            for word_x in word_split:
                if word_x in word:
                    w_indexes.append(word_idx)
                    indexes.append(word_split.index(word_x))
    if len(word_split) != len(set(indexes)):
        return ''
    out = find_lexical_sequence_1(indexes, w_indexes, value_check=True)
    row_points = dict()
    row_points[w_c] = []
    for i in out:
        pts = get_points([words_with_location[i[0]], words_with_location[i[-1]]])
        main_pts = (pts, (i[0], i[-1]))
        row_points[w_c].append(main_pts)
    if len(row_points) > 0:
        print(row_points)
        row_points = clean_row_points(row_points, word_list, w_c)
    return row_points


def look_for_columns(word_list, word, words_with_location):
    print()
    print(word, 'I am column')
    w_c = word
    word_split = [i for i in word.split(" ") if len(i) > 0]
    indexes = []
    w_indexes = []
    for word_idx, word in enumerate(word_list):
        if word in word_split:
            w_indexes.append(word_idx)
            indexes.append(word_split.index(word))
    out = find_lexical_sequence_1(indexes, w_indexes, value_check=True)
    print(out)
    col_points = dict()
    col_points[w_c] = []
    for col in out:
        pts = get_points([words_with_location[col[0]], words_with_location[col[-1]]])
        main_pts = (pts, (col[0], col[-1]))
        col_points[w_c].append(main_pts)
    # print('***** col points *****')
    # print(col_points)
    # print('-' * 10)
    return col_points


def detect_values_1(row_points, column_points, word_list, final_words):
    matches = []
    if type(row_points) != dict:
        return matches
    row_match = []
    col_vals = list(column_points.values())[0]
    col_x0 = ([i[0] for i in col_vals])
    sorted_x0 = sorted(col_x0)
    new_col_vals = [col_vals[col_x0.index(i)] for i in sorted_x0]
    row_vals = list(row_points.values())[0]
    word = list(row_points.keys())[0]
    col_vals = new_col_vals
    for row in row_vals:
        row_x0 = row[0][0]
        row_x1 = row[0][2]
        row_y0 = row[0][1]
        row_y1 = row[0][-1]
        start = row[-1][-1] + 1
        for col in col_vals:
            if row[-1] not in row_match:
                col_x0 = col[0][0]
                col_x1 = col[0][2]
                min_x = min(col_x0, col_x1)
                max_x = max(col_x0, col_x1)
                for word_index, word_element in enumerate(final_words[start:]):
                    word_x0 = word_element['x0']
                    word_x1 = word_element['x1']
                    word_y0 = word_element['y0']
                    word_y1 = word_element['y1']
                    word_center = word_x0 + ((word_x1 - word_x0) / 2)
                    word_y_center = word_y0 + ((word_y1 - word_y0) / 2)
                    append_idx = start + word_index
                    if append_idx in matches:
                        continue
                    elif min_x < word_center < max_x and row_y0 < word_y_center < row_y1:
                        #                         print("x_center:",word_center,"y_center:",word_y_center)
                        #                         print("x_boundaries:",col_x0,col_x1)
                        #                         print("y_boundaries:",row_y0,row_y1)
                        matches.append(start + word_index)
                        print(word_element)
                        row_match.append(row[-1])
                        break
            else:
                break
    print(matches)
    print(row_match)
    return matches


def get_values(word_list, words_with_location):
    table_header_found = ''
    for table_header in table_headers:
        if type(table_header) == str:
            word_split = [i for i in table_header.split(" ") if len(i) > 0]
            indexes = []
            w_indexes = []
            for word_idx, word in enumerate(word_list):
                if word in word_split:
                    w_indexes.append(word_idx)
                    indexes.append(word_split.index(word))
            if len(indexes) > 0:
                lex_out = find_lexical_sequence_1(indexes, w_indexes)
                if len(lex_out) > 0 and len(lex_out[0]) == len(word_split):
                    table_header_found = table_header
                    break
    if len(table_header_found) > 0:
        print(f'found this header {table_header_found} and lex is {lex_out}')
        search_keys = reference[reference['PDF Table Header'].isin(table_headers)]
        search_columns = reference[reference['PDF Table Header'].isin(table_headers)]['columns_to_extract']
        search_keys = search_keys['PDF Key'].to_list()
        final_output = {}
        if len(search_columns) > 0:
            search_columns = [i for i in search_columns.to_list() if type(i) == str]
            need_to_search = []
            for i in search_columns:
                for j in i.split(','):
                    j = j.strip()
                    if j not in need_to_search:
                        need_to_search.append(j)
            print(need_to_search, '*******************')
            for col in need_to_search:
                c_pts = look_for_columns(word_list, col, words_with_location)
                print(search_keys)
                a = {}
                v = []
                z = []
                v_m = {}
                for word in search_keys:
                    print('*************')
                    print(word)
                    print(c_pts)
                    output = look_for_key_vals(word_list, word, words_with_location)
                    word_out = detect_values_1(output, c_pts, word_list, words_with_location)
                    if word_out is not None and len(word_out) > 0:
                        print(word)
                        print(word_list[word_out[0]])
                        if word not in final_output:
                            final_output[word] = {}
                        final_output[word][col] = word_list[word_out[0]]
                        print('///////////////')
                    else:
                        print('---------------')
        print(':) final output')
        print(final_output)
        if len(final_output) > 0:
            subset = reference[['PDF Table Header Org', 'PDF Key Org', 'PDF Key', 'columns_to_extract_org',
                                'columns_to_extract', 'Classification category']].copy()
            subset['key_values'] = subset['PDF Key'].apply(lambda x: final_output.get(x))
            # print(type(subset.key_values.iloc[0]))
            buffer = []
            for i in range(len(subset)):
                key = subset['columns_to_extract'].iloc[i]
                val_store = subset['key_values'].iloc[i]
                if type(val_store) == dict:
                    buffer.append(val_store.get(key))
                else:
                    buffer.append(None)
            subset['needed_values'] = buffer
            return subset
        #     if output is not None and len(output) > 0:
        #         idx = row_index.index(word)
        #         a[word] = output
        #         v.append(idx)
        #         v_m[idx] = output
        # if len(a) > 0:
        #     values = list(range(len(reference)))
        #     temp = []
        #     # print(values)
        #     for i in values:
        #         if i in v_m:
        #             temp.append(v_m[i])
        #         else:
        #             temp.append(None)
        #     subset = reference[['PDF Table Header Org', 'PDF Key Org']].copy()
        #     subset['Extracted_values'] = temp
        #     return subset

        # look_for_key_vals(word_list, search_keys[1])




def run_extraction(file_name, reference_name):
    global reference
    global table_headers
    
    global page_cnt
    import random
    global patyh
    global ss
    rn1 = random.randrange(0, 10000, 3)
    pdf_ob = pdfplumber.open(file_name)
    page_cnt = list(range(len(pdf_ob.pages)))
    reference = pd.read_csv(reference_name)
    if reference['PDF Key'].isnull().sum() >=1:
   
        ss = reference[~reference['PDF Key'].notnull()]
        ss = ss[['PDF Table Header','Table_ending_key', 'Table_ending_column']]
        ss = ss.reset_index(drop= True)
        ss = ss.applymap(str.lower)
    reference = reference[reference['PDF Key'].notnull()]

    if len(reference[reference['PDF Key'].notnull()])>=1:
    
        reference = reference[['Brokerage', 'PDF Table Header', 'PDF Key', 'Classification category','columns_to_extract',]]
        reference = reference.reset_index(drop= True) 
        reference.drop_duplicates(inplace=True)
        reference['PDF Table Header Org'] = reference['PDF Table Header']
        reference['PDF Table Header'] = reference['PDF Table Header'].apply(lambda x: cleanse_key(x))
        reference['PDF Key Org'] = reference['PDF Key']
        reference['PDF Key'] = reference['PDF Key'].apply(lambda x: cleanse_key(x))
        reference['columns_to_extract_org'] = reference['columns_to_extract']
        reference['columns_to_extract'] = reference['columns_to_extract'].apply(
            lambda x: x.lower() if type(x) == str else x)
        global row_index
        row_index = reference['PDF Key'].to_list()
        table_headers = []
        for i in reference['PDF Table Header'].to_list():
            if type(i) == str and i not in table_headers:
                table_headers.append(i)
        final_out = []
        for i in page_cnt:
            print(f'page number in pdf is {i + 1} and in code is {i} ****************')
            page_2 = pdf_ob.pages[i].extract_words(extra_attrs=['y0', 'y1'])
            words = [i for i in page_2 if len(i['text'].strip()) > 1]
            final_words = []
            new_words = []
            for word in words:
                word_x = word['text'].lower()
                if word_x.isalpha() and len(set(word_x)) == 1:
                    continue
                else:
                    new_words.append(word)
                    final_words.append(word_x)
            words = new_words
            out = get_values(final_words, words)
            if out is not None:
                out['Page Number'] = i + 1
                final_out.append(out)
        if len(final_out) > 0:
            final_out = pd.concat(final_out)
            final_out.dropna(subset=['needed_values'], inplace=True)
            final_out['Final Values'] = final_out['needed_values']
            final_out = final_out[
                ['PDF Table Header Org', 'PDF Key Org', 'columns_to_extract_org', 'Classification category', 'Final Values',
                 'Page Number']]
            final_out.index = list(range(1, len(final_out) + 1))
            
            csv = convert_df(final_out)
            st.dataframe(final_out)
            patyh = "output"+str(rn1)+".csv"
            # cf = st.button("Confirm And Verify")
            # if cf:
                
            final_out.to_csv(patyh)
            st.download_button(label="Download The Output",data=csv,file_name='pdf_output.csv' ,mime='text/csv',)
            Run_number = str(rn1)
            project_id = str(int(pp))
            input_loc  = ref 
            output_file_name = str("output:1->>  ")+str(patyh)
            Status = str("success")
            if Status:
                st.success("successfully extracted key_values")
                st.write(pd.DataFrame({'Key_Name': ["username","Run_number: " ,"project_id:","Input_File_location:","output_file_names"],'Value_Name': [username,Run_number,project_id,input_loc,output_file_name],}))
                add_project_history(username,Run_number,project_id,input_loc,output_file_name,Status)

                    
                    
                    
                    
    else:
        st.warning("No keys to extract values")

            # elif ss is None:
            #     st.write("No Keys to Extract Table data")
            #     su = st.success("Extraction completed successfully")
                        
            #     # create_project_history()
            #     Run_number = str(rn2)
            #     project_id = str(int(pp))
            #     input_loc  = ref 
            #     output_file_name = str("output:1->>  ")+str(patyh)
            #     Status = str("success")
            #     if Status:
            #         st.write(pd.DataFrame({'Key_Name': ["username","Run_number: " ,"project_id:","Input_File_location:","output_file_names"],'Value_Name': [username,Run_number,project_id,input_loc,output_file_name],}))
            #         add_project_history(username,Run_number,project_id,input_loc,output_file_name,Status)










def tab_ext():
    global rowss
    if ss is not None:

        st.warning("Table Extracton Started Please Wait...")
        rowss = ss.values.tolist()
        # st.write(rowss)
        multi()
        # su = st.success("Extraction completed successfully")
            
    # create_project_history()
        Run_number = str(rn)
        project_id = str(int(pp))
        input_loc  = ref 
        # output_file_name = str("output:1->>  ")+str(patyh)+str("<<>>")+str("output:2->> ")+str(fname)
        output_file_name = str("output:2->> ")+str(fname)
        Status1 = str("success")
        # cf1 = st.button("Confirm And Verify")
        if Status1:
            st.write(pd.DataFrame({'Key_Name': ["username","Run_number: " ,"project_id:","Input_File_location:","output_file_names"],'Value_Name': [username,Run_number,project_id,input_loc,output_file_name],}))
            add_project_history(username,Run_number,project_id,input_loc,output_file_name,Status1)

    elif ss is None:
        st.write("No keys to extract table")

def truncate_tb(trt):
    global zx
    heads1 = st.text_input("Enter Table Name to get Truncate Values")
    if heads1:
        lengths = st.text_input("Enter length of Table right side")
        if lengths:
            lengths = int(lengths)
            heads1 = heads1.lower()
            zx = [i for i in trt if heads1 in i]
            zx = zx[0]
            # st.write(zx)

            multi_tru(lengths)
















reg = "^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{6,20}$"
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable1(username TEXT,email Text,password TEXT)')



def add_userdata(username,email,password):
    c.execute('INSERT INTO userstable1(username,email,password) VALUES (?,?,?)',(username,email,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable1 WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable1')
    data = c.fetchall()
    return data

def delete_user(users):
    c.execute('DELETE FROM userstable1 WHERE username = ?',(users))
    conn.commit()


def create_project_table():
    c.execute('CREATE TABLE IF NOT EXISTS project3(project_id INTEGER PRIMARY KEY AUTOINCREMENT,username Text ,project_name TEXT,File_Location Text)')

def add_project_data(username,project_name,File_Location):
    c.execute('INSERT INTO project3(username,project_name,File_Location) VALUES (?,?,?)',(username,project_name,File_Location))
    conn.commit()

def delete_project(pro_id):
    c.execute('DELETE FROM project3 WHERE project_id = ?',(pro_id))
    conn.commit()
    # conn.close()

def view_all_projects():
    c.execute('SELECT * FROM project3')
    data = c.fetchall()
    return data




def update_project(location,pr_id):
    c.execute('UPDATE project3 SET File_Location = ? WHERE project_id = ?',(location,pr_id))
    conn.commit()

def create_project_history():
    c.execute('CREATE TABLE IF NOT EXISTS history1(username Text,Run_number TEXT,project_id TEXT,input_loc TEXT,output_file_name Text,Status TEXT)')

def view_all_his():
    c.execute('SELECT * FROM history1')
    data = c.fetchall()
    return data
def add_project_history(username,Run_number,project_id,input_loc,output_file_name,Status):
    c.execute('INSERT INTO history1(username,Run_number,project_id,input_loc,output_file_name,Status) VALUES (?,?,?,?,?,?)',(username,Run_number,project_id,input_loc,output_file_name,Status))
    conn.commit()

def save_uploadedfile(path,uploadedfile):
     with open(os.path.join(path,uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     
def main():
    global fname
    global rn
    global rn2
    global pp
    global ref
    global username
    global page_cnt2
    
    import random
    
    rn2 = random.randrange(0, 100, 1)
    global upload_file
    st.title("Transaction Classification Tool RSM US")
    st.sidebar.image("ifusion.jpg", use_column_width="auto")
    menu = ["Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)

    # if choice == "Home":
    #     st.subheader("Home")
    #     st.warning("Please Find All Operations On << Sidebar")

    if choice == "Login":
        # st.title("Login")
        # st.subheader("Login Section")


        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):

            # lg = st.sidebar.button("Logout")

            # if lg :
            #     st.write("Logout")

            # else:
                # if password == '12345':
            create_usertable()
            create_project_history()
            create_project_table()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:

                st.success("Logged In as {}".format(username))

                task = st.sidebar.selectbox("Project list",["Deselect","Create project","Modify project"])
                
                if task == "Deselect":
                    print("")

                elif task == "Create project":
                    
        
                    p_name = st.text_input("Enter project name")
                    if p_name:

                        saver = st.file_uploader('Please Upload Your Mapping File',type=['csv'])
                        if saver is not None:
                            st.success("File Uploaded")
                            file_details = saver.name
                            df  = pd.read_csv(saver)
                            path2 = "csvdump/"+str(username)
                            os.makedirs(path2 ,exist_ok=True)
                            ot = os.path.join("csvdump/"+str(username)+"/"+str(file_details))
                            # save_uploadedfile(saver)
                            df.to_csv(ot)
                            # st.write(file_details)
                            Sve = st.button("save")
                            if Sve:
                                add_project_data(username,p_name,file_details)
                                project_result = view_all_projects()
                                clean_pr = pd.DataFrame(project_result,columns=["project_id","username","project_name","File_Location"])
                                ref = clean_pr.values.tolist()
                                ref = ref[-1]
                                ref = ref[0]
                                st.success("saved project_id  {} successfully".format(ref))





                elif task =="Modify project":
                    st.subheader("View/Delete Mapping Record")
                    task1 = st.selectbox("Task list",['View','Delete'])
                    # if task1 == "Modify":
                    #     st.subheader("Modify project")

                    #     upd = st.text_input(r"please enter updated file location")
                    #     ids = st.text_input("pleaser enter project_id")
                    #     # st.write(type(ids))
                    #     if ids:
                    #         ut = st.button("Update")
                    #         if ut:
                    #             update_project(upd,ids)
                    #             st.success(" modified mapping file successfully")


                    if task1 =='View':
                        st.subheader("View projects")
                        project_result = view_all_projects()
                        clean_pr = pd.DataFrame(project_result,columns=["project_id","username","project_name","File_Location"])
                        clean_pr = clean_pr[clean_pr ['username']==username]
                        view = st.button("Click to view all projects")
                        if view:

                            st.dataframe(clean_pr)
                            
                    elif task1 =='Delete':
                        st.subheader("Delete project")
                        del_p = st.text_input("Please enter project_id to delete")
                        if del_p :
                            delete_project(del_p)
                            st.warning("project_id {} Deleted Successfully".format(del_p))




                    
                   
                        






              

                tasknew = st.sidebar.selectbox("Data Extraction",["Deselect","Extraction","Truncate","Run_History"])
                if tasknew == "Deselect":
                    print("")
                elif tasknew == "Run_History":
                        hs = view_all_his()
                        clean_p = pd.DataFrame(hs,columns=["username","Run_number","project_id","input_loc","output_file_name","Status"])
                        clean_p = clean_p[clean_p['username']==username]
                        st.dataframe(clean_p)


                elif tasknew =="Extraction":
                    st.subheader("Extraction")

                    pp=st.text_input("Please Enter project_id")

                    if pp:

                        project_results = view_all_projects()
                        
                        
                        clean_id = pd.DataFrame(project_results,columns=["project_id","username","project_name","File_Location"])
                        
                        match = clean_id[clean_id['project_id']==int(pp)]
                        
                        st.dataframe(match)
                        st.warning("Please Dont Upload PDF File If project_id is Empty")
                        st.warning("Please Upload Your Brokerage PDF File Matches to Mapping File")
                        ref = match.values.tolist()
                        ref = ref[-1]
                        ref = ref[3]
                        # st.write(ref)
                        ler = 'C:/Users/IM-RT-LP-823/Desktop/updated_RSM_Streamlit_new/csvdump'+"/"+str(username)
                        # ler2 = file_details
                        ll= ler+str("/")+str(ref)
                        ld = pd.read_csv(ll)
                        ld = ld[['PDF Table Header', 'PDF Key', 'columns_to_extract', 'Classification category','Table_ending_key', 'Table_ending_column']]
                        st.success("Mapping File")
                        st.write(ld)
                        st.header("RSM")
              
                        upload_file = st.file_uploader('Please Upload Your Brokerage  PDF File')
                       
                        if upload_file:
                            st.success("File Uploaded")
                            rn = random.randrange(0, 10000, 2)
                            path3 = "pdfdump/"+str(username)
                            os.makedirs(path3 ,exist_ok=True)

                            save_uploadedfile(path3,upload_file)
                            fname = upload_file.name.rstrip("pdf").rstrip("PDF").rstrip(".")+str("_output")+str(rn)+str(".xlsx")

                           
                            # bar = st.progress(100)
                            # latest_iteration = st.empty()
                            # for i in range(100):
                            #     latest_iteration.text(f'Loading {i+1}')
                            #     bar.progress(i + 1)
                            #     time.sleep(0.001)
                            # with st.spinner("Loading the file"):
                            #     time.sleep(1)
                            #     st.success("File Uploaded")
                            run_extraction(upload_file,ll)
                            tab_ext()
                            
                elif tasknew =="Truncate":
                    st.subheader("Extraction ++")
                    pp=st.text_input("Please Enter project_id")

                    if pp:

                        project_results = view_all_projects()
                        
                        
                        clean_id = pd.DataFrame(project_results,columns=["project_id","username","project_name","File_Location"])
                        
                        match = clean_id[clean_id['project_id']==int(pp)]
                        
                        st.dataframe(match)
                        st.warning("Please Dont Upload PDF File If project_id is Empty")
                        st.warning("Please Upload Your Brokerage PDF File Matches to Mapping File")
                        ref = match.values.tolist()
                        ref = ref[-1]
                        ref = ref[3]
                        # st.write(ref)
                        ler = 'C:/Users/IM-RT-LP-823/Desktop/updated_RSM_Streamlit_new/csvdump'+"/"+str(username)
                        # ler2 = file_details
                        ll2= ler+str("/")+str(ref)
                        ld = pd.read_csv(ll2)
                        trt = ld[~ld['PDF Key'].notnull()]
                        trt= trt[['PDF Table Header','Table_ending_key', 'Table_ending_column']]
                        trt = trt.applymap(str.lower)
                        st.write(trt)
                        # st.write(trt)

                        st.header("RSM")
                        trt = trt.values.tolist()
                        # st.write(trt)
              
                        upload_file = st.file_uploader('Please Upload Your Brokerage PDF File')
                         
                       
                        if upload_file:
                            pdf_ob2 = pdfplumber.open(upload_file)
                            page_cnt2 = list(range(len(pdf_ob2.pages)))
                            truncate_tb(trt)

     
                        
                    

                            
                taskn = st.sidebar.selectbox("Users",["Deselect","Profiles"])
                if taskn == "Deselect":
                    print("")

                elif taskn == "Profiles":
                    st.subheader("User Profiles")
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["Username",'Email',"Password"])
                    clean_db = clean_db[clean_db['Username']==username]
                    st.dataframe(clean_db)
                    # st.subheader("Delete user")
                    # de = st.text_input("Please enter username")
                    # if de :
                    #     delete_user(de)
                    #     st.warning("username  {}  deleted successfully".format(de))

            else:
                st.warning("Incorrect Username/Password")





    elif choice == "SignUp":
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        Email = st.text_input('Email')
        new_password = st.text_input("Password",type='password')
        confirm_password = st.text_input("confirm Password",type='password')
        if confirm_password:
            pat = re.compile(reg)
            mat = re.search(pat, new_password)
        # if mat:
        #     print("")
        # else:
        #     # st.warning(")

        if st.button("Signup"):
            if new_password == confirm_password and mat:

                create_usertable()
                add_userdata(new_user,Email,make_hashes(new_password))
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")
            else:
                st.warning("Password not matched or Please enter Password(atleast 1.lowercase 1.uppercase 1.special char")




if __name__ == '__main__':
    main()
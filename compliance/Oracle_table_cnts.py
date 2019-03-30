import zipfile
import pandas as pd
import datetime
import os

def remove_file(filename):
    ## delete only if file exists ##
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("Unable to remove, %s file." % filename)

def is_string_part_of(input_strinc, target):
    return target in input_strinc

def get_table_name_rows_from_file(input_file):
    colnames = ["OWNER", "TABLE_NAME", "NUM_ROWS", "BLOCKS", "AVG_ROW_LEN", "CHAIN_CNT", "TIMESTAMP",
                "SAMPLE_SIZE SAMPLE_PERCENT", "e1", "e2"]
    df = pd.read_csv(input_file, skiprows=3, header=None, delim_whitespace=True, names=colnames, engine='python')
    df = df[["TABLE_NAME", "NUM_ROWS"]]
    #print(df.head(20))
    return df

def add_coumns_to_df(df,Cust_num,Case_num,Customer_name,Upload_date):
    df['Cust_Num']=Cust_num
    df['Case_Num']=Case_num
    df['Customer_Name']=Customer_name
    df['Upload_Date']=Upload_date
    return df

def save_data_file(df,data_dir,file_name):
    current_date=datetime.datetime.today().strftime('%d-%m-%Y')

file="/Users/Stephen/Downloads/oracle_perf_html.zip"
unzip_dir="/tmp/unzips"
data_dir="/tmp/table_data"
zip_ref = zipfile.ZipFile(file, 'r')

for file_in_zip in zip_ref.namelist():
    if (is_string_part_of(file_in_zip,"Tables.txt")):
        print ("Extracting file:  ",file_in_zip,"\n")
        zip_ref.extract(file_in_zip,unzip_dir)
        file_to_read_in=unzip_dir+"/"+file_in_zip

        df=get_table_name_rows_from_file(file_to_read_in)
        df=add_coumns_to_df(df,341,333,'PTC','1-2-2019')

        print (df.columns)
        print(datetime.datetime.today().strftime('%d-%m-%Y'))
        #remove_file(file_to_read_in)
zip_ref.close()
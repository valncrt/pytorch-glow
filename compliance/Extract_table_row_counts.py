import zipfile
import pandas as pd
import datetime
import os
import requests, zipfile, io
import cx_Oracle
import re
import codecs
import urllib

pd.options.mode.chained_assignment = None

def remove_file_temp_file(filename):
    ## delete only if file exists ##
    if os.path.exists(filename):
        os.remove(filename)
    else:
        print("Unable to remove, %s file." % filename)

def is_string_part_of(input_strinc, target):
    return target in input_strinc

def get_table_name_rows_from_file_oracle(input_file):
    colnames = ["OWNER", "TABLE_NAME", "NUM_ROWS", "BLOCKS", "AVG_ROW_LEN", "CHAIN_CNT", "TIMESTAMP",
                "SAMPLE_SIZE SAMPLE_PERCENT", "e1", "e2"]
    df = pd.read_csv(input_file, skiprows=3, header=None, delim_whitespace=True, names=colnames, engine='python')
    df = df[["TABLE_NAME", "NUM_ROWS"]]
    #print(df.head(20))
    return df

def get_table_name_rows_from_file_sqlserver(input_file):
    colnames = ["TABLE_NAME", "NUM_ROWS", "DataSpace MB", "IndexSpace MB", "Data Compression", "inMemory",
                "SAMPLE_SIZE SAMPLE_PERCENT", "e1", "e2"]
    df = pd.read_csv(input_file, skiprows=3, header=None, delim_whitespace=True, names=colnames, engine='python')
    df = df[["TABLE_NAME", "NUM_ROWS"]]
    #print(df.head(20))
    return df

def get_table_name_rows_from_file_sqlserver_gather_info(input_file):
    colnames = ["TABLE_NAME", "NUM_ROWS", "DataSpace MB", "IndexSpace MB", "Data Compression", "inMemory",
                "SAMPLE_SIZE SAMPLE_PERCENT", "e1", "e2","e3","e4","e5"]
    df = pd.read_csv(input_file, skiprows=3, header=None, delim_whitespace=True, names=colnames, engine='python')
    df = df[["TABLE_NAME", "NUM_ROWS"]]
    #print(df.head(20))
    return df
def add_coumns_to_df(df,cust_num,case_num,company_name,upload_date):
    df['Cust_Num']=cust_num
    df['Case_Num']=case_num
    df['Company_Name']=company_name
    df['Upload_Date']=upload_date
    return df

def remove_unneeded_lines(df):
    to_drop = ['----']
    #print ("remove_unneeded_lines",df.columns,df.head())
    df=df[~df['TABLE_NAME'].astype(str).str.contains('|'.join(to_drop))]
    return df

def remove_commas_from_table_row_counts(df):
    print(df['TABLE_NAME'].head(10))
    #df['NUM_ROWS'] = df['NUM_ROWS'].replace(np.nan, '', regex=True)
    df['NUM_ROWS']=df['NUM_ROWS'].astype(str).str.replace(',','')
    #df.NUM_ROWS = pd.to_numeric(df.NUM_ROWS, errors='coerce',downcast='integer')
    return df

def save_data_file(df,data_dir,cust_num):
    current_date=datetime.datetime.today().strftime('%m-%d-%Y-%f')
    file_name=cust_num+"-"+current_date
    columns=["Table_name","CNT","CUST_NUM","CASE_NUM","COMPANY_NAME","FILE_UPLOAD_DATE"]
    df=remove_unneeded_lines(df)
    df= remove_commas_from_table_row_counts(df)
    print ("Saving file ...: ",data_dir+"/"+file_name)
    df.to_csv(data_dir+"/"+file_name,index=False,index_label=columns)

def get_file_from_url(link_to_attachement,unzip_dir):
    if(link_to_attachement.find("http")==0 ): #file is a URL
        temp_file=unzip_dir +"/"+"zip_temp.zip"
        print ("Making tempory file:", temp_file)
        #urllib.request.urlretrieve(link_to_attachement, temp_file)
        try:
            res = requests.get(link_to_attachement,stream=True)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:  # This is the correct syntax
            print (e)
            pass
            #continue
        with open(temp_file, 'wb') as playFile:
            for chunk in res.iter_content(1024):
                playFile.write(chunk)
        return temp_file
    else:
        return link_to_attachement #file is local (for testing)

def sqlserver_or_oracle(file_name):
    with codecs.open(file_name, 'r', encoding='utf-8',errors='ignore') as file:
        head = [next(file) for x in range(10)]
        print ("head: ", head)
        print("Inside sqlserver_or_oracle")
        head = ''.join(head)  #Convert from list to string for searching
        if "OWNER" in head: #Owner is only in Oracle Tables file
            print("Found an Oracle Tables Report ...")
            return True
        else:
            print ("Found a SQL Server Tables Rerport ...")
            return False
    print(head)

def is_a_sql_server_report(file_name,number_of_rows_to_check=100):
    num_lines_in_file = sum(1 for line in codecs.open(file_name, 'r', encoding='utf-8',errors='ignore'))
    if (num_lines_in_file >number_of_rows_to_check):
        with codecs.open(file_name, 'r', encoding='utf-8',errors='ignore') as file:
            head = [next(file) for x in range(number_of_rows_to_check)]
            print ("head: ", head)
            head = ''.join(head)  #Convert from list to string for searching
            if "Microsoft SQL Server" in head: #
                print("Microsoft SQL Server Report  ...")
                return True
            else:
                return False
    else:
        return False

def is_an_oracle_report(file_name, number_of_rows_to_check=10):
    num_lines_in_file = sum(1 for line in codecs.open(file_name, 'r', encoding='utf-8',errors='ignore'))
    if(num_lines_in_file>number_of_rows_to_check):
        with codecs.open(file_name, 'r', encoding='utf-8',errors='ignore') as file:
            head = [next(file) for x in range(number_of_rows_to_check)]
            #print ("head: ", head)
            head = ''.join(head)  #Convert from list to string for searching
            if "Oracle Database " in head: #
                print("Oracle Report  ...")
                return True
            else:
                return False
    else:
        return False
    #print(head)

def extraxt_file_from_zip(file_in_zip,zip_ref):
    error_code=0
    print("Checking file:  ", file_in_zip, "\n")
    try:
        zip_ref.extract(file_in_zip, unzip_dir)
    except:
        error_code=1
        pass
    file_to_read_in = unzip_dir + "/" + file_in_zip
    return file_to_read_in,error_code

def remove_first_n_lines(file_name,number_of_lines_to_remove):
    with open(file_name, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(file_name, 'w') as fout:
        fout.writelines(data[number_of_lines_to_remove:])

def remove_last_n_lines(file_name,number_of_lines_to_remove):
    num_lines_in_file = sum(1 for line in open(file_name))
    with open(file_name, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(file_name, 'w') as fout:
        lines_to_keep =num_lines_in_file -int(number_of_lines_to_remove)
        fout.writelines(data[0:lines_to_keep])

def get_table_row_data_from_gather_info_sqlserver(sql_server_report,output_file='/tmp/unzips/sql_server_tables_temp.txt'):
    print ("Inside get_table_row_data_from_gather_info_sqlserver ... ",sql_server_report )

    with codecs.open(sql_server_report, 'r', encoding='utf-8',errors='ignore') as infile , open(output_file, 'w') as outfile:
        copy = False
        for line in infile:
            #print ("line ... ", line)
            if (re.match("(.*)Size of all tables, order by name(.*)", line) or re.match("(.*)Report on all tables(.*)", line) ): #start of table section  "Size of all tables, order by name"
                outfile.write(line) # add this
                print(line)
                copy = True
            elif re.match("(.*)Report on tables that have more than 10,000 rows(.*)", line): #end of table section  "Report on tables that have more than 10,000 rows"
                outfile.write(line) # add this
                copy = False
            elif copy:
                outfile.write(line)
    remove_first_n_lines(output_file, 2)
    remove_last_n_lines(output_file, 6)
    return output_file

def get_table_row_data_from_gather_info_oracle(oracle_report,output_file='/tmp/unzips/oracle_tables_temp.txt'):
    print ("Inside get_table_row_data_from_gather_info_oracle ...",oracle_report)
    with codecs.open(oracle_report, 'r', encoding='utf-8',errors='ignore') as infile, open(output_file, 'w') as outfile:
        copy = False
        for line in infile:
            #print (line)
            #print ("Line2: ", infile.readline())
            if re.match("(.*)Report End: Windchill Release IDs(.*)" ,line): #start of table section
                outfile.write(line) # add this
                copy = True
            elif re.match("(.*)Report End on Tables(.*)", line): #end of table section "**Report End on Tables***"
                outfile.write(line) # add this
                copy = False
            elif copy:
                outfile.write(line)
    remove_first_n_lines(output_file,3)
    remove_last_n_lines(output_file,5)
    return output_file


def process_zip_file(customer_number, case_number, cust_name,  upload_date, attachement_link,unzip_dir,data_dir):
    zip_file=get_file_from_url(attachement_link,unzip_dir)
    if (os.path.getsize(zip_file) >1000 and attachement_link.endswith(".zip")): #make sure the file isn't empty and is a zip
        zip_ref = zipfile.ZipFile(zip_file, 'r')

        for file_in_zip in zip_ref.namelist():
            if (is_string_part_of(file_in_zip,"Tables.txt")):  #new gather info report where each report is in a file

                file_to_read_in,error_code=extraxt_file_from_zip(file_in_zip, zip_ref)
                if(error_code ==0):# no problems
                    if(sqlserver_or_oracle(file_to_read_in)):
                        df = get_table_name_rows_from_file_oracle(file_to_read_in)
                    else:
                        df = get_table_name_rows_from_file_sqlserver(file_to_read_in)

                    df = add_coumns_to_df(df, customer_number, case_number, cust_name, upload_date)
                    print (df.columns)
                    print("Saving file for: ",customer_number, case_number, cust_name, upload_date)
                    save_data_file(df,data_dir,customer_number)
                    remove_file_temp_file(file_to_read_in)

            elif (file_in_zip.endswith(".txt")): #look for old style gather info reoprts
                file_to_read_in,error_code = extraxt_file_from_zip(file_in_zip, zip_ref)
                if(error_code==0): #no problems reading the file
                    if(is_a_sql_server_report(file_to_read_in)):
                        temp_tables_file=get_table_row_data_from_gather_info_sqlserver(file_to_read_in)
                        df = get_table_name_rows_from_file_sqlserver_gather_info(temp_tables_file)
                        df = add_coumns_to_df(df, customer_number, case_number, cust_name, upload_date)
                        print(df.columns)
                        print("Saving file for: ", customer_number, case_number, cust_name, upload_date)
                        save_data_file(df, data_dir, customer_number)
                        remove_file_temp_file(file_to_read_in)

                    elif(is_an_oracle_report(file_to_read_in)):

                        temp_tables_file=get_table_row_data_from_gather_info_oracle(file_to_read_in)
                        df = get_table_name_rows_from_file_oracle(temp_tables_file)
                        df = add_coumns_to_df(df, customer_number, case_number, cust_name, upload_date)
                        print(df.columns)
                        print("Saving file for: ", customer_number, case_number, cust_name, upload_date)
                        save_data_file(df, data_dir, customer_number)
                        remove_file_temp_file(file_to_read_in)
                    else:
                        remove_file_temp_file(file_to_read_in)
        zip_ref.close()


def process_text_file(customer_number, case_number, cust_name,  upload_date, attachement_link,unzip_dir,data_dir): #old style gather info reports attached as txt files
    file="/tmp/unzips/temp_report.txt"
    error_code=0
    try:
        urllib.request.urlretrieve(attachement_link, file)
    except:
        error_code=1
        pass

    if(error_code==0 and os.path.getsize(file) >1000): #no problems downloading text file
        if(is_a_sql_server_report(file)):
            temp_tables_file=get_table_row_data_from_gather_info_sqlserver(file)
            df = get_table_name_rows_from_file_sqlserver_gather_info(temp_tables_file)
            df = add_coumns_to_df(df, customer_number, case_number, cust_name, upload_date)
            print(df.columns)
            print("Saving file for: ", customer_number, case_number, cust_name, upload_date)
            save_data_file(df, data_dir, customer_number)

        elif(is_an_oracle_report(file)):
            temp_tables_file=get_table_row_data_from_gather_info_oracle(file)
            df = get_table_name_rows_from_file_oracle(temp_tables_file)
            df = add_coumns_to_df(df, customer_number, case_number, cust_name, upload_date)
            print(df.columns)
            print("Saving file for: ", customer_number, case_number, cust_name, upload_date)
            save_data_file(df, data_dir, customer_number)
        else:
            remove_file_temp_file(file)

        remove_file_temp_file(file)



def get_cursor():
    connection = cx_Oracle.connect('SCQUERY/inquiry@oradbprod3.ptc.com/psnapdw.world')
    # print (connection.version)
    # connection.close()
    cursor = connection.cursor()

    cursor.execute("""
     select distinct cas.customer_number__c customer_number ,cas.casenumber case_number,act.NAME cust_name, att.CREATEDDATE upload_date,att.attachment_link__c attachement_link
      from SFDC_SNAP.SF_CASE_ATTACHMENT__C att, SFDC_SNAP.SF_CASE cas, sf_product2 prod, SFDC_SNAP.sf_account act
      where   cas.id=att.case__c   and prod.id =cas.PRODUCT__C and act.id = cas.ACCOUNTID and upper(prod.name) like '%WIND%' and
        (upper(ATTACHMENT_NAME__C) like '%REPORT%' or upper(ATTACHMENT_NAME__C) like '%SQL%' or upper(ATTACHMENT_NAME__C) like '%ORACLE%') and
      (upper(ATTACHMENT_NAME__C) like '%.ZIP%' or upper(ATTACHMENT_NAME__C) like '%.TXT%')-- or upper(ATTACHMENT_NAME__C) like '%.RAR%' or upper(ATTACHMENT_NAME__C) like '%.7Z%')
      and cas.createddate >(sysdate -31) 
      minus
         select distinct cas.customer_number__c customer_number,cas.casenumber case_number,act.NAME cust_name, att.CREATEDDATE upload_date,att.attachment_link__c attachement_link
      from SFDC_SNAP.SF_CASE_ATTACHMENT__C att, SFDC_SNAP.SF_CASE cas, sf_product2 prod, SFDC_SNAP.sf_account act
      where   cas.id=att.case__c   and prod.id =cas.PRODUCT__C and act.id = cas.ACCOUNTID and upper(prod.name) like '%WIND%' and
    (upper(ATTACHMENT_NAME__C) like '%VAULT%' or upper(ATTACHMENT_NAME__C) like '%WINDU%' or upper(ATTACHMENT_NAME__C) like '%AUDIT%'
    or upper(ATTACHMENT_NAME__C) like '%QUEUE%' or upper(ATTACHMENT_NAME__C) like '%SQLHC%' or upper(ATTACHMENT_NAME__C) like '%ACL%'
    or upper(ATTACHMENT_NAME__C) like '%UPGRADE%' or upper(ATTACHMENT_NAME__C) like '%SQLHC%' or upper(ATTACHMENT_NAME__C) like '%ACL%'
    or upper(ATTACHMENT_NAME__C) like '%LICENS%' or upper(ATTACHMENT_NAME__C) like '%PRINCIPAL%' or upper(ATTACHMENT_NAME__C) like '%CABINET%'
    or upper(ATTACHMENT_NAME__C) like '%MS%' or upper(ATTACHMENT_NAME__C) like '%TEMPLATE%' or upper(ATTACHMENT_NAME__C) like '%INSTALL%') 
    and cas.createddate >(sysdate -31)""")
    return cursor

def execute(data_dir,unzip_dir):
    cursor=get_cursor()
    #customer_number, case_number, cust_name,  upload_date, attachement_link,unzip_dir,data_dir
    for customer_number, case_number,cust_name,upload_date, attachement_link in cursor:
        print("customer :", customer_number," ",cust_name, " case num: " ,case_number, attachement_link ,upload_date)
        upload_date=str(upload_date)
        year_month_day=upload_date[0: upload_date.find(' ')]
        if(attachement_link.find('?')>0): #bypass links with question marks
            print("Not downloading file for link: ",attachement_link)
        else:
            if(attachement_link.endswith(".zip") ):
                process_zip_file(customer_number, case_number,cust_name, year_month_day , attachement_link, unzip_dir, data_dir)
            elif (attachement_link.endswith(".txt")):
                process_text_file(customer_number, case_number, cust_name, year_month_day, attachement_link, unzip_dir,data_dir)

file="/Users/Stephen/Downloads/oracleperf_gpdmp02_GTT_Production.zip"
unzip_dir="/tmp/unzips"
data_dir="/tmp/table_data"
#url_no_tables="http://internal.ptc.com/salesforce/attachments/cases/14/83/18/98/remis_reporting_cogstartup.zip"
#url="http://internal.ptc.com/salesforce/attachments/cases/14/80/54/82/oracleperf_output.zip"
#url="http://internal.ptc.com/salesforce/attachments/cases/14/81/35/23/oracle_perf_html.zip"
#url="http://internal.ptc.com/salesforce/attachments/cases/14/83/18/98/remis_reporting_cogstartup.zip"
#sql_server_zip="/Users/Stephen/Downloads/C14682851_sqlperf-v7.zip"
gather_info_text="http://internal.ptc.com/salesforce/attachments/cases/13/77/93/91/gather_info_script_report.txt"
gather_info_text="http://internal.ptc.com/salesforce/attachments/cases/13/79/15/04/report.txt"
oracle_gather_info_report="/Users/Stephen/Downloads/report.txt.zip"
sql_server_gather_info="/Users/Stephen/Downloads/sql2014.20181016.report.zip"


#process_text_file("111", "2222", "made_up_customer",  '1-1-2019', gather_info_text, unzip_dir, data_dir)

execute(data_dir,unzip_dir)

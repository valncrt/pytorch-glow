
oracle_report ="/Users/Stephen/Downloads/report.txt"
sql_server_report="/Users/Stephen/Downloads/SQLPerf/RITM0035114.txt"
sql_server_report2="/tmp/unzips/sql2014.20181016.report.txt"

to_find_1 = "Size of all tables, order by name"
to_find_2 = "Report on tables that have more than 10,000 rows"

import re
#get_position_of_string_in_file(sql_server_report,to_find_2)

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

def get_table_row_data_from_gather_info_sqlserver(sql_server_report,output_file):
    with open(sql_server_report,"r") as infile, open(output_file, 'w') as outfile:
        copy = False
        for line in infile:
            if (re.match("(.*)Size of all tables, order by name(.*)", line) or re.match("(.*)Report on all tables(.*)",
                                                                                        line)): #start of table section  "Size of all tables, order by name"
                outfile.write(line) # add this
                copy = True
            elif re.match("(.*)Report on tables that have more than 10,000 rows(.*)", line): #end of table section  "Report on tables that have more than 10,000 rows"
                outfile.write(line) # add this
                copy = False
            elif copy:
                outfile.write(line)
    remove_first_n_lines(output_file, 2)
    remove_last_n_lines(output_file, 6)

get_table_row_data_from_gather_info_sqlserver(sql_server_report2,'/tmp/unzips/test_table_rows.out')




#remove_first_n_lines('/tmp/test_table_rows.out',2)
#remove_last_n_lines('/tmp/test_table_rows.out',4)


def get_table_row_data_from_gather_info_oracle(oracle_report,output_file):
    with open(oracle_report,"r") as infile, open(output_file, 'w') as outfile:
        copy = False
        for line in infile:
            #print (line)
            #print ("Line2: ", infile.readline())
            if re.match("(.*)Report End: Windchill Release IDs(.*)", line) : #start of table section "**Report End: Windchill Release IDs"
                outfile.write(line) # add this
                copy = True
            elif re.match("(.*)Report End on Tables(.*)", line): #end of table section "**Report End on Tables***"
                outfile.write(line) # add this
                copy = False
            elif copy:
                outfile.write(line)
    remove_first_n_lines('/tmp/test_table_rows.out',3)
    remove_last_n_lines('/tmp/test_table_rows.out',5)

#get_table_row_data_from_gather_info_oracle(oracle_report,'/tmp/unzips/test_table_rows.out')


def test():
    import re
    report = open(oracle_report, "r")

    for line in report:
        if re.match("(.*)Report End on Tables(.*)", line):
            print (line)

#test()
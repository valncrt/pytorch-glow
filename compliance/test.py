import os
str="http://internal.ptc.com/salesforce/attachments/cases/14/80/49/27/20190318_SQL??.zip"
zip_file="/tmp/unzips/zip_temp.zip"
str2=  ['\n', 'OWNER                TABLE_NAME                           NUM_ROWS       BLOCKS AVG_ROW_LEN    CHAIN_CNT LAST_ANALYZED          SAMPLE_SIZE SAMPLE_PER\n', '-------------------- ------------------------------ -------------- ------------ ----------- ------------ ------------------- -------------- ----------\n', 'ABBPROD              ABSCOLLECTIONCRITERIAKEY                   13            5         223            0 2017-09-26 02:00:08             13        100\n', 'ABBPROD              ABSTRACTSITE\n', 'ABBPROD              ACCEPTEDSTRATEGY\n', 'ABBPROD              ACCESSCONTROLSURROGATE\n', 'ABBPROD              ACCESSCONTROLSURROGATEMASTER\n', 'ABBPROD              ACCESSPOLICYRULE                          141           13         370            0 2017-09-26 02:00:17            141        100\n', 'ABBPROD              ACTIONITEMSUBJECTLINK\n']
str3='OWNER 333'

str4 = ''.join(str2)
if "TABLE_NAME"  in str4:
    print ("Found it")
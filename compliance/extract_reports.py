from __future__ import print_function

import cx_Oracle

connection = cx_Oracle.connect('SCQUERY/inquiry@oradbprod3.ptc.com/psnapdw.world')
#print (connection.version)
#connection.close()

cursor = connection.cursor()

cursor.execute("""
 select distinct cas.customer_number__c customer_number ,cas.casenumber case_number,act.NAME cust_name, att.CREATEDDATE upload_date,att.attachment_link__c attachement_link
  from SFDC_SNAP.SF_CASE_ATTACHMENT__C att, SFDC_SNAP.SF_CASE cas, sf_product2 prod, SFDC_SNAP.sf_account act
  where   cas.id=att.case__c   and prod.id =cas.PRODUCT__C and act.id = cas.ACCOUNTID and upper(prod.name) like '%WIND%' and
    (upper(ATTACHMENT_NAME__C) like '%REPORT%' or upper(ATTACHMENT_NAME__C) like '%SQL%' or upper(ATTACHMENT_NAME__C) like '%ORACLE%') and
  (upper(ATTACHMENT_NAME__C) like '%.ZIP%' or upper(ATTACHMENT_NAME__C) like '%.RAR%' or upper(ATTACHMENT_NAME__C) like '%.7Z%')
  and cas.createddate >(sysdate -365) 
  minus
     select distinct cas.customer_number__c customer_number,cas.casenumber case_number,act.NAME cust_name, att.CREATEDDATE upload_date,att.attachment_link__c attachement_link
  from SFDC_SNAP.SF_CASE_ATTACHMENT__C att, SFDC_SNAP.SF_CASE cas, sf_product2 prod, SFDC_SNAP.sf_account act
  where   cas.id=att.case__c   and prod.id =cas.PRODUCT__C and act.id = cas.ACCOUNTID and upper(prod.name) like '%WIND%' and
(upper(ATTACHMENT_NAME__C) like '%VAULT%' or upper(ATTACHMENT_NAME__C) like '%WINDU%' or upper(ATTACHMENT_NAME__C) like '%AUDIT%'
or upper(ATTACHMENT_NAME__C) like '%QUEUE%' or upper(ATTACHMENT_NAME__C) like '%SQLHC%' or upper(ATTACHMENT_NAME__C) like '%ACL%'
or upper(ATTACHMENT_NAME__C) like '%UPGRADE%' or upper(ATTACHMENT_NAME__C) like '%SQLHC%' or upper(ATTACHMENT_NAME__C) like '%ACL%') 
and cas.createddate >(sysdate -365)""")

for customer_number__c, case_number,cust_name, attachement_link,upload_date in cursor:
    print("customer :", customer_number__c," ",cust_name, " case num: " ,case_number, attachement_link ,upload_date)
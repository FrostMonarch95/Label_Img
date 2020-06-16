import mysql.connector
from mysql.connector import Error
import os
import time
import datetime

class my_data_base:

    #返回属性列中的不同项
    #request_col = "satelite ","sensor","latitude","longtitude","shottime",
    #id,spectrum
    def select_distinct_lines(self,request_col):
        global db_error
        command = "select distinct "+request_col+" from remote_sensing_core_table"
        try:
            self.cursor.execute(command)
            records = self.cursor.fetchall()
            return records  
        except Error as e:
            db_error = e
            print("select distinct error")
            print(e)
            return -1
    #返回表中有多少行数据.
    # 返回 -1 表示出现错误，可以用来检查表是否可用 
    # 返回 其它数字代表行数.
    def select_count_lines(self):
        global db_error
        command = "select count(*) from remote_sensing_core_table"
        try:
            self.cursor.execute(command)
            records=self.cursor.fetchall();
            return records
        except Error as e:
            print(e)
            db_error=e
            return -1
    #kwargs["satelite"] = 'GF1','GF2' ......
    #kwargs["longtitude"] = (-180,180) 这是一个元组表明我们搜索的经度范围为 [-180,180]
    #kwargs["latitude"] = (-90,90) 这是一个元组表明我们搜索的纬度范围为 [-180,180]
    #kwargs["shottime"] = [datetime_object1,datetime_object2]   
    def select_from_lines(self,**kwargs):
        global db_error
        keys = ["satelite","sensor","latitude","longtitude","shottime","id","spectrum"]
        suc=0
        for key in keys:
            if key in kwargs:
                suc = 1
        if not suc:
            command = "select * from remote_sensing_core_table "
        else: command = "select * from remote_sensing_core_table where"
        with_and = 0
        paralist = []
        if  keys[0] in kwargs:
            command+=" satelite = %s"
            paralist.append(kwargs[keys[0]])
            with_and = 1
        if  keys[1] in kwargs:
            if with_and:command+=" and"
            command+=" sensor =  %s"
            paralist.append(kwargs[keys[1]])
            with_and=1
        if keys[2] in kwargs:
            if with_and:command+=" and"
            command += " latitude between %s and %s"
            paralist.append(str(kwargs[keys[2]][0]))
            paralist.append(str(kwargs[keys[2]][1]))
            with_and=1
        if keys[3] in kwargs:
            if with_and:command+=" and"
            command += " longtitude between %s and %s"
            paralist.append(str(kwargs[keys[3]][0])) 
            paralist.append(str(kwargs[keys[3]][1]))
            with_and=1
        
        if keys[4] in kwargs:
            if kwargs[keys[4]][0] > kwargs[keys[4]][1]:
                self.warning("input datetime left small right big")
                return -1;
            if with_and:command+=" and"
            first = kwargs[keys[4]][0]
            second = kwargs[keys[4]][1]
            command+= " shottime between %s and %s "
            paralist.append(first.strftime("%Y-%m-%d"))
            paralist.append(second.strftime("%Y-%m-%d"))
            with_and=1
        if keys[5] in kwargs:
            if with_and:command+=" and"
            command+=" id = %s"
            paralist.append(kwargs[keys[5]])
            with_and=1
        if keys[6] in kwargs:
            if with_and:command+=" and"
            command+=" spectrum = %s"
            paralist.append(kwargs[keys[6]])
            with_and=1
        try:
            print(command)
            self.cursor.execute(command,paralist)
            records = self.cursor.fetchall()
            return records
            #how to deal with return data
            # for row in records:
            # print("Id = ", row[0], )
            # print("Name = ", row[1])
            # print("Price  = ", row[2])
            # print("Purchase date  = ", row[3], "\n")
        except Error as e:
            db_error = e
            print("select error ",e)
            return -1
    def warning(self,st):

        print(st)
    def drop_table(self):
        global db_error
        self.cursor.execute("drop table if exists remote_sensing_core_table")
        self.connection.commit()
    def delete_line(self,pk):
        global db_error
        try:
            sql_delete = "DELETE FROM remote_sensing_core_table WHERE file_description = %s"
            sql_data = (pk,)

            self.cursor.execute(sql_delete, sql_data)
            self.connection.commit()
        except Error as e:
            db_error = e;
            print("delete row failed ",e)
       
        
    #location:用户选择的扫描的路径.
    #cursor:输入指令的cursor
    #table:需要插入的表的名称

    #注意存放的格式：
    #东经为正，西经为负
    #北纬为正，南纬为负
    def scan_and_insert_to_table(self,location):
        global db_error
        if location == '':location =None
        command = """create table if not exists remote_sensing_core_table (file_description varchar(300) primary key ,
                satelite varchar(10),
                sensor varchar(10),
                longtitude decimal(10,5),
                latitude decimal(10,5),
                shottime date,
                id varchar(20),
                spectrum varchar(10)
                );"""
        self.cursor.execute(command);
        support_extension = ["jpg","png","tif","tiff"]
        files=[]
        for file in os.listdir(location):
            suc=0
            for ext in support_extension:
                if file.split(".")[-1] == ext:
                    suc = 1;
                    break;
            if suc:files.append(file)
        final = []
        for file in files:
            ls = file.split("_");
            ret = [ os.path.join(location,file)  if location is not None else file]
            if len(ls)<5:
                self.warning("format not correct for file {} support only 4 _".format(os.path.join(location,file)))
                continue
            if ls[0][0:2]!="GF":
                self.warning("format not correct for file {} only support GF".format(os.path.join(location,file)))
                continue
            ret.append(ls[0]);
            ret.append(ls[1]);
            if not ( ls[2][0] == 'W' or ls[2][0] == 'E'):
                self.warning("format not correct for file {} file should in Exxx.xx or Wxxx.xx".format(os.path.join(location,file)) )
                continue;
            try: 
                longitude = float(ls[2][1:])
            except Exception as e:
                self.warning("format not correct for file {} xxx.xx not a float number in Exxx.xx".format(os.path.join(location,file)))
                continue;
            if longitude <0 or longitude >180:continue;
            if ls[2][0] == 'W':longitude=0-longitude
            ret.append(longitude);
            
            if not ( ls[3][0] == 'N' or ls[3][0] == 'S'):
                self.warning("format not correct for file {} file should in Nxxx.xx or Sxxx.xx".format(os.path.join(location,file)) )
                continue;
            try: 
                latitude = float(ls[3][1:])
            except Exception as e:
                self.warning("format not correct for file {} xxx.xx not a float number in Nxxx.xx".format(os.path.join(location,file)))
                continue;
            if latitude <0 or latitude >90:continue;
            if ls[3][0] == 'S':latitude=0-latitude
            ret.append(latitude)
            try:
                datetime = str(int(ls[4][0:4]))+'-'+str(int(ls[4][4:6]))+'-'+str(int(ls[4][6:8]))
            except Exception as e:
                self.warning("datetime format not correct")
                print(e)
                continue
            ret.append(datetime)
            tmp = ls[5].split("-")    
            if len(tmp)<2:
                self.warning("- format not correct")
                continue;
            ret.append(tmp[0])
            ret.append(tmp[1].split(".")[0]);
            ret=tuple(ret);
            final.append(ret)
        print("total files ",len(final))
    # for idx,ele in enumerate(final):

        mySql_insert_query = """INSERT INTO remote_sensing_core_table (file_description, satelite, sensor, longtitude ,latitude,shottime,id,spectrum) 
                            VALUES (%s, %s, %s, %s,%s, %s, %s, %s) ON DUPLICATE KEY UPDATE file_description = file_description"""

        # recordTuple = ele
        try:
            self.cursor.executemany(mySql_insert_query, final)
            self.connection.commit()
            print("insert data OK")
        except Error as e:
            db_error = e
            print("insert data failed ")
            print(e)
        
            
            
    def login(self,host,database,user_name,password):
        global db_error
        if not host:
            host = 'localhost'
        if not database:
            database = 'ts2'
        if not user_name:
            user_name = 'root'
        if not password:
            password = '1029wang'
        try:
            self.connection = mysql.connector.connect(host=host,
                                                 user=user_name,
                                                 password=password,
                                                 auth_plugin='mysql_native_password')
            if self.connection.is_connected():
                db_Info = self.connection.get_server_info()
                print("Connected to MySQL Server version ", db_Info)
                
                
                self.cursor = self.connection.cursor()
                self.cursor.execute("CREATE DATABASE IF NOT EXISTS {} ".format(database))
                self.cursor.execute("USE {}".format(database))
                self.cursor.execute("select database();")
                record = self.cursor.fetchone()
                print("You're connected to database: ", record)
                #self.scan_and_insert_to_table('')
                #drop_table()
                #delete_line("GF2_PMS1_E117.4_N36.7_20150926_L1A0001064469-MSS1.jpg");
                #kwargs["satelite"] = 'GF1','GF2' ......
                #kwargs["longtitude"] = (-180,180) 这是一个元组表明我们搜索的经度范围为 [-180,180]
                #kwargs["latitude"] = (-90,90) 这是一个元组表明我们搜索的纬度范围为 [-180,180]
                #kwargs["shottime"] = [datetime_object1,datetime_object2]   
                
                #print(select_from_lines(satelite ="GF2",shottime = [datetime.date(2013,9,28),datetime.date(2019,4,13)],latitude = [0,40],longtitude = [0,116.9]));
                #print("distinct")
                #print(self.select_distinct_lines("shottime"))
                
                    
                return 0
        except Error as e:
            db_error = e
            print("Error while connecting to MySQL", e)
            return -1
    def close_db(self):
        global db_error
        try: 
            self.cursor.close()
        except Error as e:
            db_error = e
            print("Error cursor close ",e);
        try:
            self.connection.close()
        except Error as e:
            db_error = e 
            print("Error while closing connection",e)
    def __init__(self):        
        self.cursor = None;
        self.connection = None;
        
singleton_data_base =  my_data_base();
db_error = ''
if __name__ == '__main__':
    singleton_data_base.login(False,False,False,False);
    singleton_data_base.scan_and_insert_to_table('');
    print(singleton_data_base.select_distinct_lines('shottime'))
    singleton_data_base.close_db();
    
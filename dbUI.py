from mydb import singleton_data_base
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialogButtonBox
from PyQt5  import QtCore 
from PyQt5 import QtGui
import mydb 
from PyQt5.QtWidgets import QFormLayout
from PyQt5 import QtSql
import pandas as pd
class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]
    
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])


class Login(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Login, self).__init__(parent)
        self.setWindowTitle("database login")
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.test_ls = ['Host',"database",'user_name','password']
        self.textname_ls = []
        self.Host = 'localhost'
        self.Database = ''
        self.user_name = 'root'
        self.password = ''
        self.resize(300,200)
        for i in range(len(self.test_ls)):
            self.textname_ls.append(QtWidgets.QLineEdit(self))
            
        self.textname_ls[-1].setEchoMode(QtWidgets.QLineEdit.Password)
        self.textname_ls[0].setText(self.Host)
        self.textname_ls[2].setText(self.user_name)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.myaccept)
        self.buttonBox.rejected.connect(self.myreject)
              
        layout = QtWidgets.QVBoxLayout(self)
        formLayout = QFormLayout()
        for i in range(len(self.test_ls)):
            formLayout.addRow(self.test_ls[i], self.textname_ls[i])
        layout.addLayout(formLayout)
        
        layout.addWidget(self.buttonBox)
    
    
    def myreject(self):
        QtWidgets.QMessageBox.warning(
                self, 'Warning', 'You can not use mysql service in this mode')
        self.reject();
    def myaccept(self):
        # self.Host = 'localhost'
        # self.Database = ''
        # self.user_name = 'root'
        # self.password = ''
        self.Host = self.textname_ls[0].text()
        self.Database = self.textname_ls[1].text();
        self.user_name = self.textname_ls[2].text();
        self.password = self.textname_ls[3].text();
        if singleton_data_base.login(self.Host,self.Database,self.user_name,self.password) == 0:
            self.accept();
        else:
            QtWidgets.QMessageBox.warning(
                self, 'Error', 'Database error \n' + str(mydb.db_error))
            return;
        

    def handleLogin(self):
        if (self.textName.text() == 'foo' and
            self.textPass.text() == 'bar'):
            self.accept()
        else:
            QtWidgets.QMessageBox.warning(
                self, 'Error', 'Bad user or password')
class select_result_window(QtWidgets.QDialog):

    def __init__(self,sql_data, parent = None):
        super(select_result_window, self).__init__(parent)
        self.table = QtWidgets.QTableView()
        
        col = [str(i+1) for i in range(len(sql_data))]
        print(col)
        data = pd.DataFrame(sql_data, columns = ["file","satelite","sensor","latitude","longtitude","shottime","id","spectrum"], index=col)

        self.model = TableModel(data)
        self.table.setModel(self.model)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.myaccept)
        self.buttonBox.rejected.connect(self.myreject)
        layout = QtWidgets.QVBoxLayout()
        
        layout.addWidget(self.table)  #table view
        layout.addWidget(self.buttonBox)  #pushbutton
        self.setLayout(layout)
        self.resize(1024,500)
    def myaccept(self):
        self.accept();
    def myreject(self):
        self.reject();
        
class search_window(QtWidgets.QDialog):
    def spectrum_cb_change(self):
        if self.spectrum_cb.currentText() != 'ALL':
            self.spectrum_text = self.spectrum_cb.currentText()
        else: self.spectrum_text = ''
    def satelite_cb_change(self):
        if self.satelite_cb.currentText() != 'ALL':
            self.satelite_text = self.satelite_cb.currentText()
        else: self.satelite_text = ''
    def sensor_cb_change(self):
        if self.sensor_cb.currentText() != 'ALL':
            self.sensor_text = self.sensor_cb.currentText()
        else:self.sensor_text = ''
    # @pyqt_slot()
    # def dateChanged_left(self):
        # self.ldate_changed = 1
    # @pyqt_slot()
    # def dateChanged_right(self):
        # self.rdate_changed = 1
    def __init__(self, parent=None):
        self.ret_data = '' #shows the return data of selecting
        self.satelite_text = '' #which satelite the user want to use
        self.sensor_text = '' #which sensor the user want to use
        self.spectrum_text = '' #which spectrum the user want to use
        # self.ldate_changed = 0  #user modify the left date
        # self.rdate_changed = 0   #user modify the right date
        
        
        super(search_window, self).__init__(parent)
        formLayout = QFormLayout()
        self.satelite_cb = QtWidgets.QComboBox()
        ret = singleton_data_base.select_distinct_lines("satelite")
        self.satelite_cb.addItem('ALL')
        for ele in ret:
            self.satelite_cb.addItem(ele[0])
        self.satelite_cb.currentIndexChanged.connect(self.satelite_cb_change)
        formLayout.addRow("satelite",self.satelite_cb)
        
        self.sensor_cb = QtWidgets.QComboBox()
        self.sensor_cb.addItem('ALL')
        ret = singleton_data_base.select_distinct_lines("sensor")
        for ele in ret:
            self.sensor_cb.addItem(ele[0])
        self.sensor_cb.currentIndexChanged.connect(self.sensor_cb_change)
        formLayout.addRow("sensor",self.sensor_cb)
        
        validator = QtGui.QDoubleValidator()
        
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("latitude"))
        self.la1 = QtWidgets.QLineEdit()
        self.la1 .setValidator(validator);
        layout.addWidget(self.la1 )
        self.la2 = QtWidgets.QLineEdit()
        self.la2 .setValidator(validator);        
        layout.addWidget(self.la2)
        formLayout.addRow(layout)
        
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("longtitude"))
        self.lo1 = QtWidgets.QLineEdit()
        self.lo1 .setValidator(validator);
        layout.addWidget(self.lo1 )
        self.lo2 = QtWidgets.QLineEdit()
        self.lo2.setValidator(validator);        
        layout.addWidget(self.lo2)
        formLayout.addRow(layout)
        
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("date time"))
        self.left_date=QtWidgets.QDateEdit()
        self.left_date.setDate(QtCore.QDate(1900, 1, 1))
        layout.addWidget(self.left_date)
        self.right_date=QtWidgets.QDateEdit()
        self.right_date.setDate(QtCore.QDate(2035, 1, 1))
        layout.addWidget(self.right_date)
        formLayout.addRow(layout)
        
        self.id = QtWidgets.QLineEdit()
        formLayout.addRow("id",self.id)
        
        self.spectrum_cb = QtWidgets.QComboBox()
        ret = singleton_data_base.select_distinct_lines("spectrum")
        print(ret)
        self.spectrum_cb.addItem('ALL')
        for ele in ret:
            self.spectrum_cb.addItem(ele[0])
        self.spectrum_cb.currentIndexChanged.connect(self.spectrum_cb_change)
        formLayout.addRow("spectrum",self.spectrum_cb)
        
        self.setLayout(formLayout)
        
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.myaccept)
        self.buttonBox.rejected.connect(self.myreject)
        formLayout.addRow(self.buttonBox);
    def myaccept(self):
        satelite = self.satelite_cb.currentText();
        dic = {}
        if satelite != 'ALL':dic["satelite"]=satelite
        l1 = -180
        if self.lo1.text() != '':l1 =  float(self.lo1.text());
        l2 = 180
        if self.lo2.text() != '':l2 = float(self.lo2.text());
        dic["longtitude"] = (l1,l2);
        l1 = -90
        l2 = 90;
        if self.la1.text() != '':l1 = float(self.la1.text())
        if self.la2.text() != '':l2 = float(self.la2.text());
        dic["latitude"] = (l1,l2)
        dic["shottime"] = [self.left_date.date().toPyDate(),self.right_date.date().toPyDate()]
        if self.sensor_cb.currentText() != 'ALL':dic["sensor"] = self.sensor_cb.currentText()
        if self.id.text() != '':dic["id"] = self.id.text();
        if self.spectrum_cb.currentText()!= 'ALL':dic["spectrum"] = self.spectrum_cb.currentText()
        self.ret_data =  singleton_data_base.select_from_lines(**dic)
        self.accept();
    def myreject(self):
        self.reject();
def  scan_and_insert_to_table(location):
    singleton_data_base.scan_and_insert_to_table(location);       

if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    
    login = Login()
    
    
    if login.exec_() == QtWidgets.QDialog.Accepted:
        swindow = search_window()
        if swindow.exec_() == QtWidgets.QDialog.Accepted:
            if swindow.ret_data != -1:
                print(swindow.ret_data);
                res_win = select_result_window(swindow.ret_data);
                res_win.exec_();
            else:
                
                print("select error")
                print(mydb.db_error)
        print("database test done")   
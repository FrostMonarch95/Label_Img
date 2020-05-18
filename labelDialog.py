try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *


BB = QDialogButtonBox
class External(QThread):
    """
    Runs a counter thread.
    """
    countChanged = pyqtSignal(int)
    def __del__(self):
        self.wait()
    def run(self):
        import time
        count = 0
        while count<100:
            count +=1
            self.countChanged.emit(count)


class LabelDialog(QDialog):

    def __init__(self, text="object label", parent=None, listItem=None):
        super(LabelDialog, self).__init__(parent)
        self.cur_class=0
        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setReadOnly(True)
        #self.edit.editingFinished.connect(self.postProcess)

        model = QStringListModel()
        model.setStringList(listItem)
        completer = QCompleter()
        completer.setModel(model)
        self.edit.setCompleter(completer)

        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)

        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        if listItem is not None and len(listItem) > 0:
            self.listWidget = QListWidget(self)
            for item in listItem:
                self.listWidget.addItem(item)
            self.listWidget.itemClicked.connect(self.listItemClick)
            self.listWidget.itemDoubleClicked.connect(self.listItemDoubleClick)
            layout.addWidget(self.listWidget)

        self.setLayout(layout)

    def validate(self):
        self.accept()

    def popUp(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.cur_class if self.exec_() else None

    def listItemClick(self, tQListWidgetItem):
        self.cur_class=self.listWidget.currentRow()
        try:
            text = tQListWidgetItem.text().trimmed()
        except AttributeError:
            # PyQt5: AttributeError: 'str' object has no attribute 'trimmed'
            text = tQListWidgetItem.text().strip()
        self.edit.setText(text)

    def listItemDoubleClick(self, tQListWidgetItem):
        self.listItemClick(tQListWidgetItem)
        self.validate()
class bandOption(QDialog):
    def __init__(self,bandList,parent=None):

         super(bandOption, self).__init__(parent)


         layout = QVBoxLayout()
         self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)

         bb.accepted.connect(self.validate)
         bb.rejected.connect(self.reject)
         layout.addWidget(bb)
         self.bandList = bandList
         if bandList is not None and len(bandList) > 0:
            self.listWidget=QListWidget(self)
            for item in bandList:
                item=str(item+1)
                item='band'+item
                self.listWidget.addItem(item)

            layout.addWidget(self.listWidget)
         self.listWidget.itemClicked.connect(self.chooseIdx)

         hlayout=QHBoxLayout()
         btnUp=QPushButton("UP")
         btnDown=QPushButton("Down")
         btnUp.clicked.connect(self.btnUp)
         btnDown.clicked.connect(self.btnDown)
         hlayout.addWidget(btnUp)
         hlayout.addWidget(btnDown)
         layout.addLayout(hlayout)
         self.setLayout(layout)
         self.selectIdx=-1
    def chooseIdx(self):
        self.selectIdx=self.listWidget.currentRow()
    def popUp(self, text='', move=True):
        if move:
            self.move(QCursor.pos())
        return self.bandList if self.exec_() else None

    def btnUp(self):
        if self.selectIdx == -1 or self.selectIdx==0:return
        print("up before")
        print(self.bandList)
        self.bandList[self.selectIdx-1],self.bandList[self.selectIdx]=self.bandList[self.selectIdx],self.bandList[self.selectIdx-1]
        print("up after")
        print(self.bandList)

        item=self.listWidget.takeItem(self.selectIdx)
        self.listWidget.insertItem(self.selectIdx-1,item)
        self.listWidget.setCurrentItem(item)
        self.selectIdx-=1
        print("up finish")
    def btnDown(self):
        if self.selectIdx == -1 or self.selectIdx==len(self.bandList)-1:return
        print("down before")
        print(self.bandList)
        self.bandList[self.selectIdx+1],self.bandList[self.selectIdx]=self.bandList[self.selectIdx],self.bandList[self.selectIdx+1]
        print("down after")
        print(self.bandList)
        item=self.listWidget.takeItem(self.selectIdx)
        self.listWidget.insertItem(self.selectIdx+1,item)
        self.listWidget.setCurrentItem(item)
        self.selectIdx+=1
    def validate(self):
        self.accept()


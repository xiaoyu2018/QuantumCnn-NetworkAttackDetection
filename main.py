import sys
from PyQt5 import QtCore 
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QLabel, QHBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.uic import loadUi
import qcnn
import base_model
import random
import PyQt5

# # 自适应高分辨率
# QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)

if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
    PyQt5.QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


# 隐藏GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


model_type=-1

#训练过程窗口
class Process_win(QWidget):
    def __init__(self,jpg_path):
        super().__init__()
        
        self.setWindowTitle("训练过程")
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.lab = QLabel()
        
        self.lab.setPixmap(QPixmap(jpg_path[0]))
        self.vbox = QHBoxLayout()
        self.vbox.addWidget(self.lab)
        self.setLayout(self.vbox)



#开辟新的线程训练模型
class RunThread(QtCore.QThread):
 # 通过类成员对象定义信号对象
    _signal = QtCore.pyqtSignal(qcnn.MyQnnModel)

    def __init__(self, model):
        super(RunThread, self).__init__()
        self.model=model

    def __del__(self):
        self.wait()

    def run(self):
        
        
        self.model.Train()
        self._signal.emit(self.model)
        
        # 记录此模型
        self.model.Evaluate()
        self.model.RandomTest()
        self.exit()

class win(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(".//newMainWindow.ui", self)
        # 丑化界面
        self.setWindowOpacity(0.98)
        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)  # 隐藏边框
        

        self.my_qmodel = qcnn.MyQnnModel()
        self.my_model=base_model.BaseModel()
        self.process_win=Process_win
        
        #信号连接
        self.comboBox_select.currentIndexChanged.connect(
            self.ChangeStat_combox)
        self.pushButton_starttrain.clicked.connect(self.StartTrain)
        self.pushButton_showprocess.clicked.connect(self.ShowProcess)
        self.pushButton_random.clicked.connect(self.RandomPredict)
        self.pushButton_all.clicked.connect(self.AllPredict)
        self.pushButton_loadmodel.clicked.connect(self.LoadModel)

        self.pushButton_trainpath.clicked.connect(self.GetTrainPath)
        self.pushButton_valpath.clicked.connect(self.GetValPath)
        self.pushButton_testpath.clicked.connect(self.GetTestPath)
        self.checkBox_save.stateChanged.connect(
            lambda: self.ChangeStat_checkbox(self.checkBox_save))
    # 仅提供量子卷积神经网络，默认是2号数据集
    def GetTrainPath(self):
        name_list = QFileDialog. getOpenFileNames(
            self, "请选择训练集数据与训练集标签", ".//dataset", "CSV Files(*.csv)")
        if(len(name_list[0])<2):
            self.statusbar.showMessage("未正确选择训练集！")
            return
        
        self.my_qmodel.traindata_path = name_list[0][0]
        self.my_qmodel.trainlabel_path = name_list[0][1]

        self.lineEdit_trainpath.setText(f"{name_list[0][0]};{name_list[0][1]}")

    # 仅提供量子卷积神经网络，默认是2号数据集
    def GetValPath(self):
        name_list = QFileDialog. getOpenFileNames(
            self, "请选择验证集数据与验证集标签", ".//dataset", "CSV Files(*.csv)")
        if(len(name_list[0]) < 2):
            self.statusbar.showMessage("未正确选择验证集！")
            return
        
        self.my_qmodel.valdata_path = name_list[0][0]
        self.my_qmodel.vallabel_path = name_list[0][1]

        self.lineEdit_valpath.setText(f"{name_list[0][0]};{name_list[0][1]}")
    
    # 仅提供量子卷积神经网络，默认是2号数据集
    def GetTestPath(self):
        name_list = QFileDialog. getOpenFileNames(
            self, "请选择验证集数据与验证集标签", ".//dataset", "CSV Files(*.csv)")
        if(len(name_list[0]) < 2):
            self.statusbar.showMessage("未正确选择测试集！")
            return

        self.my_qmodel.testdata_path = name_list[0][0]
        self.my_qmodel.testlabel_path = name_list[0][1]

        self.lineEdit_testpath.setText(f"{name_list[0][0]};{name_list[0][1]}")

    # 切换算法后更新状态栏
    def ChangeStat_combox(self):
        self.statusbar.showMessage(
            "即将训练的算法已更换为 "+f"{self.comboBox_select.currentText()}"+"!")

    def ChangeStat_checkbox(self,a):
        if(a.isChecked()==True):
            self.statusbar.showMessage("将保存本次训练结果！")
        else:
            self.statusbar.showMessage("不保存本次训练结果！")

    # 子线程的回调
    def CallBack(self,msg):
        self.my_qmodel=msg
    
    def StartTrain(self):
        # 仅提供量子卷积神经网络的模型训练
        self.lineEdit_trainacc.setText("")
        self.lineEdit_trainloss.setText("")
        self.lineEdit_valacc.setText("")
        self.lineEdit_valloss.setText("")
        self.lineEdit_traintime.setText("")
        
        self.my_qmodel.is_save = self.checkBox_save.isChecked()
            
        self.statusbar.showMessage("正在训练模型...")

        
        if(self.comboBox_select.currentText() == "Hybrid model with a single quantum filter"):
            self.my_qmodel.model_name = "HQcnn_s"
        elif(self.comboBox_select.currentText() == "Hybrid convolution with multiple quantum filters"):
            self.my_qmodel.model_name = "HQcnn_m"

        self.my_qmodel.epochs = int(self.lineEdit_epoch.text())
        self.my_qmodel.batch_size = int(self.lineEdit_batch.text())
        self.my_qmodel.num_classes = int(self.lineEdit_class.text())
        self.my_qmodel.features = int(self.lineEdit_feature.text())
        self.my_qmodel.LoadData()
        
        

        self.thread = RunThread(self.my_qmodel)
        self.thread._signal.connect(self.CallBack)
        self.thread.start()
        
        #等待子线程运行完毕
        while(True):
            QApplication.processEvents()
            if(self.my_qmodel.train_over==True):
                self.lineEdit_trainacc.setText(
                    "%.7f" % self.my_qmodel.train_accuracy)
                self.lineEdit_valacc.setText(
                    "%.7f" % self.my_qmodel.val_accuracy)
                self.lineEdit_trainloss.setText(
                    "%.7f" % self.my_qmodel.train_loss)
                self.lineEdit_valloss.setText(
                    "%.7f" % self.my_qmodel.val_loss)
                self.lineEdit_traintime.setText(
                    "%ds" % self.my_qmodel.train_time)
               
                self.statusbar.showMessage(f"模型训练完毕！")
                break
        
        #重置训练状态
        self.my_qmodel.train_over = False


    def ShowProcess(self):
        jpg_path = QFileDialog.getOpenFileName(
            self, "请选择某个模型的训练过程图", ".//mymodles", "JPG Files(*.jpg)")
        if(jpg_path[0]==""):
            self.statusbar.showMessage("未正确选择图片！")
            return
        self.process_win = Process_win(jpg_path)
        self.process_win.show()
        self.statusbar.showMessage("过程图片已显示！")

    def AllPredict(self):
        # 传统神经网络模型
        if(model_type == 0):
            self.statusbar.showMessage(f"正在检测全部测试集，共 {len(self.my_model.test_label)} 条数据...")
            
            self.my_model.Evaluate()
            self.lineEdit_testacc.setText("%.7f" % self.my_model.test_accuracy)
            self.lineEdit_testloss.setText("%.7f" % self.my_model.test_loss)
            self.statusbar.showMessage("测试集检测完毕！")
        
        # 量子神经网络模型
        elif(model_type == 1):
            with open(f".//mymodles//{self.my_qmodel.model_name}_alltest.txt", "r") as f:
                test_loss, test_accuracy = f.readlines()
            
            self.lineEdit_testloss.setText(
                "%.7f" % float(test_loss))
            self.lineEdit_testacc.setText(
                "%.7f" % float(test_accuracy))
        else:
            self.statusbar.showMessage("请先加载模型！")
    
    def RandomPredict(self):
        # 传统神经网络模型
        if(model_type==0):
            self.statusbar.showMessage("正在随机抽检...")

            predict,real=self.my_model.RandomTest()
            
            self.lineEdit_predictresult.setText(predict)
            self.lineEdit_realresult.setText(real)

            self.statusbar.showMessage("抽检完毕！")
        
        # 量子神经网络模型
        elif(model_type==1):
            with open(f".//mymodles//{self.my_qmodel.model_name}_randomtest.txt", "r") as f:
                lines = f.readlines()
            num=random.randint(0, len(lines))
            real,predict=lines[num].split(" ")

            self.lineEdit_predictresult.setText(predict)
            self.lineEdit_realresult.setText(real)
        else:
            self.statusbar.showMessage("请先加载模型！")
        

    def LoadModel(self):
        self.statusbar.showMessage("正在加载模型...")
        model_path=QFileDialog.getOpenFileName(self, "请选择预载模型",".//mymodles", "H5 Files(*.h5)")

        
        if(model_path[0]==""):
            self.statusbar.showMessage("模型未正确而载入！")
            return
        
        self.lineEdit_loadmodel.setText(model_path[0])
        
        global model_type
        
        # 加载传统神经网络模型
        if(("Bp_model" in str(model_path[0])) or ("Cnn_model" in str(model_path[0]))):
            self.my_model.LoadModle(str(model_path[0]))
            
            if("Bp"in str(model_path[0])):
                self.my_model.model_name = "BP"
            else:
                self.my_model.model_name = "CNN"
            
            
            # 使用规定好的测试集
            if('1'in str(model_path[0])):
                flag = 1
            elif('2'in str(model_path[0])):
                flag = 2
            elif('3' in str(model_path[0])):
                flag = 3
            
            self.my_model.data_mode=flag
            
            # 读取训练结果文件
            with open(f".//mymodles//{self.my_model.model_name}_{self.my_model.data_mode}.txt", "r") as f:
                train_loss, train_accuracy, val_loss, val_accuracy, train_time = f.readlines()
                
            # 显示训练结果
            self.lineEdit_trainacc.setText(
                "%.7f" % float(train_accuracy))
            self.lineEdit_valacc.setText(
                "%.7f" % float(val_accuracy))
            self.lineEdit_trainloss.setText(
                "%.7f" % float(train_loss))
            self.lineEdit_valloss.setText(
                "%.7f" % float(val_loss))
            self.lineEdit_traintime.setText(
                "%ds" % int(train_time))
            
            # 载入测试集数据
            self.my_model.LoadData()
            # 界面显示测试集路径
            self.lineEdit_testpath.setText(
                ".//dataset//test_data_%d.csv;.//dataset//test_label_%d.csv" % (flag,flag))
            
            # 禁止选取测试集操作
            self.pushButton_testpath.setEnabled(False)
            
            model_type = 0
        # 加载量子卷积神经网络模型
        else:
            self.pushButton_testpath.setEnabled(True)
            
            # self.my_qmodel.LoadModle(str(model_path[0]))
            if("HQcnn_s" in str(model_path[0])):
                self.my_qmodel.model_name = "HQcnn_s"
            else:
                self.my_qmodel.model_name = "HQcnn_m"
            
            # print(model_path[0],self.my_qmodel.model_name)
            # 读取训练结果文件
            with open(f".//mymodles//{self.my_qmodel.model_name}.txt", "r") as f:
                train_loss, train_accuracy, val_loss, val_accuracy, train_time = f.readlines()

            # 显示训练结果
            self.lineEdit_trainacc.setText(
                "%.7f" % float(train_accuracy))
            self.lineEdit_valacc.setText(
                "%.7f" % float(val_accuracy))
            self.lineEdit_trainloss.setText(
                "%.7f" % float(train_loss))
            self.lineEdit_valloss.setText(
                "%.7f" % float(val_loss))
            self.lineEdit_traintime.setText(
                "%ds" % int(train_time))
            
            # 界面显示默认测试集路径
            self.lineEdit_testpath.setText(
                ".//dataset//test_data_2.csv;.//dataset//test_label_2.csv")
            
            model_type = 1
        
        # 显示模型加载状态
        if(model_type == 0):
            self.statusbar.showMessage("传统神经网络模型载入成功！")
        elif(model_type==1):
            self.statusbar.showMessage("量子神经网络模型载入成功！")
        else:
            self.statusbar.showMessage("模型未正确而载入！")


def RunWindow():
    app = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    RunWindow()

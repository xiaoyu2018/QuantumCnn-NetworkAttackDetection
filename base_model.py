from tensorflow_core import keras
from tensorflow_core import argmax
from tensorflow_core import newaxis
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np


intrusion_list_1 = ['normal.',  # normal
              'back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.',  # DOS
              'ipsweep.', 'nmap.', 'portsweep.', 'satan.',  # PROBE
              'ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezclient.', 'warezmaster.',  # R2L
              'buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']  # U2R

intrusion_list_2 = ['normal.',  # normal
                  'back.', 'neptune.', 'teardrop.',   # DOS
                  'ipsweep.',  'satan.',  # PROBE
                  'warezclient.', 'guess_passwd.'  # R2L
                  ]

intrusion_list_3 = ['normal.',  # normal
              'back.', 'neptune.', 'pod.', 'smurf.', 'teardrop.',  # DOS
              'ipsweep.', 'nmap.', 'portsweep.', 'satan.'  # PROBE
              ] 

class BaseModel():
    def __init__(self):
        self.is_save_model = False  # 是否保存训练模型
        
        self.train_accuracy = 2.5
        self.val_accuracy = 1.0
        self.test_accuracy = 1.0
        
        self.train_loss=1.0
        self.val_loss=1.0
        self.test_loss=1.0
        
        self.train_time = 0
        self.history = keras.callbacks.History
        self.model = keras.models.Model
        
        self.batch_size = 64 #批次大小
        self.epochs=100 #迭代轮数
        self.input_shape = ()
        self.num_classes= -1    #最终分类数

        self.model_name=""  
        self.data_mode = 2  # 选取数据集

    def LoadData(self):
        if(self.data_mode == 1):
            # 1号数据集39个特征，23分类
            self.input_shape = (39, 1)
            self.num_classes = 23

            self.train_data = pd.read_csv(
                './/dataset//train_data_1.csv', header=None).values
            self.train_label = pd.read_csv(
                './/dataset//train_label_1.csv', header=None).values
            self.val_data = pd.read_csv(
                './/dataset//val_data_1.csv', header=None).values
            self.val_label = pd.read_csv(
                './/dataset//val_label_1.csv', header=None).values
            self.test_data = pd.read_csv(
                './/dataset//test_data_1.csv', header=None).values
            self.test_label = pd.read_csv(
                './/dataset//test_label_1.csv', header=None).values
        
        elif(self.data_mode == 2):
            # 2号数据集10个特征，10分类
            self.input_shape = (12, 1)
            self.num_classes = 8

            self.train_data = pd.read_csv(
                './/dataset//train_data_2.csv', header=None).values
            self.train_label = pd.read_csv(
                './/dataset//train_label_2.csv', header=None).values
            self.val_data = pd.read_csv(
                './/dataset//val_data_2.csv', header=None).values
            self.val_label = pd.read_csv(
                './/dataset//val_label_2.csv', header=None).values
            self.test_data = pd.read_csv(
                './/dataset//test_data_2.csv', header=None).values
            self.test_label = pd.read_csv(
                './/dataset//test_label_2.csv', header=None).values

        elif(self.data_mode == 3):
            # 3号数据集19个特征，10分类
            self.input_shape = (19, 1)
            self.num_classes = 10

            self.train_data = pd.read_csv(
                './/dataset//train_data_3.csv', header=None).values
            self.train_label = pd.read_csv(
                './/dataset//train_label_3.csv', header=None).values
            self.val_data = pd.read_csv(
                './/dataset//val_data_3.csv', header=None).values
            self.val_label = pd.read_csv(
                './/dataset//val_label_3.csv', header=None).values
            self.test_data = pd.read_csv(
                './/dataset//test_data_3.csv', header=None).values
            self.test_label = pd.read_csv(
                './/dataset//test_label_3.csv', header=None).values
        # 调整数据输入形状
        self.Reshpae()

    def Reshpae(self):
        #因为送入神经网络是成batch的
        #所以要把数据变shape为（self.train_data.shape[0], 39, 1）
        #一维卷积只能在列维度移动所以要（self.train_data.shape[0], 39, 1）
        if(self.data_mode == 1):
            self.train_data = self.train_data.reshape(
                self.train_data.shape[0], 39, 1)
            self.test_data = self.test_data.reshape(
                self.test_data.shape[0], 39, 1)
            self.val_data = self.val_data.reshape(
                self.val_data.shape[0], 39, 1)
        elif(self.data_mode == 2):
            self.train_data = self.train_data.reshape(
                self.train_data.shape[0], 12, 1)
            self.test_data = self.test_data.reshape(
                self.test_data.shape[0], 12, 1)
            self.val_data = self.val_data.reshape(
                self.val_data.shape[0], 12, 1)
        elif(self.data_mode == 3):
            self.train_data = self.train_data.reshape(
                self.train_data.shape[0], 19, 1)
            self.test_data = self.test_data.reshape(
                self.test_data.shape[0], 19, 1)
            self.val_data = self.val_data.reshape(
                self.val_data.shape[0], 19, 1)

    def LoadModle(self, path):
        self.model = keras.models.load_model(path)

    def RandomTest(self):
        num = random.randint(0, len(self.test_data)-1)
        # 变为模型能接受的形式
        x_predict = self.test_data[num]
        x_predict = x_predict[newaxis,...]

        predict = self.model.predict(x_predict)
        print(predict,self.test_label[num])
        if(self.data_mode == 1):
            real = intrusion_list_1[int(self.test_label[num])]
            pred = intrusion_list_1[argmax(predict[0], axis=-1)]
        elif(self.data_mode == 2):
            real = intrusion_list_2[int(self.test_label[num])]
            pred = intrusion_list_2[argmax(predict[0], axis=-1)]
        elif(self.data_mode == 3):
            real = intrusion_list_3[int(self.test_label[num])]
            pred = intrusion_list_3[argmax(predict[0], axis=-1)]
        
        print('真实值：', real)
        print('检测值：', pred)

        return real,pred

    def Evaluate(self):
        
        # 将测试集输入到训练好的模型中，查看测试集的误差
        score = self.model.evaluate(self.test_data, self.test_label,
                                    verbose=1, batch_size=64)
        self.test_accuracy = score[1]
        self.test_loss = score[0]
        print('Test loss:', score[0])
        print('Test accuracy: %.2f%%' % (score[1] * 100))

    def SaveTrainProcess(self):
        # 保存训练结果txt文件
        with open(f".//mymodles//{self.model_name}_{self.data_mode}.txt", "w") as f:
            f.writelines(line+'\n' for line in[str(round(self.train_loss,7)),str(self.train_accuracy),
                                               str(round(self.val_loss, 7)), str(self.val_accuracy),str(self.train_time)])
        

        # 保存训练过程图片
        acc = self.history.history['sparse_categorical_accuracy']
        val_acc = self.history.history['val_sparse_categorical_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        
        plt.subplot(2, 1, 1)
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.title('Training and Validation Accuracy of '+self.model_name)
        plt.legend()
        #设置坐标轴刻度
        my_x_ticks = np.arange(0, self.epochs, 1)
        my_y_ticks = np.arange(0.5, 1, 0.05)
        # plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)

        plt.subplot(2, 1, 2)
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss of '+self.model_name)
        plt.legend()
        #设置坐标轴刻度
        my_x_ticks = np.arange(0, self.epochs, 1)
        my_y_ticks = np.arange(0, 2, 0.15)
        # plt.xticks(my_x_ticks)
        plt.yticks(my_y_ticks)

        # 调整图片使不重叠
        plt.tight_layout()
        plt.savefig('.//mymodles//'+self.model_name+"_"+str(self.data_mode)+'.jpg')
        plt.clf()

        # 等待子类实现
        def Train(self):
            pass

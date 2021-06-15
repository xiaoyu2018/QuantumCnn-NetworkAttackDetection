import tensorflow_quantum as tfq
from tensorflow_core import argmax
from tensorflow_core import newaxis
from tensorflow_core import dtypes
from tensorflow_core import keras

from datetime import datetime
import matplotlib.pyplot as plt

import cirq
import sympy
import numpy as np

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

intrusion_list = ['normal.',  # normal
                  'back.', 'neptune.', 'teardrop.',   # DOS
                    'ipsweep.',  'satan.',  # PROBE
                    'warezclient.', 'guess_passwd.'  # R2L
                    ]

intrusion_list_3 = ['normal.',  # normal
                    'back.', 'neptune.', 'pod.', 'smurf.', 'teardrop.',  # DOS
                    'ipsweep.', 'nmap.', 'portsweep.', 'satan.'  # PROBE
                    ]

class MyQnnModel():
    def __init__(self):
        
        self.train_accuracy = 2.5
        self.train_loss = 2.5
        self.val_accuracy = 1.0
        self.val_loss = 1.0
        self.test_accuracy = 1.0
        self.test_loss = 1.0
        self.train_time = 0

        self.train_over = False
        self.is_save = False

        self.model = keras.models.Model
        self.history = keras.callbacks.History

        self.model_name = ""
        self.batch_size = 16  # 批次大小
        self.epochs = 70  # 迭代轮数
        self.num_classes = 8 #分类数
        self.features = 12 #数据特征数
        
        # 默认使用2号数据集
        self.traindata_path = ".//dataset//train_data_2.csv"
        self.trainlabel_path = ".//dataset//train_label_2.csv"
        self.valdata_path = ".//dataset//val_data_2.csv"
        self.vallabel_path = ".//dataset//val_label_2.csv"
        self.testdata_path = ".//dataset//test_data_2.csv"
        self.testlabel_path = ".//dataset//test_label_2.csv"


        self.train_data = []
        self.train_label = []
        self.val_data = []
        self.val_label = []
        self.test_data = []
        self.test_label = []

        # 初始量子比特
        self.quantum_bits = cirq.GridQubit.rect(1, self.features)

    # 将经典数据设置为旋转z门系数，8个量子比特首先进入旋转Z门
    # 达到将经典数据转化为量子数据的目的
    def ThetasAppend(self, bits, classic_data):
        circuit = cirq.Circuit()
        for i in range(self.features):
            circuit += [cirq.rz(classic_data[i])(bits[i])]
        # 返回的量子线路将作为输入
        return circuit

    # 载入数据
    def LoadData(self):
        print(self.features,
              len(self.trainlabel_path), self.valdata_path)
        # 更新量子比特
        self.quantum_bits = cirq.GridQubit.rect(1, self.features)

        train_data = np.genfromtxt(
            self.traindata_path, delimiter=",")
        train_label = np.genfromtxt(
            self.trainlabel_path, delimiter=",")
        val_data = np.genfromtxt(
            self.valdata_path, delimiter=",")
        val_label = np.genfromtxt(
            self.vallabel_path, delimiter=",")
        test_data = np.genfromtxt(
            self.testdata_path, delimiter=",")
        test_label = np.genfromtxt(
            self.testlabel_path, delimiter=",")

        thetas = []
        n_data = len(train_label)
        # 逐条将经典数据转换为量子数据
        for n in range(n_data):
            thetas.append(self.ThetasAppend(
                self.quantum_bits, train_data[n]))
        # 将量子线路转换为dtype为string的张量形式
        self.train_data = tfq.convert_to_tensor(thetas)
        self.train_label = np.array(train_label)

        thetas = []
        n_data = len(val_label)
        for n in range(n_data):
            thetas.append(self.ThetasAppend(
                self.quantum_bits, val_data[n]))
        self.val_data = tfq.convert_to_tensor(thetas)
        self.val_label = np.array(val_label)

        thetas = []
        n_data = len(test_label)
        for n in range(n_data):
            thetas.append(self.ThetasAppend(
                self.quantum_bits, test_data[n]))
        self.test_data = tfq.convert_to_tensor(thetas)
        self.test_label = np.array(test_label)

    # 加载模型
    def LoadModle(self, path):
        self.model.load_weights(path)

    # 全集检测
    def Evaluate(self):
        if(self.is_save == False):
            print('未保存')
            return
        # 将测试集输入到训练好的模型中，查看测试集的误差
        score = self.model.evaluate(self.test_data, self.test_label,
                                    verbose=1, batch_size=128)
        self.test_accuracy = score[1]
        self.test_loss = score[0]
        print('Test loss:', score[0])
        print('Test accuracy: %.2f%%' % (score[1] * 100))
        
        # 保存训练结果txt文件
        with open(f".//mymodles//{self.model_name}_alltest.txt", "w") as f:
            f.writelines(line+'\n' for line in[str(round(self.test_loss, 7)), str(self.test_accuracy)])
    # 随机检测
    def RandomTest(self):
        if(self.is_save==False):
            print('未保存')
            return
        # 获取所有单条检测值保存到txt文件
        with open(f".//mymodles//{self.model_name}_randomtest.txt", "w") as f:
            for num in range(len(self.test_data)):

                # 变为模型能接受的形式
                x_predict = self.test_data[num]
                x_predict = x_predict[newaxis, ...]

                predict = self.model.predict(x_predict)
                print(predict, self.test_label[num])
                real = intrusion_list[int(self.test_label[num])]
                pred = intrusion_list[argmax(predict[0], axis=-1)]

                print('真实值：', real)
                print('检测值：', pred)
                f.write(f"{real} {pred}\n")

    # 开始训练
    def Train(self):

        if(self.model_name == "HQcnn_s"):
            model = self.GetHModel_s()
        elif(self.model_name == "HQcnn_m"):
            model = self.GetHModel_m()

        self.model = model
        

        start_time = datetime.now()

        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=['sparse_categorical_accuracy'])

        # # 存储模型的回调函数
        # cp_callback= keras.callbacks.ModelCheckpoint(filepath=f".//mymodles//{self.model_name}_model.ckpt",
        #                                              save_weights_only=True,
        #                                              save_best_only=True)

        self.history = model.fit(self.train_data,
                            self.train_label,
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=1,
                            validation_data=(self.val_data, self.val_label)
                            )
        end_time = datetime.now()
        self.train_time = (end_time-start_time).seconds

        self.train_accuracy = self.history.history['sparse_categorical_accuracy'][self.epochs-1]
        self.val_accuracy = self.history.history['val_sparse_categorical_accuracy'][self.epochs-1]
        self.train_loss = self.history.history['loss'][self.epochs-1]
        self.val_loss = self.history.history['val_loss'][self.epochs-1]
        print(self.train_accuracy, self.val_accuracy)
        
        if(self.is_save==True):
            
            #保存训练过程图片
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
            plt.xticks(my_x_ticks)
            plt.yticks(my_y_ticks)

            plt.subplot(2, 1, 2)
            plt.plot(loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.title('Training and Validation Loss of '+self.model_name)
            plt.legend()
            #设置坐标轴刻度
            my_x_ticks = np.arange(0, self.epochs, 1)
            my_y_ticks = np.arange(0, 2, 0.15)
            plt.xticks(my_x_ticks)
            plt.yticks(my_y_ticks)
            # 调整图片使不重叠
            plt.tight_layout()
            plt.savefig('.//mymodles//'+self.model_name +'.jpg')
            plt.clf()

            # 保存训练结果txt文件
            with open(f".//mymodles//{self.model_name}.txt", "w") as f:
                f.writelines(line+'\n' for line in[str(round(self.train_loss, 7)), str(self.train_accuracy),
                                                str(round(self.val_loss, 7)), str(self.val_accuracy), str(self.train_time)])
        
        self.train_over = True

    
    # 基础量子线路 
    # 量子态编码线路(QSEC),每个量子比特都经过一个Hadamard门
    # 将初始为0态的量子比特振幅为（根号2，根号2）的叠加态
    def quantum_state_encoding_circuit(self,bits):
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(bits))
        return circuit

    # 单比特量子门,symbols为参数
    def one_qubit_unitary(self, bit, symbols):
        return cirq.Circuit(
            cirq.X(bit)**symbols[0],
            cirq.Y(bit)**symbols[1],
            cirq.Z(bit)**symbols[2])

    # 双比特量子门
    def two_qubit_unitary(self, bits, symbols):
        circuit = cirq.Circuit()
        circuit += self.one_qubit_unitary(bits[0], symbols[0:3])
        circuit += self.one_qubit_unitary(bits[1], symbols[3:6])
        circuit += [cirq.ZZ(*bits)**symbols[6]]
        circuit += [cirq.YY(*bits)**symbols[7]]
        circuit += [cirq.XX(*bits)**symbols[8]]
        circuit += self.one_qubit_unitary(bits[0], symbols[9:12])
        circuit += self.one_qubit_unitary(bits[1], symbols[12:])
        return circuit

    # 双比特池化门
    def two_qubit_pool(self, source_qubit, sink_qubit, symbols):
        pool_circuit = cirq.Circuit()
        sink_basis_selector = self.one_qubit_unitary(sink_qubit, symbols[0:3])
        source_basis_selector = self.one_qubit_unitary(
            source_qubit, symbols[3:6])
        pool_circuit.append(sink_basis_selector)
        pool_circuit.append(source_basis_selector)
        pool_circuit.append(cirq.CNOT(control=source_qubit, target=sink_qubit))
        pool_circuit.append(sink_basis_selector**-1)
        return pool_circuit

    # 量子卷积
    def quantum_conv_circuit(self, bits, symbols):
        circuit = cirq.Circuit()
        for first, second in zip(bits[0::2], bits[1::2]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        for first, second in zip(bits[1::2], bits[2::2] + [bits[0]]):
            circuit += self.two_qubit_unitary([first, second], symbols)
        return circuit

    # 量子池化
    def quantum_pool_circuit(self, source_bits, sink_bits, symbols):
        circuit = cirq.Circuit()
        for source, sink in zip(source_bits, sink_bits):
            circuit += self.two_qubit_pool(source, sink, symbols)
        return circuit

    
    # 量子卷积神经网络模型
    # 量子卷积池化线路
    def multi_readout_model_circuit(self, qubits):
        model_circuit = cirq.Circuit()
        symbols = sympy.symbols('qconv0:21')
        model_circuit += self.quantum_conv_circuit(qubits, symbols[0:15])
        model_circuit += self.quantum_pool_circuit(qubits[:int(self.features/2)], qubits[int(self.features/2):],
                                                   symbols[15:21])
        return model_circuit

    # 带单量子滤波器的量子卷积神经网络
    def GetHModel_s(self):

        # 在Cirq中创建qubits以及测量操作
        readouts = [cirq.Z(bit)
                    for bit in self.quantum_bits[int(self.features/2):]]

        qdata_input = keras.Input(
            shape=(), dtype=dtypes.string)

        qdata_state = tfq.layers.AddCircuit()(
            qdata_input, prepend=self.quantum_state_encoding_circuit(self.quantum_bits))

        quantum_model = tfq.layers.PQC(
            self.multi_readout_model_circuit(self.quantum_bits),
            readouts)(qdata_state)

        dense_1 = keras.layers.Dense(
            16, activation='relu')(quantum_model)

        dense_2 = keras.layers.Dense(self.num_classes,
                                     activation='softmax')(dense_1)

        hybrid_model = keras.Model(
            inputs=[qdata_input], outputs=[dense_2])

        return hybrid_model

    # 带多量子滤波器的量子卷积神经网络
    def GetHModel_m(self):

        # 在Cirq中创建qubits以及测量操作
        readouts = [cirq.Z(bit) for bit in self.quantum_bits[int(self.features/2):]]

        qdata_input = keras.Input(
            shape=(), dtype=dtypes.string)

        qdata_state = tfq.layers.AddCircuit()(
            qdata_input, prepend=self.quantum_state_encoding_circuit(self.quantum_bits))

        # 实现三个量子滤波器
        quantum_model_multi1 = tfq.layers.PQC(
            self.multi_readout_model_circuit(self.quantum_bits),
            readouts)(qdata_state)

        quantum_model_multi2 = tfq.layers.PQC(
            self.multi_readout_model_circuit(self.quantum_bits),
            readouts)(qdata_state)

        quantum_model_multi3 = tfq.layers.PQC(
            self.multi_readout_model_circuit(self.quantum_bits),
            readouts)(qdata_state)

        # 将测量所得的输出输入到一个经典神经网络中
        concat_out = keras.layers.concatenate(
            [quantum_model_multi1, quantum_model_multi2, quantum_model_multi3])

        dense_1 = keras.layers.Dense(16, 
                                    activation='relu')(concat_out)

        dense_2 = keras.layers.Dense(self.num_classes,
                                    activation='softmax')(dense_1)

        multi_qconv_model = keras.Model(inputs=[qdata_input],
                                           outputs=[dense_2])
        return multi_qconv_model



if __name__ == "__main__":
    model = MyQnnModel()
    model.model_name = "HQcnn_s"
    model.epochs=50
    model.LoadData()
    model.Train()
    model.Evaluate()
    model.RandomTest()




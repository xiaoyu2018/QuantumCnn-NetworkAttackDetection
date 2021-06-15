from sklearn import preprocessing
from cirq.contrib.svg import SVGCircuit
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import math

#数据载入
def thetas_append(bits, datas_minmax):
    circuit = cirq.Circuit()
    circuit += [cirq.rz(datas_minmax[0])(bits[0])]  # -2 * math.pi *
    circuit += [cirq.rz(datas_minmax[1])(bits[1])]
    circuit += [cirq.rz(datas_minmax[2])(bits[2])]
    circuit += [cirq.rz(datas_minmax[3])(bits[3])]
    circuit += [cirq.rz(datas_minmax[4])(bits[4])]
    circuit += [cirq.rz(datas_minmax[5])(bits[5])]
    return circuit


def datasets(bits):
    dataTotal = np.genfromtxt(
        ".//dataset//1back_ipsweep_normal_3000 -6 _1.csv", delimiter=",")
    datas = dataTotal[:, 0:6]
    labels = dataTotal[:, -1]
    datas_minmax = preprocessing.MinMaxScaler().fit_transform(datas)
    thetas = []
    n_data = len(labels)
    for n in range(n_data):
        thetas.append(thetas_append(bits, datas_minmax[n]))
    permutation = np.random.permutation(range(n_data))

    split_ind = int(n_data * 0.7)

    train_datas = thetas[:split_ind]
    test_datas = thetas[split_ind:]

    train_labels = labels[:split_ind]
    test_labels = labels[split_ind:]

    return tfq.convert_to_tensor(train_datas), np.array(train_labels), \
        tfq.convert_to_tensor(test_datas), np.array(test_labels)



# 1. 量子态编码线路(QSEC)
def quantum_state_encoding_circuit(bits):
    """根据`bits`构建并返回量子态编码线路."""
    circuit = cirq.Circuit()
    circuit.append(cirq.H.on_each(bits))
    return circuit


# 2. 变分量子线路(VQC)
# 单条R门变分量子线路
def one_qubit_unitary(bit, symbols):
    return cirq.Circuit(
        cirq.Y(bit)**symbols[0],
        cirq.Z(bit)**symbols[1])


# R门变分量子线路
def variational_quantum_circuit_R(bits, symbols):
    circuit = cirq.Circuit()
    circuit += one_qubit_unitary(bits[0], symbols[0:2])
    circuit += one_qubit_unitary(bits[1], symbols[2:4])
    circuit += one_qubit_unitary(bits[2], symbols[4:6])
    circuit += one_qubit_unitary(bits[3], symbols[6:8])
    circuit += one_qubit_unitary(bits[4], symbols[8:10])
    circuit += one_qubit_unitary(bits[5], symbols[10:12])
    return circuit

# CZ门变分量子线路
def variational_quantum_circuit_CZ(bits):
    circuit = cirq.Circuit()
    for this_bit, next_bit in zip(bits, bits[1:] + [bits[0]]):
        circuit.append(cirq.CZ(this_bit, next_bit))
    return circuit


# 变分量子线路
def variational_quantum_circuit(bits, symbols):
    circuit = cirq.Circuit()
    circuit += variational_quantum_circuit_R(bits, symbols[0:12])
    circuit += variational_quantum_circuit_CZ(bits)
    circuit += variational_quantum_circuit_R(bits, symbols[12:24])
    circuit += variational_quantum_circuit_CZ(bits)
    circuit += variational_quantum_circuit_R(bits, symbols[24:36])
    circuit += variational_quantum_circuit_CZ(bits)
    circuit += variational_quantum_circuit_R(bits, symbols[36:48])
    circuit += variational_quantum_circuit_CZ(bits)
    circuit += variational_quantum_circuit_R(bits, symbols[48:60])
    circuit += variational_quantum_circuit_CZ(bits)
    circuit += variational_quantum_circuit_R(bits, symbols[60:72])

    return circuit


# 3. 模型构建

# 创建自定义层
def create_model_circuit(qubits):
    model_circuit = cirq.Circuit()
    symbols = sympy.symbols('w0:72')
    # Cirq通过sympy.Symbols将符号映射为待学习变量
    # TensorFlow Quantum扫描电路的输入并将其替换为TensorFlow变量
    model_circuit += variational_quantum_circuit(qubits, symbols[0:72])

    return model_circuit


# 在Cirq中创建qubits以及测量操作
establish_qbits = cirq.GridQubit.rect(1, 6)
readout_operators0 = cirq.Z(establish_qbits[0])
readout_operators1 = cirq.Z(establish_qbits[1])
readout_operators2 = cirq.Z(establish_qbits[2])
readout_operators3 = cirq.Z(establish_qbits[3])
readout_operators4 = cirq.Z(establish_qbits[4])
readout_operators5 = cirq.Z(establish_qbits[5])


# 将经典数据输入QSEC并作为AddCircuit的一部分
datas_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
datas_state = tfq.layers.AddCircuit()(
    datas_input, prepend=quantum_state_encoding_circuit(establish_qbits))

quantum_model = tfq.layers.PQC(create_model_circuit(
    establish_qbits), readout_operators0)(datas_state)

#quantum_model = tfq.layers.PQC(create_model_circuit(establish_qbits),[readout_operators0, readout_operators1, readout_operators2, readout_operators3])(datas_state)

vqnn_model = tf.keras.Model(inputs=[datas_input], outputs=[quantum_model])


# 4训练模型
# 获取训练集与测试集
train_datas, train_labels, test_datas, \
test_labels = datasets(establish_qbits)

# 参数设置
@tf.function
def custom_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0.5 else 0.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))


vqnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   loss=tf.losses.mse,
                   metrics=[custom_accuracy])

history = vqnn_model.fit(x=train_datas,
                         y=train_labels,
                         #batch_size=140,
                         epochs=20,
                         verbose=1,
                         validation_data=(test_datas, test_labels))


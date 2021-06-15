from base_model import BaseModel
from tensorflow_core import keras
from datetime import datetime







class MyCnnModel(BaseModel):
    def __init__(self) :
        BaseModel.__init__(self)
        self.model_name="CNN"   

    def Train(self):
        start_time=datetime.now()

        
        #将标签转换为独热编码
        # self.train_label=keras.utils.to_categorical(self.train_label)
        # self.test_label=keras.utils.to_categorical(self.test_label)


        self.model = keras.Sequential()  # sequential序贯模型:多个网络层的线性堆叠
        #输出的维度（卷积滤波器的数量）filters=32；1D卷积窗口的长度kernel_size=3；激活函数activation   模型第一层需指定input_shape：
        # data_format默认channels_last（39，1）
        self.model.add(keras.layers.Conv1D(
            32, 3, activation='relu',  input_shape=self.input_shape, name='convolution_layer'))
        # 池化层：最大池化  池化窗口大小pool_size=2
        self.model.add(keras.layers.MaxPooling1D(
            pool_size=(2), name='pooling_layer'))
        self.model.add(keras.layers.Flatten(
            name='flatten_layer'))  # 展平一个张量，返回一个调整为1D的张量
        
        # 防止过拟合
        # 需要丢弃的输入比例=0.25    dropout正则化-减少过拟合
        # self.model.add(keras.layers.Dropout(0.25))
        
        self.model.add(keras.layers.Dense(64, activation='relu',
                                    name='hidden_layer'))  # 全连接层
        # 需要丢弃的输入比例=0.25    dropout正则化-减少过拟合
        # self.model.add(keras.layers.Dropout(0.15))
        self.model.add(keras.layers.Dense(
            self.num_classes, activation='softmax', name='softmax_layer'))

        #编译，损失函数:多类对数损失，用于多分类问题， 优化函数：adadelta， 模型性能评估是准确率
        #输入特征经softmax层转换为概率分布，from_logits=False
        #label未转换为独热码，特征为概率分布，metrics用sparse_categorical_accuracy
        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           optimizer=keras.optimizers.Adam(), metrics=['sparse_categorical_accuracy'])
        #运行 ， verbose=1输出进度条记录      epochs训练的轮数     batch_size:指定进行梯度下降时每个batch包含的样本数
        self.history = self.model.fit(self.train_data, self.train_label, batch_size=self.batch_size,
                                 epochs=self.epochs, verbose=1, validation_data=(self.val_data,self.val_label))
        end_time=datetime.now()
        self.train_time=(end_time-start_time).seconds
        
        #保存
        self.train_accuracy = self.history.history['sparse_categorical_accuracy'][self.epochs-1]
        self.val_accuracy = self.history.history['val_sparse_categorical_accuracy'][self.epochs-1]
        self.train_loss = self.history.history['loss'][self.epochs-1]
        self.val_loss = self.history.history['val_loss'][self.epochs-1]

        if(self.is_save_model):
            self.model.save(".//mymodles//Cnn_model_"+str(self.data_mode)+".h5")

if __name__ == "__main__":
    mymodel=MyCnnModel()
    
    mymodel.data_mode = 2
    mymodel.epochs = 50
    mymodel.batch_size = 64
    mymodel.is_save_model = False

    mymodel.LoadData()
    mymodel.Train()
    print(mymodel.train_time)
    print(mymodel.train_accuracy)
    # mymodel.SaveTrainProcess()
    # mymodel.LoadModle(f'.//mymodles//Cnn_model_{mymodel.data_mode}.h5')
    
    mymodel.Evaluate()
    # mymodel.model.summary()
    mymodel.RandomTest()



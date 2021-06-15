from base_model import BaseModel
from tensorflow_core import keras
from datetime import datetime

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class MyBPModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
        self.model_name = "BP"

    def Train(self):
        start_time = datetime.now()

        

        self.model = keras.Sequential()  

        # 展平一个张量，返回一个调整为1D的张量
        self.model.add(keras.layers.Flatten(
            input_shape=self.input_shape, name="flatten_layer"))

        #防止过拟合
        #model.add(Dropout(0.25))  #需要丢弃的输入比例=0.25    dropout正则化-减少过拟合
        self.model.add(
            keras.layers.Dense(64, activation='relu',name="hidden_layer"))
        self.model.add(keras.layers.Dense(
            self.num_classes, activation='softmax', name='softmax_layer'))

        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                           optimizer=keras.optimizers.Adam(), metrics=['sparse_categorical_accuracy'])
        
        self.history = self.model.fit(self.train_data, self.train_label, batch_size=self.batch_size,
                                      epochs=self.epochs, verbose=1, validation_data=(self.val_data, self.val_label))

        end_time = datetime.now()
        self.train_time = (end_time-start_time).seconds

        #保存
        self.train_accuracy = self.history.history['sparse_categorical_accuracy'][self.epochs-1]
        self.val_accuracy = self.history.history['val_sparse_categorical_accuracy'][self.epochs-1]
        self.train_loss = self.history.history['loss'][self.epochs-1]
        self.val_loss = self.history.history['val_loss'][self.epochs-1]

        if(self.is_save_model):
            self.model.save(".//mymodles//Bp_model_"+str(self.data_mode)+".h5")


if __name__ == "__main__":
    mymodel = MyBPModel()
    mymodel.data_mode=2
    mymodel.epochs=50
    mymodel.batch_size=128
    mymodel.is_save_model=False

    mymodel.LoadData()
    mymodel.Train()
    print(mymodel.train_time)
    print(mymodel.train_accuracy)
    # mymodel.SaveTrainProcess()
    # mymodel.LoadModle(f'.//mymodles//Bp_model_{mymodel.data_mode}.h5')
    
    mymodel.Evaluate()
    # mymodel.model.summary()
    # mymodel.RandomTest()

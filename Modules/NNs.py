#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
#import time

from FUNCTIONS import load_dic_from_json
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import Input        
from tensorflow.keras.layers import Activation, AveragePooling2D, AveragePooling1D, BatchNormalization
from tensorflow.keras.layers import Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, LSTM, MaxPooling1D, MaxPooling2D


class NN_model(object):
    
    def __init__(self):
        
        self.diretorio = os.getcwd()


    def set_parameters(self, parameters_dic, architecture = None):

        #general parameters
        self.n_architecture = architecture
        self.parameters = parameters_dic
        self.feature = parameters_dic['feature']
        self.section_name = parameters_dic['section_name']
        self.machine_type = parameters_dic['machine_type']
        self.sent_batch_size = parameters_dic['sent_batch_size']
        
        if self.feature is None:
            self.training_mode = 'sections'
            self.subdescription = self.section_name
        else:
            self.training_mode = 'sentences'        
            self.subdescription = self.feature
              
    
    def get_model(self):

        
        model_save_folder = None
        #caso já exista o modelo treinado para sentenças
        if self.section_name is None:
            if os.path.exists(self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'):
                model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
        
        #caso já exista o modelo treinado para sections    
        else:
            if os.path.exists(self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'):
                model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
        
        #carregando o modelo
        if model_save_folder:
            model = load_model(model_save_folder)
            print('\nCarregando NN já criada...')
            print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
            print('Carregando o arquivo h5 com o modelo...')
            model.summary()
            return model
        
        #try:
        if self.machine_type == 'conv1d':
                        
            print('Setting CONV1D...')
            
            #determinando o input_shape
            input_shape = self.parameters['input_shape']['wv']
            
            #inputs
            inputs = Input(shape = input_shape)
            print('(Input) inputs.shape: ', inputs.shape)
                
            #l1 - conv1d para os word-vectors (inputs1)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv1d']['l1'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l1'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d']['l1'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d']['l1'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d']['l1'],
                            strides = self.parameters['strides']['conv1d']['l1']
                            )
            x1 = conv1D(inputs)
            print('(l1) Conv1D x1.shape: ', x1.shape)
            x1 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last")(x1)
            print('(l1) Pooling1D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l1) BatchNormalization x1.shape: ', x1.shape)
            
            #l2
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d']['l2'],
                            kernel_size = self.parameters['kernel_size']['conv1d']['l2'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l2'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d']['l2'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d']['l2'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d']['l2'],
                            strides = ( self.parameters['strides']['conv1d']['l2'] )
                            )
            x1 = conv1D(x1)
            print('(l2) Conv1d x1.shape: ', x1.shape)
            x1 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last" )(x1)
            print('(l2) Pooling1D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l2) BatchNormalization x1.shape: ', x1.shape)
            x1 = Flatten()(x1)
            print('(l2) Flatten x1.shape: ', x1.shape)
            
            #l3
            x1 = Dense(units = self.parameters['units']['conv1d']['l3'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l3'],
                       activation = self.parameters['activation']['conv1d']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv1d']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)

            #l4
            x1 = Dense(units = self.parameters['units']['conv1d']['l4'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l4'],
                       activation = self.parameters['activation']['conv1d']['l4'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv1d']['l4'])(x1)
            print('(l4) Dense x1.shape: ', x1.shape)

            #l5
            x1 = Dense(units = self.parameters['units']['conv1d']['l5'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l5'],
                       activation = self.parameters['activation']['conv1d']['l5'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv1d']['l5'])(x1)
            print('(l5) Dense x1.shape: ', x1.shape)

            #l6
            outputs = Dense(units = self.parameters['units']['conv1d']['l6'], 
                            activation = self.parameters['activation']['conv1d']['l6'])(x1)
            print('(l6) Dense x1.shape: ', outputs.shape)

            #time.sleep(30)

            #compile
            model = Model(inputs=inputs, outputs=outputs, name='conv1d')
            if self.parameters['optimizer']['conv1d'].lower() == 'sgd':
                    from tensorflow.keras.optimizers import SGD
                    chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['conv1d'].lower() == 'adam':
                    from tensorflow.keras.optimizers import Adam
                    chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['conv1d'].lower() == 'rms':                    
                    from tensorflow.keras.optimizers import RMSprop
                    chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['conv1d'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()

            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)            
            
            return model


        elif self.machine_type == 'conv2d':

            #determinando o input_shape
            input_shape = self.parameters['input_shape']['wv']
            
            #inputs
            inputs = Input(shape = input_shape )
            print('(Input) inputs.shape: ', inputs.shape)
                
            #l1 - conv2d para os word-vectors (inputs1)
            conv2D = Conv2D(filters = self.parameters['filters']['conv2d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv2d']['l1'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv2d']['l1'],
                            padding = 'valid',
                            data_format = 'channels_last',
                            activation = self.parameters['activation']['conv2d']['l1'],
                            strides = ( self.parameters['strides']['conv2d']['l1'], self.parameters['strides']['conv2d']['l1'] )
                            )
            x1 = conv2D(inputs)
            print('(l1) Conv2D x1.shape: ', x1.shape)
            x1 = MaxPooling2D(pool_size=(2,2), strides=1, padding="valid", data_format="channels_last")(x1)
            print('(l1) Pooling2D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l1) BatchNormalization x1.shape: ', x1.shape)

            #l2 - conv2d
            conv2D = Conv2D(filters = self.parameters['filters']['conv2d']['l2'],
                            kernel_size = self.parameters['kernel_size']['conv2d']['l2'],
                            padding = 'valid',
                            data_format = 'channels_last',
                            activation = self.parameters['activation']['conv2d']['l2'],
                            strides = ( self.parameters['strides']['conv2d']['l2'], self.parameters['strides']['conv2d']['l2'] )
                            )
            x1 = conv2D(x1)
            print('(l2) Conv2D x1.shape: ', x1.shape)
            x1 = MaxPooling2D(pool_size=(3,3), strides=2, padding="valid", data_format="channels_last")(x1)
            print('(l2) Pooling2D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l2) BatchNormalization x1.shape: ', x1.shape)
            x1 = Flatten()(x1)
            print('(l2) Flatten x1.shape: ', x1.shape)
            
            #l3 - dense
            x1 = Dense(units = self.parameters['units']['conv2d']['l3'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d']['l3'],
                       activation = self.parameters['activation']['conv2d']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)

            #l4 - dense
            x1 = Dense(units = self.parameters['units']['conv2d']['l4'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d']['l4'],
                       activation = self.parameters['activation']['conv2d']['l4'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l4'])(x1)
            print('(l4) Dense x1.shape: ', x1.shape)

            #l5 - dense
            x1 = Dense(units = self.parameters['units']['conv2d']['l5'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d']['l5'],
                       activation = self.parameters['activation']['conv2d']['l5'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l5'])(x1)
            print('(l5) Dense x1.shape: ', x1.shape)

            #l6 - dense
            outputs = Dense(units = self.parameters['units']['conv2d']['l6'], 
                            activation = self.parameters['activation']['conv2d']['l6'])(x1)
            print('(l6) Dense x1.shape: ', outputs.shape)

            #compile
            model = Model(inputs=inputs, outputs=outputs, name='conv2d')
            if self.parameters['optimizer']['conv2d'].lower() == 'sgd':
                    from tensorflow.keras.optimizers import SGD
                    chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['conv2d'].lower() == 'adam':
                    from tensorflow.keras.optimizers import Adam
                    chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['conv2d'].lower() == 'rms':                    
                    from tensorflow.keras.optimizers import RMSprop
                    chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['conv2d'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()

            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)            
            
            return model


        elif self.machine_type == 'lstm':
            
            print('Setting LSTM...')

            #determinando o input_shape em função do vetor usado
            input_shape = self.parameters['input_shape']['wv']
            
            #inputs
            inputs = Input(shape = input_shape)
            
            #l1 - lstm
            lstm = LSTM(units = self.parameters['units']['lstm']['l1'],
                        kernel_initializer = self.parameters['kernel_initializer']['lstm']['l1'],
                        activation = self.parameters['activation']['lstm']['l1'],
                        dropout = self.parameters['dropout']['lstm']['l1'],
                        return_sequences = True)
            
            x1 = lstm(inputs)
            print('(l1) LSTM x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l1) BatchNormalization x1.shape: ', x1.shape)

            #l2 - lstm
            lstm = LSTM(units = self.parameters['units']['lstm']['l2'],
                        kernel_initializer = self.parameters['kernel_initializer']['lstm']['l2'],
                        activation = self.parameters['activation']['lstm']['l2'],
                        dropout = self.parameters['dropout']['lstm']['l2'])
            
            x1 = lstm(x1)
            print('(l2) LSTM x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l2) BatchNormalization x1.shape: ', x1.shape)
            
            #l3 - dense
            x1 = Dense(units = self.parameters['units']['lstm']['l3'], 
                       kernel_initializer=self.parameters['kernel_initializer']['lstm']['l3'],
                       activation = self.parameters['activation']['lstm']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['lstm']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)

            #l4 - dense
            x1 = Dense(units = self.parameters['units']['lstm']['l4'], 
                       kernel_initializer=self.parameters['kernel_initializer']['lstm']['l4'],
                       activation = self.parameters['activation']['lstm']['l4'])(x1)
            x1 = Dropout(self.parameters['dropout']['lstm']['l4'])(x1)
            print('(l4) Dense x1.shape: ', x1.shape)

            #l5 - dense
            x1 = Dense(units = self.parameters['units']['lstm']['l5'], 
                       kernel_initializer=self.parameters['kernel_initializer']['lstm']['l5'],
                       activation = self.parameters['activation']['lstm']['l5'])(x1)
            x1 = Dropout(self.parameters['dropout']['lstm']['l5'])(x1)
            print('(l5) Dense x1.shape: ', x1.shape)

            #l6 - dense
            outputs = Dense(units = self.parameters['units']['lstm']['l6'], 
                            activation = self.parameters['activation']['lstm']['l6'])(x1)
            print('(l6) Dense x1.shape: ', outputs.shape)


            #compile
            model = Model(inputs=inputs, outputs=outputs, name='lstm')
            if self.parameters['optimizer']['lstm'].lower() == 'sgd':
                from tensorflow.keras.optimizers import SGD
                chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['lstm'].lower() == 'adam':
                from tensorflow.keras.optimizers import Adam
                chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['lstm'].lower() == 'rms':                    
                from tensorflow.keras.optimizers import RMSprop
                chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['lstm'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)
            
            return model                
    
    
        elif self.machine_type == 'conv1d_lstm':
                    
            #inputs
            inputs1 = Input(shape = self.parameters['input_shape']['wv'] )
            inputs2 = Input(shape = self.parameters['input_shape']['tv'] )
            print('(Input1) inputs1.shape: ', inputs1.shape)
            print('(Input2) inputs2.shape: ', inputs2.shape)
            
            #l1 - conv1d para os word-vectors (inputs1)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d_lstm']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv1d_lstm']['l1'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d_lstm']['l1'],
                            strides = self.parameters['strides']['conv1d_lstm']['l1']            
                            )
            x1 = conv1D(inputs1)
            print('(l1) Conv1D x1.shape: ', x1.shape)
            x1 = GlobalMaxPooling1D()(x1)
            print('(l1) Pooling1D x1.shape: ', x1.shape)
            
            #l2
            x1 = Dense(units = self.parameters['units']['conv1d_lstm']['l2'], activation = self.parameters['activation']['conv1d_lstm']['l2'])(x1)
            x1 = Dropout(self.parameters['units']['conv1d_lstm']['l2'])(x1)
            print('(l2) Dense x1.shape: ', x1.shape)
            
            #l3
            x1 = Dense(units = self.parameters['units']['conv1d_lstm']['l3'], activation = self.parameters['activation']['conv1d_lstm']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)
            
            #l4 - lstm para os topic vectors (inputs2)
            lstm = LSTM(units = self.parameters['units']['conv1d_lstm']['l4'], 
                        return_sequences = True
                        )
            x2 = lstm(inputs2)
            x2 = Dropout(self.parameters['dropout']['conv1d_lstm']['l4'])(x2)
            x2 = Flatten()(x2)
            print('(l4) LSTM x2.shape: ', x2.shape)
            
            #l5
            x2 = Dense(units = self.parameters['units']['conv1d_lstm']['l5'], activation = self.parameters['activation']['conv1d_lstm']['l5'])(x2)
            print('(l5) Dense x2.shape: ', x2.shape)
            
            #concatenando os dois inputs
            concatX = Concatenate(axis=1)([x1, x2])
            print('(Concat) concatX.shape: ', concatX.shape)
            
            #l6 - dense
            x3 = Dense(units = self.parameters['units']['conv1d_lstm']['l6'], activation = self.parameters['activation']['conv1d_lstm']['l6'])(concatX)
            x3 = Dropout(self.parameters['dropout']['conv1d_lstm']['l6'])(x3)
            print('(l6) Dense x3.shape: ', x3.shape)
            
            #l7 - dense
            outputs = Dense(units = self.parameters['units']['conv1d_lstm']['l7'], activation = self.parameters['activation']['conv1d_lstm']['l7'])(x3)    
            print('(l7) Dense outputs.shape: ', outputs.shape)
            
            #compile
            model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='conv1d_lstm')
            if self.parameters['optimizer']['conv1d_lstm'].lower() == 'sgd':
                from tensorflow.keras.optimizers import SGD
                chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['conv1d_lstm'].lower() == 'adam':
                from tensorflow.keras.optimizers import Adam
                chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['conv1d_lstm'].lower() == 'rms':                    
                from tensorflow.keras.optimizers import RMSprop
                chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['conv1d_lstm'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)            
            
            return model            


        elif self.machine_type == 'conv1d_conv1d':

            print('Setting CONV1D...')

            #inputs
            inputs1 = Input(shape = self.parameters['input_shape']['wv'] )
            inputs2 = Input(shape = self.parameters['input_shape']['tv'] )
            print('(Input1) inputs1.shape: ', inputs1.shape)
            print('(Input2) inputs2.shape: ', inputs2.shape)
            
            #l1 - conv1d para os word-vectors (inputs1)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d_conv1d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv1d_conv1d']['l1'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l1'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d_conv1d']['l1'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d_conv1d']['l1'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d_conv1d']['l1'],
                            strides = self.parameters['strides']['conv1d_conv1d']['l1']
                            )
            x1 = conv1D(inputs1)
            print('(l1) Conv1D x1.shape: ', x1.shape)
            x1 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last")(x1)
            print('(l1) Pooling1D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l1) BatchNormalization x1.shape: ', x1.shape)

            #l2
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d_conv1d']['l2'],
                            kernel_size = self.parameters['kernel_size']['conv1d_conv1d']['l2'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l2'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d_conv1d']['l2'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d_conv1d']['l2'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d_conv1d']['l2'],
                            strides = ( self.parameters['strides']['conv1d_conv1d']['l2'] )
                            )
            x1 = conv1D(x1)
            print('(l2) Conv1d x1.shape: ', x1.shape)
            x1 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last" )(x1)
            print('(l2) Pooling1D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l2) BatchNormalization x1.shape: ', x1.shape)
            x1 = Flatten()(x1)
            print('(l2) Flatten x1.shape: ', x1.shape)

            
            #l3
            x1 = Dense(units = self.parameters['units']['conv1d_conv1d']['l3'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l3'],
                       activation = self.parameters['activation']['conv1d_conv1d']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv1d_conv1d']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)
            
            #l4 - conv1d para os topic-vectors (inputs2)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d_conv1d']['l4'],
                            kernel_size = self.parameters['kernel_size']['conv1d_conv1d']['l4'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l4'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d_conv1d']['l4'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d_conv1d']['l4'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d_conv1d']['l4'],
                            strides = self.parameters['strides']['conv1d_conv1d']['l4']
                            )
            x2 = conv1D(inputs2)
            print('(l4) Conv1D x2.shape: ', x2.shape)
            x2 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last")(x2)
            print('(l4) Pooling1D x2.shape: ', x2.shape)
            x2 = BatchNormalization(center=True, scale=True)(x2)
            print('(l4) BatchNormalization x2.shape: ', x2.shape)

            #l5
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d_conv1d']['l5'],
                            kernel_size = self.parameters['kernel_size']['conv1d_conv1d']['l5'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l5'],
                            kernel_regularizer=self.parameters['kernel_regularizer']['conv1d_conv1d']['l5'],
                            bias_regularizer=self.parameters['bias_regularizer']['conv1d_conv1d']['l5'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d_conv1d']['l5'],
                            strides = ( self.parameters['strides']['conv1d_conv1d']['l5'] )
                            )
            x2 = conv1D(x2)
            print('(l5) Conv1d x2.shape: ', x2.shape)
            x2 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last" )(x2)
            print('(l5) Pooling1D x2.shape: ', x2.shape)
            x2 = BatchNormalization(center=True, scale=True)(x2)
            print('(l5) BatchNormalization x2.shape: ', x2.shape)
            x2 = Flatten()(x2)
            print('(l5) Flatten x1.shape: ', x2.shape)

            #l6
            x2 = Dense(units = self.parameters['units']['conv1d_conv1d']['l6'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l6'],
                       activation = self.parameters['activation']['conv1d_conv1d']['l6'])(x2)
            x2 = Dropout(self.parameters['dropout']['conv1d_conv1d']['l6'])(x2)
            print('(l6) Dense x2.shape: ', x2.shape)

            #concatenando os dois inputs
            concatX = Concatenate(axis=1)([x1, x2])
            print('(Concat) concatX.shape: ', concatX.shape)

            #l7
            concatX = Dense(units = self.parameters['units']['conv1d_conv1d']['l7'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l7'],
                       activation = self.parameters['activation']['conv1d_conv1d']['l7'])(concatX)
            concatX = Dropout(self.parameters['dropout']['conv1d_conv1d']['l7'])(concatX)
            print('(l7) Dense concatX.shape: ', concatX.shape)

            #l8
            concatX = Dense(units = self.parameters['units']['conv1d_conv1d']['l8'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv1d_conv1d']['l8'],
                       activation = self.parameters['activation']['conv1d_conv1d']['l8'])(concatX)
            x1 = Dropout(self.parameters['dropout']['conv1d_conv1d']['l8'])(concatX)
            print('(l8) Dense concatX.shape: ', concatX.shape)

            #l9
            outputs = Dense(units = self.parameters['units']['conv1d_conv1d']['l9'], 
                            activation = self.parameters['activation']['conv1d_conv1d']['l9'])(concatX)
            print('(l9) Dense concatX.shape: ', outputs.shape)
            
            #compile
            model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='conv1d_conv1d')
            if self.parameters['optimizer']['conv1d_conv1d'].lower() == 'sgd':
                from tensorflow.keras.optimizers import SGD
                chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['conv1d_conv1d'].lower() == 'adam':
                from tensorflow.keras.optimizers import Adam
                chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['conv1d_conv1d'].lower() == 'rms':                    
                from tensorflow.keras.optimizers import RMSprop
                chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['conv1d_conv1d'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)            
            
            return model


        elif self.machine_type == 'conv2d_conv1d':

            print('Setting CONV1D...')

            #inputs
            inputs1 = Input(shape = self.parameters['input_shape']['wv'] )
            inputs2 = Input(shape = self.parameters['input_shape']['tv'] )
            print('(Input1) inputs1.shape: ', inputs1.shape)
            print('(Input2) inputs2.shape: ', inputs2.shape)
            
            #l1 - conv2d para os word-vectors (inputs1)
            conv2D = Conv2D(filters = self.parameters['filters']['conv2d_conv1d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv2d_conv1d']['l1'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l1'],
                            padding = 'valid',
                            data_format = 'channels_last',
                            activation = self.parameters['activation']['conv2d_conv1d']['l1'],
                            strides = ( self.parameters['strides']['conv2d_conv1d']['l1'], self.parameters['strides']['conv2d_conv1d']['l1'] )
                            )
            x1 = conv2D(inputs1)
            print('(l1) Conv2D x1.shape: ', x1.shape)
            x1 = MaxPooling2D(pool_size=(2,2), strides=1, padding="valid", data_format="channels_last")(x1)
            print('(l1) Pooling2D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l1) BatchNormalization x1.shape: ', x1.shape)

            #l2
            conv2D = Conv2D(filters = self.parameters['filters']['conv2d_conv1d']['l2'],
                            kernel_size = self.parameters['kernel_size']['conv2d_conv1d']['l2'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l2'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv2d_conv1d']['l2'],
                            strides = ( self.parameters['strides']['conv2d_conv1d']['l2'] )
                            )
            x1 = conv2D(x1)
            print('(l2) Conv2D x1.shape: ', x1.shape)
            x1 = MaxPooling2D(pool_size=(2,2), strides=1, padding="valid", data_format="channels_last")(x1)
            print('(l2) Pooling2D x1.shape: ', x1.shape)
            x1 = BatchNormalization(center=True, scale=True)(x1)
            print('(l2) BatchNormalization x1.shape: ', x1.shape)
            x1 = Flatten()(x1)
            print('(l2) Flatten x1.shape: ', x1.shape)

            
            #l3
            x1 = Dense(units = self.parameters['units']['conv2d_conv1d']['l3'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l3'],
                       activation = self.parameters['activation']['conv2d_conv1d']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d_conv1d']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)
            
            #l4 - conv1d para os topic-vectors (inputs2)
            conv1D = Conv1D(filters = self.parameters['filters']['conv2d_conv1d']['l4'],
                            kernel_size = self.parameters['kernel_size']['conv2d_conv1d']['l4'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l4'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv2d_conv1d']['l4'],
                            strides = self.parameters['strides']['conv2d_conv1d']['l4']
                            )
            x2 = conv1D(inputs2)
            print('(l4) Conv1D x2.shape: ', x2.shape)
            x2 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last")(x2)
            print('(l4) Pooling1D x2.shape: ', x2.shape)
            x2 = BatchNormalization(center=True, scale=True)(x2)
            print('(l4) BatchNormalization x2.shape: ', x2.shape)

            #l5
            conv1D = Conv1D(filters = self.parameters['filters']['conv2d_conv1d']['l5'],
                            kernel_size = self.parameters['kernel_size']['conv2d_conv1d']['l5'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l5'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv2d_conv1d']['l5'],
                            strides = ( self.parameters['strides']['conv2d_conv1d']['l5'] )
                            )
            x2 = conv1D(x2)
            print('(l5) Conv1d x2.shape: ', x2.shape)
            x2 = MaxPooling1D(pool_size=2, strides=1, padding="valid", data_format="channels_last" )(x2)
            print('(l5) Pooling1D x2.shape: ', x2.shape)
            x2 = BatchNormalization(center=True, scale=True)(x2)
            print('(l5) BatchNormalization x2.shape: ', x2.shape)
            x2 = Flatten()(x2)
            print('(l5) Flatten x1.shape: ', x2.shape)

            #l6
            x2 = Dense(units = self.parameters['units']['conv2d_conv1d']['l6'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l6'],
                       activation = self.parameters['activation']['conv2d_conv1d']['l6'])(x2)
            x2 = Dropout(self.parameters['dropout']['conv2d_conv1d']['l6'])(x2)
            print('(l6) Dense x2.shape: ', x2.shape)

            #concatenando os dois inputs
            concatX = Concatenate(axis=1)([x1, x2])
            print('(Concat) concatX.shape: ', concatX.shape)

            #l7
            concatX = Dense(units = self.parameters['units']['conv2d_conv1d']['l7'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l7'],
                       activation = self.parameters['activation']['conv2d_conv1d']['l7'])(concatX)
            concatX = Dropout(self.parameters['dropout']['conv2d_conv1d']['l7'])(concatX)
            print('(l7) Dense concatX.shape: ', concatX.shape)

            #l8
            concatX = Dense(units = self.parameters['units']['conv2d_conv1d']['l8'], 
                       kernel_initializer=self.parameters['kernel_initializer']['conv2d_conv1d']['l8'],
                       activation = self.parameters['activation']['conv2d_conv1d']['l8'])(concatX)
            x1 = Dropout(self.parameters['dropout']['conv2d_conv1d']['l8'])(concatX)
            print('(l8) Dense concatX.shape: ', concatX.shape)

            #l9
            outputs = Dense(units = self.parameters['units']['conv2d_conv1d']['l9'], 
                            activation = self.parameters['activation']['conv2d_conv1d']['l9'])(concatX)
            print('(l9) Dense concatX.shape: ', outputs.shape)
            
            #compile
            model = Model(inputs=[inputs1, inputs2], outputs=outputs, name='conv2d_conv1d')
            if self.parameters['optimizer']['conv2d_conv1d'].lower() == 'sgd':
                from tensorflow.keras.optimizers import SGD
                chosen_opt = SGD(learning_rate=1e-5, nesterov=True, momentum=0.9)
            elif self.parameters['optimizer']['conv2d_conv1d'].lower() == 'adam':
                from tensorflow.keras.optimizers import Adam
                chosen_opt = Adam(learning_rate=1e-6)
            elif self.parameters['optimizer']['conv2d_conv1d'].lower() == 'rms':                    
                from tensorflow.keras.optimizers import RMSprop
                chosen_opt = RMSprop(learning_rate=0.001)
            model.compile(loss = self.parameters['loss']['conv2d_conv1d'], optimizer = chosen_opt, metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/models/{self.training_mode}_{self.subdescription}_{self.machine_type}_ch_{self.sent_batch_size}_arch_{self.n_architecture}.h5'
            model.save(model_save_folder)            
            
            return model

        #except KeyError:
        #    print('O dicionário de paramêtros não é compatível com o tipe de NN inserido.')
        #    print('> Abortando função: NN_model.get_model')
        

def complete_NN_parameters_dic(NN_parameters_dic, machine_type = None):

    #NN parameters
    NN_parameters_dic['activation'] = {}
    NN_parameters_dic['activation'][machine_type] = {}
    NN_parameters_dic['dropout'] = {}
    NN_parameters_dic['dropout'][machine_type] = {}
    NN_parameters_dic['filters'] = {}
    NN_parameters_dic['filters'][machine_type] = {}
    NN_parameters_dic['kernel_initializer'] = {}
    NN_parameters_dic['kernel_initializer'][machine_type] = {}
    NN_parameters_dic['kernel_regularizer'] = {}
    NN_parameters_dic['kernel_regularizer'][machine_type] = {}
    NN_parameters_dic['bias_regularizer'] = {}
    NN_parameters_dic['bias_regularizer'][machine_type] = {}
    NN_parameters_dic['kernel_size'] = {}
    NN_parameters_dic['kernel_size'][machine_type] = {}
    NN_parameters_dic['loss'] = {}
    NN_parameters_dic['loss'][machine_type] = {}
    NN_parameters_dic['optimizer'] = {}
    NN_parameters_dic['optimizer'][machine_type] = {}
    NN_parameters_dic['strides'] = {}
    NN_parameters_dic['strides'][machine_type] = {}
    NN_parameters_dic['units'] = {}
    NN_parameters_dic['units'][machine_type] = {}
    
    if machine_type.lower() == 'conv1d':
        #model conv1d
        #-----------------------
        #l1 - conv1d
        NN_parameters_dic['activation'][machine_type]['l1'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l1'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l1'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l1'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l1'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l1'] = 3
        NN_parameters_dic['strides'][machine_type]['l1'] = 1

        #l2 - conv1d
        NN_parameters_dic['activation'][machine_type]['l2'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l2'] = 200
        NN_parameters_dic['kernel_initializer'][machine_type]['l2'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l2'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l2'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l2'] = 5
        NN_parameters_dic['strides'][machine_type]['l2'] = 3
        
        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l3'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.3
        NN_parameters_dic['units'][machine_type]['l3'] = 200

        #l4 - dense
        NN_parameters_dic['activation'][machine_type]['l4'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l4'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l4'] = 0.3
        NN_parameters_dic['units'][machine_type]['l4'] = 100

        #l5 - dense
        NN_parameters_dic['activation'][machine_type]['l5'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l5'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l5'] = 0.3
        NN_parameters_dic['units'][machine_type]['l5'] = 10
        
        #l6 - dense
        NN_parameters_dic['activation'][machine_type]['l6'] = 'sigmoid'
        NN_parameters_dic['units'][machine_type]['l6'] = 1       
        
        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'
    
    
    elif machine_type.lower() == 'conv2d':

        #model conv2d
        #-----------------------        
        #l1 - conv2d
        NN_parameters_dic['activation'][machine_type]['l1'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l1'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l1'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l1'] = 3
        NN_parameters_dic['strides'][machine_type]['l1'] = 1

        #l2 - conv2d
        NN_parameters_dic['activation'][machine_type]['l2'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l2'] = 200
        NN_parameters_dic['kernel_initializer'][machine_type]['l2'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l2'] = 5
        NN_parameters_dic['strides'][machine_type]['l2'] = 3
        
        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l3'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.2
        NN_parameters_dic['units'][machine_type]['l3'] = 500

        #l4 - dense
        NN_parameters_dic['activation'][machine_type]['l4'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l4'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l4'] = 0.2
        NN_parameters_dic['units'][machine_type]['l4'] = 100

        #l5 - dense
        NN_parameters_dic['activation'][machine_type]['l5'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l5'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l5'] = 0.2
        NN_parameters_dic['units'][machine_type]['l5'] = 10
        
        #l6 - dense
        NN_parameters_dic['activation'][machine_type]['l6'] = 'sigmoid'
        NN_parameters_dic['units'][machine_type]['l6'] = 1       
        
        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'


    elif machine_type.lower() == 'lstm':

        #model lstm
        #-----------------------        
        #l1 - lstm
        NN_parameters_dic['activation'][machine_type]['l1'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l1'] = 'he_normal'
        NN_parameters_dic['units'][machine_type]['l1'] = 100
        NN_parameters_dic['dropout'][machine_type]['l1'] = 0.2
        
        #l2 - lstm
        NN_parameters_dic['activation'][machine_type]['l2'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l2'] = 'he_normal'
        NN_parameters_dic['units'][machine_type]['l2'] = 100
        NN_parameters_dic['dropout'][machine_type]['l2'] = 0.2

        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'elu'    
        NN_parameters_dic['units'][machine_type]['l3'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l3'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.2

        #l4 - dense
        NN_parameters_dic['activation'][machine_type]['l4'] = 'elu'
        NN_parameters_dic['units'][machine_type]['l4'] = 50
        NN_parameters_dic['kernel_initializer'][machine_type]['l4'] = 'he_normal'        
        NN_parameters_dic['dropout'][machine_type]['l4'] = 0.2
        
        #l5 - dense
        NN_parameters_dic['activation'][machine_type]['l5'] = 'elu'    
        NN_parameters_dic['units'][machine_type]['l5'] = 50
        NN_parameters_dic['kernel_initializer'][machine_type]['l5'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l5'] = 0.2
        
        #l6 - dense
        NN_parameters_dic['activation'][machine_type]['l6'] = 'sigmoid'    
        NN_parameters_dic['units'][machine_type]['l6'] = 1

        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'


    elif machine_type.lower() == 'conv1d_lstm':

        #model #conv1d_lstm
        #-----------------------        
        #l1 - convid
        NN_parameters_dic['activation'][machine_type]['l1'] = 'relu'
        NN_parameters_dic['filters'][machine_type]['l1'] = 250
        NN_parameters_dic['kernel_size'][machine_type]['l1'] = 3
        NN_parameters_dic['strides'][machine_type]['l1'] = 1
        
        #l2 - dense
        NN_parameters_dic['activation'][machine_type]['l2'] = 'sigmoid'
        NN_parameters_dic['dropout'][machine_type]['l2'] = 0.2
        NN_parameters_dic['units'][machine_type]['l2'] = 250        
        
        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'sigmoid'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.2
        NN_parameters_dic['units'][machine_type]['l3'] = 50
        
        #l4 - lstm
        NN_parameters_dic['activation'][machine_type]['l4'] = 'sigmoid'
        NN_parameters_dic['dropout'][machine_type]['l4'] = 0.2
        NN_parameters_dic['units'][machine_type]['l4'] = 250

        #l5 - dense
        NN_parameters_dic['activation'][machine_type]['l5'] = 'sigmoid'
        NN_parameters_dic['dropout'][machine_type]['l5'] = 0.2
        NN_parameters_dic['units'][machine_type]['l5'] = 50
        
        #l6 - dense (concateando os outputs de l1 + l2)
        NN_parameters_dic['activation'][machine_type]['l6'] = 'sigmoid'
        NN_parameters_dic['dropout'][machine_type]['l6'] = 0.2
        NN_parameters_dic['units'][machine_type]['l6'] = 10

        #l7- dense
        NN_parameters_dic['activation'][machine_type]['l7'] = 'sigmoid'
        NN_parameters_dic['units'][machine_type]['l7'] = 1
        
        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'


    elif machine_type.lower() == 'conv1d_conv1d':

        #model #conv1d_conv1d
        #-----------------------        
        #l1 - conv1d WV
        NN_parameters_dic['activation'][machine_type]['l1'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l1'] = 50
        NN_parameters_dic['kernel_initializer'][machine_type]['l1'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l1'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l1'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l1'] = 3
        NN_parameters_dic['strides'][machine_type]['l1'] = 1

        #l2 - conv1d
        NN_parameters_dic['activation'][machine_type]['l2'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l2'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l2'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l2'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l2'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l2'] = 5
        NN_parameters_dic['strides'][machine_type]['l2'] = 3
        
        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l3'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.2
        NN_parameters_dic['units'][machine_type]['l3'] = 200
        
        #l4 - conv1d TV
        NN_parameters_dic['activation'][machine_type]['l4'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l4'] = 50
        NN_parameters_dic['kernel_initializer'][machine_type]['l4'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l4'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l4'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l4'] = 3
        NN_parameters_dic['strides'][machine_type]['l4'] = 1

        #l5 - conv1d
        NN_parameters_dic['activation'][machine_type]['l5'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l5'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l5'] = 'he_normal'
        NN_parameters_dic['kernel_regularizer'][machine_type]['l5'] = 'l1'
        NN_parameters_dic['bias_regularizer'][machine_type]['l5'] = 'l1'
        NN_parameters_dic['kernel_size'][machine_type]['l5'] = 5
        NN_parameters_dic['strides'][machine_type]['l5'] = 3
        
        #l6 - dense
        NN_parameters_dic['activation'][machine_type]['l6'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l6'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l6'] = 0.2
        NN_parameters_dic['units'][machine_type]['l6'] = 200

        #l7 - dense
        NN_parameters_dic['activation'][machine_type]['l7'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l7'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l7'] = 0.2
        NN_parameters_dic['units'][machine_type]['l7'] = 100

        #l8 - dense
        NN_parameters_dic['activation'][machine_type]['l8'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l8'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l8'] = 0.2
        NN_parameters_dic['units'][machine_type]['l8'] = 10

        #l9 - dense
        NN_parameters_dic['activation'][machine_type]['l9'] = 'sigmoid'
        NN_parameters_dic['units'][machine_type]['l9'] = 1
        
        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'


    elif machine_type.lower() == 'conv2d_conv1d':

        #model #conv2d_conv1d
        #-----------------------        
        #l1 - conv2d WV
        NN_parameters_dic['activation'][machine_type]['l1'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l1'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l1'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l1'] = 3
        NN_parameters_dic['strides'][machine_type]['l1'] = 1

        #l2 - conv2d
        NN_parameters_dic['activation'][machine_type]['l2'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l2'] = 200
        NN_parameters_dic['kernel_initializer'][machine_type]['l2'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l2'] = 5
        NN_parameters_dic['strides'][machine_type]['l2'] = 3
        
        #l3 - dense
        NN_parameters_dic['activation'][machine_type]['l3'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l3'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l3'] = 0.2
        NN_parameters_dic['units'][machine_type]['l3'] = 500
        
        #l4 - conv1d TV
        NN_parameters_dic['activation'][machine_type]['l4'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l4'] = 100
        NN_parameters_dic['kernel_initializer'][machine_type]['l4'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l4'] = 3
        NN_parameters_dic['strides'][machine_type]['l4'] = 1

        #l5 - conv1d
        NN_parameters_dic['activation'][machine_type]['l5'] = 'elu'
        NN_parameters_dic['filters'][machine_type]['l5'] = 200
        NN_parameters_dic['kernel_initializer'][machine_type]['l5'] = 'he_normal'
        NN_parameters_dic['kernel_size'][machine_type]['l5'] = 5
        NN_parameters_dic['strides'][machine_type]['l5'] = 3
        
        #l6 - dense
        NN_parameters_dic['activation'][machine_type]['l6'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l6'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l6'] = 0.2
        NN_parameters_dic['units'][machine_type]['l6'] = 500

        #l7 - dense
        NN_parameters_dic['activation'][machine_type]['l7'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l7'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l7'] = 0.2
        NN_parameters_dic['units'][machine_type]['l7'] = 100

        #l8 - dense
        NN_parameters_dic['activation'][machine_type]['l8'] = 'elu'
        NN_parameters_dic['kernel_initializer'][machine_type]['l8'] = 'he_normal'
        NN_parameters_dic['dropout'][machine_type]['l8'] = 0.2
        NN_parameters_dic['units'][machine_type]['l8'] = 10

        #l9 - dense
        NN_parameters_dic['activation'][machine_type]['l9'] = 'sigmoid'
        NN_parameters_dic['units'][machine_type]['l9'] = 1
        
        #compile
        NN_parameters_dic['loss'][machine_type] = 'binary_crossentropy'
        NN_parameters_dic['optimizer'][machine_type] = 'sgd'


    return NN_parameters_dic
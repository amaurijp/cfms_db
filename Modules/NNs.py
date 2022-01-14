#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class NN_model(object):
    
    def __init__(self):
        
        import os        
        self.diretorio = os.getcwd()


    def set_parameters(self, parameters_dic):

        #general paramters
        self.parameters = parameters_dic
        self.feature = parameters_dic['feature']
        self.section_name = parameters_dic['section_name']
        self.vector_type = parameters_dic['vector_type']
        self.replace_Ngrams = parameters_dic['replace_Ngrams']
        self.filter_stopwords = parameters_dic['filter_stopwords']
        self.machine_type = parameters_dic['machine_type']
        self.sent_batch_size = parameters_dic['sent_batch_size']
        
        if self.feature is None:
            self.training_mode = 'sections'
            self.subdescription = self.section_name
        else:
            self.training_mode = 'sentences'        
            self.subdescription = self.feature
              
    
    def get_model(self):
        
        from keras.models import Model, Sequential, load_model
        from keras import Input        
        from keras.layers import Activation, AveragePooling2D, AveragePooling1D, BatchNormalization
        from keras.layers import Concatenate, Conv1D, Conv2D, Dense, Dropout, Flatten
        from keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, LSTM, MaxPooling1D, MaxPooling2D
        import os
        
        model_save_folder = None
        #caso já exista o modelo treinado para sentenças
        if self.section_name is None:
            if os.path.exists(self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5'):
                model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5'
        
        #caso já exista o modelo treinado para sections    
        else:
            if os.path.exists(self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.h5'):
                model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.h5'
        #carregando o modelo
        if model_save_folder:
            model = load_model(model_save_folder)            
            print(f'Modelo {self.machine_type} encontrado para os parâmetros inseridos.')
            print('Carregando o arquivo h5 com o modelo...')
            return model        
        
        #try:
        if self.machine_type == 'conv1d':
                        
            print('Setting CONV1D...')
            
            #determinando o input_shape em função do vetor usado
            if self.parameters['vector_type'] == 'wv':
                input_shape = self.parameters['input_shape']['wv']
            elif self.parameters['vector_type'] == 'tv':
                input_shape = self.parameters['input_shape']['tv']
            
            #inputs
            inputs = Input(shape = input_shape)
            print('(Input) inputs.shape: ', inputs.shape)
                
            #l1 - conv1d para os word-vectors (inputs1)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv1d']['l1'],
                            kernel_initializer=self.parameters['kernel_initializer']['conv1d']['l1'],
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
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d']['l2'],
                            strides = ( self.parameters['strides']['conv1d']['l2'] )
                            )
            x1 = conv1D(x1)
            print('(l2) conv1d x1.shape: ', x1.shape)
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
            print('(l6) Dense x1.shape: ', x1.shape)

            #compile
            model = Model(inputs=inputs, outputs=outputs, name='conv1d')
            model.compile(loss = self.parameters['loss']['conv1d'], optimizer = self.parameters['optimizer']['conv1d'], metrics=['accuracy'])
            model.summary()

            #save
            model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.h5'
            model.save(model_save_folder)            
            
            return model


        elif self.machine_type == 'conv2d':

            #determinando o input_shape em função do vetor usado
            if self.parameters['vector_type'] == 'wv':
                input_shape = self.parameters['input_shape']['wv']
            elif self.parameters['vector_type'] == 'tv':
                input_shape = self.parameters['input_shape']['tv']
            
            #inputs
            inputs = Input(shape = input_shape )
            print('(Input) inputs.shape: ', inputs.shape)
                
            #l1 - conv1d para os word-vectors (inputs1)
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
            
            #l2
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
            x1 = Flatten()(x1)
            print('(l2) Flatten x1.shape: ', x1.shape)
            
            #l3
            x1 = Dense(units = self.parameters['units']['conv2d']['l3'], activation = self.parameters['activation']['conv2d']['l3'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l3'])(x1)
            print('(l3) Dense x1.shape: ', x1.shape)

            #l4
            x1 = Dense(units = self.parameters['units']['conv2d']['l4'], activation = self.parameters['activation']['conv2d']['l4'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l4'])(x1)
            print('(l4) Dense x1.shape: ', x1.shape)

            #l5
            x1 = Dense(units = self.parameters['units']['conv2d']['l5'], activation = self.parameters['activation']['conv2d']['l5'])(x1)
            x1 = Dropout(self.parameters['dropout']['conv2d']['l5'])(x1)
            print('(l5) Dense x1.shape: ', x1.shape)

            #l6
            outputs = Dense(units = self.parameters['units']['conv2d']['l6'], activation = self.parameters['activation']['conv2d']['l6'])(x1)
            print('(l6) Dense x1.shape: ', x1.shape)

            #compile
            model = Model(inputs=inputs, outputs=outputs, name='conv2d')
            model.compile(loss = self.parameters['loss']['conv2d'], optimizer = self.parameters['optimizer']['conv2d'], metrics=['accuracy'])
            model.summary()

            #save
            model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_ch_{self.sent_batch_size}_rNgram_{self.replace_Ngrams}.h5'
            model.save(model_save_folder)            
            
            return model


        elif self.machine_type == 'lstm':
            
            print('Setting LSTM...')

            #determinando o input_shape em função do vetor usado
            if self.parameters['vector_type'] == 'wv':
                input_shape = self.parameters['input_shape']['wv']
            elif self.parameters['vector_type'] == 'tv':
                input_shape = self.parameters['input_shape']['tv']
            
            #montando a rede LSTM
            model=Sequential()
            
            #l1 - lstm
            model.add(LSTM(units = self.parameters['units']['lstm']['l1'], 
                            return_sequences = True,
                            input_shape = input_shape)
                      )
            model.add(Dropout(self.parameters['dropout']['lstm']['l1']))
            model.add(Flatten())
            
            #l2 - dense
            model.add(Dense(units = self.parameters['units']['lstm']['l2'], activation = self.parameters['activation']['lstm']['l2']))
            
            #compile
            model.compile(loss = self.parameters['loss']['lstm'], optimizer = self.parameters['optimizer']['lstm'], metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5'
            model.save(model_save_folder)
            
            return model                
    
    
        elif self.machine_type == 'conv1d_lstm':
                    
            #inputs
            inputs1 = Input(shape = self.parameters['input_shape']['wv'] )
            inputs2 = Input(shape = self.parameters['input_shape']['tv'] )
            print('(Input1) inputs1.shape: ', inputs1.shape)
            print('(Input2) inputs2.shape: ', inputs2.shape)
            
            #l1 - conv1d para os word-vectors (inputs1)
            conv1D = Conv1D(filters = self.parameters['filters']['conv1d']['l1'],
                            kernel_size = self.parameters['kernel_size']['conv1d']['l1'],
                            padding = 'valid',
                            activation = self.parameters['activation']['conv1d']['l1'],
                            strides = self.parameters['strides']['conv1d']['l1']            
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
            model.compile(loss = self.parameters['loss']['conv1d_lstm'], optimizer = self.parameters['optimizer']['conv1d_lstm'], metrics=['accuracy'])
            model.summary()
            
            #save
            model_save_folder = self.diretorio + f'/Outputs/Models/{self.training_mode}_{self.subdescription}_{self.machine_type}_{self.vector_type}_SW_{self.filter_stopwords}_rNgram_{self.replace_Ngrams}.h5'
            model.save(model_save_folder)            
            
            return model            

    
        #except KeyError:
        #    print('O dicionário de paramêtros não é compatível com o tipe de NN inserido.')
        #    print('> Abortando função: NN_model.get_model')
        
        
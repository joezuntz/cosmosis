import os
import numpy as np
import tensorflow as tf
from sklearn.decomposition import IncrementalPCA
import pickle
from tqdm import trange
dtype = tf.float32


# checking that we are using a GPU
device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
print('using', device, 'device \n')

class ResBlockBN(tf.keras.layers.Layer):
    """
    Residual block with two Dense layers and a skip connection.
    """
    def __init__(self, in_size, channel, out_size):
        super(ResBlockBN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(
            channel,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(1e-2)
        )
        self.layer2 = tf.keras.layers.Dense(
            out_size,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(1e-2)
        )

        if in_size == out_size:
            self.skip_layer = lambda x: x
        else:
            self.skip_layer = tf.keras.layers.Dense(
                out_size,
                use_bias=False,
                kernel_initializer=tf.keras.initializers.Zeros()
            )

    def call(self, x, training=False):
        h = tf.nn.relu(self.layer1(x))
        y = tf.nn.relu(self.layer2(h) * 0.1 + self.skip_layer(x))
        return y


class LINNATF(tf.keras.Model):
    """
    TensorFlow equivalent of ChtoModelv2 from LINNA with residual blocks.
    """
    def __init__(self, in_size, out_size, linearmodel=None):
        super(LINNATF, self).__init__()
        self.channel = 16
        hidden_size = max(32, int(out_size * 32))
        if out_size > 30:
            hidden_size = 1000

        self.layer1 = tf.keras.layers.Dense(
            hidden_size, activation="relu", kernel_initializer="he_normal"
        )
        self.layer2 = ResBlockBN(hidden_size, self.channel, hidden_size // 2)
        hidden_size //= 2
        self.layer3 = ResBlockBN(hidden_size, self.channel * 2, hidden_size // 2)
        hidden_size //= 2
        self.layer4 = ResBlockBN(hidden_size, self.channel * 4, hidden_size // 2)
        hidden_size //= 2
        self.layer6 = tf.keras.layers.Dense(
            hidden_size * 4, activation="relu", kernel_initializer="he_normal"
        )
        self.layer7 = tf.keras.layers.Dense(
            out_size, activation="relu", kernel_initializer="he_normal"
        )
        self.layer8 = tf.keras.layers.Dense(out_size, kernel_initializer="he_normal")

        self.linearmodel = linearmodel

    def call(self, inputs, training=False):
        x = self.layer1(inputs)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.layer6(x)
        x = self.layer7(x)
        if self.linearmodel is not None:
            out = self.layer8(x) + self.linearmodel(inputs)
        else:
            out = self.layer8(x)
        return out

class CosmoPowerNN(tf.keras.Model):
    def __init__(self, 
                 parameters=None, 
                 modes=None, 
                 parameters_mean=None, 
                 parameters_std=None, 
                 features_mean=None, 
                 features_std=None, 
                 n_hidden=[512,512,512], 
                 restore=False, 
                 restore_filename=None, 
                 trainable=True,
                 optimizer=None,
                 verbose=False, 
                 architecture_type="MLP"
                 ):
        """
        Constructor
        """
        # super
        super(CosmoPowerNN, self).__init__()
        self.architecture_type = architecture_type
        # restore
        if restore is True:
            self.restore(restore_filename)

        # else set variables from input arguments
        else:
            # attributes
            self.parameters = parameters
            self.n_parameters = len(self.parameters)
            self.modes = modes
            self.n_modes = len(self.modes)
            self.n_hidden = n_hidden

            # input parameters mean and std
            self.parameters_mean_ = parameters_mean 
            self.parameters_std_ = parameters_std 

            # (log)-spectra or pca mean and std
            self.features_mean_ = features_mean 
            self.features_std_ = features_std 

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean_, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std_, dtype=dtype, name='parameters_std')


        # (log)-spectra or pca mean and std
        self.features_mean = tf.constant(self.features_mean_, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std_, dtype=dtype, name='features_std')

        ## weights, biases and activation function parameters for each layer of the network
        #self.W = []
        #self.b = []
        #self.alphas = []
        #self.betas = [] 
        #for i in range(self.n_layers):
        #    self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
        #    self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name = "b_" + str(i), trainable=trainable))
        #for i in range(self.n_layers-1):
        #    self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "alphas_" + str(i), trainable=trainable))
        #    self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name = "betas_" + str(i), trainable=trainable))

        ## restore weights if restore = True
        #if restore is True:
        #    for i in range(self.n_layers):
        #      self.W[i].assign(self.W_[i])
        #      self.b[i].assign(self.b_[i])
        #    for i in range(self.n_layers-1):
        #      self.alphas[i].assign(self.alphas_[i])
        #      self.betas[i].assign(self.betas_[i])

        # build network
        if self.architecture_type == "MLP":
            # architecture
            self.architecture = [self.n_parameters] + self.n_hidden + [self.n_modes]
            self.n_layers = len(self.architecture) - 1

            self.W, self.b, self.alphas, self.betas = [], [], [], []
            for i in range(self.n_layers):
                self.W.append(tf.Variable(tf.random.normal([self.architecture[i], self.architecture[i+1]], 0., 1e-3), name="W_" + str(i), trainable=trainable))
                self.b.append(tf.Variable(tf.zeros([self.architecture[i+1]]), name="b_" + str(i), trainable=trainable))
            for i in range(self.n_layers-1):
                self.alphas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name="alphas_" + str(i), trainable=trainable))
                self.betas.append(tf.Variable(tf.random.normal([self.architecture[i+1]]), name="betas_" + str(i), trainable=trainable))

        elif self.architecture_type == "ResMLP":
            #self.model = tf.keras.Sequential([
            #    tf.keras.layers.InputLayer(input_shape=(self.n_parameters,)),
            #    tf.keras.layers.Dense(512, activation="relu"),
            #    tf.keras.layers.BatchNormalization(),
            #    tf.keras.layers.Dropout(0.1),
            #    tf.keras.layers.Dense(512, activation="relu"),
            #    tf.keras.layers.BatchNormalization(),
            #    tf.keras.layers.Dropout(0.1),
            #    tf.keras.layers.Dense(512, activation="relu"),
            #    tf.keras.layers.Dense(self.n_modes)  # output
            #])
            self.model = LINNATF(self.n_parameters, self.n_modes)

        # optimizer
        self.optimizer = optimizer or tf.keras.optimizers.Adam()
        self.verbose= verbose

        # print initialization info, if verbose
        if self.verbose:
            multiline_str = f"\nInitialized {self.architecture_type} model, \n" \
                            f"mapping {self.n_parameters} input parameters to {self.n_modes} output modes, \n" \
                            f"using {len(self.n_hidden)} hidden layers, \n" \
                            f"with {list(self.n_hidden)} nodes, respectively. \n"
            print(multiline_str)

    def activation(self, 
                   x, 
                   alpha, 
                   beta
                   ):
        return tf.multiply(tf.add(beta, tf.multiply(tf.sigmoid(tf.multiply(alpha, x)), tf.subtract(1.0, beta)) ), x)

    @tf.function
    def predictions_tf(self, 
                       parameters_tensor
                       ):
        #outputs = []
        #layers = [tf.divide(tf.subtract(parameters_tensor, self.parameters_mean), self.parameters_std)]
        #for i in range(self.n_layers - 1):
        #    # linear network operation
        #    outputs.append(tf.add(tf.matmul(layers[-1], self.W[i]), self.b[i]))
        #    # non-linear activation function
        #    layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))
        ## linear output layer
        #layers.append(tf.add(tf.matmul(layers[-1], self.W[-1]), self.b[-1]))#
        ## rescale -> output predictions
        #return tf.add(tf.multiply(layers[-1], self.features_std), self.features_mean)
        x = (parameters_tensor - self.parameters_mean) / self.parameters_std
        if self.architecture_type == "MLP":
            outputs = []
            layers = [x]
            for i in range(self.n_layers - 1):
                outputs.append(tf.matmul(layers[-1], self.W[i]) + self.b[i])
                layers.append(self.activation(outputs[-1], self.alphas[i], self.betas[i]))
            layers.append(tf.matmul(layers[-1], self.W[-1]) + self.b[-1])
            return layers[-1] * self.features_std + self.features_mean
        elif self.architecture_type == "ResMLP":
            return self.model(x) * self.features_std + self.features_mean


    def forward_pass_np(self, 
                        parameters_arr
                        ):
        # forward pass through the network
        act = []
        layers = [(parameters_arr - self.parameters_mean_)/self.parameters_std_]
        for i in range(self.n_layers-1):

            # linear network operation
            act.append(np.dot(layers[-1], self.W_[i]) + self.b_[i])

            # pass through activation function
            layers.append((self.betas_[i] + (1.-self.betas_[i])*1./(1.+np.exp(-self.alphas_[i]*act[-1])))*act[-1])

        # final (linear) layer -> (standardised) predictions
        layers.append(np.dot(layers[-1], self.W_[-1]) + self.b_[-1])

        # rescale and output
        return layers[-1]*self.features_std_ + self.features_mean_
     
    # Numpy array predictions
    def predictions_np(self, 
                       parameters_dict
                       ):
        parameters_arr = self.dict_to_ordered_arr_np(parameters_dict)
        return self.forward_pass_np(parameters_arr)

    def update_emulator_parameters(self):
        r"""
        Update emulator parameters before saving them
        """
        # put network parameters to numpy arrays
        #self.W_ = [self.W[i].numpy() for i in range(self.n_layers)]
        #self.b_ = [self.b[i].numpy() for i in range(self.n_layers)]
        #self.alphas_ = [self.alphas[i].numpy() for i in range(self.n_layers-1)]
        #self.betas_ = [self.betas[i].numpy() for i in range(self.n_layers-1)]

        # put mean and std parameters to numpy arrays
        #self.parameters_mean_ = self.parameters_mean.numpy()
        #self.parameters_std_ = self.parameters_std.numpy()
        #self.features_mean_ = self.features_mean.numpy()
        #self.features_std_ = self.features_std.numpy()
        if self.architecture_type == "MLP":
            self.emulator_parameters = {
                "W": [w.numpy() for w in self.W],
                "b": [b.numpy() for b in self.b],
                "alphas": [a.numpy() for a in self.alphas],
                "betas": [b.numpy() for b in self.betas],
            }
        elif self.architecture_type == "ResMLP":
            self.emulator_parameters = self.model.get_weights()

    # save
    def save(self, 
             filename,
             diagnostics
             ):
        r"""
        Save network parameters

        Parameters:
            filename (str):
                filename tag (without suffix) where model will be saved
        """
        # attributes
        #attributes = [self.W_, 
        #              self.b_, 
        #              self.alphas_, 
        #              self.betas_, 
        #              self.parameters_mean_, 
        #              self.parameters_std_,
        #              self.features_mean_,
        #              self.features_std_,
        #              self.n_parameters,
        #              self.parameters,
        #              self.n_modes,
        #              self.modes,
        #              self.n_hidden,
        #              self.n_layers,
        #              self.architecture,
        #              diagnostics
        #              ]
        ## save attributes to file
        #with open(filename + ".pkl", 'wb') as f:
        #    pickle.dump(attributes, f)
        save_dict = {
            "architecture_type": self.architecture_type,
            "diagnostics": diagnostics,
            "parameters_mean": self.parameters_mean_.tolist(),
            "parameters_std": self.parameters_std_.tolist(),
            "features_mean": self.features_mean_.tolist(),
            "features_std": self.features_std_.tolist()
        }
        if self.architecture_type == "MLP":
            save_dict["weights"] = {
                "W": [w.numpy().tolist() for w in self.W],
                "b": [b.numpy().tolist() for b in self.b],
                "alphas": [a.numpy().tolist() for a in self.alphas],
                "betas": [b.numpy().tolist() for b in self.betas]
            }
        elif self.architecture_type == "ResMLP":
            save_dict["weights"] = self.model.get_weights()

        np.savez(filename, **save_dict)
        print(f"Model saved to {filename}")


    # restore attributes
    def restore(self, 
                filename
                ):
        r"""
        Load pre-trained model

        Parameters:
            filename (str):
                filename tag (without suffix) where model was saved
        """
        # load attributes
        #with open(filename + ".pkl", 'rb') as f:
        #    self.W_, self.b_, self.alphas_, self.betas_, \
        #    self.parameters_mean_, self.parameters_std_, \
        #    self.features_mean_, self.features_std_, \
        #    self.n_parameters, self.parameters, \
        #    self.n_modes, self.modes, \
        #    self.n_hidden, self.n_layers, self.architecture, _, _, _ = pickle.load(f)
        data = np.load(filename, allow_pickle=True)
        arch = str(data["architecture_type"])

        self.parameters_mean_ = data["parameters_mean"]
        self.parameters_std_ = data["parameters_std"]
        self.features_mean_ = data["features_mean"]
        self.features_std_ = data["features_std"]

        if arch == "MLP":
            weights = data["weights"].item()
            for i in range(len(weights["W"])):
                self.W[i].assign(weights["W"][i])
                self.b[i].assign(weights["b"][i])
            for i in range(len(weights["alphas"])):
                self.alphas[i].assign(weights["alphas"][i])
                self.betas[i].assign(weights["betas"][i])
        elif arch == "ResMLP":
            self.model.set_weights(data["weights"])

        print(f"Model restored from {filename} with architecture={arch}")

    def summary(self):
        print("\n====================================================")
        print(f" Model Summary ({self.architecture_type})")
        print("====================================================")

        if self.architecture_type == "MLP":
            total_params = 0

            # Header
            print(f"{'Layer (type)':30} {'Output Shape':20} {'Param #':10}")
            print("="*65)

            input_dim = self.n_parameters
            for i in range(self.n_layers):
                output_dim = self.architecture[i+1]
                w_params = input_dim * output_dim
                b_params = output_dim
                param_count = w_params + b_params
                total_params += param_count

                print(f"Dense_{i:<26} ({output_dim:>3},)         {param_count:>10}")

                if i < len(self.alphas):
                    # alphas and betas
                    a_params = output_dim
                    b_params_extra = output_dim
                    total_params += (a_params + b_params_extra)
                    print(f"ActivationParams_{i:<15} ({output_dim:>3},)         {a_params + b_params_extra:>10}")

                input_dim = output_dim  # update for next layer

            print("="*65)
            print(f"Total params: {total_params}")
            print("Trainable params: all")
            print("Non-trainable params: 0")
            print("====================================================\n")

        elif self.architecture_type == "ResMLP":
            self.model.summary()

    
    # auxiliary function to sort input parameters
    def dict_to_ordered_arr_np(self, 
                               input_dict, 
                               ):
        if self.parameters is not None:
            return np.stack([input_dict[k] for k in self.parameters], axis=1)
        else:
            return np.stack([input_dict[k] for k in input_dict], axis=1)

    
    @tf.function
    def compute_loss(self, training_parameters, training_features):
        return tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.predictions_tf(training_parameters), training_features)))


    @tf.function
    def compute_loss_weighted_w_cov(self, training_parameters, training_features):
        diff = tf.subtract(self.predictions_tf(training_parameters), training_features)
        return tf.sqrt(tf.reduce_mean(tf.multiply(tf.multiply(diff, self.data_inv_cov), diff)))
    
    @tf.function
    def compute_loss_weighted_w_like(self, training_parameters, training_features):
        diff = tf.subtract(self.predictions_tf(training_parameters), training_features)
        diff_data = tf.subtract(training_features, self.datavector)
        like = tf.multiply(tf.multiply(diff_data, self.data_inv_cov), diff_data)
        return tf.sqrt(tf.reduce_mean(tf.divide(tf.multiply(tf.multiply(diff, self.data_inv_cov), diff), like)))



    @tf.function
    def compute_loss_and_gradients(self, training_parameters, training_features):
        # compute loss on the tape
        with tf.GradientTape() as tape:
            # loss
            loss = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(self.predictions_tf(training_parameters), training_features))) 
        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        return loss, gradients
    
    def training_step(self, training_parameters,training_features):
        # compute loss and gradients
        loss, gradients = self.compute_loss_and_gradients(training_parameters, training_features)
        # apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss
    
    def train(self,
              training_parameters,
              training_features,
              filename_saved_model,
              # cooling schedule
              validation_split=0.1,
              learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes=[1024, 1024, 1024, 1024, 1024],
              gradient_accumulation_steps = [1, 1, 1, 1, 1], 
              # early stopping set up
              patience_values = [100, 100, 100, 100, 100],
              max_epochs = [1000, 1000, 1000, 1000, 1000],
             ):
        # check correct number of steps
        assert len(learning_rates)==len(batch_sizes)\
               ==len(gradient_accumulation_steps)==len(patience_values)==len(max_epochs), \
               'Number of learning rates, batch sizes, gradient accumulation steps, patience values and max epochs are not matching!'

        # training start info, if verbose
        if self.verbose:
            multiline_str = "Starting cosmopower training, \n" \
                            f"using {int(100*validation_split)} per cent of training samples for validation. \n" \
                            f"Performing {len(learning_rates)} learning steps, with \n" \
                            f"{list(learning_rates)} learning rates \n" \
                            f"{list(batch_sizes)} batch sizes \n" \
                            f"{list(gradient_accumulation_steps)} gradient accumulation steps \n" \
                            f"{list(patience_values)} patience values \n" \
                            f"{list(max_epochs)} max epochs \n"
            print(multiline_str)

        # from dict to array
        training_parameters = self.dict_to_ordered_arr_np(training_parameters)

        # parameters standardisation
        self.parameters_mean = np.mean(training_parameters, axis=0)
        self.parameters_std = np.std(training_parameters, axis=0)

        # input parameters mean and std
        self.parameters_mean = tf.constant(self.parameters_mean, dtype=dtype, name='parameters_mean')
        self.parameters_std = tf.constant(self.parameters_std, dtype=dtype, name='parameters_std')

        
        # features standardisation
        self.features_mean = np.mean(training_features, axis=0)
        self.features_std = np.std(training_features, axis=0)
        
        # (log)-spectra mean and std
        self.features_mean = tf.constant(self.features_mean, dtype=dtype, name='features_mean')
        self.features_std = tf.constant(self.features_std, dtype=dtype, name='features_std')
        
        # casting
        training_parameters = tf.convert_to_tensor(training_parameters, dtype=dtype)
        training_features = tf.convert_to_tensor(training_features, dtype=dtype)


        # training/validation split
        n_validation = int(training_parameters.shape[0] * validation_split)
        n_training = training_parameters.shape[0] - n_validation

        diagnostics = {}
        # train using cooling/heating schedule for lr/batch-size
        for i in range(len(learning_rates)):

            print('learning rate = ' + str(learning_rates[i]) + ', batch size = ' + str(batch_sizes[i]))

            # set learning rate
            self.optimizer.lr = learning_rates[i]

            # split into validation and training sub-sets
            training_selection = tf.random.shuffle([True] * n_training + [False] * n_validation)

            # create iterable dataset (given batch size)
            training_data = tf.data.Dataset.from_tensor_slices((training_parameters[training_selection], training_features[training_selection])).shuffle(n_training).batch(batch_sizes[i])


            # set up training loss
            training_loss = [np.infty]
            validation_loss = [np.infty]
            best_loss = np.infty
            early_stopping_counter = 0


            if 'learning cycle ' + str(i) not in diagnostics:
                diagnostics['learning cycle ' + str(i)] = {}
            diagnostics['learning cycle ' + str(i)]['epochs'] = []
            diagnostics['learning cycle ' + str(i)]['validation_loss'] = []
            diagnostics['learning cycle ' + str(i)]['training_loss'] = []
            # loop over epochs
            early_stopping = False
            with trange(max_epochs[i]) as t:
                if early_stopping==False:
                    for epoch in t:
                        diagnostics['learning cycle ' + str(i)]['epochs'].append(epoch)
                        # loop over batches
                        #counter = 0
                        for theta, feats in training_data:
                            #print('batch ', counter)
                            #print(theta.shape)  # (batch_sizes[i], parameter_dim)
                            #print(feats.shape)    # (batch_sizes[i], feature_dim)
                            #counter +=1
                            loss = self.training_step(theta, feats)
                            #print('loss: ', loss)
                            training_loss.append(loss)
                        diagnostics['learning cycle ' + str(i)]['training_loss'].append(training_loss)
                        # compute validation loss at the end of the epoch
                        validation_loss.append(self.compute_loss(training_parameters[~training_selection], training_features[~training_selection]).numpy())

                        # update the progressbar
                        t.set_postfix(loss=validation_loss[-1])
                        diagnostics['learning cycle ' + str(i)]['validation_loss'].append(validation_loss)
                        # early stopping condition
                        if validation_loss[-1] < best_loss:
                            best_loss = validation_loss[-1]
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                        if early_stopping_counter >= patience_values[i]:
                            self.update_emulator_parameters()
                            self.save(filename_saved_model, diagnostics)
                            print('Validation loss = ' + str(best_loss))
                            print('Model saved.')
                            early_stopping = True
                            break
                    self.update_emulator_parameters()
                    self.save(filename_saved_model, diagnostics)
                    print('Reached max number of epochs. Validation loss = ' + str(best_loss))
                    print('Model saved.')
        self.summary()

class CPEmulator:
    def __init__(self, model_parameters, modes, nn_model='MLP', itteration=1, data_trafo='log_norm', n_pca=64, datavector=None, inv_cov=None):
        self.trained = False
        self.model_parameters = model_parameters
        self.modes = modes
        self.data_trafo = data_trafo
        self.datavector = datavector
        self.data_inv_cov = inv_cov
        self.n_pca = n_pca
        print('DATA TRANSFORMATION: ', data_trafo)
        self.pca_transform_matrix = None
        self.parameters_filenames = ['parameters_filenames'+str(i) for i in range(itteration)]
        self.features_filenames = ['features_filenames'+str(i) for i in range(itteration)]
        self.n_batches_pca = len(self.parameters_filenames)
        self.nn_model = nn_model

    def transform(self, model_datavector):
        if self.data_trafo == 'log_norm':
            y = np.log10(model_datavector)
            self.y_mean = np.mean(y, axis=0)
            self.y_std = np.std(y, axis=0)
            self.features_mean = self.y_mean
            self.features_std = self.y_std
            y_train = (y - self.y_mean) / self.y_std
            return y_train
        elif self.data_trafo == 'norm':
            self.y_mean = np.mean(model_datavector, axis=0)
            self.y_std = np.std(model_datavector, axis=0)
            self.features_mean = self.y_mean
            self.features_std = self.y_std
            y_train = (model_datavector - self.y_mean) / self.y_std
            return y_train
        elif self.data_trafo == 'PCA':
            y_mean = np.mean(model_datavector, axis=0)
            y_std = np.std(model_datavector, axis=0)
            # PCA object
            PCA = IncrementalPCA(n_components=self.n_pca)
            with trange(self.n_batches_pca) as t:
                for i in t:
                    # load (log)-spectra and mean+std
                    features = np.load(self.features_filenames[i] + ".npz")['features']
                    normalised_features = (features - y_mean)/y_std
                    # partial PCA fit
                    PCA.partial_fit(normalised_features)
            # set the PCA transform matrix
            self.pca_transform_matrix = PCA.components_
            # transform the (log)-spectra to PCA basis
            training_pca = np.concatenate([PCA.transform((np.load(self.features_filenames[i] + ".npz")['features'] - y_mean)/y_std) for i in range(self.n_batches_pca)])
            self.features_mean = y_mean
            self.features_std = y_std

            # mean and std of PCA basis
            self.y_mean = np.mean(training_pca, axis=0)
            self.y_std = np.std(training_pca, axis=0)
            y_train = (training_pca - self.y_mean) / self.y_std
            return y_train 


    
    def backtransform(self, model_datavector):
        if self.data_trafo == 'log_norm':
            return 10**model_datavector
        elif self.data_trafo == 'norm':
            return model_datavector
        elif self.data_trafo == 'PCA':
            return np.dot(model_datavector, self.pca_transform_matrix)*self.features_std + self.features_mean


    def train(self, X, y, model_filename, test_split=0.1, batch_size=32, n_cycles=5):
        print('within train-function')
        self.X_mean = {key: np.mean(X[key], axis=0) for key in X.keys()}
        self.X_std = {key: np.std(X[key], axis=0) for key in X.keys()}
        X_train = {key: (X[key]-self.X_mean[key]) / self.X_std[key] for key in X.keys()}

        y_train = self.transform(y)
        X_mean_arr = np.array([self.X_mean[key] for key in self.model_parameters])
        X_std_arr = np.array([self.X_std[key] for key in self.model_parameters])
        self.cp_nn = CosmoPowerNN(parameters=self.model_parameters, 
                      modes=self.n_pca if self.data_trafo=='PCA' else self.modes, 
                      parameters_mean=X_mean_arr, 
                      parameters_std=X_std_arr, 
                      features_mean=self.y_mean, 
                      features_std=self.y_std, 
                      n_hidden = [512, 512, 512, 512], # 4 hidden layers, each with 512 nodes
                      verbose=True, # useful to understand the different steps in initialisation and training
                      )


        with tf.device(device):
            # train
            self.cp_nn.train(training_parameters=X_train,  
                        training_features=y_train,  
                        filename_saved_model=model_filename, 
                        # cooling schedule
                        validation_split=test_split, #percentage of samples from the training set that will be used for validation
                        learning_rates= [10**(-2-i) for i in range(n_cycles)], #[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                        batch_sizes= list(np.full(n_cycles, batch_size)), #TODO: add option to vary
                        gradient_accumulation_steps = list(np.full(n_cycles, 1)), 
                        # early stopping set up
                        patience_values = list(np.full(n_cycles, 100)), 
                        max_epochs = list(np.full(n_cycles, 1000)) 
                        )
        # attributes
        attributes = [self.X_mean, 
                      self.X_std, 
                      self.y_mean, 
                      self.y_std,
                      self.features_mean,
                      self.features_std,
                      self.pca_transform_matrix]

        # save attributes to file
        with open(model_filename + "_means.pkl", 'wb') as f:
            pickle.dump(attributes, f)    
        self.trained = True


    def load(self, filename):
        self.trained = True
        self.cp_nn = CosmoPowerNN(restore=True, 
                      restore_filename=filename,
                      )
        with open(filename + "_means.pkl", 'rb') as f:
            self.X_mean, self.X_std, self.y_mean, self.y_std, self.features_mean, self.features_std, self.pca_transform_matrix = pickle.load(f)

    def predict(self, X):
        assert self.trained, "The emulator needs to be trained first before predicting"
        X_norm_dic = {key: [(X[key]-self.X_mean[key]) / self.X_std[key]] for key in X.keys()}
        #print('within predict-function')
        #print( 'X: ', X_norm_dic )
        y_pred = self.cp_nn.predictions_np(X_norm_dic)
        #print('y_pred: ', y_pred)
        y_pred = y_pred * self.y_std + self.y_mean
        #print('before: ', y_pred)
        y_pred = self.backtransform(y_pred)
        #print('after: ', y_pred)
        return y_pred  


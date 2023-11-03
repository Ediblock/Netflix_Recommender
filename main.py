import keras.backend
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import metrics
import data_access as da
import os
import time


movie_titles_file_path = "./data/movie_titles.txt"
probe_file_path = "./data/probe.txt"
qualifying_file_path = "./data/qualifying.txt"
training_set_file_path = "./data/training_set/"
save_weights_file_path = "./save_model_weights/cp.cpkt"
tensorboard_path = os.path.join(os.curdir, "tens_logs")

def get_run_logdir():
    run_id = time.strftime("event_%Y_%m_%d-%H_%M_%S")
    return os.path.join(tensorboard_path, run_id)


def create_model(vocabulary: int) -> tf.keras.Model:
    emb_initializer = tf.keras.initializers.random_uniform(minval=-1, maxval=1)
    emb_outputsize = 5

    input_l = tf.keras.layers.Input(shape=(vocabulary,))
    # emb = tf.keras.layers.Embedding(input_dim=vocabulary, output_dim=emb_outputsize,
    #                                 embeddings_initializer=emb_initializer, input_length=vocabulary)(input)
    # resh = tf.keras.layers.Flatten()(emb)
    l1 = tf.keras.layers.Dense(300, activation="relu", use_bias=True, kernel_initializer=
                               tf.keras.initializers.random_normal(mean=0.5, stddev=3))(input_l)
    drop2 = tf.keras.layers.Dropout(0.2)(l1)
    # l3 = tf.keras.layers.Dense(2000, activation="sigmoid", use_bias=True,
    #                            kernel_initializer=tf.keras.initializers.random_uniform(minval=-0.2, maxval=0.2))(drop2)
    # l4 = tf.keras.layers.Dense(5000, activation="tanh", use_bias=True, kernel_initializer=
    #                            tf.keras.initializers.random_uniform(minval=-0.5, maxval=0.5))(l3)
    l5 = tf.keras.layers.Dense(100, activation="relu", use_bias=True, kernel_initializer=
                                   tf.keras.initializers.random_uniform(minval=0, maxval=1))(drop2)
    # drop3 = tf.keras.layers.Dropout(0.5)(l5)
    l6 = tf.keras.layers.Dense(100, activation="sigmoid", use_bias=True, kernel_initializer=
                               tf.keras.initializers.random_uniform(minval=0.2, maxval=0.8))(l5)
    output = tf.keras.layers.Dense(vocabulary, activation="relu", use_bias=False, kernel_initializer=
                                   tf.keras.initializers.random_uniform(minval=0, maxval=1))(l6)
    model = tf.keras.models.Model(inputs=input_l, outputs=output)
    return model


def create_model2(movie_vocabulary: int) -> tf.keras.Model:
    emb_outputsize = 200
    input_size =(200,)

    #movie path
    movie_input_layer = tf.keras.layers.Input(shape=(1,))
    emb1 = tf.keras.layers.Embedding(input_dim=movie_vocabulary + 1, output_dim=emb_outputsize, input_length=1,
                                     embeddings_initializer=tf.keras.initializers.he_normal()
                                    )(movie_input_layer)
    fl1 = tf.keras.layers.Flatten()(emb1)

    #rating path
    rating_input_layer = tf.keras.layers.Input(shape=(movie_vocabulary,))
    rl1 = tf.keras.layers.Dense(256, activation='elu', use_bias=True,
                               kernel_initializer=tf.keras.initializers.he_uniform(),
                               bias_initializer=tf.keras.initializers.he_normal()
                              )(rating_input_layer)
    rl2 = tf.keras.layers.Dense(emb_outputsize, activation='elu', use_bias=True,
                               kernel_initializer=tf.keras.initializers.he_normal(),
                               bias_initializer=tf.keras.initializers.he_normal()
                              )(rl1)

    #dot for these two paths
    dot = tf.keras.layers.Dot(axes=1)([fl1, rl2])
    #concatenate
    cat = tf.keras.layers.Concatenate()([fl1, rl2, dot])
    drop1 = tf.keras.layers.Dropout(0.2)(cat)
    ol1 = tf.keras.layers.Dense(256, activation="elu", use_bias=True,
                                kernel_initializer=tf.keras.initializers.he_normal(),
                                bias_initializer=tf.keras.initializers.he_normal()
                               )(drop1)
    output_layer = tf.keras.layers.Dense(1)(ol1)

    model = tf.keras.models.Model(inputs=[movie_input_layer, rating_input_layer], outputs=output_layer)
    return model


def compute_loss_from_model(model:tf.keras.Model, loss_fn:tf.keras.losses.Loss,
                            x_input_data, y_data_true,
                            weights = None,
                            should_print:bool = True):
    """x_input_data and y_data_true and weights should be a list or a numpy array"""
    if isinstance(x_input_data, tf.data.Dataset) and isinstance(y_data_true, tf.data.Dataset) and isinstance(weights, tf.data.Dataset):
        x_input_data = x_input_data.as_numpy_iterator()
        y_data_true = y_data_true.as_numpy_iterator()
        weights = weights.as_numpy_iterator()
    y_predicted = model(x_input_data)
    loss_value = loss_fn(y_data_true, y_predicted, weights)
    if should_print:
        print(f"{loss_fn.name} it's loss value is: {loss_value}")
    return loss_value


should_train = True
movie_titles = pd.read_csv(movie_titles_file_path, delimiter=";", encoding="windows-1252",
                           names=["id", "year", "title"], dtype={"id":np.int32, "year":np.int32})
highest_score = 5.0

training_dataset = tf.data.Dataset.list_files(training_set_file_path + "*.txt")
if should_train:
    model = create_model2(movie_titles.count()[0])
    training_data = da.MovieDataset(training_dataset, movie_titles.count()[0], highest_score)
    training_data.set_batch_size(64)
    training_data.set_weight_sample(weight_sample=False)
    training_data.set_data_test_percentage(10)
    training_data.set_data_validation_percentage(10)
    training_data.split_train_test_validation_data(shuffle=True)
    training_data.set_hot_encoding(True)
    # loss_fn = metrics.CustomMSE()
    loss_fn = tf.keras.losses.MeanSquaredError()


    # Training loop
    epochs = 10
    initial_learning_rate = 1e-3
    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate, decay_steps=50,
    #                                                                decay_rate=1/10)
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    # exp_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_learning_rate,
    #                                                                    decay_steps=training_data.get_train_length(),
    #                                                                      decay_rate=4e6)
    train_loss_list = []
    test_loss_list = []
    learning_rate_list = []
    rmse_metric = tf.keras.metrics.RootMeanSquaredError()

    # train_batch_summary_writer = tf.summary.create_file_writer(get_run_logdir() + "_batch")
    train_summary_writer = tf.summary.create_file_writer(get_run_logdir())

    for epoch in range(epochs):
        print(f"Start of epoch {epoch}")
        training_data.shuffle_train_data()
        # optimizer._set_hyper("learning_rate", learning_rate(epoch).numpy())
        # keras.backend.set_value(optimizer.learning_rate, learning_rate(epoch).numpy())

        for step in range(int(training_data.get_train_length())):
            # x_batch, y_batch, weights_batch = training_data.get_train_data(step)
            x1_batch, x2_batch, y_batch = training_data.get_train_data(step)
            x1_batch, x2_batch, y_batch = np.array(x1_batch), np.array(x2_batch), np.array(y_batch)
            # weights_batch = np.array(weights_batch, dtype="float32")
            # if (epoch*int(training_data.get_train_length()/120) + step) % 10 == 0:
            #     optimizer._set_hyper("learning_rate", initial_learning_rate/2.)
            # optimizer._set_hyper("learning_rate", learning_rate(epoch*int(training_data.get_train_length()/120) + step).numpy())
            # optimizer._set_hyper("learning_rate", initial_learning_rate)
            # optimizer.learning_rate = exp_learning_rate(step)
            # optimizer.learning_rate = 10

            with tf.GradientTape() as tape:
                # Forward pass
                logits = model([x2_batch, x1_batch], training=True)

                # Computing loss value
                # loss_value = tf.constant(10*np.log10(loss_fn(y_batch, logits, mask_loss=weights_batch)), dtype="float32")
                # Calculating x*log10(y)

                # loss_value = tf.math.multiply(tf.math.divide(tf.math.log(loss_fn(y_batch, logits, mask_loss=weights_batch)),
                #                                              tf.math.log(tf.constant(10, dtype="float32"))),
                #                               tf.constant(10, dtype="float32"))

                loss_value = loss_fn(y_batch, logits)

            # Updating gradients with regards to the loss
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Updating weights for the model
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_loss_list.append(loss_value)
            # learning_rate_list.append(optimizer._decayed_lr("float32").numpy())

            # Saving to tensorboard
            # with train_batch_summary_writer.as_default():
            #     tf.summary.scalar("loss", loss_value, step=step)
            #     tf.summary.scalar("learning_rate", optimizer.learning_rate, step=step)
            #     for layer in model.layers:
            #         for i in range(len(layer.variables)):
            #             tf.summary.histogram(layer.variables[i].name, layer.variables[i].numpy(), step=step)
            # tf.summary.histogram(layer.)

            if step % 10 == 0:
                print(f"Loss values on the training data is: {loss_value}")

        # Evaluating test rmse
        # For shorter evaluation time testing data is divided by 10
        rmse_metric.reset_state()
        training_data.shuffle_test_data()
        for step in range(training_data.get_test_length() // 10):
            x1_batch, x2_batch, y_batch = training_data.get_test_data(step)
            x1_batch, x2_batch, y_batch = np.array(x1_batch), np.array(x2_batch), np.array(y_batch)

            pred = model([x2_batch, x1_batch])
            rmse_metric(y_batch, pred)

        with train_summary_writer.as_default():
            tf.summary.scalar("Train_loss", loss_value, step=epoch)
            tf.summary.scalar("Test_loss", rmse_metric.result().numpy(), step=epoch)
            tf.summary.scalar("learning_rate", optimizer.learning_rate, step=epoch)
            for layer in model.layers:
                for i in range(len(layer.variables)):
                    tf.summary.histogram(layer.variables[i].name, layer.variables[i].numpy(), step=step)

        #TODO: Add early stopping

        # log every 50 epoch
        # if epoch % 2 == 0:
        #     if epoch == 0:
        #         lowest_loss_val = 0
        #     loss_sum = []
        #     number_per_slices = 20
        #     training_data.shuffle_validation_data()
        #     for i in range(int(np.ceil(training_data.get_test_length()/number_per_slices))):
        #         if i < int(np.floor(training_data.get_test_length()/number_per_slices)):
        #             x_batch, y_batch, weights_batch = training_data.get_test_data(slice(i*number_per_slices, (i + 1)*number_per_slices, None))
        #             x_batch = np.array(x_batch)
        #             y_batch = np.array(y_batch)
        #             weights_batch = np.array(weights_batch, dtype="float32")
        #         else:
        #             x_batch, y_batch, weights_batch = training_data.get_test_data(slice(i*number_per_slices,
        #                                                                                 training_data.get_test_length(), None))
        #             x_batch = np.array(x_batch)
        #             y_batch = np.array(y_batch)
        #             weights_batch = np.array(weights_batch, dtype="float32")

        #         loss_val = compute_loss_from_model(model=model, loss_fn=loss_fn,
        #                                            x_input_data=x_batch, y_data_true=y_batch,
        #                                            weights=weights_batch, should_print=False)
        #         loss_sum.append(loss_val)
        #     loss_sum = np.array(loss_sum).sum()/len(loss_sum)
        #     test_loss_list.append(10*np.log10(loss_sum))
        #     print(f"Loss value on the test data is: {loss_sum}")
        #     if lowest_loss_val == 0:
        #         lowest_loss_val = loss_sum
        #     if loss_sum < lowest_loss_val:
        #         model.save_weights(save_weights_file_path, overwrite=True, save_format="tf")
        #         lowest_loss_val = loss_sum
    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(range(len(train_loss_list)), train_loss_list)
    # axs[0].set_title("Train loss = f(step*epochs)")
    # axs[1].plot(range(len(test_loss_list)), test_loss_list)
    # axs[1].set_title("Test loss = f(every second epoch)")
    # axs[2].plot(range(len(learning_rate_list)), learning_rate_list)
    # axs[2].set_title("learning rate = f(step*epochs)")
    # fig.set_constrained_layout(True)
    # plt.show()

else:
    model = create_model(movie_titles.count()[0])
    model.load_weights(save_weights_file_path)

    recommender = da.Recommender(model, training_dataset, movie_titles, movie_titles.count()[0], highest_score)
    recommended_movies = recommender.recommend(2557870, 5)
    print(recommended_movies)



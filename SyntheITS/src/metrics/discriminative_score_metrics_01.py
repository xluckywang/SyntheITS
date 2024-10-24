import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


def discriminative_score_metrics(dataX, dataX_hat):
    """
    Calculate the discriminative score between original and synthetic data using a post-hoc RNN classifier.

    Args:
    - dataX: Original data
    - dataX_hat: Synthetic data

    Returns:
    - Discriminative score (absolute difference between classification accuracy and 0.5)
    """

    tf.random.set_seed(1)

    No = len(dataX)
    data_dim = len(dataX[0][0, :])

    dataT = []
    max_seq_len = 0
    for i in range(No):
        max_seq_len = max(max_seq_len, len(dataX[i][:, 0]))
        dataT.append(len(dataX[i][:, 0]))

    hidden_dim = max(int(data_dim), 1)
    iterations = 2000
    batch_size = 128

    X = tf.keras.layers.Input(shape=(max_seq_len, data_dim), name="myinput_x")
    X_hat = tf.keras.layers.Input(shape=(max_seq_len, data_dim), name="myinput_x_hat")

    # Times
    T = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="myinput_t")
    T_hat = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="myinput_t_hat")

    def discriminator(X, T):
        gru = tf.keras.layers.GRU(units=hidden_dim, activation='tanh', name='cd_cell')
        dense = tf.keras.layers.Dense(units=1, activation=None)

        def call(inputs):
            outputs = gru(inputs)
            Y_hat = dense(outputs)
            Y_hat_final = tf.nn.sigmoid(Y_hat)
            return Y_hat, Y_hat_final

        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                Y_hat, Y_hat_final = call(inputs)
                d_loss = tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True)(targets, Y_hat))

            d_vars = gru.trainable_variables + dense.trainable_variables

            gradients = tape.gradient(d_loss, d_vars)
            optimizer = tf.keras.optimizers.Adam()
            optimizer.apply_gradients(zip(gradients, d_vars))

            return Y_hat, Y_hat_final, d_vars

        X_input = tf.keras.Input(shape=X.shape[1:])  # 定义输入张量
        Y_hat, Y_hat_final, d_vars = train_step(X_input, T)

        return Y_hat, Y_hat_final, d_vars

    def train_test_divide(dataX, dataX_hat, dataT):
        No = len(dataX)
        idx = np.random.permutation(No)
        train_idx = idx[:int(No * 0.8)]
        test_idx = idx[int(No * 0.8):]

        trainX = [dataX[i] for i in train_idx]
        trainX_hat = [dataX_hat[i] for i in train_idx]

        testX = [dataX[i] for i in test_idx]
        testX_hat = [dataX_hat[i] for i in test_idx]

        trainT = [dataT[i] for i in train_idx]
        testT = [dataT[i] for i in test_idx]

        return trainX, trainX_hat, testX, testX_hat, trainT, testT

    Y_real, Y_pred_real, d_vars = discriminator(X, T)
    Y_fake, Y_pred_fake, _ = discriminator(X_hat, T_hat)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_real, labels=tf.ones_like(Y_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Y_fake, labels=tf.zeros_like(Y_fake)))
    D_loss = D_loss_real + D_loss_fake

    D_optimizer = tf.keras.optimizers.Adam()
    D_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def D_train_step(X, T, X_hat, T_hat):
        with tf.GradientTape() as tape:
            Y_real, _, _ = discriminator(X, T)
            Y_fake, _, _ = discriminator(X_hat, T_hat)
            D_loss_real = D_loss_fn(tf.ones_like(Y_real), Y_real)
            D_loss_fake = D_loss_fn(tf.zeros_like(Y_fake), Y_fake)
            D_loss = D_loss_real + D_loss_fake
        gradients = tape.gradient(D_loss, d_vars)
        D_optimizer.apply_gradients(zip(gradients, d_vars))
        return D_loss

    trainX, trainX_hat, testX, testX_hat, trainT, testT = train_test_divide(dataX, dataX_hat, dataT)

    train_dataset = tf.data.Dataset.from_tensor_slices((trainX, trainT, trainX_hat, trainT))
    train_dataset = train_dataset.shuffle(buffer_size=len(trainX)).batch(batch_size)

    for itt in range(iterations):
        for X_mb, T_mb, X_hat_mb, T_hat_mb in train_dataset:
            X_mb = tf.squeeze(X_mb, axis=1)  # 减少维度使其匹配GRU层的要求
            X_hat_mb = tf.squeeze(X_hat_mb, axis=1)  # 减少维度使其匹配GRU层的要求
            D_loss = D_train_step(X_mb, T_mb, X_hat_mb, T_hat_mb)

        if itt % 500 == 0:
            print("[step: {}] loss - d loss: {}".format(itt, np.round(D_loss.numpy(), 4)))

    Y_pred_real_curr, Y_pred_fake_curr = discriminator(testX, testT)[1], discriminator(testX_hat, testT)[1]

    Y_pred_final = np.squeeze(np.concatenate((Y_pred_real_curr, Y_pred_fake_curr), axis=0))
    Y_label_final = np.concatenate((np.ones([len(Y_pred_real_curr), ]), np.zeros([len(Y_pred_real_curr), ])), axis=0)

    Acc = accuracy_score(Y_label_final, Y_pred_final > 0.5)

    Disc_Score = np.abs(0.5 - Acc)

    return Disc_Score

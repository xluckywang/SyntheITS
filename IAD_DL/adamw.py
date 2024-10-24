from keras.optimizers import Optimizer
from keras import backend as K

class AdamW(Optimizer):
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=1e-4, name='AdamW', **kwargs):
        super(AdamW, self).__init__(name=name, **kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('decay', decay)
        self._set_hyper('weight_decay', weight_decay)
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply_dense(self, grad, var):
        lr = self._get_hyper('learning_rate')
        if self.initial_decay > 0:
            lr *= (1. / (1. + self._get_hyper('decay') * K.cast(self.iterations, K.dtype(self._get_hyper('decay')))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self._get_hyper('beta_2'), t)) / (1. - K.pow(self._get_hyper('beta_1'), t)))

        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')

        m_t = (self._get_hyper('beta_1') * m) + (1. - self._get_hyper('beta_1')) * grad
        v_t = (self._get_hyper('beta_2') * v) + (1. - self._get_hyper('beta_2')) * K.square(grad)
        var_update = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - lr * self._get_hyper('weight_decay') * var

        self.iterations.assign_add(1)
        self._updates.append(K.update(m, m_t))
        self._updates.append(K.update(v, v_t))
        self._updates.append(K.update(var, var_update))

    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

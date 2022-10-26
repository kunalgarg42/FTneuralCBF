import numpy as np
import torch


class Dataset_with_Grad(object):

    def __init__(self, n_state, m_control, buffer_size=100000):
        self.n_state = n_state
        self.m_control = m_control
        self.buffer_size = buffer_size
        self.buffer_data_s = torch.tensor([]).reshape(0, n_state)
        self.buffer_data_u_NN = torch.tensor([]).reshape(0, m_control)
        self.buffer_data_u = torch.tensor([]).reshape(0, m_control)
        self.dang_count = 0
        self.safe_count = 0
        self.mid_count = 0

    def add_data(self, state, u, u_nominal):
        """
        args:
            state (n_state,): state of the agent
            obstacle (k_obstacle, n_state): K obstacles
            u_nominal (m_control,): the nominal control
            state_next (n_state,): state of the agent at the next timestep
        """

        self.buffer_data_s = torch.vstack((self.buffer_data_s, state.clone()))
        self.buffer_data_u_NN = torch.vstack((self.buffer_data_u_NN, u.clone()))
        self.buffer_data_u = torch.vstack((self.buffer_data_u, u_nominal.clone()))

        self.buffer_data_s = self.buffer_data_s[-self.buffer_size:]
        self.buffer_data_u_NN = self.buffer_data_u_NN[-self.buffer_size:]
        self.buffer_data_u = self.buffer_data_u[-self.buffer_size:]

    def sample_data(self, batch_size, index):

        # s, u_NN, u = self.sample_data_from_buffer(batch_size, self.buffer_data, index)

        indices_init = (index - 1) * batch_size

        indices_end = index * batch_size
        if indices_end > self.buffer_data_s.shape[1]:
            indices = np.random.randint(len(self.buffer_data_s), size=batch_size)
        else:
            indices = np.arange(indices_init, indices_end, 1)

        s = self.buffer_data_s[indices, :]
        u_NN = self.buffer_data_u_NN[indices, :]
        u = self.buffer_data_u[indices, :]
        u = np.array(u)
        return s, u_NN, u

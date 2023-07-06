import numpy as np
import torch


class Dataset_with_Grad(object):

    def __init__(self, y_state, n_state, m_control, train_u, buffer_size=200000, traj_len = 100):
        self.n_state = n_state
        self.train_u = train_u
        self.m_control = m_control
        self.buffer_size = buffer_size
        self.traj_len = traj_len
        # self.ns = int(self.buffer_size / self.traj_len)
        self.ns = self.buffer_size
        self.buffer_data_s = torch.tensor([]).reshape(0, y_state)
        self.buffer_data_s_diff = torch.tensor([]).reshape(0, n_state)
        self.buffer_data_u_NN = torch.tensor([]).reshape(0, m_control)
        self.buffer_data_u = torch.tensor([]).reshape(0, 1)
        self.dang_count = 0
        self.safe_count = 0
        self.mid_count = 0

        # Maintain a list of permuted indices so that we can scramble the data on each
        # epoch
        self.permuted_indices = torch.tensor([])

    def add_data(self, state, state_diff, u, u_nominal):
        """
        args:
            state (n_state,): state of the agent
            obstacle (k_obstacle, n_state): K obstacles
            u_nominal (m_control,): the nominal control
            state_next (n_state,): state of the agent at the next timestep
        """

        if self.buffer_data_s.shape[0] == 0:
            self.buffer_data_s = state.clone()
        else:
            self.buffer_data_s = torch.vstack((self.buffer_data_s, state.clone()))
        
        if self.buffer_data_s_diff.shape[0] == 0:
            self.buffer_data_s_diff = state_diff.clone()
        else:
            self.buffer_data_s_diff = torch.vstack((self.buffer_data_s_diff, state_diff.clone()))
        
        if self.buffer_data_u_NN.shape[0] == 0:
            self.buffer_data_u_NN = u.clone()
        else:
            self.buffer_data_u_NN = torch.vstack((self.buffer_data_u_NN, u.clone()))
        
        if self.buffer_data_u.shape[0] == 0:
            self.buffer_data_u = u_nominal.clone()
        else:
            self.buffer_data_u = torch.vstack((self.buffer_data_u, u_nominal.clone()))

        self.buffer_data_s = self.buffer_data_s[-self.buffer_size:]
        self.buffer_data_s_diff = self.buffer_data_s_diff[-self.buffer_size:]
        self.buffer_data_u_NN = self.buffer_data_u_NN[-self.buffer_size:]
        self.buffer_data_u = self.buffer_data_u[-self.ns:]

        # Get a new set of permuted indices
        self.permuted_indices = torch.randperm(self.n_pts)

    @property
    def n_pts(self):
        if self.buffer_data_s.shape[0] == 0:
            return self.buffer_data_s_diff.shape[0]
        else:
            return self.buffer_data_s.shape[0]

    @property
    def n_pts_gamma(self):
        return self.buffer_data_u.shape[0]

    def sample_data(self, batch_size, index):
        """
        Sample batch_size data points from the data buffers.

        args:
            batch_size: how many points to sample
            index: the index of the batch to sample (so that we can sample without
                replacement)
        returns:
            a random selection of batch_size data points, sampled without replacement.
        """

        # s, u_NN, u = self.sample_data_from_buffer(batch_size, self.buffer_data, index)

        # We'll sample these points by pulling a range of indices from the list of
        # permuted indices. Start by getting the start and end points of this range
        indices_init = index * batch_size
        indices_end = (index + 1) * batch_size

        # If the end of the range exceeds the number of available data points,
        # shift both the start and the end back until the batch fits
        if indices_end > self.n_pts:
            extra_pts_needed = indices_end - self.n_pts
            indices_init -= extra_pts_needed
            indices_end -= extra_pts_needed

        # Get the slice of randomly permuted indices
        indices = self.permuted_indices[indices_init:indices_end]
        # print(index)
        # print(batch_size)
        # print((indices_init, indices_end))
        # print(indices[:10])

        # Sample data from those indices.
        s = self.buffer_data_s[indices, :]

        # Not sure what's happening here. Looks like we're not sampling control values?
        # Probably OK since these return values aren't being used.
        # if self.train_u > 0:
        #     u_NN = self.buffer_data_u_NN[indices, :]
        #     u = self.buffer_data_u[indices, :]
        #     u = np.array(u)
        # else:
        u_NN = []
        u = []
        return s, u_NN, u

    def sample_data_all(self, batch_size, index):
            """
            Sample batch_size data points from the data buffers.

            args:
                batch_size: how many points to sample
                index: the index of the batch to sample (so that we can sample without
                    replacement)
            returns:
                a random selection of batch_size data points, sampled without replacement.
            """

            # indices_init = index * batch_size

            # indices_end = (index + 1) * batch_size

            # if indices_end > self.buffer_data_s.shape[1]:
            #     indices_s = np.arange(0, batch_size, 1)
            #     indices_gamma = np.arange(0, ns, 1)
            # else:
            #     indices_init_gamma = index * ns
            #     indices_end_gamma = (index + 1) * ns
            #     indices_s = np.arange(indices_init, indices_end, 1)
            #     indices_gamma = np.arange(indices_init_gamma, indices_end_gamma, 1)

            indices_init = index * batch_size
            indices_end = (index + 1) * batch_size

            # If the end of the range exceeds the number of available data points,
            # shift both the start and the end back until the batch fits
            if indices_end > self.n_pts:
                extra_pts_needed = indices_end - self.n_pts
                indices_init -= extra_pts_needed
                indices_end -= extra_pts_needed

            # Get the slice of randomly permuted indices
            indices = self.permuted_indices[indices_init:indices_end]

            s = self.buffer_data_s[indices, :]
            s_diff = self.buffer_data_s_diff[indices, :]
            u = self.buffer_data_u_NN[indices, :]

            gamma = self.buffer_data_u[indices]

            return s, s_diff, u, gamma

    def sample_only_res(self, batch_size, index):
            """
            Sample batch_size data points from the data buffers.

            args:
                batch_size: how many points to sample
                index: the index of the batch to sample (so that we can sample without
                    replacement)
            returns:
                a random selection of batch_size data points, sampled without replacement.
            """

            indices_init = index * batch_size
            indices_end = (index + 1) * batch_size

            # If the end of the range exceeds the number of available data points,
            # shift both the start and the end back until the batch fits
            if indices_end > self.n_pts:
                extra_pts_needed = indices_end - self.n_pts
                indices_init -= extra_pts_needed
                indices_end -= extra_pts_needed

            # Get the slice of randomly permuted indices
            indices = self.permuted_indices[indices_init:indices_end]

            s_diff = self.buffer_data_s_diff[indices, :]

            gamma = self.buffer_data_u[indices]

            return s_diff, gamma
import numpy as np
import torch


class Dataset(object):

    def __init__(self, n_state, m_control, n_pos, buffer_size=10000, safe_alpha=0.3, dang_alpha=0.4):
        self.n_state = n_state
        self.m_control = m_control
        self.n_pos = n_pos
        self.safe_alpha = safe_alpha
        self.dang_alpha = dang_alpha
        self.buffer_size = buffer_size
        self.buffer_safe = []
        self.buffer_dang = []
        self.buffer_mid = []
        self.dang_count = 0
        self.safe_count = 0
        self.mid_count = 0

    def add_data(self, state, u_nominal, state_next, state_error):
        """
        args:
            state (n_state,): state of the agent
            obstacle (k_obstacle, n_state): K obstacles
            u_nominal (m_control,): the nominal control
            state_next (n_state,): state of the agent at the next timestep
        """
        # print(state[0][0])
        # print(obstacle[0][0])
        alpha = state[:,1]
        # print(state)
        # dist = np.linalg.norm(obstacle - state)
        min_alpha = torch.min(alpha)
        # print(min_alpha)
        max_alpha = torch.max(alpha)
        # print(max_alpha)
        data = [np.copy(state).astype(np.float32), 
                np.copy(u_nominal).astype(np.float32), np.copy(state_next).astype(np.float32),
                np.copy(state_error).astype(np.float32)]
        
        # print(min_dist)
        # print(data.shape)

        # print(self.dang_dist)
        if min_alpha < - self.dang_alpha or max_alpha > self.dang_alpha:
            self.buffer_dang.append(data)
            self.buffer_dang = self.buffer_dang[-self.buffer_size:]
            # print('added in dang')
            self.dang_count = self.dang_count+1
        elif min_alpha > - self.safe_alpha and max_alpha < self.safe_alpha:
            self.buffer_safe.append(data)
            self.buffer_safe = self.buffer_safe[-self.buffer_size:]
            # print('added in safe')
            self.safe_count= self.safe_count+1
        else:
            self.buffer_mid.append(data)
            self.buffer_mid = self.buffer_mid[-self.buffer_size:]
            # print('added in mid')
            self.mid_count = self.mid_count+1

    def sample_data(self, batch_size):
        num_safe = batch_size // 3 
        num_dang = batch_size // 3
        num_mid = batch_size - num_safe - num_dang
        # print(self.mid_count)
        # print(self.safe_count)
        # print(self.dang_count)
        # print(np.array(self.buffer_safe,dtype = object).shape)
        # print(np.array(self.buffer_mid,dtype = object).shape)
        # print(np.array(self.buffer_dang,dtype = object).shape)
        
        n_state = self.n_state
        m_control = self.m_control

        if self.dang_count>0:
            s_dang, u_dang, s_next_dang, e_dang = self.sample_data_from_buffer(num_dang, self.buffer_dang)
        else:
            if self.safe_count>0:
                num_safe = num_safe + num_dang
            s_dang, u_dang, s_next_dang, e_dang = np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,m_control),np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,n_state)

        if self.mid_count>0:
            s_mid, u_mid, s_next_mid, e_mid = self.sample_data_from_buffer(num_mid, self.buffer_mid)
        else:
            if self.safe_count>0:
                num_safe = num_safe + num_mid
            s_mid, u_mid, s_next_mid, e_mid = np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,m_control),np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,n_state)

        # print(self.dang_count)
        # print(self.mid_count)
        # print(self.safe_count)
        if self.safe_count>0:
            s_safe, u_safe, s_next_safe, e_safe = self.sample_data_from_buffer(num_safe, self.buffer_safe)
            # print(s_safe.shape)
        else:
            s_safe, u_safe, s_next_safe, e_safe = np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,m_control),np.array([],dtype = np.float32).reshape(0,n_state),np.array([],dtype = np.float32).reshape(0,n_state)

        s = np.concatenate([s_safe, s_dang, s_mid], axis=0)
        u = np.concatenate([u_safe, u_dang, u_mid], axis=0)
        s_next = np.concatenate([s_next_safe, s_next_dang, s_next_mid], axis=0)
        e = np.concatenate([e_safe, e_dang, e_mid], axis=0)

        return s,  u, s_next, e

    def sample_data_from_buffer(self, batch_size, buffer):
        indices = np.random.randint(len(buffer), size=(batch_size))
        s = np.zeros((batch_size, self.n_state), dtype=np.float32)
        u = np.zeros((batch_size, self.m_control), dtype=np.float32)
        s_next = np.zeros((batch_size, self.n_state), dtype=np.float32)
        e = np.zeros((batch_size, self.n_state), dtype=np.float32)
        for i, ind in enumerate(indices):
            state, u_nominal, state_next, state_error = buffer[ind]
            s[i] = state
            u[i] = u_nominal
            s_next[i] = state_next
            e[i] = state_error
        return s, u, s_next, e


class Dataset_with_Grad(object):

    def __init__(self, n_state, m_control, n_pos, buffer_size=100000, safe_alpha=0.3, dang_alpha=0.4):
        self.n_state = n_state
        self.m_control = m_control
        self.n_pos = n_pos
        self.safe_alpha = safe_alpha
        self.dang_alpha = dang_alpha
        self.buffer_size = buffer_size
        self.buffer_data = []
        self.buffer_safe = []
        self.buffer_dang = []
        self.buffer_mid = []
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
        alpha = state[:,1].clone()
        min_alpha = torch.min(alpha)
        max_alpha = torch.max(alpha)

        data = [state.clone(), u.clone(), u_nominal.clone()]
        # print(min_dist)

        self.buffer_data.append(data)
        self.buffer_dang = self.buffer_dang[-self.buffer_size:]
        # print(self.dang_dist)
        # if min_alpha < - self.dang_alpha or max_alpha > self.dang_alpha:
        #     self.buffer_dang.append(data)
        #     self.buffer_dang = self.buffer_dang[-self.buffer_size:]
        #     self.dang_count = self.dang_count+1
        # elif min_alpha > - self.safe_alpha and max_alpha < self.safe_alpha:
        #     self.buffer_safe.append(data)
        #     self.buffer_safe = self.buffer_safe[-self.buffer_size:]
        #     self.safe_count= self.safe_count+1
        # else:
        #     self.buffer_mid.append(data)
        #     self.buffer_mid = self.buffer_mid[-self.buffer_size:]
        #     self.mid_count = self.mid_count+1
        # print(np.array([self.buffer_safe]).shape)

    def sample_data(self, batch_size,index):
        num_safe = batch_size // 3 
        num_dang = batch_size // 3
        num_mid = batch_size - num_safe - num_dang
        n_state = self.n_state
        m_control = self.m_control
        s, u_NN, u = self.sample_data_from_buffer(batch_size,self.buffer_data,index)
        # if self.dang_count>0:
        #     s_dang, u_NN_dang, u_dang = self.sample_data_from_buffer(num_dang, self.buffer_dang)
        # else:
        #     if self.safe_count>0:
        #         num_safe = num_safe + num_dang
        #     s_dang, u_NN_dang, u_dang = torch.zeros(0, self.n_state).reshape(0,n_state),torch.zeros(0,self.m_control),np.array([],dtype = np.float32).reshape(0,m_control)

        # if self.mid_count>0:
        #     s_mid, u_NN_mid, u_mid = self.sample_data_from_buffer(num_mid, self.buffer_mid)
        # else:
        #     if self.safe_count>0:
        #         num_safe = num_safe + num_mid
        #     s_mid, u_NN_mid, u_mid = torch.zeros(0, self.n_state).reshape(0,n_state),torch.zeros(0,self.m_control),np.array([],dtype = np.float32).reshape(0,m_control)

        # if self.safe_count>0:
        #     s_safe, u_NN_safe, u_safe = self.sample_data_from_buffer(num_safe, self.buffer_safe)
        # else:
        #     s_safe, u_NN_safe, u_safe = torch.zeros(0, self.n_state).reshape(0,n_state),torch.zeros(0,self.m_control),np.array([],dtype = np.float32).reshape(0,m_control)

        # s = torch.cat([s_safe, s_dang, s_mid], axis=0)
        # u_NN = torch.cat([u_NN_safe, u_NN_dang, u_NN_mid], axis=0)
        # u = np.concatenate([u_safe, u_dang, u_mid], axis=0)

        return s, u_NN, u

    def sample_data_from_buffer(self, batch_size, buffer,index):
        # indices = np.random.randint(len(buffer), size=(batch_size))
        indices_init = (index-1) * batch_size
        indices_end = (index) * batch_size
        if indices_end > np.array([buffer]).shape[1]:
            indices = np.random.randint(len(buffer), size=(batch_size))
        else:
            indices = np.arange(indices_init, indices_end, 1)
        s = torch.zeros(batch_size, self.n_state)
        u_NN = torch.zeros(batch_size, self.m_control)
        u = np.zeros((batch_size, self.m_control), dtype=np.float32)
        # print(np.array([buffer]).shape)
        for i, ind in enumerate(indices):
            state, u_neural, u_nominal = buffer[ind]
            s[i] = state
            u_NN[i] = u_neural
            u[i] = u_nominal
        return s, u_NN, u
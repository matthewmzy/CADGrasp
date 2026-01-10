import os
import sys
from cadgrasp.optimizer.IBSAdam import IBSAdam
import torch
from tqdm import tqdm


class AdamOptimizer:
    def __init__(self, hand_name, writer, max_iters=200,
                 learning_rate=5e-3, lr_decay=0.5, decay_every=100, 
                 parallel_num=10, device='cuda'):
        self.writer = writer
        self.hand_name = hand_name
        self.device = device
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.parallel_num = parallel_num
        self.opt_model = IBSAdam(hand_name=hand_name, 
                                 parallel_num=self.parallel_num, 
                                 device=device)

    def run_adam(self, ibs_triplets, running_name, cone_viz_num=0, cone_mu=0, filt_or_not=True):
        q_trajectory = []
        self.opt_model.reset(ibs_triplets, running_name, cone_viz_num=cone_viz_num, cone_mu=cone_mu, filt_or_not=filt_or_not)
        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())

        energy_dict = {'E_joint': [], 
                       'E_spen' : [], 
                       'E_cont_1' : [], 
                       'E_cont_2' : [], 
                       'E_cont_3' : [], 
                       'E_cont_4' : [], 
                       'E_pen' : [], 
                       'E_trans' : [], 
                       'E_distal' : []}
        
        self.opt_model.set_opt_weight(1, 0.6, 0.35, self.learning_rate, self.decay_every, self.lr_decay)
        for i_iter in tqdm(range(self.max_iters), desc=f'{running_name}'):
            if i_iter == self.max_iters//2:
                self.opt_model.set_opt_weight(2, 0.25, 0.3, self.learning_rate, self.decay_every, self.lr_decay)
            self.opt_model.step(energy_dict)
            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())

            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {f'{i_energy}': energy[i_energy] for i_energy in range(len(energy))}
                self.writer.add_scalars(main_tag=f'energy/{running_name}', tag_scalar_dict=tag_scaler_dict, global_step=i_iter)
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        num_particles, traj_len, n_dofs = q_trajectory.shape
        num_particles = num_particles // self.parallel_num
        q_trajectory = q_trajectory.reshape(num_particles, self.parallel_num, traj_len, n_dofs)
        energy_list = self.opt_model.energy.detach().cpu().reshape(-1, self.parallel_num)
        energy_list[energy_list == float('inf')] = 1e10
        energy_list[energy_list != energy_list] = 1e10
        min_indices = torch.argmin(energy_list, dim=1)
        q_trajectory = q_trajectory[torch.arange(num_particles), min_indices].reshape(-1, traj_len, n_dofs)
        for key in energy_dict.keys():
            if len(energy_dict[key]) == 0:
                continue
            energy_dict[key] = torch.stack(energy_dict[key], dim=0) \
                .detach().cpu() \
                .reshape(-1, num_particles, self.parallel_num) \
                [:,torch.arange(num_particles), min_indices] \
                .reshape(-1, num_particles)

        return q_trajectory, energy_dict
    
    
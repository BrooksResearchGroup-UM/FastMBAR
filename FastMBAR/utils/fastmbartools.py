import numpy as np
import simtk.unit as unit
from FastMBAR import *
import sys
import os

__author__ = 'Shuming Liu'

def umbrella_bias(x, param):
    # param should be np.array([kappa1, x0_1, kappa2, x0_2, ...])
    param = np.reshape(param, (-1, 2))
    kappa = param[:, 0]
    x0 = param[:, 1]
    bias = 0.5*np.sum(kappa*((x - x0)**2), axis=1)
    return bias


def linear_bias(x, param):
    # param should be np.array([kappa1, x0_1, kappa2, x0_2, ...])
    param = np.reshape(param, (-1, 2))
    kappa = param[:, 0]
    x0 = param[:, 1]
    bias = np.sum(kappa*(x - x0), axis=1)
    return bias


class FastMBARSolver:
    def __init__(self, temp, traj_bias_cv, kB, bias_param, bias_func='umbrella', unbiased_energy=None):
        # initialize
        self.traj_bias_cv = traj_bias_cv
        # temp can be a list or array, or if all the trajectories share the same temperature, then set temp as a float
        if np.isscalar(temp):
            temp = [temp]*len(self.traj_bias_cv)
        self.temp = np.array(temp)
        self.kB = kB
        self.bias_param = bias_param
        if bias_func == 'umbrella':
            print('Use umbrella bias for all the trajectories')
            self.bias_func = [umbrella_bias]*len(self.traj_bias_cv)
        elif bias_func == 'linear':
            print('Use linear bias for all the trajectories')
            self.bias_func = [linear_bias]*len(self.traj_bias_cv)
        else:
            if type(bias_func) is not list:
                bias_func = [bias_func]*len(self.traj_bias_cv)
            self.bias_func = bias_func
        self.unbiased_energy = unbiased_energy
        if self.unbiased_energy is None:
            print('Set self.unbiased_energy as None')
            print('Be careful this requires all the simulations and target PMF to be at the same temperature!')
        
    def reset_kB(self, kB):
        print('Be careful that kB is a global parameter!')
        self.kB = kB
    
    def add_traj(self, temp, traj_bias_cv, bias_param, bias_func='umbrella', unbiased_energy=None):
        # add new trajectories
        self.traj_bias_cv += traj_bias_cv
        if np.isscalar(temp):
            temp = [temp]*len(traj_bias_cv)
        self.temp = np.array(self.temp.tolist() + temp)
        self.bias_param += bias_param
        if bias_func == 'umbrella':
            print('Use umbrella bias for all the trajectories')
            bias_func = [umbrella_bias]*len(traj_bias_cv)
        elif bias_func == 'linear':
            print('Use linear bias for all the trajectories')
            bias_func = [linear_bias]*len(traj_bias_cv)
        else:
            if type(bias_func) is not list:
                bias_func = [bias_func]*len(traj_bias_cv)
        self.bias_func += bias_func
        if (self.unbiased_energy is None) or (unbiased_energy is None):
            self.unbiased_energy = None
            print('Set self.unbiased_energy as None')
            print('Be careful this requires all the simulations and target PMF to be at the same temperature!')
        else:
            self.unbiased_energy += unbiased_energy
    
    def solve(self, bootstrap=True, cuda=False, verbose=False):
        self.M = len(self.traj_bias_cv) # the number of thermodynamic states
        self.n_samples = np.array([len(each) for each in self.traj_bias_cv]) # the number of samples in each thermodynamic state
        self.X = np.concatenate(self.traj_bias_cv, axis=0) # all the samples 
        self.N = self.X.shape[0] # the total number of samples
        A = np.zeros((self.M, self.N))
        self.unique_temp = None
        if self.unbiased_energy is None:
            self.E = 0
            # check if all the temperatures are the same
            if np.all(self.temp == self.temp[0]):
                # the simulation and target PMF are all at self.unique_temp
                self.unique_temp = self.temp[0]
            else:
                print('Error: temperatures are not identical, so we have to input unbiased energies')
                sys.exit(1)
        else:
            self.E = self.unbiased_energy
        A += self.E
        
        # add bias and convert to reduced potential energy
        for i in range(self.M):
            A[i, :] += self.bias_func[i](self.X, self.bias_param[i])
            A[i, :] *= 1/(self.kB*self.temp[i])
        self.A = A
        
        # solve
        self.fastmbar = FastMBAR(energy=self.A, num_conf=self.n_samples, cuda=cuda, bootstrap=bootstrap, verbose=verbose)
        
    
    def computePMF(self, traj_target_cv, target_temp, bins):
        # compute PMF for the target CV
        # check temperature
        if self.unique_temp is not None:
            if self.unique_temp != target_temp:
                print(f'Error: as we do not input unbiased energy, we can only compute PMF at temperature {self.unique_temp}')
                sys.exit(1)
        
        # check size of traj_target_cv
        if len(traj_target_cv) != len(self.traj_bias_cv):
            print('Error: the number of trajectories in traj_target_cv and self.traj_bias_cv are not consistent')
            sys.exit(1)
        for i in range(len(traj_target_cv)):
            if traj_target_cv[i].shape[0] != self.traj_bias_cv[i].shape[0]:
                print(f'Error: the number of samples in the traj_target_cv[{i}] and self.traj_bias_cv[{i}] are not consistent')
                sys.exit(1)
        
        # concatenate samples    
        self.Y = np.concatenate(traj_target_cv, axis=0)
        
        # set bins for the target CVs
        # bins should follow np.histogramdd
        self.bins = bins
        
        # get the number of rows for perturbed reduced potential energy matrix
        hist_0, edges = np.histogramdd(self.Y[[0]], self.bins)
        self.edges = edges
        L = 1
        for each in self.edges:
            L *= (len(each) - 1)
        self.L = L
        
        # get the centers of all the grids
        centers = []
        for each in self.edges:
            centers.append(0.5*(each[1:] + each[:-1]))
        self.centers = centers
        
        # get the extended coordinate of grid centers 
        # the extended coordinate is of shape (self.L, self.Y.shape[1])
        # we only consider self.Y.shape[1] <= 3
        if self.Y.shape[1] == 1:
            self.ext_centers = self.centers
        elif self.Y.shape[1] == 2:
            ext_centers = []
            for c1 in self.centers[0]:
                for c2 in self.centers[1]:
                    ext_centers.append([c1, c2])
            self.ext_centers = np.array(ext_centers)
        elif self.Y.shape[1] == 3:
            ext_centers = []
            for c1 in self.centers[0]:
                for c2 in self.centers[1]:
                    for c3 in self.centers[3]:
                        ext_centers.append([c1, c2, c3])
            self.ext_centers = np.array(ext_centers)
        else:
            print('Warning: the number of dimensions for the target CV is > 3')
            print('The method does not compute self.ext_centers')
            
        # put samples into the grid and set restraints
        R = np.zeros((self.L, self.N))
        for i in range(self.N):
            hist_i, edges = np.histogramdd(self.Y[[i]], self.bins)
            hist_i[hist_i == 0] = np.inf
            hist_i[hist_i == 1] = 0
            R[:, i] += np.reshape(hist_i, -1)
        self.R = R
        self.B = self.E + self.R
        
        # compute PMF
        self.PMF, _ = self.fastmbar.calculate_free_energies_of_perturbed_states(self.B)
        
    def writePMF(self, output_path=None, header='', remove_nan=True, PMF_min_as_zero=True):
        output = np.zeros((self.L, self.Y.shape[1] + 1))
        output[:, :-1] = self.ext_centers
        output[:, -1] = self.PMF
        if remove_nan:
            output = output[~np.isnan(self.PMF)]
            if PMF_min_as_zero:
                output[:, -1] -= np.amin(output[:, -1])
        self.output = output
        if output_path is not None:
            np.savetxt(output_path, self.output, header=header)
    
        

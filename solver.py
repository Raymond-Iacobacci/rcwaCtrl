import sys
import typing
import scipy as sp  # This shouldn't have to be in here, it breaks minimal imports

import autograd.numpy as npa
import numpy as np
import numpy.typing as npt
import ff

logging = False
# Everything in this file is set up for a single dimension of movement. It must be altered to allow for a second dimension.


def extended_redheffer_star_product(mat1: npt.ArrayLike, mat2: npt.ArrayLike) -> np.ndarray:
    '''
    Assumes buffer between two systems.
    Compute the system scattering matrix and take the input from the bottom to be 0. Then use the reflected and the inputs to calculate backwards (while moving around the matrices) what the next coupling coefficients should be.
    '''
    sa00, sa01, sa10, sa11 = ff.quar(mat1)
    sb00, sb01, sb10, sb11 = ff.quar(mat2)
    identity = np.eye(sb00.shape[0])
    # This matrix accounts for the infinite reflections (geometric) that occur at each interface
    inf1 = np.linalg.inv(identity - sb00 @ sa11)
    inf2 = np.linalg.inv(identity - sa11 @ sb00)
    '''
    Assuming that each scattering matrix has been computed (shown here), then the updates to the mode coefficients are simply those modes * the scattering matrices themselves.
    '''
    sc00 = sa00 + sa01@inf1@sb00@sa10
    sc01 = sa01@inf1@sb01
    sc10 = sb10@inf2@sa10
    sc11 = sb11 + sb10@inf2@sa11@sb01
    sc = np.block([[sc00, sc01], [sc10, sc11]])
    return sc

class Layer:
    """
    Class for defining a single layer of a layer stack used in a simulation
    """

    def __init__(self, permeability: npt.ArrayLike, permittivity: npt.ArrayLike, thickness: float, n_harmonics: int = 1):
        """
        No crystal should be created beforehand. We interpret everything the same way so debugging is easier.
        """
        self.permittivity = permittivity
        self.permeability = permeability
        self.thickness = thickness
        self.n_harmonics = n_harmonics
        self.is_vacuum = np.max(np.abs(self.permittivity)) == 1 and np.min(np.abs(self.permittivity)) == 1 and np.max(np.abs(self.permeability)) == 1 and np.min(np.abs(self.permeability)) == 1

    def layer_distribution_convolution_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        # Initialize convolution matrices
        size = 2 * self.n_harmonics + 1
        permittivity_convolution_matrix = np.zeros((size, size), dtype=np.cdouble)
        permeability_convolution_matrix = np.zeros((size, size), dtype=np.cdouble)

        # Compute Fourier representations with FFT
        permittivity_fourier = np.fft.fftshift(np.fft.fft(self.permittivity, axis=0))
        permeability_fourier = np.fft.fftshift(np.fft.fft(self.permeability, axis=0))

        # Extract relevant harmonics
        zero_harmonic = len(self.permittivity) // 2
        range_slice = slice(zero_harmonic - 2 * self.n_harmonics, zero_harmonic + 2 * self.n_harmonics + 1)
        permittivity_fourier = permittivity_fourier[range_slice] / len(permittivity_fourier)
        permeability_fourier = permeability_fourier[range_slice] / len(permeability_fourier)

        # Fill the convolution matrices using Toeplitz symmetry
        for x in range(size):
            for y in range(size):
                offset = y - x + 2 * self.n_harmonics
                permittivity_convolution_matrix[y, x] = permittivity_fourier[offset]
                permeability_convolution_matrix[y, x] = permeability_fourier[offset]

        return permittivity_convolution_matrix, permeability_convolution_matrix

class Solver:
    
    def zero_block(self) -> np.ndarray:
        return np.zeros(shape=(self.graph_harmonics, self.graph_harmonics))

    def id_block(self, scale: str = "single") -> np.ndarray:
        return np.eye(self.graph_harmonics) if scale == "single" else np.eye(2 * self.graph_harmonics)
    
    def pq_matrices(self, layer: Layer, kx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ky = self.kx_matrix(0)
        permittivity_convolution, permeability_convolution = layer.layer_distribution_convolution_matrices()
        p00 = kx @ np.linalg.inv(permittivity_convolution) @ ky
        p01 = permeability_convolution - kx @ np.linalg.inv(permittivity_convolution) @ kx
        p10 = ky @ np.linalg.inv(permittivity_convolution) @ ky - permeability_convolution
        p11 = -ky @ np.linalg.inv(permittivity_convolution) @ kx
        p = np.block([[p00, p01], [p10, p11]])

        q00 = kx @ np.linalg.inv(permeability_convolution) @ ky
        q01 = permittivity_convolution - kx @ np.linalg.inv(permeability_convolution) @ kx
        q10 = ky @ np.linalg.inv(permeability_convolution) @ ky - permittivity_convolution
        q11 = -ky @ np.linalg.inv(permeability_convolution) @ kx
        q = np.block([[q00, q01], [q10, q11]])
        return p, q
    
    def kx_matrix(self, k0: float) -> np.ndarray:
        kx = np.zeros(shape = (self.graph_harmonics))
        kx[self.n_harmonics] = k0
        kx += np.array([2 * np.pi / self.grating_period * n for n in range(-self.n_harmonics, self.n_harmonics + 1)])
        return np.diag(kx)
    
    def __init__(self, layer_stack: typing.List[Layer], grating_period: float, wavelength: float, n_harmonics: int = 0, theta: float = 0):
        self.layer_stack = layer_stack
        assert self.layer_stack[0].is_vacuum and self.layer_stack[-1].is_vacuum
        self.n_harmonics = n_harmonics
        self.grating_period = grating_period
        self.wavelength = wavelength
        self.theta = theta
        self.graph_harmonics = 2 * self.n_harmonics + 1
        self.kx0 = self.kx_matrix(np.sin(self.theta))

        vac_p, vac_q = self.pq_matrices(self.layer_stack[0], self.kx0)
        vac_lambda_w_sqr, vac_w = np.linalg.eig(vac_p @ vac_q)
        
        vac_lambda_w = np.sqrt(vac_lambda_w_sqr)
        
        vac_fwd_lambda_w = np.where(np.real(vac_lambda_w) < 0, -vac_lambda_w, vac_lambda_w)
        vac_bwd_lambda_w = np.where(np.real(vac_lambda_w) >= 0, -vac_lambda_w, vac_lambda_w)
        
        vac_trns_matrix = np.block([[vac_w, vac_w], [vac_q @ vac_w @ np.linalg.inv(np.diag(vac_bwd_lambda_w)) * -1j, vac_q @ vac_w @ np.linalg.inv(np.diag(vac_fwd_lambda_w)) * -1j]]) # NOTE: sign convention

        print("Vacuum transfer matrix:")
        ff.printdf(vac_trns_matrix)
        
        vac_trns_matrix2 = np.block([[vac_w, vac_w], [vac_q @ vac_w @ np.linalg.inv(np.diag(vac_bwd_lambda_w)) * 1j, vac_q @ vac_w @ np.linalg.inv(np.diag(vac_fwd_lambda_w)) * 1j]]) # NOTE: sign convention
        
        print(f'Inverse vacuum:')
        ff.printdf(vac_trns_matrix2)
        
        for i, layer in enumerate(self.layer_stack[1:]):
            p, q = self.pq_matrices(layer, self.kx0)
            
            lambda_w_sqr, w = np.linalg.eig(p @ q)
            print(p,q)
            print(w)
            lambda_w = np.sqrt(lambda_w_sqr)
            print(f'Lambda results (propagation):\n{lambda_w}')
            
            fwd_lambda_w = np.where(np.real(lambda_w) < 0, -lambda_w, lambda_w)
            fwd_prop_matrix = sp.linalg.expm(np.diag(fwd_lambda_w) * -layer.thickness * 2 * np.pi / self.wavelength)

            bwd_lambda_w = np.where(np.real(lambda_w) >= 0, -lambda_w, lambda_w)
            bwd_prop_matrix = sp.linalg.expm(np.diag(bwd_lambda_w) * layer.thickness * 2 * np.pi / self.wavelength) # NOTE: check with Declan since this restriction means that we deal with vertically isotropic media only -- this is true in our case, but it is not true in general...
            
            trns_matrix = np.block([[w, w], [q @ w @ np.linalg.inv(np.diag(bwd_lambda_w)) * -1j, q @ w @ np.linalg.inv(np.diag(fwd_lambda_w)) * -1j]])

            M1 = np.linalg.inv(trns_matrix) @ vac_trns_matrix
            # print('Layer transfer matrix:')
            # ff.printdf(trns_matrix)
            # print(f'First interface transfer matrix:')
            # ff.printdf(np.abs(M1))
            m11, m12, m21, m22 = ff.quar(M1) # this is not the answer since the transfer matrix works off of sx, sy, ux, and uy
            fref_coefs = -np.linalg.inv(m22) @ m21
            unit_power = np.zeros(2 * self.graph_harmonics)
            unit_power[self.n_harmonics] = 1
            # print('Unit power:')
            # print(np.linalg.inv(vac_w) @ unit_power)
            # print(vac_w)
            # print(np.linalg.inv(vac_w))
            
            # print(f'Scattering matrix shape:\n{M1.shape}')
            
            trns_prop_matrix = np.block([[w @ bwd_prop_matrix, w @ fwd_prop_matrix], [q @ w @ np.linalg.inv(np.diag(bwd_lambda_w)) * 1j @ bwd_prop_matrix, q @ w @ np.linalg.inv(np.diag(fwd_lambda_w)) * -1j @ fwd_prop_matrix]])
            M2 = np.linalg.inv(vac_trns_matrix2) @ trns_prop_matrix # NOTE: bug in this last one, it should not reverse the transfer relations from the first one
            print(vac_trns_matrix)
            print(trns_matrix)
            print(np.abs(trns_prop_matrix))
            print(vac_trns_matrix2)
            # print(f'Second layer propagation transfer matrix:')
            # ff.printdf(trns_prop_matrix)
            # print(np.linalg.det(trns_prop_matrix))
            m11, m12, m21, m22 = ff.quar(M2)
            fref_coefs = -np.linalg.inv(m22) @ m21
            print('Second half transfer matrix:')
            ff.printdf(np.abs(M2))

            M2 @= M1
            
            
            # print(f'Normal harmonics: {self.n_harmonics}, {3 * self.n_harmonics + 1}')
            # print(np.sum(np.abs(fref_coefs)))
            # for i in range(fref_coefs.shape[0]):
            #     print(i, fref_coefs[i,i])
            
            # ff.printdf(np.abs(M1))
            
            break
            # ...they should all be reflected/refracted with the same magnitudes...right?
            # We can do a test by computing the off-diagonal 4-blocks of the scattering matrix, if the magnetic fields and electric fields don't intersect (or maybe the smaller off-diagonal sub-blocks since we're looking at the intersection of polarities)

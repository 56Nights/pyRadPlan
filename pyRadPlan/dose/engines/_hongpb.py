from typing import cast
import array_api_compat
from ._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract
from ...machines import ParticlePencilBeamKernel
import math
"""
has_gpu=False
if has_gpu:
    import cupy as self.cp
"""

class ParticleHongPencilBeamEngine(ParticlePencilBeamEngineAbstract):
    # constants
    short_name = "HongPB"
    name = "Hong Particle Pencil-Beam"
    possible_radiation_modes = ["protons", "helium", "carbon", "VHEE"]
    """
    def wrapper(self,sigma_1,sigma_2,sigma_ini_sq,radial_dist_sq,weight,lateral):
     if self._has_gpu:
      def calc_particle_bixel_double_gpu(sigma_1,sigma_2,sigma_ini_sq,radial_dist_sq,weight,lateral):
        i=self.cuda.grid(1)
        if i >= sigma_1.size:  # Avoid accessing out-of-bounds memory
         return
        sigma_sq_narrow = sigma_1[i] ** 2 + sigma_ini_sq
        sigma_sq_broad = sigma_2[i] ** 2 + sigma_ini_sq

        l_narr = math.exp(-radial_dist_sq[i] / (2 * sigma_sq_narrow)) / (
        2 * math.pi * sigma_sq_narrow
    )
        l_bro = math.exp(-radial_dist_sq[i] / (2 * sigma_sq_broad)) / (
        2 * math.pi * sigma_sq_broad
    )
        lateral[i] = (1 - weight[i]) * l_narr + weight[i] * l_bro
      gpu_kernel = self.cuda.jit(calc_particle_bixel_double_gpu)

      return gpu_kernel
  """

    # private methods
    def _calc_particle_bixel(self, bixel):
        kernels = self._interpolate_kernels_in_depth(bixel)
        # print('plus censé passer par la')
        pb_kernel = cast(ParticlePencilBeamKernel, bixel["kernel"])

        xp = array_api_compat.array_namespace(bixel["radial_dist_sq"])
        # print('doit pas etre la')
        # Lateral Component
        if self.lateral_model == "single":
            # Compute lateral sigma
            sigma_sq = kernels["sigma"] ** 2 + bixel["sigma_ini_sq"]
            lateral = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq)) / (2 * xp.pi * sigma_sq)
        elif self.lateral_model == "double":
            # Compute lateral sigmas
            sigma_sq_narrow = kernels["sigma_1"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_broad = kernels["sigma_2"] ** 2 + bixel["sigma_ini_sq"]

            # Calculate lateral profile
            l_narr = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_narrow)) / (
                2 * xp.pi * sigma_sq_narrow
            )
            l_bro = xp.exp(-bixel["radial_dist_sq"] / (2 * sigma_sq_broad)) / (
                2 * xp.pi * sigma_sq_broad
            )
            lateral = (1 - kernels["weight"]) * l_narr + kernels["weight"] * l_bro
           
        elif self.lateral_model == "multi":
            sigma_sq = kernels["sigma_multi"] ** 2 + bixel["sigma_ini_sq"]
            lateral = xp.sum(
                (
                    xp.column_stack(
                        (1 - xp.sum(kernels["weight_multi"], axis=1), kernels["weight_multi"])
                    )
                    * xp.exp(-bixel["radial_dist_sq"][:, xp.newaxis] / (2 * sigma_sq))
                    / (2 * xp.pi * sigma_sq)
                ),
                axis=1,
            )
        elif self.lateral_model == "singleXY":
            # Extract squared distances in the two lateral axes
            x_sq = bixel["lat_dists"][:, 0] ** 2
            y_sq = bixel["lat_dists"][:, 1] ** 2

            sigma_sq_x = kernels["sigma_x"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_y = kernels["sigma_y"] ** 2 + bixel["sigma_ini_sq"]
            sigma_x = xp.sqrt(sigma_sq_x)
            sigma_y = xp.sqrt(sigma_sq_y)

            # Anisotropic 2D Gaussian
            lateral = xp.exp(-(x_sq / (2 * sigma_sq_x) + y_sq / (2 * sigma_sq_y))) / (
                2 * xp.pi * sigma_x * sigma_y
            )

        else:
            raise ValueError("Invalid Lateral Model")

        bixel["physical_dose"] = pb_kernel.lateral_cut_off.comp_fac * lateral * kernels["idd"]

        # Check if we have valid dose values
        if xp.any(xp.isnan(bixel["physical_dose"])) or xp.any(bixel["physical_dose"] < 0):
            raise ValueError("Error in particle dose calculation.")

        if self.calc_let:
            bixel["let_dose"] = bixel["physical_dose"] * kernels["let"]

        if self.calc_bio_dose:
            # TODO: correct / adaptive alpha / beta values given tissue indices
            bixel_alpha = kernels["alpha"][0]
            bixel_beta = kernels["beta"][0]

            # Multiple with dose
            bixel["alpha_dose"] = bixel["physical_dose"] * bixel_alpha
            bixel["sqrt_beta_dose"] = bixel["physical_dose"] * xp.sqrt(bixel_beta)
            
    # private methods
    def _calc_particle_bixel_gpu(self, bixel, bixel_gpu):
        kernels = self._interpolate_kernels_in_depth_gpu(bixel, bixel_gpu)
        pb_kernel = cast(ParticlePencilBeamKernel, bixel["kernel"])
        # print('ok passer par la')
        # Lateral Component
        if self.lateral_model == "single":
            # Compute lateral sigma
            sigma_sq = kernels["sigma"] ** 2 + bixel["sigma_ini_sq"]
            lateral = self.cp.exp(-bixel_gpu["radial_dist_sq"] / (2 * sigma_sq)) / (2 * self.cp.pi * sigma_sq)
        elif self.lateral_model == "double":
            # Compute lateral sigmas
            
            sigma_sq_narrow = kernels["sigma_1"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_broad = kernels["sigma_2"] ** 2 + bixel["sigma_ini_sq"]

            # Calculate lateral profile
            l_narr = self.cp.exp(-bixel_gpu["radial_dist_sq"] / (2 * sigma_sq_narrow)) / (
                2 * self.cp.pi * sigma_sq_narrow
            )
            l_bro = self.cp.exp(-bixel_gpu["radial_dist_sq"] / (2 * sigma_sq_broad)) / (
                2 * self.cp.pi * sigma_sq_broad
            )
            lateral = (1 - kernels["weight"]) * l_narr + kernels["weight"] * l_bro
            
            """
            s=kernels["weight"].shape[0]
            lateral=self.cp.empty(s,dtype=self.cp.float32)


            self.calc_particle_bixel_double_gpu[(s+128-1)//128,128](kernels["sigma_1"],kernels["sigma_2"],bixel["sigma_ini_sq"],bixel_gpu["radial_dist_sq"],kernels["weight"],lateral)
            """
        elif self.lateral_model == "multi":
            sigma_sq = kernels["sigma_multi"] ** 2 + bixel["sigma_ini_sq"]
            lateral = self.cp.sum(
                (
                    self.cp.column_stack(
                        (1 - self.cp.sum(kernels["weight_multi"], axis=1), kernels["weight_multi"])
                    )
                    * self.cp.exp(-bixel_gpu["radial_dist_sq"][:, self.cp.newaxis] / (2 * sigma_sq))
                    / (2 * self.cp.pi * sigma_sq)
                ),
                axis=1,
            )
        elif self.lateral_model == "singleXY":
            # Extract squared distances in the two lateral axes
            x_sq = bixel_gpu["lat_dists"][:, 0] ** 2
            y_sq = bixel_gpu["lat_dists"][:, 1] ** 2

            sigma_sq_x = kernels["sigma_x"] ** 2 + bixel["sigma_ini_sq"]
            sigma_sq_y = kernels["sigma_y"] ** 2 + bixel["sigma_ini_sq"]
            sigma_x = self.cp.sqrt(sigma_sq_x)
            sigma_y = self.cp.sqrt(sigma_sq_y)

            # Anisotropic 2D Gaussian
            lateral = self.cp.exp(-(x_sq / (2 * sigma_sq_x) + y_sq / (2 * sigma_sq_y))) / (
                2 * self.cp.pi * sigma_x * sigma_y
            )

        else:
            raise ValueError("Invalid Lateral Model")
        bixel["physical_dose"] = pb_kernel.lateral_cut_off.comp_fac * lateral * kernels["idd"]
        # Check if we have valid dose values
        if self.cp.any(self.cp.isnan(bixel["physical_dose"])) or self.cp.any(bixel["physical_dose"] < 0):
            raise ValueError("Error in particle dose calculation.")
        if self.calc_let:
            bixel["let_dose"] = bixel["physical_dose"] * kernels["let"]
        if self.calc_bio_dose:
            # TODO: correct / adaptive alpha / beta values given tissue indices
            bixel_alpha = kernels["alpha"][0]
            bixel_beta = kernels["beta"][0]
            # Multiple with dose
            bixel["alpha_dose"] = bixel["physical_dose"] * bixel_alpha
            bixel["sqrt_beta_dose"] = bixel["physical_dose"] * self.cp.sqrt(bixel_beta)
    
    
    @staticmethod
    def is_available(pln, machine):
        available, msg = [], []
        return available, msg

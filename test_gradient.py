#!/usr/bin/env python
"""Script description
"""

from __future__ import division
import numpy as np
import simulateAtmoGeneral as sa
import pickle
import time


def main():
    """entry point"""

    #
    # Load the MISR database.
    #
    with open('misr.pkl', 'rb') as f:
        misr = pickle.load(f)
    
    particles_list = misr.keys()
    particle = misr[particles_list[0]]
    aerosol_params = {
        "k_RGB": np.array(particle['k']) / np.max(np.array(particle['k'])),#* 10**-12,
        "w_RGB": particle['w'],
        "g_RGB": (particle['g']),
        "visibility": sa.VISIBILITY,
        "air_typical_h": 8,
        "aerosols_typical_h": 8,        
    }

    
    sky_params = sa.SKY_PARAMS

    X, H = np.meshgrid(
        np.arange(0, sky_params['width'], sky_params['dxh']),
        np.arange(0, sky_params['height'], sky_params['dxh'])[::-1]
        )

    #
    # Create the distributions of air and aerosols
    #
    ATMO_aerosols = np.exp(-H/aerosol_params["aerosols_typical_h"])
    ATMO_aerosols[:, :int(H.shape[1]/2)] = 0
    ATMO_air = np.exp(-H/aerosol_params["air_typical_h"])

    img = sa.calcRadianceGradient(ATMO_aerosols, ATMO_air, aerosol_params, sky_params)


if __name__=='__main__':
    main()

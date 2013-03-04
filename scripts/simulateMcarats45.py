"""
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from atmotomo import getResourcePath, Mcarats, SOLVER_F3D, loadGRADS
import amitibo
import os


def main(photon_num=1e10, solver=SOLVER_F3D):
    """Main doc"""
    
    camera_params = amitibo.attrClass(
        img_x=400,
        img_y=300,
        theta=0,
        phi=0
    )
    
    Atm_ext1d = []
    Atm_abs1d = []
    
    dx = 30
    dy = 30
    z_coords = np.array((0.000000E+00,  3.000000E+01,  6.000000E+01,  9.000000E+01,  1.200000E+02,  1.500000E+02,  1.800000E+02,  2.100000E+02,  2.300000E+02,  2.500000E+02,  2.600000E+02,  2.700000E+02,  2.800000E+02,  2.900000E+02,  3.000000E+02,  3.100000E+02,  3.200000E+02,  3.300000E+02,  3.400000E+02,  3.500000E+02,  3.600000E+02,  3.700000E+02,  3.800000E+02,  3.900000E+02,  4.000000E+02,  4.100000E+02,  4.200000E+02,  4.300000E+02,  4.400000E+02,  4.500000E+02,  4.600000E+02,  6.000000E+02,  1.000000E+03,  2.000000E+03,  3.000000E+03,  4.000000E+03,  6.000000E+03,  8.000000E+03,  1.000000E+04,  1.500000E+04,  2.000000E+04,  3.000000E+04))

    Atm_ext1d.append(np.array((2.45113E-05,  2.44561E-05,  2.43958E-05,  2.43346E-05,  2.42734E-05,  2.42117E-05,  2.41498E-05,  2.40980E-05,  2.40568E-05,  2.40256E-05,  2.40046E-05,  2.39831E-05,  2.39609E-05,  2.39383E-05,  2.39154E-05,  2.38922E-05,  2.38690E-05,  2.38456E-05,  2.38222E-05,  2.37988E-05,  2.37755E-05,  2.37520E-05,  2.37279E-05,  2.37015E-05,  2.36700E-05,  2.36279E-05,  2.35715E-05,  2.35065E-05,  2.34441E-05,  2.33933E-05,  2.31618E-05,  2.24001E-05,  2.07798E-05,  1.87793E-05,  1.69788E-05,  1.45587E-05,  1.17577E-05,  9.41149E-06,  6.07198E-06,  2.91507E-06,  9.61213E-07)))
    Atm_omg1d = np.ones_like(Atm_ext1d[0])
    Atm_apf1d = -Atm_omg1d
    Atm_abs1d.append(np.array((7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  7.94729E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  5.96045E-09,  5.96047E-09,  5.96047E-09,  8.51495E-09,  8.94071E-09,  8.94074E-09,  9.05995E-09,  9.35797E-09,  9.95407E-09,  1.11164E-08,  1.27258E-08,  2.29610E-08,  4.34327E-08,  4.78620E-08)))

    Atm_ext1d.append(np.array((1.09831E-05,  1.09583E-05,  1.09313E-05,  1.09039E-05,  1.08765E-05,  1.08488E-05,  1.08211E-05,  1.07979E-05,  1.07794E-05,  1.07655E-05,  1.07561E-05,  1.07464E-05,  1.07365E-05,  1.07263E-05,  1.07161E-05,  1.07057E-05,  1.06953E-05,  1.06848E-05,  1.06743E-05,  1.06639E-05,  1.06534E-05,  1.06429E-05,  1.06321E-05,  1.06202E-05,  1.06061E-05,  1.05873E-05,  1.05620E-05,  1.05329E-05,  1.05049E-05,  1.04821E-05,  1.03784E-05,  1.00371E-05,  9.31107E-06,  8.41467E-06,  7.60792E-06,  6.52350E-06,  5.26843E-06,  4.21713E-06,  2.72075E-06,  1.30620E-06,  4.30703E-07)))
    Atm_abs1d.append(np.array((2.38419E-07,  2.38419E-07,  2.38419E-07,  2.38419E-07,  2.38419E-07,  2.38419E-07,  2.38419E-07,  2.35439E-07,  2.35439E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.38418E-07,  2.38419E-07,  2.38419E-07,  2.36803E-07,  2.35450E-07,  2.34691E-07,  2.38805E-07,  2.46674E-07,  2.62210E-07,  2.92864E-07,  3.34434E-07,  6.02819E-07,  1.14080E-06,  1.25665E-06)))

    Atm_ext1d.append(np.array((4.98723E-06,  4.97600E-06,  4.96373E-06,  4.95129E-06,  4.93883E-06,  4.92627E-06,  4.91369E-06,  4.90315E-06,  4.89475E-06,  4.88841E-06,  4.88414E-06,  4.87976E-06,  4.87525E-06,  4.87065E-06,  4.86598E-06,  4.86127E-06,  4.85654E-06,  4.85179E-06,  4.84703E-06,  4.84227E-06,  4.83752E-06,  4.83275E-06,  4.82783E-06,  4.82246E-06,  4.81606E-06,  4.80749E-06,  4.79602E-06,  4.78279E-06,  4.77009E-06,  4.75975E-06,  4.71266E-06,  4.55768E-06,  4.22799E-06,  3.82096E-06,  3.45462E-06,  2.96221E-06,  2.39230E-06,  1.91492E-06,  1.23545E-06,  5.93121E-07,  1.95575E-07)))
    Atm_abs1d.append(np.array((1.31131E-07,  1.29144E-07,  1.31131E-07,  1.29144E-07,  1.31131E-07,  1.27157E-07,  1.27157E-07,  1.31130E-07,  1.30436E-07,  1.29911E-07,  1.29970E-07,  1.29990E-07,  1.30191E-07,  1.30102E-07,  1.30134E-07,  1.30180E-07,  1.30157E-07,  1.30059E-07,  1.29916E-07,  1.29869E-07,  1.29774E-07,  1.29761E-07,  1.29559E-07,  1.29140E-07,  1.27481E-07,  1.25349E-07,  1.24210E-07,  1.23527E-07,  1.23432E-07,  1.23466E-07,  1.28675E-07,  1.28004E-07,  1.26966E-07,  1.27801E-07,  1.31258E-07,  1.38898E-07,  1.54907E-07,  1.76819E-07,  3.18650E-07,  6.03011E-07,  6.64254E-07)))

    atm_file = [getResourcePath('les05_00%i.atm.ctl' % i)  for i in (45, 55, 67)]
    
    #
    # Create the results folder
    #
    results_path = amitibo.createResultFolder()

    #
    # Create the test
    #
    out_files = []
    for ch in range(3):
        mc = Mcarats(results_path, base_name='base%d'%ch)
        mc.setAtmosphereDims(shape=(60, 60, 41), dx=dx, dy=dy, z_coords=z_coords, iz3l=10, nz3=21, tmp_prof=0)
        mc.add1Ddistribution(
            ext1d=Atm_ext1d[ch],
            omg1d=Atm_omg1d,
            apf1d=Atm_apf1d,
            abs1d=Atm_abs1d[ch]
        )
        atm_dict = loadGRADS(atm_file[ch])        
        mc.add3Ddistribution(
            ext3d=atm_dict['extp3d'],
            omg3d=atm_dict['omgp3d'],
            apf3d=atm_dict['apfp3d'],
            abs3d=atm_dict['abst3d']
        )
        mc.addCamera(camera_params)
        mc.setSolarSource(theta=120.0, phi=180.0)
        
        #
        # Run the test
        #
        out_files.append(mc.run(photon_num=photon_num, solver=solver))
    
    img = np.dstack(Mcarats.calcImg(out_files)[::-1])
    
    #
    # Show the results
    #
    plt.imshow(img)
    plt.show()
    
    
if __name__ == '__main__':
    main()

    
    
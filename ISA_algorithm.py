#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:38:07 2020

@author: rbenda

Computation of ISA multipoles
"""

import numpy as np
import time
import math

import matplotlib
from matplotlib import pyplot


from extraction_QM_info import computes_nb_primitive_GTOs  
from extraction_QM_info import import_correspondence_pGTOs_atom_index
from extraction_QM_info import import_basis_set_exponents_primitive_GTOs
from extraction_QM_info import import_position_nuclei_associated_basis_function_pGTO
from extraction_QM_info import computes_nb_contracted_GTOs 
from extraction_QM_info import computes_angular_momentum_numbers_primitive_GTOs
from extraction_QM_info import import_nb_contracted_shells
from extraction_QM_info import import_nb_primitive_shells

from extraction_QM_info import compute_correspondence_index_contracted_GTOs_primitive_GTOs
from extraction_QM_info import compute_correspondence_index_primitive_GTOs_contracted_GTOs
from extraction_QM_info import import_contraction_scheme_matrix_pSHELLS_to_cSHELLS
from extraction_QM_info import correspondence_index_contracted_GTOs_contracted_shells
from extraction_QM_info import correspondence_index_primitive_GTOs_primitive_shells

from extraction_QM_info import import_density_matrix_contracted_GTOs
from extraction_QM_info import compute_density_matrix_coefficient_pGTOs
from extraction_QM_info import compute_scalar_product_two_primitive_GTOs
from extraction_QM_info import compute_scalar_product_two_contracted_GTOs
from extraction_QM_info import compute_normalization_coefficient_primitive_GTO

from auxiliary_functions import reads_Lebedev_grid

from ISA_auxiliary_functions import precompute_molecular_density_TEST_DIATOMIC_SUM_GAUSSIANS

from ISA_auxiliary_functions import  lgwt
#from LiH_dissociation_plots import reads_1_body_DM_QP


from ISA_diatomic_Newton import ISA_iterative_diatomic_Cances_BIS
from ISA_diatomic_Newton import compute_local_multipoles_ISA_DIATOMIC_NEWTON
from ISA_diatomic_Newton import compute_atomic_charge_DIATOMIC_NEWTON
from ISA_diatomic_Newton import compute_atomic_dipole_DIATOMIC_NEWTON
from ISA_diatomic_Newton import compute_atomic_quadrupole_xx_DIATOMIC_NEWTON

from GISA import  GISA_classic_algorithm
from GISA import  precompute_molecular_density_DIATOMIC
from GISA import  compute_mass_rho_MOLECULAR_GISA
from GISA import compute_atomic_charge_GISA
from GISA import compute_atomic_dipole_GISA
from GISA import compute_atomic_quadrupole_xx_GISA
from GISA import compute_atomic_quadrupole_zz_GISA

from GISA import  precompute_molecular_density_DIATOMIC_BIS
from GISA import compute_mass_rho_MOLECULAR_GISA_BIS
from GISA import compute_local_multipoles_GISA
from GISA import generate_random_guess_GISA

from LISA import  LISA_algorithm

from MBISA import MBISA_classic_algorithm
from MBISA import compute_local_multipoles_MBISA

from ISA_radial import ISA_radial_Lilestolen
from ISA_radial import ISA_radial_Lilestolen_DIATOMIC
from ISA_radial import compute_atomic_charge_ISA_radial_DIATOMIC
from ISA_radial import precompute_molecular_density_DIATOMIC_ISA

from cclib.io import ccopen


############################
#CONVERSION CONSTANTS :
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
# For test densities (non-dimensonal, not originating from a QM code output)
#conversion_bohr_angstrom=1.
############################

#Directory of QM output files :
QM_outputs_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/QM_output_files/'

ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'

Lebedev_grid_points_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/Lebedev_grid_points_dir/'

GL_points_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/Gauss_Legendre_points/'

ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'

ISA_outputs_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_outputs/'

One_body_DM_dir ='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/One_body_DM_dir/'

######################################################################################
######################################################################################
######################################################################################

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


###########################################################
#Importation of the data of the total molecular density (QM calculation)

## QM output file considered :

#Gaussian :
#filename = "H2O_SP_PBE0_cc-pVDZ_no_PCM.log"
#filename = "H2O_Opt_PBE0_6-31Gd_no_PCM.log"
#filename = "O2_Opt_PBE0_6-31Gd_z_axis.log"


#GAMESS :
#filename = "Benzene_Opt_PBE0_ACCT.out"


#########
# EXAMPLES FOR ISA - diatomic case

#filename = "LiH_HF_6-31G.out"
#filename = "LiH_HF_STO_3G.out"

#filename = "O2_RHF_6-31G.out"
#filename = "O2_RHF_6-31Gd.out"

filename = "CO_RHF_6-31G.out"
#filename = "CO_RHF_ACCD.out"

#filename = "N2_RHF_6-31Gd.out"

## BEWARE : ClO(-) has to be along z axis :
#filename = "CLO_ion_Opt_PBE0_6-31Gd_no_PCM.out"
#filename = "CLO_ion_Opt_PBE0_6-31Gd_no_PCM_z_axis.out"

######################
#To deal with the specificities of the basis treatment depending on the
#different quantum chemistry codes, we precise the code which
#has generated the input for the ISA calculation (output of a quantum code) 
#Example : 'SP' shells (in 6-31G(d) Pople's type basis sets) : count only for 
#one contracted shell in Gaussian / GAMESS
#while they count for 2 contracted shells (one S, one P contracted shell) in Psi4

#input_type='Gaussian'
input_type='GAMESS'
#input_type='Psi4'

# FCI/6-31G one-body density matrix file
# computed at each specific Li--H distance
# Donct forget to change "atomic_coordinates" accordingly (dissociation of the 2 atoms!)

#file_DM= "1-body_DM_LiH_Opt_RHF_6-31G.txt"
file_DM= "1-body_DM_LiH_1_50Ang_FCI_6-31G_PYSCF.txt"
#file_DM= "1-body_DM_LiH_1_70Ang_FCI_6-31G_pt2max_0.txt"

#file_DM= "1-body_DM_LiH_1_80Ang_FCI_6-31G_pt2_max_1e-9.txt"
#file_DM= "1-body_DM_LiH_FCI_6-31G_Opt_RHF_6-31G_pt2_max_1e-9.txt"

position_atomH_LiH=1.50
#position_atomH_LiH = 1.31975755 +0.31975755

# Density matrix type (regular or coming from Quantum Package):
# (case of LiH, calcultions FCI/6-31G)
DM_type="regular"
#DM_type="QP"

######################################################
# Permutation matrix to swith the one-body DM imported from QP or PySCF
# into the same order of indexation as GAMESS :

# => system and basis-set dependent !

# For LiH, 6-31G :
permut_DM_AO_order=np.identity(11)

for k in range(2,6):
    permut_DM_AO_order[k,k]=0

permut_DM_AO_order[2,3]=1
permut_DM_AO_order[3,4]=1
permut_DM_AO_order[4,5]=1
permut_DM_AO_order[5,2]=1

######################################################""

##################################
#Indicates whether with the given basis set, and with the QM code used for the calculation,
#d-shells are of Cartesian or of Spherical form :
boolean_cartesian_D_shells=True

#Examples : 
# *** In Gaussian program : 
#    - for 6-31G(d) Pople's type basis set 
#           => boolean_cartesian_D_shells = True
#    - for (aug)-cc-pVTZ Dunning's type basis set 
#           => boolean_cartesian_D_shells = False

# *** In Psi4 program : 
#Indicated by 'Spherical Harmonics' variable in Psi4 output.dat file 
#cf . "Spherical Harmonics    = FALSE" (e.g. for 6-31G(d) Pople's style basis set)
#cf.  "Spherical Harmonics    = TRUE" (e.g. for cc-pVDZ Dunning's type basis set)

# *** In GAMESS program : 
##Indicated by 'ispher' variable (in the $contrl section) in GAMESS
##(by default : ispher = -1 : cartesian basis functions / 
#ispher=1 : spherical basis functions -- 'pure' shells for p, d, f, etc. -type shells)
##################################

##################################
#Indicates whether with the given basis set, and with the QM code used for the calculation,
#f-shells are of Cartesian or of Spherical form :
  
# *** In Psi4 program : 
#- 6-31G(d) or Pople's type basis sets 
#=> CARTESIAN f-basis functions are used
#==> boolean_cartesian_F_shells=False
#(i.e. 10 basis functions of f-type, for each F contracted shell)

# *** In Gaussian program : 
#6-31G(d), Pople's type basis set or (aug)-cc-pVXZ Dunning's type basis sets
# => 7 basis functions of f-type (spherical form)
#=> boolean_cartesian_F_shells=False
boolean_cartesian_F_shells=True
##################################

##################################
#*** Psi4 exceptions for indexation orders of p-type GTOs or cartesian d-type GTOs :
#("exceptions" relatively to Gaussan or GAMESS primitive GTOs indexation orders)

#psi4_exception_P_GTOs_indexation_order=True if indexation order : p0, p+1, p-1 
#instead of px, py, pz
#psi4_exception_P_GTOs_indexation_order=False : usual order PX, PY, PZ (i.e. p+1, p-1, p0)
psi4_exception_P_GTOs_indexation_order=False

#psi4_exception_D_GTOs_indexation_order=True
# if indexation order : dxx, dxy, dxz, dyy, dyz, dzz
#instead of dxx, dyy, dzz, dxy, dxz, dyz (case of Gaussian / GAMESS)
#psi4_exception_D_GTOs_indexation_order=False : usual order for cartesian D shells 
#(in the case boolean_cartesian_D_shells=True) : XX, YY, ZZ, XY, XZ, YZ
#or usual order for spherical d GTOs (d0, d+1, d-1, d+2, d-2), in the case 
#boolean_cartesian_D_shells=False
psi4_exception_D_GTOs_indexation_order=False
##################################

    
parser = ccopen(QM_outputs_files_dir+filename)

QM_data = parser.parse()

########################################################################
#Useful informations are extracted from the QM output (stored in 'QM_data')

#Coordinates read from the last optimization step of the QM output file :
#Evaluation of 'atomcoords' attribute at the last index 
#(case of a geometry optimization => last [optimal] geometry)

if (DM_type=="regular"):
    atomic_coordinates=QM_data.atomcoords[len(QM_data.atomcoords)-1]

elif (DM_type=="QP"):
    atomic_coordinates=[[ 0.   ,       0.    ,     0.0],
                        [ 0.    ,      0.    ,     position_atomH_LiH]]

nb_atoms=len(atomic_coordinates)

print('nb_atoms')
print(nb_atoms)

print('atomic_coordinates :')
print(" ")
print(str(atomic_coordinates))
print(" ")
    
nb_primitive_GTOs=computes_nb_primitive_GTOs(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
total_nb_primitive_GTOs=nb_primitive_GTOs[1]
print('nb_primitive_GTOs')
print(nb_primitive_GTOs)

#Index of the atom to which each primitive GTO is associated  :
correspondence_basis_pGTOs_atom_index=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms)

##nb_primitive_GTOs=len(correspondence_basis_pGTOs_atom_index)

basis_set_exponents_primitive_GTOs=import_basis_set_exponents_primitive_GTOs(QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
print('basis_set_exponents_primitive_GTOs')
#print(basis_set_exponents_primitive_GTOs)

position_nuclei_associated_basis_function_pGTO=import_position_nuclei_associated_basis_function_pGTO(atomic_coordinates,nb_primitive_GTOs,nb_atoms,QM_data)
print('position_nuclei_associated_basis_function_pGTO')
#print(position_nuclei_associated_basis_function_pGTO)
      
angular_momentum_numbers_primitive_GTOs=computes_angular_momentum_numbers_primitive_GTOs(QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,input_type,psi4_exception_P_GTOs_indexation_order,psi4_exception_D_GTOs_indexation_order)
print('angular_momentum_numbers_primitive_GTOs')
#print(angular_momentum_numbers_primitive_GTOs)

nb_contracted_shells=import_nb_contracted_shells(QM_data,input_type)[1]

print('nb_contracted_shells')
print(nb_contracted_shells)

nb_primitive_shells=import_nb_primitive_shells(QM_data)[1]
print('nb_primitive_shells')
print(nb_primitive_shells)

nb_contracted_GTOs=computes_nb_contracted_GTOs(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
print('nb_contracted_GTOs')
print(nb_contracted_GTOs)

nb_primitive_GTOs=computes_nb_primitive_GTOs(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
print('nb_primitive_GTOs')
print(nb_primitive_GTOs)


logfile_ISA_name='ISA_'+str(filename.split('.')[0])+'_test.log'

#Opening the file of specified name, in 'writing' mode
logfile_ISA  = open(ISA_output_files_dir+logfile_ISA_name,"w")

if (DM_type=="regular"):
    density_matrix_coefficient_contracted_GTOs=import_density_matrix_contracted_GTOs(QM_data,logfile_ISA)
    
    print('density_matrix_coefficient_contracted_GTOs')
    #print(density_matrix_coefficient_contracted_GTOs)
    #print(' ')
    
elif (DM_type=="QP"):
    
     density_matrix_cGTOs_QP_indexation_order = reads_1_body_DM_QP(One_body_DM_dir,file_DM)
     
     print('density_matrix_cGTOs_QP_indexation_order')
     print(density_matrix_cGTOs_QP_indexation_order)
     
     density_matrix_coefficient_contracted_GTOs = [np.dot(permut_DM_AO_order,np.dot(density_matrix_cGTOs_QP_indexation_order,np.transpose(permut_DM_AO_order)))]
          
     #print('density_matrix_coefficient_contracted_GTOs')
     #print(density_matrix_coefficient_contracted_GTOs)
     #print(' ')

     
correspondence_index_contracted_GTOs_primitive_GTOs=compute_correspondence_index_contracted_GTOs_primitive_GTOs(QM_data,input_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells)

print('correspondence_index_contracted_GTOs_primitive_GTOs')
print(correspondence_index_contracted_GTOs_primitive_GTOs)

    
correspondence_index_primitive_GTOs_contracted_GTOs=compute_correspondence_index_primitive_GTOs_contracted_GTOs(correspondence_index_contracted_GTOs_primitive_GTOs,total_nb_primitive_GTOs)

#Information used by 'compute_scalar_product_two_primitive_GTOs()'
#For each primitive GTO : gives the index of the atom on which it is centered
correspondence_basis_pGTOs_atom_index=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms)
    
#Contraction coefficients given in e/bohr^(3/2) (atomic units) by Gaussian output file
contraction_coefficients_pSHELLS_to_cSHELLS=import_contraction_scheme_matrix_pSHELLS_to_cSHELLS(QM_data,input_type,nb_contracted_shells,input_type)

tab_correspondence_index_contracted_GTOs_contracted_shells=correspondence_index_contracted_GTOs_contracted_shells(QM_data,input_type,nb_contracted_GTOs,boolean_cartesian_D_shells,boolean_cartesian_F_shells)
    
tab_correspondence_index_primitive_GTOs_primitive_shells=correspondence_index_primitive_GTOs_primitive_shells(QM_data,input_type,total_nb_primitive_GTOs,boolean_cartesian_D_shells,boolean_cartesian_F_shells)

if (DM_type=="regular"):

    #Density matrix written in the basis of primitive GTOs (as that needed for DMA calculations) :
    density_matrix_coefficient_pGTOs=compute_density_matrix_coefficient_pGTOs(QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,density_matrix_coefficient_contracted_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,correspondence_index_contracted_GTOs_primitive_GTOs,correspondence_index_primitive_GTOs_contracted_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,total_nb_primitive_GTOs,nb_contracted_GTOs,nb_primitive_shells,nb_contracted_shells,logfile_ISA)

##########
##Check that Tr(S_{pGTOs}*D^{pGTOs}) = N =nb electrons
##########
scalar_products_primitive_GTOs=[[compute_normalization_coefficient_primitive_GTO(alpha,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
                                 *compute_normalization_coefficient_primitive_GTO(beta,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)*
                                 compute_scalar_product_two_primitive_GTOs(alpha,beta,boolean_cartesian_D_shells,boolean_cartesian_F_shells,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO) for beta in range(total_nb_primitive_GTOs)] for alpha in range(total_nb_primitive_GTOs)]


print('density_matrix_coefficient_pGTOs')
#print(density_matrix_coefficient_pGTOs)

#######################
#Check total charge = Tr(S*D) 
#(1) for the primitive GTOs basis :
if (DM_type=="regular"):
    
    Q_tot_bis=0
    for i in range(total_nb_primitive_GTOs): 
               
        #normalization_coeff_pGTO_i=extraction_QM_info.compute_normalization_coefficient_primitive_GTO(i,QM_data,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
    
        for j in range(i,total_nb_primitive_GTOs):
                    
            #normalization_coeff_pGTO_j=extraction_QM_info.compute_normalization_coefficient_primitive_GTO(j,QM_data,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
    
            Q_tot_bis+=density_matrix_coefficient_pGTOs[i][j]*scalar_products_primitive_GTOs[i][j]
        
    print('Total charge with D^{pGTOs} = Tr(S_p*D) = '+str(Q_tot_bis))
    print(" ")
    print('(S_p = primitive GTOs overlap matrix)')
    print(" ")
#######################    


#######################
#Check total charge = Tr(S*D) 
#(2) for the associated contracted GTOs basis :
    
scalar_products_contracted_GTOs=[[compute_scalar_product_two_contracted_GTOs(i,j,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO) for j in range(nb_contracted_GTOs)] for i in range(nb_contracted_GTOs)]

print('scalar_products_contracted_GTOs')
#print(scalar_products_contracted_GTOs)
print(' ')

#P=np.linalg.inv(permut_DM_AO_order)
#print(np.dot(P,np.dot(scalar_products_contracted_GTOs,np.transpose(P))))


#Tr(S*D) = sum_{mu,nu} (S_{mu,nu} * D_{mu,nu})
Q_tot=0
    
for mu in range(nb_contracted_GTOs):
        
    for nu in range(nb_contracted_GTOs):
            
        #Sum over alpha and beta density matrices :
        for sigma in range(len(density_matrix_coefficient_contracted_GTOs)):
                
            Q_tot+=density_matrix_coefficient_contracted_GTOs[sigma][mu][nu]*scalar_products_contracted_GTOs[mu][nu]
    
print("Total charge with D^{cGTOs} = Tr(S_c*D) = "+str(Q_tot))
print(" ")
print('(S_c = contracted GTOs overlap matrix)')
print(" ")
#######################


tab_normalization_constants_pGTOs=[compute_normalization_coefficient_primitive_GTO(i,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs) for i in range(total_nb_primitive_GTOs)]

print('tab_normalization_constants_pGTOs')
#print(tab_normalization_constants_pGTOs)


#Enf of importation of QM data / density and of some elementary checks
######################################################################################

# END of QM output importation and associated tests
######################################################################################
######################################################################################
######################################################################################


######################################################################
######################################################################
# PARAMETERS OF the ISA family of algorithms :
   
# Radial grid used both for the computations of the integral from 0 to infinity (R parameter = infinity)
# AND used for the convergence criterion computation (sum_a ||w_a^(m+1)-w_a^(m)||_{L2,GL})
Ng_radial=40

Ng_theta=25


##############
# Choice of the real molecular density or of a test density (non-dimensonal, not originating from a QM code output) :

#density_type = "TEST"
density_type = "Molecular"
##############

# Gauss-Legendre angular grid on theta computed once and for all :
print('Gauss-Legendre angular grid on theta :')
x_GL_0_1, w_GL_0_1 = lgwt(Ng_theta,0,1)

print('Gauss-Legendre radial grid on r :')
x_GL_0_R, w_GL_0_R = lgwt(Ng_radial,0,1)

# R (Ang) --> infinity (R >> |R2-R1| the distance between the two atoms)
# BEWARE : consistency with r_cutoff in interpolate_P1()
Rmax=5

# Scaled radial Gauss-Legendre grid :
x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]

    
tab_r_plot_FINAL = [i*0.001 for i in range(3000)]

###############

atomic_numbers = [QM_data.atomnos[0],QM_data.atomnos[1]]

print('atomic_numbers')
print(atomic_numbers)

# Fixed parameters :

# BEWARE TO convert everything to atomic units ( 1/Bohr**2 ) )!!

# GISA exponents for atom H given in the paper Verstraelen 2012 :
exponents_GISA_H_atom = [5.672, 1.505, 0.5308, 0.2204]

# GISA exponents for atom C given in the paper Verstraelen 2012 :
exponents_GISA_C_atom = [148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]

# GISA exponents for atom N given in the paper Verstraelen 2012 :
exponents_GISA_N_atom = [178.0,52.42, 19.87, 1.276, 0.6291, 0.2857]

# GISA exponents for atom O given in the paper Verstraelen 2012 :
exponents_GISA_O_atom = [220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]


################
# GISA / L-ISA coefficients :
    
# Initially : sum_a (sum_{k=1..Ka} c_(a,k)^(0)) = sum_a N_a^(0) = N (total number of electrons)
    
coeffs_c_init=[]

###################
# TO BE CHANGED FOR EACH MOLECULAR SYSTEM :
    
###################
# LiH
"""
tab_K_GISA =[6,4]

# Choice of coefficients c and exponents alpha for LiH : (default values of Li)

coeffs_c_init.append([0.5 for k in range(tab_K_GISA[0])])

coeffs_c_init.append([0.25 for k in range(tab_K_GISA[1])])

# Choice 1 : even-tempered exponents (WHEN NO EXPONENTS AVAILABLE  : e.g. for Li atom):
#exponents_GISA=[[0.5*2**i for i in range(tab_K_GISA[0])],
#                [2**i for i in range(tab_K_GISA[1])]]

# Choice 2 :
# Innermost shell : alpha_(a,k=0) = (2Z_a)/a0 where a0 = 0.529... (1 Bohr expressed in Angstroms)
# Outermost shell : alpha_(a,k=Ka-1) = 2/a0
# Shells in-between : alpha_(a,k) = (2*Z_a**(1-((i-1)/(Ka-1))))/a0 for i=1 .. Ka-2

a0 = conversion_bohr_angstrom

# These exponents pose problem in the radial quadrature if R_max is too large (e.g. in L-ISA)
exponents_GISA_Li=[]

for i in range(1,tab_K_GISA[0]+1):
    
    exponents_GISA_Li.append((2*atomic_numbers[0]**(1-((i-1)/(tab_K_GISA[0]-1))))/a0)
                
exponents_GISA=[exponents_GISA_Li,
                exponents_GISA_H_atom]

"""

###################
# O2
"""
# Number of shells for each O atom :
tab_K_GISA =[6,6]
 
# Choice of coefficients c and exponents alpha for O2 : (from published exponents values)
# No available guess values for c_(a,k)^(0):
 
coeffs_c_init.append([8./tab_K_GISA[0] for k in range(tab_K_GISA[0])])

coeffs_c_init.append([8./tab_K_GISA[1] for k in range(tab_K_GISA[1])])


exponents_GISA=[exponents_GISA_O_atom,
                exponents_GISA_O_atom]

"""
###################
# CO

########################
# Usual choice (same as in GISA paper) :

if (density_type=="Molecular"):
    
    tab_K_GISA =[6,6]
    
    coeffs_c_init.append([1. for k in range(tab_K_GISA[0])])
    
    coeffs_c_init.append([8./6. for k in range(tab_K_GISA[1])])

    exponents_GISA=[exponents_GISA_C_atom,
                    exponents_GISA_O_atom]
    
    
elif (density_type=="TEST"):
    
    #tab_K_GISA =[2,2]
    
    tab_K_GISA =[1,1]
    
    # Different possible initial guess for the atomic shell cofficients, for GISA :
    
    tab_alpha1 = [0.5]
    tab_alpha2 = [0.5]
    
    coeffs_c_init.append([1.0])  
    coeffs_c_init.append([1.0])
    
    
###################
# N2
"""
tab_K_GISA =[6,6]
 

# Choice of coefficients c and exponents alpha for O2 : (from published exponents values)
# No available guess values for c_(a,k)^(0):
 
coeffs_c_init.append([7./6. for k in range(tab_K_GISA[0])])

coeffs_c_init.append([7./6. for k in range(tab_K_GISA[1])])


exponents_GISA=[exponents_GISA_N_atom,
                exponents_GISA_N_atom]

"""
###################

###################
# ClO
"""
tab_K_GISA =[6,6]
 

# Choice of coefficients c and exponents alpha for O2 : (from published exponents values)
# No available guess values for c_(a,k)^(0):
 
    
coeffs_c_init.append([17.5/6. for k in range(tab_K_GISA[0])])

coeffs_c_init.append([8.5/6. for k in range(tab_K_GISA[1])])

exponents_GISA_Cl=[]

for i in range(1,tab_K_GISA[0]+1):
    
    exponents_GISA_Cl.append((2*atomic_numbers[0]**(1-((i-1)/(tab_K_GISA[0]-1))))/a0)
     
    
exponents_GISA=[exponents_GISA_Cl,
                exponents_GISA_O_atom]

"""
###################

print('exponents_GISA')
print(exponents_GISA)
print(' ')

print('coeffs_c_init')
#print(coeffs_c_init)
print(' ')
#print('N_1 = '+str(np.sum(coeffs_c_init[0]))+' ; N_2 = '+str(np.sum(coeffs_c_init[1])))
print(' ')

####################################################
####################################################

if (density_type=="Molecular"):
        
    print('GISA / MB-ISA diatomic : Precomputation molecular density on atomic grids (invariance along phi) ')
    # NB : useful also for the calculation of atomic local multipoles in the case of the ISA-DIATOMIC NEWTON algorithm
    
    
    
    # When rho(.) [molecular density] computed at the Gauss-Legendre quadrature radial x angular grid
    values_rho_around_atom_1, values_rho_around_atom_2 = precompute_molecular_density_DIATOMIC(Ng_radial,Ng_theta,Rmax,
                                                                                               x_GL_0_1, w_GL_0_1,
                                                                                               x_GL_0_R, w_GL_0_R,
                                                                                               atomic_coordinates,
                                                                                               QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                               density_matrix_coefficient_contracted_GTOs,
                                                                                               position_nuclei_associated_basis_function_pGTO,
                                                                                               nb_contracted_GTOs,
                                                                                               total_nb_primitive_GTOs,
                                                                                               correspondence_basis_pGTOs_atom_index,
                                                                                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                               correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                               contraction_coefficients_pSHELLS_to_cSHELLS)

    
    values_rho=[values_rho_around_atom_1,values_rho_around_atom_2]

    print('End precomputation rho (invariance along phi) : starting GISA algorithm')

####################################################

####################################################
# Test density : rho(r) = exp(-alpha_1*(r-R1)**2) + exp(-alpha_2*(r-R2)**2)

elif (density_type=="TEST"):
    
    print('Precomputation TEST molecular density on atomic grids')
    
    # Given in atomic nuits (Bohr**(-2))

    tab_alpha1 = [10.]
    tab_alpha2 = [10.]
    
    values_rho_around_atom_1, values_rho_around_atom_2 = precompute_molecular_density_TEST_DIATOMIC_SUM_GAUSSIANS(Ng_radial,Ng_theta,Rmax,
                                                                                                                  x_GL_0_1, w_GL_0_1,
                                                                                                                  x_GL_0_R, w_GL_0_R,
                                                                                                                  atomic_coordinates,
                                                                                                                  tab_alpha1,tab_alpha2)
    values_rho=[values_rho_around_atom_1,values_rho_around_atom_2]
    
    print('END precomputation TEST molecular density on atomic grids')

####################################################

##############################################################################################
##############################################################################################
##############################################################################################
# LAUNCHING DIFFERENT ISA algorithms :
    
    
##############################################################################################
# GISA classic iterative procedure (Step 1 and Step 2 [L2 norm minimization])
# and compute local atomic charges, dipoles and quadrupolar tensors AT CONVERGENCE
# using the obtained converged w_a(.) [here, obtained by the GISA method]
# Maximal radial integration limit : Rmax (previously defined)
output_file_GISA='GISA_output_'+str(filename.split('.')[0])+'_'+str(tab_K_GISA[0])+'_shells_Ng_radial_'+str(Ng_radial)+'_Ng_theta_'+str(Ng_theta)+'.txt'


#Opening the file of specified name, in 'writing' mode
logfile_GISA  = open(ISA_outputs_dir+output_file_GISA,"w")
    
logfile_GISA.write('----------------------------------------')
logfile_GISA.write('\n')
logfile_GISA.write('GISA multi-center decomposition of a molecular density')
logfile_GISA.write("\n")
logfile_GISA.write("\n")
logfile_GISA.write("Robert Benda, Eric Cancès, Virginie Ehrlacher, Benjamin Stamm 2021")
logfile_GISA.write("\n")
logfile_GISA.write('----------------------------------------')
logfile_GISA.write("\n")
logfile_GISA.write("\n")
    
coeffs_c_GISA_FINAL, w_1_times_r2_values_FINAL_GISA, w_2_times_r2_values_FINAL_GISA = GISA_classic_algorithm(tab_K_GISA,Ng_radial,
                                                                                                             Ng_theta,
                                                                                                             coeffs_c_init,exponents_GISA,
                                                                                                             Rmax,values_rho,
                                                                                                             x_GL_0_1, w_GL_0_1,
                                                                                                             x_GL_0_R, w_GL_0_R,
                                                                                                             atomic_coordinates,
                                                                                                             atomic_numbers,
                                                                                                             logfile_GISA,
                                                                                                             tab_r_plot_FINAL)
                                                                                                            
    
    
    
    
    
logfile_GISA.write('\n')
logfile_GISA.write("Number of shells per-atom :")
logfile_GISA.write('\n')
logfile_GISA.write(str(tab_K_GISA[0]))
logfile_GISA.write('\n')
logfile_GISA.write("Optimal GISA coefficients :")
logfile_GISA.write("\n")
logfile_GISA.write(str(coeffs_c_GISA_FINAL))
logfile_GISA.write('\n')
logfile_GISA.write("GISA exponents :")
logfile_GISA.write("\n")
logfile_GISA.write(str(exponents_GISA))
logfile_GISA.write("\n")
logfile_GISA.write("\n")
logfile_GISA.write(" Values of r--> np.log(4*pi*r²*w1(r)) :")
logfile_GISA.write("\n")
logfile_GISA.write(str(w_1_times_r2_values_FINAL_GISA))
logfile_GISA.write("\n")
logfile_GISA.write(" Values of r--> np.log(4*pi*r²*w2(r)) :")
logfile_GISA.write("\n")
logfile_GISA.write(str(w_2_times_r2_values_FINAL_GISA))
    
logfile_GISA.close()


# END GISA classic algorithm
##############

#####################################################################

##############################################################################################
# L-ISA algorithm (Kullback-Leibler entropy minimization)

output_file_LISA='LISA_output_'+str(filename.split('.')[0])+'_Ng_radial_'+str(Ng_radial)+'_Ng_theta_'+str(Ng_theta)+'.txt'

#Opening the file of specified name, in 'writing' mode
logfile_LISA  = open(ISA_outputs_dir+output_file_LISA,"w")

logfile_LISA.write('----------------------------------------')
logfile_LISA.write('\n')
logfile_LISA.write('L-ISA multi-center decomposition of a molecular density')
logfile_LISA.write("\n")
logfile_LISA.write("Robert Benda, Eric Cancès, Virginie Ehrlacher, Benjamin Stamm 2021")
logfile_LISA.write("\n")
logfile_LISA.write('----------------------------------------')
logfile_LISA.write("\n")
logfile_LISA.write("\n")
logfile_LISA.write("\n")


coeffs_c_LISA, coeffs_c_a_list_steps_LISA, w_1_values_FINAL, w_2_values_FINAL, w_1_times_r2_values_FINAL_LISA, w_2_times_r2_values_FINAL_LISA = LISA_algorithm(tab_K_GISA,Ng_radial,Ng_theta,
                                                                                                                                                               coeffs_c_init,exponents_GISA,
                                                                                                                                                               Rmax,values_rho,
                                                                                                                                                               x_GL_0_1, w_GL_0_1,
                                                                                                                                                               x_GL_0_R, w_GL_0_R,
                                                                                                                                                               atomic_coordinates,
                                                                                                                                                               atomic_numbers,
                                                                                                                                                               logfile_LISA,
                                                                                                                                                               tab_r_plot_FINAL)

logfile_LISA.write('\n')
logfile_LISA.write("Optimal L-ISA coefficients :")
logfile_LISA.write("\n")
logfile_LISA.write(str(coeffs_c_LISA[0][:]))
logfile_LISA.write("\n")
logfile_LISA.write(str(coeffs_c_LISA[1][:]))
logfile_LISA.write("\n")
logfile_LISA.write("\n")
logfile_LISA.write("\n")
logfile_LISA.write("\n")
logfile_LISA.write("L-ISA exponents :")
logfile_LISA.write("\n")
logfile_LISA.write(str(exponents_GISA))
logfile_LISA.write("\n")
logfile_LISA.write("\n")
logfile_LISA.write(" Values of r--> np.log(4*pi*r²*w1(r)) :")
logfile_LISA.write("\n")
logfile_LISA.write(str(w_1_times_r2_values_FINAL_LISA))
logfile_LISA.write("\n")
logfile_LISA.write(" Values of r--> np.log(4*pi*r²*w2(r)) :")
logfile_LISA.write("\n")
logfile_LISA.write(str(w_2_times_r2_values_FINAL_LISA))

logfile_LISA.close()

# END L-ISA algorithm
#####################################################################

##############################################################################################
# MB-ISA classic algorithm (Kullback-Leibler entropy minimization ? Iterative scheme
# inspiredby a Lagrangian formulation)

w_1_times_r2_values_FINAL_MBISA_ALL =[]

w_2_times_r2_values_FINAL_MBISA_ALL =[]

for index_nb_shells in range(1):
    
    tab_K_MBISA =[6+6*index_nb_shells,6+6*index_nb_shells]
     
    coeffs_c_init =[]
    
    coeffs_c_init.append([1. for k in range(tab_K_MBISA[0])])
    
    coeffs_c_init.append([8./6. for k in range(tab_K_MBISA[1])])

    exponents_MB_ISA_C_atom = []
    exponents_MB_ISA_O_atom = []
    
    a0 = conversion_bohr_angstrom
    
    for i in range(1,tab_K_MBISA[0]+1):
    
        exponents_MB_ISA_C_atom.append((2*atomic_numbers[0]**(1-((i-1)/(tab_K_MBISA[0]-1))))/a0)
 
    for i in range(1,tab_K_MBISA[1]+1):
    
        exponents_MB_ISA_O_atom.append((2*atomic_numbers[1]**(1-((i-1)/(tab_K_MBISA[1]-1))))/a0)
                              
    exponents_MBISA_init=[exponents_MB_ISA_C_atom,
                          exponents_MB_ISA_O_atom]

        
    
    output_file_MBISA='MBISA_output_'+str(tab_K_MBISA[0])+'_shells_'+str(filename.split('.')[0])+'_Ng_radial_'+str(Ng_radial)+'_Ng_theta_'+str(Ng_theta)+'.txt'
    
    #Opening the file of specified name, in 'writing' mode
    logfile_MBISA  = open(ISA_outputs_dir+output_file_MBISA,"w")
    
    logfile_MBISA.write('----------------------------------------')
    logfile_MBISA.write('\n')
    logfile_MBISA.write('MB-ISA multi-center decomposition of a molecular density')
    logfile_MBISA.write("\n")
    logfile_MBISA.write("Robert Benda, Eric Cancès, Virginie Ehrlacher, Benjamin Stamm 2021")
    logfile_MBISA.write("\n")
    logfile_MBISA.write('----------------------------------------')
    logfile_MBISA.write("\n")
    logfile_MBISA.write("\n")

    
    coeffs_c_MBISA_FINAL, exponents_MBISA_FINAL, w_1_times_r2_values_FINAL_MBISA, w_2_times_r2_values_FINAL_MBISA  = MBISA_classic_algorithm(tab_K_MBISA,coeffs_c_init,
                                                                                                                                             exponents_MBISA_init,
                                                                                                                                             values_rho,
                                                                                                                                             Ng_theta,
                                                                                                                                             x_GL_0_1, w_GL_0_1,
                                                                                                                                             Ng_radial,
                                                                                                                                             x_GL_0_R, w_GL_0_R,
                                                                                                                                             Rmax,
                                                                                                                                             atomic_numbers,
                                                                                                                                             atomic_coordinates,
                                                                                                                                             logfile_MBISA,
                                                                                                                                             tab_r_plot_FINAL)
    
    w_1_times_r2_values_FINAL_MBISA_ALL.append(w_1_times_r2_values_FINAL_MBISA)

    w_2_times_r2_values_FINAL_MBISA_ALL.append(w_2_times_r2_values_FINAL_MBISA)
    
    
    logfile_MBISA.write('\n')
    logfile_MBISA.write("Number of shells per-atom :")
    logfile_MBISA.write('\n')
    logfile_MBISA.write(str(tab_K_MBISA[0]))
    logfile_MBISA.write('\n')
    logfile_MBISA.write("Optimal MB-ISA coefficients :")
    logfile_MBISA.write("\n")
    logfile_MBISA.write(str(coeffs_c_MBISA_FINAL))
    logfile_MBISA.write("\n")
    logfile_MBISA.write("\n")
    logfile_MBISA.write("Optimal MB-ISA exponents :")
    logfile_MBISA.write("\n")
    logfile_MBISA.write(str(exponents_MBISA_FINAL))
    logfile_MBISA.write("\n")
    logfile_MBISA.write(" Values of r--> np.log(4*pi*r²*w1(r)) :")
    logfile_MBISA.write("\n")
    logfile_MBISA.write(str(w_1_times_r2_values_FINAL_MBISA))
    logfile_MBISA.write("\n")
    logfile_MBISA.write(" Values of r--> np.log(4*pi*r²*w2(r)) :")
    logfile_MBISA.write("\n")
    logfile_MBISA.write(str(w_2_times_r2_values_FINAL_MBISA))
    logfile_MBISA.close()

#############################################################################################


######################################################################
######################################################################




############################################################################
############################################################################
# ISA-DIATOMIC algorithm (Cancès) : Newton algorithm on y --> F1(r,y) and y --> F2(r,y)
# (NEW ALGORITHM BASED ON F1(r,y) and F2(r,y) functions)
# For checking purposes only (gives the same result as ISA-radial)
"""
# Discretization on r_1 and r_2 (r_(1,j), r_(2,j) for j=1..N)
N=Ng_radial

# Discretization on t :
Ng=Ng_theta

R1=atomic_coordinates[0][2]
R2=atomic_coordinates[1][2]

# BEWARE : for O2 : [ [ 0.          0.          0.59666961],
#                     [ 0.          0.         -0.59666961]]
# => R1 > R2 

R=abs(R2-R1)

print('Interatomic distance R = '+str(R))
print(' ')

# Radial discretization grid :
radial_grid_DIATOMIC_Newton = [x_GL_0_R_scaled,x_GL_0_R_scaled]

print('radial_grid_DIATOMIC_Newton')
print(radial_grid_DIATOMIC_Newton)
print(' ')

# Gauss-Legendre nodes (points) BETWEEN 0 and 1 :
x_GL, w_GL = lgwt(Ng,0,1)

print('Gauss Legendre nodes')
#rint(x_GL)
print('Gauss Legendre weights')
#print(w_GL)
print(' ')

# Number of GL quadrature points used to compute the spherical average of the 
# molecular density, in order to define the initial guess for w_1^(0)(r) = 0.5*<rho>_1(r) and w_2^(0)(r) = 0.5*<rho>_2(r)
Ng_avg = 30

x_GL_minus1_plus1,w_GL_minus1_plus1 = lgwt(Ng,-1,1)


atomic_weights_atom_1_current,atomic_weights_atom_2_current, w_1_times_r2_values_FINAL_ISA_Newton, w_2_times_r2_values_FINAL_ISA_Newton = ISA_iterative_diatomic_Cances_BIS(N,
                                                                                                                                                                            Ng,
                                                                                                                                                                            x_GL_minus1_plus1,w_GL_minus1_plus1,
                                                                                                                                                                            R,R1,R2,
                                                                                                                                                                            Ng_avg,
                                                                                                                                                                            radial_grid_DIATOMIC_Newton,
                                                                                                                                                                            atomic_numbers,
                                                                                                                                                                            QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                                                                                                            density_matrix_coefficient_pGTOs,
                                                                                                                                                                            density_matrix_coefficient_contracted_GTOs,
                                                                                                                                                                            position_nuclei_associated_basis_function_pGTO,
                                                                                                                                                                            nb_contracted_GTOs,
                                                                                                                                                                            total_nb_primitive_GTOs,
                                                                                                                                                                            correspondence_basis_pGTOs_atom_index,
                                                                                                                                                                            angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                                                                                                            correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                                                                                                            tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                                                                                                            tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                                                                                                            contraction_coefficients_pSHELLS_to_cSHELLS,
                                                                                                                                                                            atomic_coordinates,
                                                                                                                                                                            tab_r_plot_FINAL,
                                                                                                                                                                            density_type)
                                                                                                                                                                            


atomic_weights_atoms_DIATOMIC_NEWTON=[atomic_weights_atom_1_current,atomic_weights_atom_2_current]


print('Precomputes molecular density rho(.) at {Ra+r_i*(sin(theta_j,0,cos(theta_j)))}_{i,j}')

# 'radial_grid_DIATOMIC_Newton' = radial grid (r_(1,j))_j = (r_(2,j))_j on which w_1(.) and w_2(.) are computed

output_file_ISA_DIATOMIC_NEWTON='ISA_DIATOMIC_NEWTON_output_'+str(filename.split('.')[0])+'_Ng_radial_'+str(Ng_radial)+'_Ng_theta_'+str(Ng_theta)+'.txt'

#Opening the file of specified name, in 'writing' mode
logfile_ISA_DIATOMIC_NEWTON  = open(ISA_outputs_dir+output_file_ISA_DIATOMIC_NEWTON,"w")

logfile_ISA_DIATOMIC_NEWTON.write('----------------------------------------')
logfile_ISA_DIATOMIC_NEWTON.write('\n')
logfile_ISA_DIATOMIC_NEWTON.write('ISA-Diatomic Newton multi-center decomposition of a molecular density')
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write("Robert Benda, Eric Cancès, Virginie Ehrlacher, Benjamin Stamm 2021")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write('----------------------------------------')
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write("\n")


# N = Ng_radial
# Radial grid on which w_1(.) and w_2(.) are known :
#radial_grid_atoms = [ [Rmax*x_GL_0_R[k] for k in range(N)],[Rmax*x_GL_0_R[k] for k in range(N)]]


compute_local_multipoles_ISA_DIATOMIC_NEWTON(atomic_weights_atoms_DIATOMIC_NEWTON,
                                             values_rho,
                                             Ng_theta,
                                             x_GL_0_1, w_GL_0_1,
                                             Ng_radial,
                                             Rmax,
                                             x_GL_0_R, w_GL_0_R,
                                             radial_grid_DIATOMIC_Newton,
                                             atomic_coordinates,
                                             atomic_numbers,
                                             logfile_ISA_DIATOMIC_NEWTON)


logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(" Values of r--> w1(r) :")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(str(atomic_weights_atom_1_current))
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(" Values of r--> w2(r) :")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(str(atomic_weights_atom_2_current))
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(" Values of r--> np.log(4*pi*r²*w1(r)) :")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(str(w_1_times_r2_values_FINAL_ISA_Newton))
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(" Values of r--> np.log(4*pi*r²*w2(r)) :")
logfile_ISA_DIATOMIC_NEWTON.write("\n")
logfile_ISA_DIATOMIC_NEWTON.write(str(w_2_times_r2_values_FINAL_ISA_Newton))
logfile_ISA_DIATOMIC_NEWTON.write("\n")

logfile_ISA_DIATOMIC_NEWTON.close()

print('END ISA-DIATOMIC Newton')
print('--------------------------------')

"""
# END ISA-diatomic
############################################################################
############################################################################
############################################################################

######################################################################################
######################################################################################
######################################################################################
#ISA--radial grid-based algorithm (Lillestolen 2008 Chem. Phys. Letters) :




#################################################################################
#################################################################################
# Case of DIATOMIC molecule 
# => radial grid (Ng_radial) times Gauss-Legendre grid 
# on theta (Ng_theta) of spherical coordinates
# => Use  'values_rho' computed above by precompute_molecular_density_DIATOMIC() function
#################################################################################


#################################################################################
# Case of DIATOMIC molecule => radial grid (Ng_radial) times Gauss-Legendre grid 
# on theta (Ng_theta) of spherical coordinates

# The 2 atoms are along the z axis
grid_points_3D_atom_1 =[ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                            0,
                            atomic_coordinates[0][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
grid_points_3D_atom_2 =[ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                            0,
                            atomic_coordinates[1][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
  
discretization_pnts_atoms_DIATOMIC = [grid_points_3D_atom_1,grid_points_3D_atom_2]
#################################################################################



# If the radial discretization grid of ISA--radial 
# is set equal to the GL radial grid also used for the quadratures on "r"
radial_Log_Int_grid = [x_GL_0_R_scaled,x_GL_0_R_scaled]

##################
# ISA radial algorithm in the DIATOMIC case :
atomic_weights_current_ISA_radial, atomic_densities_current_ISA_radial, w_1_times_r2_values_FINAL_ISA_radial, w_2_times_r2_values_FINAL_ISA_radial = ISA_radial_Lilestolen_DIATOMIC(Ng_radial,x_GL_0_R,w_GL_0_R,Rmax,
                                                                                                                                                                                    Ng_theta,x_GL_0_1,w_GL_0_1,
                                                                                                                                                                                    discretization_pnts_atoms_DIATOMIC,
                                                                                                                                                                                    values_rho,
                                                                                                                                                                                    radial_Log_Int_grid,
                                                                                                                                                                                    QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                                                                                                                    density_matrix_coefficient_contracted_GTOs,
                                                                                                                                                                                    position_nuclei_associated_basis_function_pGTO,
                                                                                                                                                                                    nb_contracted_GTOs,
                                                                                                                                                                                    total_nb_primitive_GTOs,
                                                                                                                                                                                    correspondence_basis_pGTOs_atom_index,
                                                                                                                                                                                    angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                                                                                                                    correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                                                                                                                    tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                                                                                                                    tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                                                                                                                    contraction_coefficients_pSHELLS_to_cSHELLS,
                                                                                                                                                                                    atomic_coordinates,
                                                                                                                                                                                    atomic_numbers,
                                                                                                                                                                                    tab_r_plot_FINAL)


#####################################   
#####################################

output_file_ISA_radial_DIATOMIC='ISA_radial_DIATOMIC_'+str(filename.split('.')[0])+'_Ng_radial_'+str(Ng_radial)+'_Ng_theta_'+str(Ng_theta)+'.txt'

#Opening the file of specified name, in 'writing' mode
logfile_ISA_radial_DIATOMIC  = open(ISA_outputs_dir+output_file_ISA_radial_DIATOMIC,"w")

logfile_ISA_radial_DIATOMIC.write('----------------------------------------')
logfile_ISA_radial_DIATOMIC.write('\n')
logfile_ISA_radial_DIATOMIC.write('ISA-radial diatomic multi-center decomposition of a molecular density')
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write("Robert Benda, Eric Cancès, Virginie Ehrlacher, Benjamin Stamm 2021")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write('----------------------------------------')
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write("\n")


compute_local_multipoles_ISA_DIATOMIC_NEWTON(atomic_weights_current_ISA_radial,
                                             values_rho,
                                             Ng_theta,
                                             x_GL_0_1, w_GL_0_1,
                                             Ng_radial,
                                             Rmax,
                                             x_GL_0_R, w_GL_0_R,
                                             radial_Log_Int_grid,
                                             atomic_coordinates,
                                             atomic_numbers,
                                             logfile_ISA_radial_DIATOMIC)


logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(" Values of r--> w1(r) :")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(str(atomic_weights_current_ISA_radial[0]))
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(" Values of r--> w2(r) :")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(str(atomic_weights_current_ISA_radial[1]))
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(" Values of r--> np.log(4*pi*r²*w1(r)) :")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(str(w_1_times_r2_values_FINAL_ISA_radial))
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(" Values of r--> np.log(4*pi*r²*w2(r)) :")
logfile_ISA_radial_DIATOMIC.write("\n")
logfile_ISA_radial_DIATOMIC.write(str(w_2_times_r2_values_FINAL_ISA_radial))
logfile_ISA_radial_DIATOMIC.write("\n")

logfile_ISA_radial_DIATOMIC.close()

print('END ISA-DIATOMIC radial')
print('--------------------------------')

###############################################################
###############################################################
###############################################################
# Superposed plots of r --> 4*pi*r²*wa(r) for the 5 methods

matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

matplotlib.pyplot.xlabel('|r-R1| (Å)', fontsize=25)
matplotlib.pyplot.ylabel('4πr²w_1(r)', fontsize=25)

matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) in CO molecule (RHF/6-31G density)", fontsize=25)


matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_GISA_,label='GISA '+str(6+6*index_nb_shells)+' shells')
matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_GISA,'bo')

matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_LISA,label='LISA')
matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_LISA,'bo')


matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_MBISA_ALL[0],label='MBISA '+str(6+6*index_nb_shells)+' shells')   matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_MBISA_ALL[0],'bo')

#matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_ISA_Newton,label='ISA-Newton NEW',color='green')
#matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_ISA_Newton,'bo')

matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_ISA_radial,label='ISA-radial',color='green')
#matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL_ISA_radial,'bo')

matplotlib.pyplot.legend(loc='upper right', fontsize='x-large')

matplotlib.pyplot.ylim(-4, 3)

pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),linestyle='--',color='black')

matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_ALL_METHODS_w_1_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

matplotlib.pyplot.show()
 
matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

matplotlib.pyplot.xlabel('|r-R2| (Å)', fontsize=25)
matplotlib.pyplot.ylabel('4πr²w_2(r)', fontsize=25)

matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) in CO molecule (RHF/6-31G density)", fontsize=25)


matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL_GISA,label='GISA '+str(6+6*index_nb_shells)+' shells')
matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL_GISA,'bo')

matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL_LISA,label='LISA')
matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL_LISA,'bo')


matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL_MBISA_ALL[0],label='MBISA '+str(6+6*index_nb_shells)+' shells')
matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL_MBISA_ALL[0],'bo')

#matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL_ISA_Newton,label='ISA-Newton NEW',color='green')
#matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL_ISA_Newton,'bo')

matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL_ISA_radial,label='ISA-radial',color='green')
#matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL_ISA_radial,'bo')

matplotlib.pyplot.legend(loc='upper right', fontsize=25)

matplotlib.pyplot.ylim(-4, 3)

pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),linestyle='--',color='black')

matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_ALL_METHODS_w_2_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

matplotlib.pyplot.show()


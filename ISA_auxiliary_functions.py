#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:53:03 2020

@author: rbenda

Functions inherited from the DMA multipole code (2020)
and functions specific to the grid-based ISA algorithm implementation (Lillestolen 2008)
[radial grid times angular Lebedev grid]
"""

import numpy as np
import math
import matplotlib

from extraction_QM_info import compute_normalization_coefficient_primitive_GTO
#from extraction_QM_info import compute_normalization_coefficient_contracted_GTO

from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms
from auxiliary_functions import computes_coefficient_solid_harmonic_l_m_list_format



############################
#CONVERSION CONSTANTS :
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
# For test densities (non-dimensonal, not originating from a QM code output)
#conversion_bohr_angstrom=1.
############################

ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'
ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'
One_body_DM_dir ='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/One_body_DM_dir/'


###############################################################################
###############################################################################
###############################################################################
#Functions fo evaluate the density at any 3D point vec_r :

######################################################################################
#Evaluate the i^{th} primitive GTO at point r=(x,y,z) given in Angstroms :
def evaluate_primitive_GTO(i,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                           correspondence_basis_pGTOs_atom_index,
                           position_nuclei_associated_basis_function_pGTO,
                           angular_momentum_numbers_primitive_GTOs,
                           basis_set_exponents_primitive_GTOs):
    
    x,y,z = vec_r
    
    #Position of the atom where this i^{th} pGTO is centered :
    X_i, Y_i, Z_i =position_nuclei_associated_basis_function_pGTO[i]
           
    #Normalization constant of the i^{th} pGTO in Bohr^{-3/2-l_i} (atomic units)
    normalization_coeff_pGTO_i=compute_normalization_coefficient_primitive_GTO(i,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
            
    #The i^{th} primitive GTO expresses as :
                
    #|(\vec{r}-\vec{R_i})|^{l_i} * Y_{l_i}^{m_i}((\vec{r}-\vec{R_i})/|\vec{r}-\vec{R_i}|) * exp(-zeta_i * (\vec{r}-\vec{R_i})^2 )
            
    #zeta-exponent in atomic units (Bohrs**(-2))
    zeta_i=basis_set_exponents_primitive_GTOs[i]
            
    l_i=angular_momentum_numbers_primitive_GTOs[i][0]
    m_i=angular_momentum_numbers_primitive_GTOs[i][1]
            
    #Importation of 'coeff_harmonics_cartesian_i' i.e. weight coefficients, exponents, and information
    #to convert the real spherical harmonics above into a sum of homogeneous polynoms :
    if (boolean_cartesian_D_shells==True):
            
        if (boolean_cartesian_F_shells==False):
            if (l_i<=4):
                coeff_harmonics_cartesian_i=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells(l_i,m_i)
            else:
                coeff_harmonics_cartesian_i=computes_coefficient_solid_harmonic_l_m_list_format(l_i,m_i)
                
        elif (boolean_cartesian_F_shells==True):
            if (l_i<=4):
                coeff_harmonics_cartesian_i=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells(l_i,m_i)
            else:
                coeff_harmonics_cartesian_i=computes_coefficient_solid_harmonic_l_m_list_format(l_i,m_i)
               
    elif (boolean_cartesian_D_shells==False):
            
        if (boolean_cartesian_F_shells==False):
                
            if (l_i<=4):
                coeff_harmonics_cartesian_i=conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l_i,m_i)
            else:
                coeff_harmonics_cartesian_i=computes_coefficient_solid_harmonic_l_m_list_format(l_i,m_i)
                
        elif (boolean_cartesian_F_shells==True):
                    
            if (l_i<=4):
                coeff_harmonics_cartesian_i=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells(l_i,m_i)
            else:
                coeff_harmonics_cartesian_i=computes_coefficient_solid_harmonic_l_m_list_format(l_i,m_i)
                    
    #Number p(l_i,m_i) of homogeneous cartesian polynoms of degree l_i entering the definition of \chi_{i}(.)^{pGTO}:
    #(=1 in the case of Gaussian (cartesian) d-shells)
    p_i=coeff_harmonics_cartesian_i[0]
        
    #[[n_1(l_i,m_i),m_1(l_i,m_i),t_1(l_i,m_i)],..[n_{p(l_i,m_i)}(l_i,m_i),m_{p(l_i,m_i)}(l_i,m_i),t_{p(l_i,m_i)}(l_i,m_i)]] :
    #tab_indexes_i[k][0] = n_k(l_i,m_i)
    #tab_indexes_i[k][1] = m_k(l_i,m_i)
    #tab_indexes_i[k][2] = t_k(l_i,m_i)
    tab_indexes_i=coeff_harmonics_cartesian_i[1]
          
    #[w_1^(l_i,m_i),w_2^(l_i,m_i),..,w_{p(l_i,m_i)}(l_i,m_i)]
    #weights_spherical_harmonics_to_cartesian_alpha[k]= w_k^{l_i,m_i}
    weights_spherical_harmonics_to_cartesian_i=coeff_harmonics_cartesian_i[2]
            
    ################################################
    #Evaluation of the values of the i^{th} pGTO at the (grid) point \vec_r :
        
    #Value |(\vec{r}-\vec{R_i})|^{l_i} * Y_{l_i}^{m_i}((\vec{r}-\vec{R_i})/|\vec{r}-\vec{R_i}|) * exp(-zeta_i * (\vec{r}-\vec{R_i})^2 ) :
    #where \vec{R_i} is the atom center where this i^{th} pGTO is centered
    value_pGTO_i=0
            
    for k in range(p_i):
                
        #Conversion in Bohrs**(l_i)
        value_pGTO_i+=weights_spherical_harmonics_to_cartesian_i[k] * ((1/conversion_bohr_angstrom)**(l_i)) *(x-X_i)**tab_indexes_i[k][0] *(y-Y_i)**tab_indexes_i[k][1] *(z-Z_i)**tab_indexes_i[k][2]
       
    #Conversion to Bohrs
    vector_R_to_R_i=np.dot((1/conversion_bohr_angstrom),[x-X_i,y-Y_i,z-Z_i])
        
    #NEED TO NORMALIZE pGTOs  ??
    #normalization_coeff_pGTO_i*
    value_pGTO_i=normalization_coeff_pGTO_i*value_pGTO_i*math.exp(-zeta_i * np.linalg.norm(vector_R_to_R_i)**2)
    
    #Result in Bohrs**(l_i)
    return value_pGTO_i             
###################################################################   


###################################################################
#Evaluate the alpha^{th} contracted GTO at point r=(x,y,z) given in Angstroms :
#chi_{alpha}^{cGTO}(r) = N_{alpha}*sum_i ( c_i^{alpha} * chi_{i}^{pGTO}(r))
#where the sum runs over the primitive GTOs associated to this contracted GTOs
def evaluate_contracted_GTO(alpha,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                            correspondence_basis_pGTOs_atom_index,
                            position_nuclei_associated_basis_function_pGTO,
                            angular_momentum_numbers_primitive_GTOs,
                            basis_set_exponents_primitive_GTOs,
                            correspondence_index_contracted_GTOs_primitive_GTOs,
                            tab_correspondence_index_contracted_GTOs_contracted_shells,
                            tab_correspondence_index_primitive_GTOs_primitive_shells,
                            contraction_coefficients_pSHELLS_to_cSHELLS):
    
    value_cGTO=0
    
    #List of pGTOs indexes associated to the contracted GTO of index 'alpha' :
    list_pGTOs=correspondence_index_contracted_GTOs_primitive_GTOs[alpha]    
    
    for m in range(len(list_pGTOs)):
        
        index_pGTO=list_pGTOs[m]
        value_pGTO=evaluate_primitive_GTO(index_pGTO,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                          correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,
                                          angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
        
        #Index of the contracted (full) shell to which the contracted GTO (l_alpha,m_alpha) 
        #[and with no gaussian zeta exponent properly defined ; sum of several (as much as 
        #the contraction degree for the 'mother' contracted shell)] belongs
        contracted_shell_alpha=tab_correspondence_index_contracted_GTOs_contracted_shells[alpha]
            
        #Index of the contracted (full) shell to which the primitive GTO (l_i,m_i,zeta_i) 
        #[of well defined first and second angular momentum numbers l_i and m_i, as well as
        #well defined Gaussian zeta exponent zeta_i]
        primitive_shell_of_pGTO=tab_correspondence_index_primitive_GTOs_primitive_shells[index_pGTO]
       
        contraction_coeff=contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_alpha)][int(primitive_shell_of_pGTO)]
        
        value_cGTO+=contraction_coeff*value_pGTO
     
    #Normalization constant of the contracted GTO : should always be equal to 1 :
    #normalization_constant_cGTO=compute_normalization_coefficient_contracted_GTO(alpha,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
    #                                                                      tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO)
    
    #print('normalization_constant_cGTO')
    #print(normalization_constant_cGTO)
    
    value_cGTO=value_cGTO
    
    return value_cGTO
###################################################################


########################################################################
########################################################################
#Evaluation of the QM density n(r) [output of a quantum calculation] at any point 
# \vec_r =[x,y,z] :
#Result in atomic units (e/Bohr^3)
###########   
#METHOD 1 : using primitive GTOs 
# n(r) = \sum_{i,j>=i pGTOs} D^{pGTOs}_{i,j} * \chi_{i}^{pGTOs}(r) * \chi_{j}^{pGTOs}(r)
# QUESTION : are primitive GTOs normalized in the QM code ?
###########   
#METHOD 2 : using contracted GTOs 
# n(r) = \sum_{alpha,beta cGTOs} D^{cGTOs}_{alpha,beta} * \chi_{alpha}^{cGTOs}(r) * \chi_{beta}^{cGTOs}(r)
###########
"""
'density_matrix_coefficient_pGTOs' : not useful (density matrix in contracted GTOs basis is enough)
"""
def compute_density_grid_point(vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                               density_matrix_coefficient_contracted_GTOs,
                               position_nuclei_associated_basis_function_pGTO,
                               nb_contracted_GTOs,
                               total_nb_primitive_GTOs,
                               correspondence_basis_pGTOs_atom_index,
                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                               correspondence_index_contracted_GTOs_primitive_GTOs,
                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                               contraction_coefficients_pSHELLS_to_cSHELLS):
    
    x,y,z = vec_r
    
    #Density at point r : n(r) :
    density_point_r = 0
    
    
    ##########################################
    #METHOD 1 : using primitive GTOs 
    # n(r) = \sum_{i,j>=i pGTOs} D^{pGTOs}_{i,j} * \chi_{i}^{pGTOs}(r) * \chi_{j}^{pGTOs}(r)
    """
    for i in range(total_nb_primitive_GTOs): 
        
   
        #Value |(\vec{r}-\vec{R_i})|^{l_i} * Y_{l_i}^{m_i}((\vec{r}-\vec{R_i})/|\vec{r}-\vec{R_i}|) * exp(-zeta_i * (\vec{r}-\vec{R_i})^2 ) :
        #where \vec{R_i} is the atom center where this i^{th} pGTO is centered
        value_pGTO_i=evaluate_primitive_GTO(i,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)
        
        for j in range(i,total_nb_primitive_GTOs):
            
            value_pGTO_j=evaluate_primitive_GTO(j,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)

            #Evaluation of the density at point r (in e/Bohr³ i.e. atomic units):
            density_point_r+=density_matrix_coefficient_pGTOs[i][j]*value_pGTO_i*value_pGTO_j
            
            #if ( abs((density_matrix_coefficient_pGTOs[i][j]*value_pGTO_i*value_pGTO_j))>0.1):
            #    print('contribution pGTO i ='+str(i)+' pGTO j ='+str(j))
            #    print(density_matrix_coefficient_pGTOs[i][j]*value_pGTO_i*value_pGTO_j)
    """        
    #END of method 1
    ##########################
    
    ##########################
    #METHOD 2: using contracted GTOs 
    # n(r) = \sum_{alpha, beta cGTOs} D^{cGTOs}_{alpha,beta} * \chi_{alpha}^{pGTOs}(r) * \chi_{beta}^{pGTOs}(r)
    
    density_point_r=0 
    
    for alpha in range(nb_contracted_GTOs):
        
        value_cGTO_alpha=evaluate_contracted_GTO(alpha,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                 correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,
                                                 angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                 correspondence_index_contracted_GTOs_primitive_GTOs,tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                 tab_correspondence_index_primitive_GTOs_primitive_shells,contraction_coefficients_pSHELLS_to_cSHELLS)

        for beta in range(nb_contracted_GTOs):
            
            value_cGTO_beta=evaluate_contracted_GTO(beta,vec_r,QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                 correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,
                                                 angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                 correspondence_index_contracted_GTOs_primitive_GTOs,tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                 tab_correspondence_index_primitive_GTOs_primitive_shells,contraction_coefficients_pSHELLS_to_cSHELLS)

            #Evaluation of the density at point r (in e/Bohr³ i.e. atomic units):
            
            #Sum over alpha and beta density matrices :
            for sigma in range(len(density_matrix_coefficient_contracted_GTOs)):
                density_point_r+=density_matrix_coefficient_contracted_GTOs[sigma][alpha][beta]*value_cGTO_alpha*value_cGTO_beta
                
                #Checking the origin of the largest contributions to the density at a given point
                #(which basis functions / cGTOs play the most ?)
                #if ( abs((density_matrix_coefficient_contracted_GTOs[sigma][alpha][beta]*value_cGTO_alpha*value_cGTO_beta))>0.1):
                #    print('contribution cGTO alpha ='+str(alpha)+' cGTO beta ='+str(beta))
                #    print(density_matrix_coefficient_contracted_GTOs[sigma][alpha][beta]*value_cGTO_alpha*value_cGTO_beta)
                #    print('value_cGTO_'+str(alpha)+' = '+str(value_cGTO_alpha))
                #    print('value_cGTO_'+str(beta)+' = '+str(value_cGTO_beta))
                
    #Result in atomic units (e/Bohr^3)
    return density_point_r

######################################################################################     

###############################################################################
###############################################################################
###############################################################################

##############################################################################
#Functions to perform spherical averages 


#########################################################
#Computes the spherical average of the total QM density 
#(evaluated at any point thanks to the function 'compute_density_grid_point')
#around R_a (any point, e.g. the origin or an atom), at a distance r of R_a
#<rho>_{R_a}(r) = (1/(4*pi)* [int_{S^2}] rho(R_a+r.sigma) d sigma
#                  = (1/(4*pi)* sum_{i=1}^M ( w_i * rho(R_a+r.sigma_i) )
#where (w_i,sigma_i) are the M Lebedev points 
#density : function
def spherical_average_density(r,R_a,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                              position_nuclei_associated_basis_function_pGTO,
                              QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                              density_matrix_coefficient_contracted_GTOs,
                              nb_contracted_GTOs,
                              total_nb_primitive_GTOs,
                              correspondence_basis_pGTOs_atom_index,
                              angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                              correspondence_index_contracted_GTOs_primitive_GTOs,
                              tab_correspondence_index_primitive_GTOs_primitive_shells,
                              tab_correspondence_index_contracted_GTOs_contracted_shells,
                              contraction_coefficients_pSHELLS_to_cSHELLS):
    
    x_a, y_a, z_a = R_a
    Lbd_magical_number=len(Lebedev_grid_points)
        
    #Table of values of the weighted density  w_i * rho(R_a+r.sigma_i)
    #at the points {R_a+r.sigma_i}_{i=1..M}
    #where {sigma_i}_{i=1..M} are the M Lebedev points of the unit sphere
    #BEWARE : rho(R_a+r.sigma_i) given in atomic units (e/Bohr^3)
    density_values=[Lebedev_grid_points[i][3]*compute_density_grid_point([x_a+r*Lebedev_grid_points[i][0],y_a+r*Lebedev_grid_points[i][1],z_a+r*Lebedev_grid_points[i][2]],
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
                                                                         contraction_coefficients_pSHELLS_to_cSHELLS)
                        for i in range(Lbd_magical_number)]
    
    #Result in atomic units (e/Bohr^3)
    return (1/(4*math.pi))*np.sum(density_values)
#########################################################  

#########################################################
#Integrand of int_{0}^{+\infty} [ r**2 * <rho>_{R_a}(r)]
def evaluate_integrand_R_3_integration_density(r,R_a,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                               position_nuclei_associated_basis_function_pGTO,
                                               QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                               density_matrix_coefficient_contracted_GTOs,
                                               nb_contracted_GTOs,
                                               total_nb_primitive_GTOs,
                                               correspondence_basis_pGTOs_atom_index,
                                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                               correspondence_index_contracted_GTOs_primitive_GTOs,
                                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                                               contraction_coefficients_pSHELLS_to_cSHELLS):
    
    #Result of the spherical averaging => in atomic units (e/Bohr^3)
    density_spherical_average=spherical_average_density(r,R_a,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                                        position_nuclei_associated_basis_function_pGTO,
                                                        QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                        density_matrix_coefficient_contracted_GTOs,
                                                        nb_contracted_GTOs,
                                                        total_nb_primitive_GTOs,
                                                        correspondence_basis_pGTOs_atom_index,
                                                        angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                        correspondence_index_contracted_GTOs_primitive_GTOs,
                                                        tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                        tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                        contraction_coefficients_pSHELLS_to_cSHELLS)
        
    return 4*math.pi*(r/conversion_bohr_angstrom)**2 * density_spherical_average
#########################################################  


#########################################################
#Integrand of int_{0}^{+\infty} [ r**2 * <rho>_{R_a}(r)]
def evaluate_integrand_test(r,R_a,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                               position_nuclei_associated_basis_function_pGTO,
                                               QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                               density_matrix_coefficient_contracted_GTOs,
                                               nb_contracted_GTOs,
                                               total_nb_primitive_GTOs,
                                               correspondence_basis_pGTOs_atom_index,
                                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                               correspondence_index_contracted_GTOs_primitive_GTOs,
                                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                                               contraction_coefficients_pSHELLS_to_cSHELLS):
         
    return 1/(1+r**2)
#########################################################  


#########################################################
#Computes the spherical average of a function f, given by its value on 
#the Lebedev points of a sphere of center R_a and radius r
#R_a  : any point, e.g. the origin or an atom
#<f>_{R_a}(r) = (1/(4*pi)* [int_{S^2}] f(R_a+r.sigma) d sigma
#                  = (1/(4*pi)* sum_{i=1}^M ( w_i * f(R_a+r.sigma_i) )
#where (w_i,sigma_i) are the M Lebedev points 
#function_values_Lbdv_grid = [ f(R_a+r.sigma_i) = f([x_a+r*Lebedev_grid_points[i][0],y_a+r*Lebedev_grid_points[i][1],z_a+r*Lebedev_grid_points[i][2]]) for i in range(nb_Lbdv_points)]
def spherical_average_function(function_values_Lbdv_grid,r,R_a,Lebedev_grid_points):
    
    x_a, y_a, z_a = R_a
    
    Lbd_magical_number=len(Lebedev_grid_points)
    
    #Table of values of the weighted density  w_i * f(R_a+r.sigma_i)
    #at the points {R_a+r.sigma_i}_{i=1..M}
    #where {sigma_i}_{i=1..M} are the M Lebedev points of the unit sphere
    function_values=[Lebedev_grid_points[i][3]*function_values_Lbdv_grid[i] 
                    for i in range(Lbd_magical_number)]

    return (1/(4*math.pi))*np.sum(function_values)
######################################################### 


#########################################################
# Checking of total charge : integral of the QM density n(r) over whole space (R³)
# has to be equal to N (total number of electrons)
# Méthode des trapèzes pour le calcul de l'intégrale :
# sum_{n=0..(p-1)} [ (r_{n+1}-r_n)*(f(r_{n+1})-f(r_n))/2. ]
# where p is the number of radial discretization points
# where f(r)= 4*pi*r² * <rho>_{origin}(r) = r² * [int_{S^2}] rho(r.sigma+R_{origin}) d sigma
# where 'origin' is the chosen center for the radial integration and spherical average
def computes_total_charge(radial_Log_Int_grid,origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                          position_nuclei_associated_basis_function_pGTO,
                          QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                          density_matrix_coefficient_contracted_GTOs,
                          nb_contracted_GTOs,
                          total_nb_primitive_GTOs,
                          correspondence_basis_pGTOs_atom_index,
                          angular_momentum_numbers_primitive_GTOs,
                          basis_set_exponents_primitive_GTOs,
                          correspondence_index_contracted_GTOs_primitive_GTOs,
                          tab_correspondence_index_primitive_GTOs_primitive_shells,
                          tab_correspondence_index_contracted_GTOs_contracted_shells,
                          contraction_coefficients_pSHELLS_to_cSHELLS):

    p=len(radial_Log_Int_grid)
    
    total_integral_rho=0

    #Radial integration :
    #Value (r_{n+1}-r_n)*(f(r_{n+1})-f(r_n))/2.
    #where f(r)=4*pi*r² * <rho>_{0}(r) = r² * [int_{S^2}] rho(r.sigma) d sigma

    #OLD (false ?)
    ##integrand_radial_contributions=[4*math.pi*radial_Log_Int_grid[n]*0.5*(spherical_average_density(radial_Log_Int_grid[n],Lebedev_grid_points,density_matrix_coefficient_pGTOs,position_nuclei_associated_basis_function_pGTO)
    ##  +spherical_average_density(radial_Log_Int_grid[n+1],Lebedev_grid_points,density_matrix_coefficient_pGTOs,position_nuclei_associated_basis_function_pGTO))*(radial_Log_Int_grid[n+1]-radial_Log_Int_grid[n]) for n in range(p-1)]

    
    #Beware : conversion of the elements r_n = radial_Log_Int_grid[n]
    #in atomic units (Bohrs)
    # BEWARE : in atomic units ! 
    # 'evaluate_integrand_R_3_integration_density' gives a result in (e/Bohr^3)
    integrand_radial_contributions=[0.5*(evaluate_integrand_R_3_integration_density(radial_Log_Int_grid[n+1],origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                                                                    position_nuclei_associated_basis_function_pGTO,
                                                                                    QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                    density_matrix_coefficient_contracted_GTOs,
                                                                                    nb_contracted_GTOs,
                                                                                    total_nb_primitive_GTOs,
                                                                                    correspondence_basis_pGTOs_atom_index,
                                                                                    angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                    correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                    tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                    tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                    contraction_coefficients_pSHELLS_to_cSHELLS)
                                         +evaluate_integrand_R_3_integration_density(radial_Log_Int_grid[n],origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                                                                     position_nuclei_associated_basis_function_pGTO,
                                                                                     QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                     density_matrix_coefficient_contracted_GTOs,
                                                                                     nb_contracted_GTOs,
                                                                                     total_nb_primitive_GTOs,
                                                                                     correspondence_basis_pGTOs_atom_index,
                                                                                     angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                     correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                     tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                     tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                     contraction_coefficients_pSHELLS_to_cSHELLS) )*(radial_Log_Int_grid[n+1]-radial_Log_Int_grid[n])*(1/conversion_bohr_angstrom) for n in range(p-1)]


    total_integral_rho=np.sum(integrand_radial_contributions)

    
    #Result in atomic units (e : charge)
    return total_integral_rho
#########################################################################

#########################################################################
# Computes :
# the values [(r_{n+1}-r_n)*(f(r_{n+1})-f(r_n))/2.]_{n=1..p-1} where p is the number of radial discretization points
# where f(r)= 4*pi*r² * <rho>_{origin}(r) = r² * [int_{S^2}] rho(r.sigma) d sigma
# i.e. 2*pi*(r_{n+1}-r_n) * (r_n² * ((1/4*pi)* sum_{j=1..N^{Ldv}} w_j^{Ldv} rho(r_n.sigma_j) )
#                             + r_{n+1}² * ((1/4*pi)* sum_{j=1..N^{Ldv}} w_j^{Ldv} rho(r_{n+1}.sigma_j)))
# with 'origin' e.g. [0,0,0] or an other reference point chosen within the function
# And plots r_n --> (r_{n+1}-r_n)*(f(r_{n+1})-f(r_n))/2.
def computes_total_charge_integrand_contributions_TRAPEZES(radial_Log_Int_grid,origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                          position_nuclei_associated_basis_function_pGTO,
                          QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                          density_matrix_coefficient_contracted_GTOs,
                          nb_contracted_GTOs,
                          total_nb_primitive_GTOs,
                          correspondence_basis_pGTOs_atom_index,
                          angular_momentum_numbers_primitive_GTOs,
                          basis_set_exponents_primitive_GTOs,
                          correspondence_index_contracted_GTOs_primitive_GTOs,
                          tab_correspondence_index_primitive_GTOs_primitive_shells,
                          tab_correspondence_index_contracted_GTOs_contracted_shells,
                          contraction_coefficients_pSHELLS_to_cSHELLS):

    p=len(radial_Log_Int_grid)
    

    ### Table of integrand contributions at the discretization points r_i :
    # BEWARE : conversion of radial_Log_Int_grid[..] in atomic units (Bohrs) ! 
    # 'evaluate_integrand_R_3_integration_density' gives a result in (e/Bohr^3)
    # => multiplied by a length³ => has to be converted in Bohrs
    integrand_radial_contributions=[0.5*(evaluate_integrand_R_3_integration_density(radial_Log_Int_grid[n+1],origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                                                                    position_nuclei_associated_basis_function_pGTO,
                                                                                    QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                    density_matrix_coefficient_contracted_GTOs,
                                                                                    nb_contracted_GTOs,
                                                                                    total_nb_primitive_GTOs,
                                                                                    correspondence_basis_pGTOs_atom_index,
                                                                                    angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                    correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                    tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                    tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                    contraction_coefficients_pSHELLS_to_cSHELLS)*radial_Log_Int_grid[n+1]**2
                                         +evaluate_integrand_R_3_integration_density(radial_Log_Int_grid[n],origin,Lebedev_grid_points,density_matrix_coefficient_pGTOs,
                                                                                     position_nuclei_associated_basis_function_pGTO,
                                                                                     QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                     density_matrix_coefficient_contracted_GTOs,
                                                                                     nb_contracted_GTOs,
                                                                                     total_nb_primitive_GTOs,
                                                                                     correspondence_basis_pGTOs_atom_index,
                                                                                     angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                     correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                     tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                     tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                     contraction_coefficients_pSHELLS_to_cSHELLS)*radial_Log_Int_grid[n]**2 )*(radial_Log_Int_grid[n+1]-radial_Log_Int_grid[n])*(1/conversion_bohr_angstrom**3) for n in range(p-1)]

    matplotlib.pyplot.figure(figsize=(10, 8), dpi=400)
    
    matplotlib.pyplot.xlabel('r_i (Å)')

    matplotlib.pyplot.ylabel(' Integrand ')

    matplotlib.pyplot.title(" Discretized integrand 4*pi*r²*<rho>_{s,a}(r) (e) ")

    matplotlib.pyplot.plot(radial_Log_Int_grid[0:len(radial_Log_Int_grid)-1],integrand_radial_contributions,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"integrand_total_charge_values_origin_z"+str(int(origin[2]*10**3)/10**3)+"_"+str(len(Lebedev_grid_points))+"_Lbdv_points_Rlow_"+str(int(radial_Log_Int_grid[0]*10**3)/10**3)+"_Rhigh_"+str(int(radial_Log_Int_grid[len(radial_Log_Int_grid)-1]*10**3)/10**3)+"_p_"+str(p)+"_radial_grid_Log.png",bbox_inches='tight')
    
    return 0
#########################################################################

##############################################################################
##############################################################################
#Functions to interpolate an atomic weight function / atomic density of a given atom b
#at a grid point of another atom a.

######################################################################
#Computes the linearly interpolated value of w_b^{interp.}(r_(i,j)^a) at a grid point 
#r_(i,j)^a associated to atom a (argument 'point' of the function):
#####
#Linear interpolation :
#Let r_{k0} be such that r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
#w_b^{interp.}(r_(i,j)^a) = w_b(r_{k0}) + [(w_b(r_{k0+1})-w_b(r_{k0}))/(r_{k0+1}-r_{k0})] * (|| r_(i,j)^a - R_b ||-r_{k0})
#########
#atomic_weights_current_atom_b = atomic_weights_current[b] = { w_b(r_{n}) }_{n=1..p}
def interpolate_linearly_weight_other_atom_grid(b,point,atomic_weights_current_atom_b,
                                                atomic_coordinates,
                                                radial_Log_Int_grid):
    
    p=len(radial_Log_Int_grid)
    
    value_w_b_interpolated=0
    
    ##############
    #Search for index k0 such that r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
    vector=[(point[k]-atomic_coordinates[b][k]) for k in range(3)]
    
    #Boolean flag : will become True only if it exists k0 such that
    #r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
    #Will remain False if forall k, r_{k} <= || r_(i,j)^a - R_b ||
    boolean_in_between_spheres=False
    
    for n in range(p-1):
    
        #Always happens once and only once IF the radial grid goes far enough from each atomic center
        if (np.linalg.norm(vector) >= radial_Log_Int_grid[n]) and (np.linalg.norm(vector) < radial_Log_Int_grid[n+1]):
            
            boolean_in_between_spheres=True
            
            r_n=radial_Log_Int_grid[n]
            r_np1=radial_Log_Int_grid[n+1]
            
            value_w_b_interpolated=atomic_weights_current_atom_b[n]+((atomic_weights_current_atom_b[n+1]-atomic_weights_current_atom_b[n])/(r_np1-r_n))*(np.linalg.norm(vector)-r_n)
    
    if (boolean_in_between_spheres==False):
        
        value_w_b_interpolated=atomic_weights_current_atom_b[p-1]
        
    return value_w_b_interpolated
######################################################################

######################################################################
#Computes the linearly interpolated value of rho_b^{interp.}(r_(i,j)^a) at a grid point 
#r_(i,j)^a associated to atom a :
#####
#Linear interpolation :
#Let r_{k0} be such that r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
#Let sigma_h the Lebedev vector of the unit sphere minimizing :
#|| sigma_k - (r_(i,j)^a - R_b)/|| r_(i,j)^a - R_b || || over k
#Therefore :
#rho_b^{interp.}(r_(i,j)^a) = rho_b(r_{k0,h}^b) + [(rho_b(r_{k0+1,h}^b)-rho_b(r_{k0,h}^b))/(r_{k0+1}-r_{k0})] * (|| r_(i,j)^a - R_b ||-r_{k0})
#########
#atomic_densities_current_atom_b = atomic_densities_current[b] = { w_b(r_{n}) }_{n=1..p}
def interpolate_linearly_density_other_atom_grid(b,point,atomic_densities_current_atom_b,
                                                 radial_Log_Int_grid,
                                                 atomic_coordinates,
                                                 Lebedev_grid_points):
    
    p=len(radial_Log_Int_grid)
    
    nb_Ldv_points=len(Lebedev_grid_points)
     
    value_rho_b_interpolated=0
    
    ##############
    #Search for index k0 such that r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
    vector=[(point[k]-atomic_coordinates[b][k]) for k in range(3)]
    
    
                
    ##############
    #Search for index h such that sigma_h is the Lebedev point of the current 
    #Lebedev grid the closest to the unit vector (r_(i,j)^a - R_b)/ || r_(i,j)^a - R_b ||
    #pointing from the atom center b to the grid point of atom a r_(i,j)^a
    distance=1000000
            
    index_optimal_Lbdv_point=-1
            
    for h in range(nb_Ldv_points):
                
        vector_difference=[(Lebedev_grid_points[h][k]-(vector[k]/np.linalg.norm(vector))) for k in range(3)]
                
        if (np.linalg.norm(vector_difference)<distance):
            index_optimal_Lbdv_point=h
    
    #Boolean flag : will become True only if it exists k0 such that
    #r_{k0} <= || r_(i,j)^a - R_b || < r_{k0+1}
    #Will remain False if forall k, r_{k} <= || r_(i,j)^a - R_b ||
    boolean_in_between_spheres=False
    
    for n in range(p-1):
    
        #Always happens once and only once IF the radial grid goes far enough from each atomic center
        if (np.linalg.norm(vector) >= radial_Log_Int_grid[n]) and (np.linalg.norm(vector) < radial_Log_Int_grid[n+1]):
            
            boolean_in_between_spheres=True
            
            # Index corresponding to the grid point "R_b+r_n.sigma_{index_optimal_Lbdv_point}" of atom b
            #in the table 'discretization_pnts_atom_b'
            i1=n*nb_Ldv_points+index_optimal_Lbdv_point
            #=> rho_b(R_b+r_n.sigma_{index_optimal_Lbdv_point}) given by 'atomic_densities_current[b][i1]'
 
            # Index corresponding to the grid point "R_b+r_{n+1}.sigma_{index_optimal_Lbdv_point}" of atom b
            #in the table 'discretization_pnts_atom_b'
            i2=(n+1)*nb_Ldv_points+index_optimal_Lbdv_point
            
            r_n=radial_Log_Int_grid[n]
            r_np1=radial_Log_Int_grid[n+1]
            
            #No need to convert distances in Bohrs (as multiplication in numerator and denominator by distances)
            value_rho_b_interpolated=atomic_densities_current_atom_b[i1]+((atomic_densities_current_atom_b[i2]-atomic_densities_current_atom_b[i1])/(r_np1-r_n))*(np.linalg.norm(vector)-r_n)
     
    if (boolean_in_between_spheres==False):
        
        #Value of rho_b(.) at R_b+r_{high}.sigma_{index_optimal_Lbdv_point} with r_{high}= r_{p-1}
        value_rho_b_interpolated=atomic_densities_current_atom_b[(p-1)*nb_Ldv_points+index_optimal_Lbdv_point]
        
    return value_rho_b_interpolated
######################################################################


######################################################################
#'atomic_densities_current' : values of rho_a(.) on their respective grids, at the current iteration
def computes_constraint_total_charge(nb_atoms,molecular_density_atomic_grids,atomic_densities_current,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points):
    
    nb_Lebedev_points=len(Lebedev_grid_points)
    
    constraint_total_charge=0
    
    for a in range(nb_atoms):
        
        #tab_contrib_atom_a=[(molecular_density_atomic_grids[a][m]
        #                     -(atomic_densities_current[a][m]+np.sum([interpolate_linearly_density_other_atom_grid(b,discretization_pnts_atoms[a][m],atomic_densities_current[b],radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points) for b in range(nb_atoms) if b!=a] ))) for m in range(len(discretization_pnts_atoms[a]))]
        
        for m in range(len(discretization_pnts_atoms[a])):
            
            #Index 'i' of r_(i,j)^a = R_a+r_i*sigma_j = discretization_pnts_atoms[a][m]
            #(radial distance from point a of this discretization point)
            i=int(m/nb_Lebedev_points)
            
            #Index 'j' of r_(i,j)^a = R_a+r_i*sigma_j = discretization_pnts_atoms[a][m]
            #(index of the Lebedev point on the sphere centered on atom a of radius r_i, indexed m
            #in the list of grid points discretization_pnts_atoms[a][m])
            j=m-i*nb_Lebedev_points
            
            weight=0
            
            #Weight : w_k^{Ldv} * r_i^2*(r_(i+1)-r_i) if i <= (p-1)
            #Weigth taken equal to 0 by convention if i=p (i.e. no contributions
            #to the total charge deviation of the points r_(p,j)^a
            #far away (radius r_p) from atom a)
            if (i<=(len(radial_Log_Int_grid)-2)):
                #BEWARE units : radial_Log_Int_grid[.] : given in Angstroms
                weight=Lebedev_grid_points[j][3]* (1/conversion_bohr_angstrom)**3  * radial_Log_Int_grid[i]**2 * (radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i])
            
            #Sum of the atomic densities 'rho_a(.)' evaluated at these points :
            
            sum_rho_a=atomic_densities_current[a][m]+np.sum([interpolate_linearly_density_other_atom_grid(b,discretization_pnts_atoms[a][m],atomic_densities_current[b],radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points) for b in range(nb_atoms) if b!=a])
                                      
            constraint_total_charge+=weight*abs(molecular_density_atomic_grids[a][m]-sum_rho_a)                                                             
                                                                                                    
    #print('CONSTRAINT TOTAL CHARGE : contribution of discretization error grid atom '+str(a)+' : ')
    #print(constraint_total_charge)
        
    constraint_total_charge=(1/nb_atoms)*constraint_total_charge
    
    return constraint_total_charge
     
######################################################################   

######################################################################
# Computes the numerator of the constraint on the total charge (rho - sum_a rho_a = 0)
# with a TRAPEZE discretization
#'atomic_densities_current' : values of rho_a(.) on their respective grids, at the current iteration
def computes_constraint_total_charge_TRAPEZES(nb_atoms,molecular_density_atomic_grids,atomic_densities_current,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points):
    
    nb_Lebedev_points=len(Lebedev_grid_points)
    
    constraint_total_charge=0
    
    for a in range(nb_atoms):
        
        for i in range(len(radial_Log_Int_grid)-1):
            
            spherical_avg_i = 0

            spherical_avg_i_plus_1 = 0

            ###########
            for j in range(nb_Lebedev_points):
                
                m1= i*nb_Lebedev_points+j
                
                sum_rho_a=atomic_densities_current[a][m1]+np.sum([interpolate_linearly_density_other_atom_grid(b,discretization_pnts_atoms[a][m1],atomic_densities_current[b],radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points) for b in range(nb_atoms) if b!=a])
                
                # Atomic densities : in atomic units (e/Bohr³)
                spherical_avg_i += Lebedev_grid_points[j][3] * abs(molecular_density_atomic_grids[a][m1]-sum_rho_a) 
                
                m2= (i+1)*nb_Lebedev_points+j
                
                sum_rho_a=atomic_densities_current[a][m2]+np.sum([interpolate_linearly_density_other_atom_grid(b,discretization_pnts_atoms[a][m2],atomic_densities_current[b],radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points) for b in range(nb_atoms) if b!=a])
                
                spherical_avg_i_plus_1 +=Lebedev_grid_points[j][3] * abs(molecular_density_atomic_grids[a][m2]-sum_rho_a) 
            ###########
            
            # Conversion of r_i and r_(i+1) from Ang to Bohrs (atomic units)
            # for the multiplication with electronic densities
            constraint_total_charge +=   (1/conversion_bohr_angstrom)**3 * 0.5 * (radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i]) * (radial_Log_Int_grid[i+1]**2 * spherical_avg_i_plus_1 + radial_Log_Int_grid[i]**2 * spherical_avg_i)
            ##########
      
    constraint_total_charge=(1/nb_atoms)*constraint_total_charge
    
    return constraint_total_charge
     
######################################################################   

######################################################################
#Should approximate the total integral of rho(.) [molecular density] 
# i.e. the total number of electrons in the system (can be used as a check of
# correct level / degree of discretization)
# Returns : (1/N)* sum_a sum_{i=1}^{p} r_i² * (r_(i+1)-r_i) * sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) )
# where w_j^{Ldv} is the Lebedev weight associated to the point sigma_j of the unit sphere such that :
# r_(i,j)^a = R_a+r_i*sigma_j 
def computes_constraint_total_charge_normalization(nb_atoms,molecular_density_atomic_grids,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points):
    
    nb_Lebedev_points=len(Lebedev_grid_points)
    
    nb_radial_points = len(radial_Log_Int_grid)
    
    constraint_total_charge=0
    
    for a in range(nb_atoms):
        
        #OLD formulation :
        #tab_contrib_atom_a=[(molecular_density_atomic_grids[a][m]
        #                     -(atomic_densities_current[a][m]+np.sum([interpolate_linearly_density_other_atom_grid(b,discretization_pnts_atoms[a][m],atomic_densities_current[b],radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points) for b in range(nb_atoms) if b!=a] ))) for m in range(len(discretization_pnts_atoms[a]))]
        

        ##################
        # Method 3 ( avec méthode des trapèzes : somme de (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
        # Cf. function 'computes_total_charge_integrand_contributions_TRAPEZES()' 
        # and 'computes_total_charge_integrand_contributions()'
        
        for i in range(nb_radial_points-1):
            
            weight_i = (1/conversion_bohr_angstrom) * (radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i])
            
            # <rho>_a(r_i) = sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) ) :
            spherical_avg_i = 0
            
            # <rho>_a(r_(i+1)) = sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i+1,j)^a) ) :
            spherical_avg_i_plus_1 = 0     
            
            for j in range(nb_Lebedev_points):
                
                # Index of the point (R_a + r_i*sigma_j) at m^{th} position of discretization_pnts_atoms[a] :
                m1=i*nb_Lebedev_points + j
                
                spherical_avg_i+=Lebedev_grid_points[j][3] * abs(molecular_density_atomic_grids[a][m1]) 

                # Index of the point (R_a + r_(i+1)*sigma_j) at m^{th} position of discretization_pnts_atoms[a] :
                m2=(i+1)*nb_Lebedev_points + j
                
                spherical_avg_i_plus_1+=Lebedev_grid_points[j][3] * abs(molecular_density_atomic_grids[a][m2]) 
                        
            # Formule des trapèzes pour calculer l'intégrale :
            # (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
            # with <rho>_a(r_i) = (1/(4*pi) ) * [ int_{S²} (rho(R_a+r_i.sigma)) d_sigma]
            #                 = (1/(4*pi) ) * sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) )
            constraint_total_charge+= 0.5 * weight_i * (1/conversion_bohr_angstrom)**2 * (radial_Log_Int_grid[i+1]**2  * spherical_avg_i_plus_1 + radial_Log_Int_grid[i]**2 *spherical_avg_i)
        
        ##################
        
        print('Total charge after atom n°'+str(a))
        print('constraint_total_charge = '+str(constraint_total_charge))   
                             
    constraint_total_charge =  (1/nb_atoms) *  constraint_total_charge
    
    return constraint_total_charge
     
######################################################################   


######################################################################
# Should approximate the total integral of rho_a(.) [atomic density obtained
#at the end of the ISA procedure] 
# Returns :  sum_{i=1}^{p} r_i² * (r_(i+1)-r_i) * sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho_a(r_(i,j)^a) )
# where w_j^{Ldv} is the Lebedev weight associated to the point sigma_j of the unit sphere such that :
# r_(i,j)^a = R_a+r_i*sigma_j 
# 'atomic_density_atomic_grid' encodes the values of { rho_a(r_(i,j)^a) }_{i=1..p,j=1..N_{Ldv}}
def computes_constraint_total_charge_atom(nb_atoms,atomic_density_atomic_grid,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points):
    
    nb_Lebedev_points=len(Lebedev_grid_points)
    
    nb_radial_points = len(radial_Log_Int_grid)
    
    total_charge_atom_a=0

    ##################
    # Méthode des trapèzes : somme de (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
    # Cf. function 'computes_total_charge_integrand_contributions_TRAPEZES()' 
    # and 'computes_total_charge_integrand_contributions()'
        
    for i in range(nb_radial_points-1):
            
        weight_i = (1/conversion_bohr_angstrom) * (radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i])
            
        # <rho>_a(r_i) = sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) ) :
        spherical_avg_i = 0
            
        # <rho>_a(r_(i+1)) = sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i+1,j)^a) ) :
        spherical_avg_i_plus_1 = 0     
            
        for j in range(nb_Lebedev_points):
                
            # Index of the point (R_a + r_i*sigma_j) at m^{th} position of discretization_pnts_atoms[a] :
            m1=i*nb_Lebedev_points + j
                
            spherical_avg_i+=Lebedev_grid_points[j][3] * abs(atomic_density_atomic_grid[m1]) 

            # Index of the point (R_a + r_(i+1)*sigma_j) at m^{th} position of discretization_pnts_atoms[a] :
            m2=(i+1)*nb_Lebedev_points + j
                
            spherical_avg_i_plus_1+=Lebedev_grid_points[j][3] * abs(atomic_density_atomic_grid[m2]) 
                        
        # Formule des trapèzes pour calculer l'intégrale :
        # (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
        # with <rho>_a(r_i) = (1/(4*pi) ) * [ int_{S²} (rho(R_a+r_i.sigma)) d_sigma]
        #                 = (1/(4*pi) ) * sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) )
        total_charge_atom_a+= 0.5 * weight_i * (1/conversion_bohr_angstrom)**2 * (radial_Log_Int_grid[i+1]**2  * spherical_avg_i_plus_1 + radial_Log_Int_grid[i]**2 *spherical_avg_i)
        
        ##################
                                 
    return total_charge_atom_a
     
######################################################################   

######################################################################
# Should approximate the total integral of rho_a(.) [atomic density obtained
#at the end of the ISA procedure] in the DIATOMIC case with invariance around z axis
# Returns :  sum_{i=1}^{p} r_i² * (r_(i+1)-r_i) * 2*pi * sum_{j=1..N_{GL}} ( w_j^{GL} * rho_a( R_a+r_i*sin(theta_j) e_x + r_i*cos(theta_j)) )
# where w_j^{Ldv} is the Lebedev weight associated to the point sigma_j of the unit sphere such that :
#  R_a+r_i*sin(theta_j) e_x + r_i*cos(theta_j) e_z
# 'atomic_density_atomic_grid' encodes the values of { rho_a(R_a+r_i*sin(theta_j) e_x + r_i*cos(theta_j)) }_{i=1..p,j=1..N_{GL}}
def computes_constraint_total_charge_atom_DIATOMIC(a,
                                                   atomic_densities_ISA,
                                                   discretization_pnts_atoms,
                                                   radial_Log_Int_grid,
                                                   atomic_coordinates,
                                                   Ng_theta,
                                                   w_GL_0_1):
        
    nb_radial_points = len(radial_Log_Int_grid[a])
    
    total_charge_atom_a=0

    ##################
    # Méthode des trapèzes : somme de (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
    # Cf. function 'computes_total_charge_integrand_contributions_TRAPEZES()' 
    # and 'computes_total_charge_integrand_contributions()'
        
    for i in range(nb_radial_points-1):
            
        weight_i = (1/conversion_bohr_angstrom) * (radial_Log_Int_grid[a][i+1]-radial_Log_Int_grid[a][i])
            
        # <rho>_a(r_i) = 2*pi * sum_{j=1..N_{GL}} ( w_j^{GL} * rho(R_a+r_i*sin(theta_j) e_x + r_i*cos(theta_j) e_z ) ) 
        spherical_avg_i = 0
            
        # <rho>_a(r_(i+1)) = 2*pi * sum_{j=1..N_{GL}} ( w_j^{GL} * rho(R_a+r_(i+1)*sin(theta_j) e_x + r_(i+1)*cos(theta_j) e_z ) ) 
        spherical_avg_i_plus_1 = 0     
            
        for j in range(Ng_theta):
                
            spherical_avg_i+= w_GL_0_1[j] * atomic_densities_ISA[a][i][j]

            spherical_avg_i_plus_1+= w_GL_0_1[j] * atomic_densities_ISA[a][i+1][j]
                        
        # Formule des trapèzes pour calculer l'intégrale :
        # (r_(i+1)-r_i)*(f(r_(i+1)+f(r_i)))/2 avec f(r) = 4*pi*r²*<rho>_a(r) ):
        # with <rho>_a(r_i) = (1/(4*pi) ) * [ int_{S²} (rho(R_a+r_i.sigma)) d_sigma]
        #                 = (1/(4*pi) ) * sum_{j=1..N_{Ldv}} ( w_j^{Ldv} * rho(r_(i,j)^a) )
        total_charge_atom_a+= 0.5 * weight_i * (1/conversion_bohr_angstrom)**2 * (radial_Log_Int_grid[a][i+1]**2  * spherical_avg_i_plus_1 + radial_Log_Int_grid[a][i]**2 *spherical_avg_i)
        
        ##################
                                 
    return total_charge_atom_a
     
######################################################################   


######################################################################
#Computes the weighted L^2 norm of (1/N)* sum_a ||w_a^{(k+1)}(.)-w_a^{(k)}(.)||
# = (1/N)* sum_a sum_{i=1}^{p-1} (r_(i+1)-r_i)*  |w_a^{(k+1)}(r_i)-w_a^{(k)}(r_i)|**2
def compute_convergence_criterion(difference_weights_iter,radial_Log_Int_grid,nb_atoms):
    
    convergence_criterion=0
    
    for a in range(nb_atoms):
        
        #Points at the sphere of largest radii are ignored 
        #(density assumed to have decreased well enough to be negligible)
        for i in range(len(radial_Log_Int_grid)-1):
            
            convergence_criterion+=(radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i])*(difference_weights_iter[a][i])**2
    
    return (1/nb_atoms)*convergence_criterion
    
######################################################################

######################################################################
#Computes the weighted L^2 norm of (1/N)* sum_a ||w_a^{(k+1)}(.)-w_a^{(k)}(.)||
# USING the 'méthode des trapèzes'
# = (1/N)* sum_a sum_{i=1}^{p-1} [ (r_(i+1)-r_i) * 0.5 * (|w_a^{(k+1)}(r_i)-w_a^{(k)}(r_i)|**2+|w_a^{(k+1)}(r_(i+1))-w_a^{(k)}(r_(i+1))|**2) ]
def compute_convergence_criterion_TRAPEZES(difference_weights_iter,radial_Log_Int_grid,nb_atoms):
    
    convergence_criterion=0
    
    for a in range(nb_atoms):
        
        #Points at the sphere of largest radii are ignored 
        #(density assumed to have decreased well enough to be negligible)
        for i in range(len(radial_Log_Int_grid)-1):
            
            convergence_criterion+=(radial_Log_Int_grid[i+1]-radial_Log_Int_grid[i])*0.5*(difference_weights_iter[a][i]**2+difference_weights_iter[a][i+1]**2)
    
    return (1/nb_atoms)*convergence_criterion
    
######################################################################

######################################################################
#Computes the weighted L^2 norm of (1/N)* sum_a ||w_a^{(k+1)}(.)-w_a^{(k)}(.)||^2
# USING the 'méthode des trapèzes'
# = (1/N)* sum_a sum_{i=1}^{p-1} [ (r_(i+1)-r_i) * 0.5 * (|w_a^{(k+1)}(r_i)-w_a^{(k)}(r_i)|**2+|w_a^{(k+1)}(r_(i+1))-w_a^{(k)}(r_(i+1))|**2) ]
def compute_convergence_criterion_TRAPEZES_DIATOMIC(difference_weights_iter,radial_Log_Int_grid,nb_atoms):
    
    convergence_criterion=0
    
    for a in range(nb_atoms):
        
        #Points at the sphere of largest radii are ignored 
        #(density assumed to have decreased well enough to be negligible)
        for i in range(len(radial_Log_Int_grid[a])-1):
                            
            convergence_criterion+=(radial_Log_Int_grid[a][i+1]-radial_Log_Int_grid[a][i])*0.5*(difference_weights_iter[a][i]**2+difference_weights_iter[a][i+1]**2)
    
    return (1/nb_atoms)*convergence_criterion
    
######################################################################


######################################################################
"""
Convergence criterion for ISA-radial (ISA, traditionnal method of Lillestolen)
"""
#Computes the weighted L^2 norm of (1/N)* sum_a ||w_a^{(k+1)}(.)-w_a^{(k)}(.)||^2
# USING a Gauss-Legendre sum on R
# difference_weights_iter_GL_grid[a] = { |w_a^{(k+1)}(Rmax*x_i^{GL})-w_a^{(k)}(Rmax*x_i^{GL})| }_{i=1..Ng_radial}
# = (1/N)* sum_a sum_{i=1}^{N_GL} w_i^{GL} *  |w_a^{(k+1)}(Rmax*x_i^{GL})-w_a^{(k)}(Rmax*x_i^{GL})|²
#
# "w_a^{(k+1)}(Rmax*x_i^{GL})-w_a^{(k)}(Rmax*x_i^{GL})" can be computed by 
# interpolation (from the values at the discretization  grid to the values
# at the (finer) quadrature grid)
def compute_convergence_criterion_GL_DIATOMIC_ISA(difference_weights_iter_GL_grid,
                                                  x_GL_0_R,
                                                  w_GL_0_R,
                                                  Rmax,
                                                  nb_atoms):
    
    convergence_criterion=0
    
    Ng_radial = len(x_GL_0_R)

    for a in range(nb_atoms):
        
        convergence_criterion += np.sum( [ w_GL_0_R[i] * difference_weights_iter_GL_grid[a][i]**2 for i in range(Ng_radial)])
    
    return (1/nb_atoms)* math.sqrt(Rmax * convergence_criterion)
    
######################################################################




################################################################          
"""
Interpolates (linearly) a function defined only on N points on a point x in-between the grid.
- Atomic radial grid radial_grid_atom = [r0,r1,...,rN] where the function is known
- Point x (strictly) in-between two of these discretization points (not coinciding with one of them)
- tab= [f(r0),...,f(rN)] : values of a given function on this radial grid (ex : w_1(.) and w_2(.))
=> tab should be of same size as radial_grid_atom.
"""
def interpolate_P1(tab,radial_grid_atom,x):

    # Value such that f(r_cutoff)=0 by definition, where
    # f is the function encoded in tab (P1 representation, on a discrete set of points)
    r_cutoff=5
    
    # Find the index i0 such that
    # r_(i0) <= x < r_(i0+1)
    i0=-1
    
    for i in range(len(radial_grid_atom)-1):
        
        if ((radial_grid_atom[i]<=x) and (x < radial_grid_atom[i+1])):
            i0=i
            
    # If for all i, x >= r_i => P1 interpolation between the last value of f(rN) and 0=f(r_cutoff)
    if (i0==-1):
        index_max=len(radial_grid_atom)-1
        #print('BEEE i0=-1')
        #print(tab[index_max]+(x-radial_grid_atom[index_max]) * (-tab[index_max])/(r_cutoff-radial_grid_atom[index_max]))
        return tab[index_max]+(x-radial_grid_atom[index_max]) * (-tab[index_max])/(r_cutoff-radial_grid_atom[index_max])

    # Otherwise : P1 interpolation (affine par morceaux)
    # BEWARE : PROBLEM if f or all i, x < r_i (i.e. if x < r_0), for instance if r_0 > 0
    else:
        #print('BEEE i0 >0 ')
        #print(tab[i0]+(x-radial_grid_atom[i0]) * (tab[i0+1]-tab[i0])/(radial_grid_atom[i0+1]-radial_grid_atom[i0]))
        return tab[i0]+(x-radial_grid_atom[i0]) * (tab[i0+1]-tab[i0])/(radial_grid_atom[i0+1]-radial_grid_atom[i0])
################################################################


################################################################
def reads_Gauss_Legendre_points(GL_points_dir,GL_points_txt_filename):
    
    file_GL_points  = open(GL_points_dir+GL_points_txt_filename,"r")
        
    GL_points_readlines=file_GL_points.readlines()
    
    nb_points=len(GL_points_readlines)
    
    # Gauss-Legendre points and weights :
        

    x_w_GL =[]
    for i in range(nb_points):
        
        tab=GL_points_readlines[i].split(" ")
        
        sub_tab=[]
        
        for k in range(len(tab)):
            if (tab[k]!=''):
                sub_tab.append(tab[k])
                
        # w_GL[i] = float(sub_tab[1])
        # x_GL[i] = float(sub_tab[2])
        
        x_w_GL.append([float(sub_tab[2]),float(sub_tab[1])])
        
      
    # Sort x_GL table :
    x_w_GL.sort(key=lambda tab: tab[0])
    
    
    x_GL=[0 for i in range(nb_points)]
    
    w_GL=[0 for i in range(nb_points)]
    
    for i in range(nb_points):
        x_GL[i] = x_w_GL[i][0]
        w_GL[i] = x_w_GL[i][1]

    return x_GL,w_GL
################################################################




######################################################################
"""
"Simple script which produces the Legendre-Gauss weights and nodes 
for computing the definite integral of a continuous function 
on some interval [a,b]."
Legendre-Gauss weights (translated in Python from Matlab, given by Ben Stamm)
 - N = truncation order (number of integration points on [a,b] -- see line below)
 - [a,b] : intervalon which to inverval a given one-variable function f(.)
 
Q : works only to integrate POSITIVE functions ?
% lgwt.m (Matlab version)
%
% This script is for computing definite integrals using Legendre-Gauss 
% Quadrature. Computes the Legendre-Gauss nodes and weights on an interval
% [a,b] with truncation order N
%
% Suppose you have a continuous function f(x) which is defined on [a,b]
% which you can evaluate at any x in [a,b]. Simply evaluate it at all of
% the values contained in the x vector to obtain a vector f. Then compute
% the definite integral using sum(f.*w);
%
% Written by Greg von Winckel - 02/25/2004
"""
def lgwt(N,a,b):

    N=N-1;
    N1=N+1; N2=N+2;

    xu=np.linspace(-1,1,N1)

    # Initial guess
    y=[(math.cos((2*k+1)*math.pi/(2*N+2))+(0.27/N1)*math.sin(math.pi*xu[k]*N/N2)) for k in range(N1)]

    # Legendre-Gauss Vandermonde Matrix (LGVM)
    L=np.zeros([N1,N2])
    
    # Derivative of LGVM : UNUSED (NOT USEFUL)
    ## Lp=np.zeros([N1,N2])

    # Compute the zeros of the N+1 Legendre Polynomial
    # using the recursion relation and the Newton-Raphson method

    y0=2
    
    epsilon=1e-10
    
    max_test = np.max([abs(y[u]-y0) for u in range(len(y))])
    
    ####################
    # Iterate until new points are uniformly within epsilon of old points
    while ((max_test>epsilon)==True):
        
        L[:,0]=1
        #Lp[:,0]=0
    
        for u in range(N1):
            L[u,1]=y[u] 
            
        for k in range(1,N1):
            # BEWARE : décalage (k-->k+1) par rapport au code Matlab qui commençait à k=2 !
            L[:,k+1] = [((2*(k+1)-1)*y[u]*L[u,k]-k*L[u,k-1])/(k+1) for u in range(N1)]
            
            
        #Lp=(N2)*( L[:,N1]-y*L[:,N2] )./(1-y**2);  
        
        # Décalage des indices Python / Matlab (de 1)
        Lp_line= [ N2*(L[u,N1-1]-y[u]*L[u,N2-1])/(1-y[u]**2) for u in range(N1) ]
        #Lp = [ [ N2*(L[u,v]-y[u]*L[u,v])/(1-y[u]**2) for v in range(N2) ] for u in range(N1)]
        
        # BEWARE : y0 becomes a vector !
        y0=y
        
        y=[ (y0[u]-L[u,N2-1]/Lp_line[u]) for u in range(N1)]
        
        max_test = np.max([abs(y[u]-y0[u]) for u in range(N1)])
    
    ####################
    
    # Linear map from[-1,1] to [a,b]
    x=[(a*(1-y[u])+b*(1+y[u]))/2. for u in range(N1)]
    # x = flipud(x) [Matlab syntax]

    x= [x[N1-1-u] for u in range(N1)]
    
    # Compute the weights
    w=[(b-a)/((1-y[u]**2)*Lp_line[u]**2)*(N2/N1)**2 for u in range(N1) ]
    #w = flipud(w) [Matlab syntax] 
    w = [w[N1-1-u] for u in range(N1)]

    return [x,w]
######################################################################


######################################################################
"""
Comuputes the integral over the sphere S²(R) of radius R of a
function f(x,y,z) :
    int_{S²(R)} f(x,y,z) dx dy dz 
    = 2*pi*R²*int_{0..pi} f(r*sin(theta),0,r*cos(theta)) dtheta
    
    using the invariance along z axis :
        f(r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta))
        = f(r*sin(theta),0,r*cos(theta)) [taking phi=0]
        for all phi=0..2pi
    
    The : uses N (parameter of the method) Gauss-Legendre points (and weights) over [0,pi]
    to compute the 1D integral int_{0..pi} (f(r*sin(theta),0,r*cos(theta)) dtheta)
        
=> Can be used to compute spherical averages in the ISA method.
Parameters :
    - R : sphere radius on which the integration is performed
    - N : number of Gauss-Legendre points (and weights) for 1D integration on theta
    - values_f = [f(grid_x[u],0,grid_z[u])  for u in range(N)] (precomputed)
"""
def computes_integral_sphere_diatomic(R,N,values_f):

    # Gauss-Legende nodes and weights :
    [theta,w] = lgwt(N,0,math.pi)

    #  Poids * jacobienne (r² sin(theta) dtheta dphi)
    # Invariance selon l'angle phi => facteur 2*pi
    weights = [math.sin(theta[u])*w[u] for u in range(N)]

    integral = 2*math.pi * R**2 *np.sum([weights[u]*values_f[u] for u in range(N)])

    return integral
######################################################################


#######################
# Test of the 'lgwt' Legendre-Gauss nodes / weights routine :
"""
# N: number of integration points
N=5

[theta,w] = lgwt(N,0,math.pi)

print('Gauss Legendre nodes (integration points)')
print(theta)
print('Gauss Legendre weights (integration points)')
print(w)
print('np.sum(w) = '+str(np.sum(w)))

# Intégration d'une fonction à symmétrie autour de l'axe z
# (ex : densité électronique d'une molécule diatomique
# => tout plan contenant z est plan de symmétrie)

# Intégration sur une sphère de rayon R
R = 1

#  Points on unit sphere on the meridian we discussed
grid_x = [R*math.sin(theta[u]) for u in range(N)]
grid_z = [R*math.cos(theta[u]) for u in range(N)]

#  Poids * jacobienne (r² sin(theta) dtheta dphi)
# Invariance selon l'angle phi => facteur 2*pi
weights = [2*math.pi*R**2*math.sin(theta[u])*w[u] for u in range(N)]

# Test functions
# values_f = [f(grid_x[u],0,grid_z[u])  for u in range(N)]
values_f = [math.cos(theta[u])  for u in range(N)] # // Result of the integral =0 by symmetry => OK
#values_f = [1  for u in range(N)] => OK : test passed !

integral = computes_integral_sphere_diatomic(R,N,values_f)

print('Integral of f on a sphere S²(R) of radius R computed with Gauss-Legendre quadratude for 1D integration on theta')
print(integral)
"""
################################################################
################################################################


################################################################
"""

Computes the atomic partial charge int_{R³} (rho_a(r) dr)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
      Both values of w_1(.) and w_2(.) are needed to integrate rho_1 = (w_1/(w_1+w_2))*rho or rho_2
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_charge(a,Rmax,values_w_a,
                          values_rho_atom,
                          Ng_theta,
                          x_GL_0_1, w_GL_0_1,
                          Ng_radial,
                          x_GL_0_R, w_GL_0_R,
                          radial_grid_atoms,
                          atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
        
    # values_w_a[a](x_GL_0_R_scaled[l]) not defined
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * values_w_a[a][l] * compute_sph_avg_atomic_charge(l,x_GL_0_R_scaled[l],a,
                                                                                                                                 values_w_a,
                                                                                                                                 values_rho_atom,
                                                                                                                                 x_GL_0_1, w_GL_0_1,
                                                                                                                                 Ng_theta,
                                                                                                                                 radial_grid_atoms,
                                                                                                                                 atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic dipole  int_{R³} (z * rho_a(r) dr) along z axis (DIATOMIC case : the 2 atoms are along z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_dipole(a,Rmax,values_w_a,
                          values_rho_atom,
                          Ng_theta,
                          x_GL_0_1, w_GL_0_1,
                          Ng_radial,
                          x_GL_0_R, w_GL_0_R,
                          radial_grid_atoms,
                          atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**3 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**3 * x_GL_0_R_scaled[l]**3 * values_w_a[a][l] * compute_sph_avg_atomic_dipole(l,x_GL_0_R_scaled[l],a,
                                                                                                                                 values_w_a,
                                                                                                                                 values_rho_atom,
                                                                                                                                 x_GL_0_1, w_GL_0_1,
                                                                                                                                 Ng_theta,
                                                                                                                                 radial_grid_atoms,
                                                                                                                                 atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic quadrupole  int_{R³} (xx * rho_a(r) dr)  =(Q_a)_{xx} = (Q_a)_{yy} by symmetry
(DIATOMIC case : the 2 atoms are along z ; invariance around z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_xx(a,Rmax,values_w_a,
                                 values_rho_atom,
                                 Ng_theta,
                                 x_GL_0_1, w_GL_0_1,
                                 Ng_radial,
                                 x_GL_0_R, w_GL_0_R,
                                 radial_grid_atoms,
                                 atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)**3 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * values_w_a[a][l] * compute_sph_avg_atomic_quadrupole_xx(l,x_GL_0_R_scaled[l],a,
                                                                                                                                        values_w_a,
                                                                                                                                        values_rho_atom,
                                                                                                                                        x_GL_0_1, w_GL_0_1,
                                                                                                                                        Ng_theta,
                                                                                                                                        radial_grid_atoms,
                                                                                                                                        atomic_coordinates) for l in range(Ng_radial)]
    
    return math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic quadrupole  int_{R³} (xx * rho_a(r) dr)  =(Q_a)_{xx} = (Q_a)_{yy} by symmetry
(DIATOMIC case : the 2 atoms are along z ; invariance around z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_zz(a,Rmax,values_w_a,
                                 values_rho_atom,
                                 Ng_theta,
                                 x_GL_0_1, w_GL_0_1,
                                 Ng_radial,
                                 x_GL_0_R, w_GL_0_R,
                                 radial_grid_atoms,
                                 atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)* cos(theta)**2 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * values_w_a[a][l] * compute_sph_avg_atomic_quadrupole_zz(l,x_GL_0_R_scaled[l],a,
                                                                                                                                        values_w_a,
                                                                                                                                        values_rho_atom,
                                                                                                                                        x_GL_0_1, w_GL_0_1,
                                                                                                                                        Ng_theta,
                                                                                                                                        radial_grid_atoms,
                                                                                                                                        atomic_coordinates) for l in range(Ng_radial)]
    
    return 2 * math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################



################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

=> r in Bohr ? Or Ang ?
If r in Ang => density evaluated at a grid in Angstroms => in the radial quadrature, no
need to convert "r²" and "dr" in a.u. (Bohrs)

- values_w_a_atom = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_charge(index_radial,r,a,
                                  values_w_a,
                                  values_rho_atom,
                                  x_GL_0_1, w_GL_0_1,
                                  Ng_theta,
                                  radial_grid_atoms,
                                  atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
        
    # NB : values_w_a[a][index_radial] = w_a(r)
    
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(values_w_a[a][index_radial]+interpolate_P1(values_w_a[b],radial_grid_atoms[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_dipole(index_radial,r,a,
                                  values_w_a,
                                  values_rho_atom,
                                  x_GL_0_1, w_GL_0_1,
                                  Ng_theta,
                                  radial_grid_atoms,
                                  atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u]) /(values_w_a[a][index_radial]+interpolate_P1(values_w_a[b],radial_grid_atoms[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta)³ * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_xx(index_radial,r,a,
                                         values_w_a,
                                         values_rho_atom,
                                         x_GL_0_1, w_GL_0_1,
                                         Ng_theta,
                                         radial_grid_atoms,
                                         atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u])**3 /(values_w_a[a][index_radial]+interpolate_P1(values_w_a[b],radial_grid_atoms[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################



################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta)² * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- radial_grid_atoms[b] = radial grid on which w_b(.) was computed.
 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_zz(index_radial,r,a,
                                         values_w_a,
                                         values_rho_atom,
                                         x_GL_0_1, w_GL_0_1,
                                         Ng_theta,
                                         radial_grid_atoms,
                                         atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u])**2  /(values_w_a[a][index_radial]+interpolate_P1(values_w_a[b],radial_grid_atoms[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
################################################################
# Test ad-hoc densities to validate GISA /L-ISA algorithms :

    
###################################
"""
Returns the value of a testsdensity mae of a sum of two gaussians centered at R1 and R2 (the two virtual nuclei) respectively :
sum_i K_(1,i)*exp(-alpha_(1,i)*||\vec{r}-\vec{R1}||²)+ sum_ j K_(2,j)*exp(-alpha_(2,j)*||\vec{r}-\vec{R2}||²)

with K_(1,i) = (alpha_(1,i)/pi)**(3/2) and K_(2,j) = (alpha_(2,j)/pi)**(3/2) normalizing coefficients
"""
def sum_gaussians_TEST_DENSITY(vec_r,R_a,R_b,tab_alpha1,tab_alpha2):
    
    point1 = [(vec_r[i]-R_a[i]) for i in range(3)]
    
    point2 = [(vec_r[i]-R_b[i]) for i in range(3)]
    
    tab_K1 = [((tab_alpha1[l]/conversion_bohr_angstrom**2)/math.pi)**(3/2) for l in range(len(tab_alpha1)) ]
    
    tab_K2 = [((tab_alpha2[l]/conversion_bohr_angstrom**2)/math.pi)**(3/2) for l in range(len(tab_alpha2)) ]
    
    # Beware of (atomic) units
    """
    contr_atom1 = np.sum([tab_K1[i]*math.exp(-tab_alpha1[i]*(np.linalg.norm(point1)/conversion_bohr_angstrom)**2) for i in range(len(tab_alpha1))])
                          
    contr_atom2 = np.sum([tab_K2[i]*math.exp(-tab_alpha2[i]*(np.linalg.norm(point2)/conversion_bohr_angstrom)**2) for i in range(len(tab_alpha2))])
    """
 
    # Faux (H_2)+ (density with a cusp) : not same normalizing factors !
    
    contr_atom1 = np.sum([math.exp(-math.sqrt(tab_alpha1[i]*point1[2]**2 + 0.001*tab_alpha1[i]*(vec_r[0]**2+vec_r[1]**2) ) /conversion_bohr_angstrom) for i in range(len(tab_alpha1))])
                          
    contr_atom2 = np.sum([math.exp(-math.sqrt(tab_alpha2[i]*point2[2]**2 + 0.001*tab_alpha2[i]*(vec_r[0]**2+vec_r[1]**2) ) /conversion_bohr_angstrom) for i in range(len(tab_alpha2))])
    
    
    """
    contr_atom1 = np.sum([math.exp(-tab_alpha1[i]*abs(point1[2])/conversion_bohr_angstrom) for i in range(len(tab_alpha1))])
                          
    contr_atom2 = np.sum([math.exp(-tab_alpha2[i]*abs(point2[2])/conversion_bohr_angstrom) for i in range(len(tab_alpha2))])
    """
    
    return contr_atom1 + contr_atom2
###################################

"""
Computes the values of a test density 

rho(r) = (alpha_1/pi)**(3/2) * exp(-alpha_1*(r-R1)²) + (alpha_2/pi)**(3/2) * exp(-alpha_2*(r-R2)²) on the GL radial grid x GL angular  grid
"""
def precompute_molecular_density_TEST_DIATOMIC_SUM_GAUSSIANS(Ng_radial,Ng_theta,Rmax,
                                                             x_GL_0_1, w_GL_0_1,
                                                             x_GL_0_R, w_GL_0_R,
                                                             atomic_coordinates,
                                                             tab_alpha_1,tab_alpha_2):
    
    # Scaled radial Gauss-Legendre grid :
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]

    # The 2 atoms are along the z axis
    grid_points_3D_atom_1 = [ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[0][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    
    grid_points_3D_atom_2 =[ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[1][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    
    # Molecular density rho(r) = exp(-alpha_1*(r-R1)) + exp(-alpha_1*(r-R1)) 
    # => rho_(j,l) evaluated at (R_a+r_j.(sin(theta_l),0,cos(theta_l))) :
        
    values_rho_around_atom_1 = [ [ sum_gaussians_TEST_DENSITY(grid_points_3D_atom_1[k][l],atomic_coordinates[0],atomic_coordinates[1],tab_alpha_1,tab_alpha_2) for l in range(Ng_theta)] for k in range(Ng_radial) ]
 
    values_rho_around_atom_2 = [ [ sum_gaussians_TEST_DENSITY(grid_points_3D_atom_2[k][l],atomic_coordinates[1],atomic_coordinates[0],tab_alpha_2,tab_alpha_1)  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    return values_rho_around_atom_1, values_rho_around_atom_2
################################################################


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 11:06:21 2021

@author: rbenda

Implementation of the ISA-radial radial x Lebedev real-space discretization) 
grid-based algorithm (Lillestolen 2008 Chem. Phys. Letters)
"""

import numpy as np
import time
import math

import matplotlib
from matplotlib import pyplot

from ISA_auxiliary_functions import compute_density_grid_point
from ISA_auxiliary_functions import computes_total_charge
from ISA_auxiliary_functions import interpolate_linearly_weight_other_atom_grid
from ISA_auxiliary_functions import interpolate_linearly_density_other_atom_grid
from ISA_auxiliary_functions import spherical_average_function
from ISA_auxiliary_functions import computes_constraint_total_charge
from ISA_auxiliary_functions import computes_constraint_total_charge_normalization
from ISA_auxiliary_functions import compute_convergence_criterion
from ISA_auxiliary_functions import compute_convergence_criterion
from ISA_auxiliary_functions import compute_convergence_criterion_TRAPEZES
from ISA_auxiliary_functions import compute_convergence_criterion_TRAPEZES_DIATOMIC
from ISA_auxiliary_functions import compute_convergence_criterion_GL_DIATOMIC_ISA

from ISA_auxiliary_functions import computes_total_charge_integrand_contributions_TRAPEZES
from ISA_auxiliary_functions import computes_constraint_total_charge_atom
from ISA_auxiliary_functions import computes_constraint_total_charge_TRAPEZES
from ISA_auxiliary_functions import interpolate_P1

############################
#CONVERSION CONSTANTS :
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
############################

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'

ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'

# Fixed constant (convergence threshold)
epsilon_convergence=1e-10


#################################################################################
#################################################################################
"""
Implementation of the ISA-radial radial x angular (theta) real-space discretization) 
grid-based algorithm (Lillestolen 2008 Chem. Phys. Letters)

Specific to the diatomic case 

-  values_rho : precomputated values { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j=1..Ng_radial,l=1..Ng_theta} 
   of the (total) molecular density on the atomic grids 
   ==> used at all the iterations of the ISA algorithm main loop 

- radial_Log_Int_grid[a] : radial grid (discretization) chosen around atom a
"""
def ISA_radial_Lilestolen_DIATOMIC(Ng_radial,x_GL_0_R,w_GL_0_R,Rmax,
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
                                   tab_r_plot):

    
    
    
    nb_atoms=len(atomic_coordinates)
    
    print('nb_atoms = '+str(nb_atoms))
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]

    
    #################################################################################
    #INITIALIZATION OF ATOMIC WEIGHTS AND ATOMIC DENSITIES
    
    ##########################
    #Initialization : construction of rho_a^{(0)}(r_(n,j)^a)
    #in the case w_a^{(0)}(.) = 1  everywhere
    #==> no need of interpolating w_b(.) on a discretization point associated to atom a
    
    #Initialization of (w_a^{(0)})_a weight functions for all atoms a :
    #[[(w_1(r_n))_{n}],[(w_2(r_n))_{n}],[(w_3(r_n))_{n}],..]
    #i.e. w_i(r_n) = value of w_i(.) on {R_i+r_n.\sigma}, \sigma \in unit sphere
    atomic_weights_init=[[] for a in range(nb_atoms)]
    
    atomic_densities_init=[[] for a in range(nb_atoms)]
    
    #Initialization of (rho_a^{(0)})_a : initial atomic densities (discretized)
    for a in range(nb_atoms):
        
        # len(radial_Log_Int_grid[a]) 
        #(=Ng_radial ONLY when the discretization grid equals the quadrature grid)
        for k in range(len(radial_Log_Int_grid[a])):
            
            atomic_densities_init[a].append([])
            
            for l in range(Ng_theta):
                                        
                #rho_a^{(0)}(\vec{R_a} +r_k*sin(theta_l) e_x + r_k*cos(theta_l) e_z ) =
                # = (1/N_{atom}) * rho_{mol}(\vec{R_a} +r_k*sin(theta_l) e_x + r_k*cos(theta_l) e_z) where 
                
                ###atomic_densities_init[a][k].append((1/nb_atoms)*values_rho[a][k][l])
                    
                                
                point_l = [radial_Log_Int_grid[a][k] * math.sin(math.pi*x_GL_0_1[l]) ,0. , atomic_coordinates[a][2] + radial_Log_Int_grid[a][k] * math.cos(math.pi*x_GL_0_1[l]) ]
                
                tab_1 = [values_rho[a][k0][l] for k0 in range(Ng_radial)]
                
                tab_radial_l = [np.linalg.norm([x_GL_0_R_scaled[k0] * math.sin(math.pi*x_GL_0_1[l]) , 0, atomic_coordinates[0][2] + x_GL_0_R_scaled[k0] * math.cos(math.pi*x_GL_0_1[l]) ]) for k0 in range(Ng_radial) ]
                
                atomic_densities_init[a][k].append((1/nb_atoms)*interpolate_P1(tab_1,tab_radial_l,np.linalg.norm(point_l)))
                
    
    #Initialization of (w_a^{(0)})_a : initial atomic weights (discretized) 
    for a in range(nb_atoms):
        for var in range(len(radial_Log_Int_grid[a])):   
            atomic_weights_init[a].append(1)
            
    #END INITIALIZATION
    #################################################################################
    
    
    #################################################################################            
    print('-----------------------------')     
    print('Iteration 0')       
    print('atomic_densities_init : rho_a^{0}=rho/N')
    print(atomic_densities_init)
    
    print(' ')
    print('atomic_weights_init')
    print(atomic_weights_init)
    print('-----------------------------')
    #################################################################################
    
    
    #################################################################################
    #MAIN LOOP naïve ISA grid-based algorithm
    
    convergence_criterion=1000000000
    
    iteration_nb=1
    
    #Initialization of 'atomic_densities_current' and 'atomic_weights_current'
    #[[(rho_1(r_(i,j)^{1}))_{i,j}],[(rho_2(r_(i,j)^{2}))_{i,j}],[(rho_3(r_(i,j)^{3}))_{i,j}],..]
    #where (rho_1(r_(i,j)^{1}))_{i,j} is the list of values of the atomic density of the 1st atom
    #on its associated grid
    
    
    atomic_densities_current=atomic_densities_init
    
    atomic_weights_current=atomic_weights_init
    
    total_entropy_list_step=[]
    
    while ((convergence_criterion > epsilon_convergence) and (iteration_nb<50)):
        
        
        atomic_weights_previous=atomic_weights_current
        
        atomic_weights_current=[[] for a in range(nb_atoms)]
        
        ##########################
        #Iteration k='iteration_nb'
        
        ######################################
        ##Construction of all w_a^{(k)}(.)
        for a in range(nb_atoms):
            
            ####################
            #Construction of  w_a^{(k)} ==> only on the radial grid
            #(w_a^{(k)}(.) have by construction spherical symmetry)
            for n in range(len(radial_Log_Int_grid[a])):
                
                #Radial distance from atom R_a :
                #r_n=radial_Log_Int_grid[a][n]
                
                ##############
                #Construction of  w_a^{(k)}(r_n) : spherical average of rho_a^{(k-1)}(.) 
                #==> uses the values {rho_a^{(k-1)}(\vec{R_a} +r_n*sin(theta_l) e_x + r_n*cos(theta_l) e_z)}_{l=1..N_g_theta}
            
                function_values_theta_grid=[atomic_densities_current[a][n][p] for p in range(Ng_theta) ]
               

                atomic_weights_current[a].append(spherical_average_function_DIATOMIC(function_values_theta_grid,
                                                                                     x_GL_0_1,w_GL_0_1))
        
        #End construction of all w_a^{(k)}(.)
        ######################################
        
        
        #Saving values of rho_a^{(k-1)}(.) : not useful as the CV criterion 
        #can be tested on ||w_a^(k+1)-w_a^(k)|| only !
        ##atomic_densities_previous=atomic_densities_current
        
        print('atomic_weights_current')
        #print(atomic_weights_current)
        print(' ')
        
        ######################################
        #Construction (actualization) of all rho_a^{(k)}(.), 
        #knowing already all w_b^{(k)}(.) stored in 'atomic_weights_current'
        
        atomic_densities_current=[[] for a in range(nb_atoms)]
    
        for a in range(nb_atoms):
            
            ########################
            #Construction of  rho_a^{(k)}(.) => on radial grid x GL grid on theta (invariance around phi [z axis])
            for n in range(len(radial_Log_Int_grid[a])):       
                
                atomic_densities_current[a].append([])

                for p in range(Ng_theta):
                    
                    # Diatomic case :
                    b=abs(a-1)
                    
                    # Point \vec{R_a} +r_n*sin(theta_p) e_x + r_n*cos(theta_p) e_z
                    point=discretization_pnts_atoms_DIATOMIC[a][n][p]
                
                    point_shifted_other_atom = [(point[u]-atomic_coordinates[b][u]) for u in range(3)]
                    
                    # Norm || \vec{R_a} +r_n*sin(theta_p) e_x + r_n*cos(theta_p) e_z - \vec{R_b} ||
                    norm_point_shifted = np.linalg.norm(point_shifted_other_atom)
                    
                    ##############
                    #Construction of  rho_a^{(k)}(.) pointwise [on all grid points]
                    #from all atomic weight functions (w_b^{(k)}(.))_{b : atoms}
                    
                    #rho_a^{(k)}(r_(n,j)^a)= w_a^{(k)}(r_(n,j)^a) / (w_a^{(k)}(r_(n,j)^a)+ sum_b (w_b^{(k),interp.}(r_(n,j)^a))) * rho_{mol}(r_(n,j)^a)
                    
                    # Interpolated weight functions of the other atom
                    # evaluated at this grid point associated to atom a :
                    # (w_b^{(k),interp.}(|\vec{R_a} +r_n*sin(theta_p) e_x + r_n*cos(theta_p) e_z - \vec{R_b}|) :
                    interpolated_weight_other_atom = interpolate_P1(atomic_weights_current[b],radial_Log_Int_grid[b],norm_point_shifted)
                    
                    #We use the PRECOMPUTED molecular density :
                    value_molecular_density=values_rho[a][n][p]
                    # CORRECTION 22/02/22 : adaptation to the case where the discretization grid is different from the quadrature grid :
                    rho_atom_a_point=(atomic_weights_current[a][n]/(atomic_weights_current[a][n]+interpolated_weight_other_atom))*value_molecular_density
                    
                    atomic_densities_current[a][n].append(rho_atom_a_point)
                
        #End construction of all rho_a^{(k)}(.)
        ###################################################
        
        print('Iteration number '+str(iteration_nb))
        print('w_a : atomic_weights_current'+'  Iteration number '+str(iteration_nb))
        #print(atomic_weights_current)
        #print(len(atomic_weights_current))
        print(' ')
        print('rho_a : atomic_densities_current'+'  Iteration number '+str(iteration_nb))
        #print(atomic_densities_current)
        #print(len(atomic_densities_current[0]))
        print(' ')
        
        
        ########################  
        # CALCUL de l'entropie S(rho^(m+1)|rho^{0,(m)}) :
            
        total_entropy_tab = [ 4*math.pi*computes_entropy_ISA(a,atomic_weights_current,
                                                             radial_Log_Int_grid,
                                                             Rmax, 
                                                             atomic_coordinates,
                                                             x_GL_0_R,w_GL_0_R,Ng_radial,
                                                             x_GL_0_1, w_GL_0_1,Ng_theta,
                                                             values_rho) for a in range(2)]
        
        
        
        total_entropy = np.sum(total_entropy_tab)
        
        print('Total_entropy ISA TERATION '+str(iteration_nb)+' = '+str(total_entropy))
        print(' ')
                
        print('Atom 1 : '+str(total_entropy_tab[0])+' ; atom 2 : '+str(total_entropy_tab[1]))
        print(' ')

        total_entropy_list_step.append(total_entropy)
        
        ######################
        ##Convergence criterion :
        
        # If the discretization grid coincides with the quadrature grid :
        #difference_weights_iter_GL_grid=[[(atomic_weights_current[a][n]-atomic_weights_previous[a][n]) for n in range(len(radial_Log_Int_grid[a]))] for a in range(nb_atoms)]
          
        # If the discretization grid is different from the quadrature grid :
        # => interpolation to estimate w_a^(n) at the points of the quadrature grids (from their values at the discretization grid)
        difference_weights_iter_GL_grid=[[(interpolate_P1(atomic_weights_current[a], radial_Log_Int_grid[a], Rmax*x_GL_0_R[k])-interpolate_P1(atomic_weights_previous[a], radial_Log_Int_grid[a], Rmax*x_GL_0_R[k])) for k in range(len(x_GL_0_R))] for a in range(nb_atoms)]
                   
        
        #L^2 norm without weigths :
        #convergence_criterion=np.linalg.norm(difference_weights_iter)
        
        # GL weight of the integral of the difference of  |w_a^(k+1) - w_a^(k)| :
        convergence_criterion= compute_convergence_criterion_GL_DIATOMIC_ISA(difference_weights_iter_GL_grid,
                                                                             x_GL_0_R,
                                                                             w_GL_0_R,
                                                                             Rmax,
                                                                             nb_atoms)
            
        print('convergence criterion :')
        print(convergence_criterion)
        print('np.linalg.norm(difference_weights_iter_GL_grid)')
        print(np.linalg.norm(difference_weights_iter_GL_grid))

        print(' END ITERATION ')
        print(' ---------------------------- ')
        
        iteration_nb+=1
        
        if (convergence_criterion <= epsilon_convergence):
            print('CONVERGENCE ACHIEVED')
            print('atomic_weights_current')
            print(atomic_weights_current)
            print(' ')
            print('atomic_densities_current')
            #print(atomic_densities_current)
            print(' ')
            
        
            print(' END ISA-radial ALGORITHM')
     
    print('-----------')
    print('TOTAL number of iterations to converge = '+str(iteration_nb)+' for epsilon = '+str(epsilon_convergence))
    print(' ')
    print(' ')
    print('total_entropy_list_step')
    print(total_entropy_list_step)
    ########################
    # Plots the computed w_a(.) :
        
    # PLOT w_a(.) on a much finer grid than the grid used for the computation 
    # (thanks to interpolation of w_a(.) in-between)
        
    atomic_weights_atom_1_current_FINAL = [interpolate_P1(atomic_weights_current[0], radial_Log_Int_grid[0], tab_r_plot[k]) for k in range(len(tab_r_plot))]
    atomic_weights_atom_2_current_FINAL = [interpolate_P1(atomic_weights_current[1], radial_Log_Int_grid[1], tab_r_plot[k]) for k in range(len(tab_r_plot))]

    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) ISA-radial method")

    matplotlib.pyplot.plot(radial_Log_Int_grid[0],atomic_weights_current[0])
    matplotlib.pyplot.scatter(radial_Log_Int_grid[0],atomic_weights_current[0])
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')
    
    matplotlib.pyplot.savefig(ISA_plots_dir+"w_1_ISA_RADIAL_Ng_radial_"+str(Ng_radial)+"Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) ISA-radial method ")

    matplotlib.pyplot.plot(radial_Log_Int_grid[1],atomic_weights_current[1])
    matplotlib.pyplot.scatter(radial_Log_Int_grid[1],atomic_weights_current[1])
    
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"w_2_ISA_RADIAL_Ng_radial_"+str(Ng_radial)+"Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()

    # END PLOTS  
    #########################################

    ############
    # Closest index to inter-atomic distance :
    distance = abs(atomic_coordinates[0][2]-atomic_coordinates[1][2])
    
    index_interatomic_distance = -1
    
    for k0 in range(Ng_radial-1):
        if (x_GL_0_R[k0] <= distance) and (distance < x_GL_0_R[k0+1]):
            index_interatomic_distance = k0
      
    print("Intercalation inter-atomic distance / GL grid")
    print('distance = '+str(distance))
    print('k0 = '+str(k0))
    print('x_GL_0_R[k0] = '+str(x_GL_0_R[k0]))
    print('x_GL_0_R[k0+1]='+str(x_GL_0_R[k0+1]))
    print(' ')
    ###########
    
    atomic_weights_atom_2_current_DERIVATIVE_FINAL = [(interpolate_P1(atomic_weights_current[1], radial_Log_Int_grid[1], tab_r_plot[k])-interpolate_P1(atomic_weights_current[1], radial_Log_Int_grid[1], distance))/(tab_r_plot[k]-distance) for k in range(len(tab_r_plot))]

    ###########
    # PLOTS    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('(w2(r)-w2(R))/(r-R)')

    matplotlib.pyplot.title("(w2(r)-w2(R))/(r-R) around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) ISA-radial method (interpolated) ")

    matplotlib.pyplot.plot(tab_r_plot,atomic_weights_atom_2_current_DERIVATIVE_FINAL)

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    #matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_radial_DIATOMIC_w_2_times_4pir2_N_radial_"+str(Ng_radial)+"_Ng_angular_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",box_inches='tight')

    matplotlib.pyplot.show()

    #########################################  
  
    # PLOT r--> 4*pi*r²*wa(r)
    
    # Using the INTERPOLATED values of wa(.)
    w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot[i])**2 * atomic_weights_atom_1_current_FINAL[i]) for i in range(len(tab_r_plot))]
    w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot[i])**2 * atomic_weights_atom_2_current_FINAL[i]) for i in range(len(tab_r_plot))]
    
    
    print('w_2_times_r2_values_FINAL ISA radial')
    #print(w_2_times_r2_values_FINAL)
    print(' ')
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_1(r) [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) ISA-radial method (interpolated) ")

    matplotlib.pyplot.plot(tab_r_plot,w_1_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot,w_1_times_r2_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_radial_DIATOMIC_w_1_times_4pir2_N_radial_"+str(Ng_radial)+"_Ng_angular_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",box_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_2(r)  [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) ISA-radial method (interpolated) ")

    matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot,w_2_times_r2_values_FINAL,'bo')


    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_radial_DIATOMIC_w_2_times_4pir2_N_radial_"+str(Ng_radial)+"_Ng_angular_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",box_inches='tight')

    matplotlib.pyplot.show()


    # Returns atomic weights (w_a(r_i))_i at radial discretization points
    # and atomic densities (rho_a(r_(i,j)))_{i,j} at radial x angular (Lebedev) discretization (grid) points
    return atomic_weights_current, atomic_densities_current, w_1_times_r2_values_FINAL, w_2_times_r2_values_FINAL
    #END naïve ISA algorithm                   
    #################################################################################
    
    
#########################################################
#Computes the spherical average of a function f, given by its value on 
#the Lebedev points of a sphere of center R_a and radius r
#R_a  : any point, e.g. the origin or an atom
#<f>_{R_a}(r) = (1/(4*pi)* [int_{S^2}] f(R_a+r.sigma) d sigma
#                  = (1/(4*pi)) * (2*pi) * pi * sum_{i=1}^M ( w_i^{GL} * f(\vec{R_a} +r*sin(theta_i) e_x + r*cos(theta_i) e_z) )
# (factor pi : because integral from 0 to math.pi on theta)
# using invariance around phi
#where (w_i,sigma_i) are the M Lebedev weights and points on [0,1] 
#function_values_Lbdv_grid = [ f(R_a+r.sigma_i) = f([x_a+r*Lebedev_grid_points[i][0],y_a+r*Lebedev_grid_points[i][1],z_a+r*Lebedev_grid_points[i][2]]) for i in range(nb_Lbdv_points)]
def spherical_average_function_DIATOMIC(function_values_theta_GL_grid,
                                        x_GL_0_1,w_GL_0_1):
        
    Ng_theta=len(function_values_theta_GL_grid)
    
    #Table of values of the weighted density  w_i * f(R_a+r.sigma_i)
    #at the points {R_a+r.sigma_i}_{i=1..M}
    #where {sigma_i}_{i=1..M} are the M Lebedev points of the unit sphere
    function_values=[w_GL_0_1[i]*function_values_theta_GL_grid[i]*math.sin(math.pi*x_GL_0_1[i])
                             for i in range(Ng_theta)]

    return 0.5 * math.pi * np.sum(function_values)
######################################################### 

"""
w_a(.) assumed to have been computed on { Rmax*x_l^{GL} }_{l=1..N^{GL}}
"""
def compute_atomic_charge_ISA_radial_DIATOMIC(a,
                                              radial_grid,
                                              values_w_a,
                                              values_rho_atom,
                                              Ng_theta,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_radial,
                                              Rmax,
                                              x_GL_0_R, w_GL_0_R,
                                              atomic_coordinates):
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    # => need of interpolation to compute w_a^(m)(Rmax*x_l^(GL)))
    
    # values_w_a[a](x_GL_0_R_scaled[l]) not defined
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * values_w_a[a][l] * compute_sph_avg_atomic_charge_ISA_radial_DIATOMIC(l,x_GL_0_R_scaled[l],
                                                                                                                                                     a,
                                                                                                                                                     radial_grid,
                                                                                                                                                     values_w_a,
                                                                                                                                                     values_rho_atom,
                                                                                                                                                     x_GL_0_1, w_GL_0_1,
                                                                                                                                                     Ng_theta,
                                                                                                                                                     atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_charge_ISA_radial_DIATOMIC(index_radial,r,
                                                      a,
                                                      radial_grid,
                                                      values_w_a,
                                                      values_rho_atom,
                                                      x_GL_0_1, w_GL_0_1,
                                                      Ng_theta,
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
    
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(values_w_a[a][index_radial]+interpolate_P1(values_w_a[b],radial_grid[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


###############################################################
# FUNCTIONS FOR ENTROPY CALCULATIONS :

   
################################################################
"""
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- index_radial : index of r in the Gauss-Legendre radial grid (r_1,...r_M=R) where M = Ng_radial 
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of (rho(.)/ (sum_b w_b))*log((sum_b w_b)) at distance r from atom a :
# Used in the KL entropy calculation (int (rho_a * log(rho_a)))
def compute_sph_avg_entropy_1_ISA(index_radial,r,
                                  radial_Log_Int_grid,
                                  Ng_theta,
                                  a,
                                  atomic_weights_current,
                                  values_rho_atom,
                                  x_GL_0_1, w_GL_0_1,
                                  atomic_coordinates):
        
    
    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[abs(a-1)][2]]  for l in range(Ng_theta)]
    

    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * (math.sin(math.pi*x_GL_0_1[u])/(interpolate_P1(atomic_weights_current[a],radial_Log_Int_grid[a],r)+interpolate_P1(atomic_weights_current[abs(a-1)],radial_Log_Int_grid[abs(a-1)],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]))))*math.log(max(10**(-8),interpolate_P1(atomic_weights_current[a],radial_Log_Int_grid[a],r)+interpolate_P1(atomic_weights_current[abs(a-1)],radial_Log_Int_grid[abs(a-1)],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]))))  for u in range(Ng_theta)]
    
    # Integral on theta only (not on phi)
    return math.pi * np.sum(values_integrand_theta) 
################################################################

################################################################
"""
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)*log(rho(.))/ (sum_b w_b) at distance r from atom a :
# Used in the KL entropy calculation (int (rho_a * log(rho_a)))
def compute_sph_avg_entropy_2_ISA(index_radial,r,
                                  radial_Log_Int_grid,
                                  Ng_theta,
                                  a,
                                  atomic_weights_current,
                                  values_rho_atom,
                                  x_GL_0_1, w_GL_0_1,
                                  atomic_coordinates):
        
        
    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[abs(a-1)][2]]  for l in range(Ng_theta)]
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u]*math.log(values_rho_atom[index_radial][u]) * math.sin(math.pi*x_GL_0_1[u]) /(interpolate_P1(atomic_weights_current[a],radial_Log_Int_grid[a],r)+interpolate_P1(atomic_weights_current[abs(a-1)],radial_Log_Int_grid[abs(a-1)],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]

    
    # Integral on theta only (not on phi)
    return math.pi * np.sum(values_integrand_theta) 
################################################################

################################################################
"""
Returns the  entropy s_{KL}(rho_a^(m)|rho_a^{0,(m)}) at step m

'atomic_weights_current' = rho_a^{0,(m)} = w_a^(m)
"""
def computes_entropy_ISA(a,atomic_weights_current,
                         radial_Log_Int_grid,
                         Rmax, 
                         atomic_coordinates,
                         x_GL_0_R,w_GL_0_R,Ng_radial,
                         x_GL_0_1, w_GL_0_1,Ng_theta,
                         values_rho):
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
        
    

    #NEW
    # With same quadrature and discretization grid :
    # Computes r**2 * rho_a^(0,(m))(r) * (- < (rho/ (sum_b rho_b^(0,(m))))*log(sum_b rho_b^(0,(m))) >_a(r) + < rho*log(rho)/ (sum_b rho_b^(0,(m)))>_a(r) )
    # on a GL radial grid (r_l)_l
    
    #With different quadrature and discretization grids (the quadrature grid being typically finer) :
    # If w_a(.) known only on 'radial_Log_Int_grid[a)' grid
    
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * interpolate_P1(atomic_weights_current[a],radial_Log_Int_grid[a],x_GL_0_R_scaled[u]) * (-compute_sph_avg_entropy_1_ISA(u,x_GL_0_R_scaled[u],
                                                                                                                                                                                                                      radial_Log_Int_grid,
                                                                                                                                                                                                                      Ng_theta,
                                                                                                                                                                                                                      a,
                                                                                                                                                                                                                      atomic_weights_current,
                                                                                                                                                                                                                      values_rho[a],
                                                                                                                                                                                                                      x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                      atomic_coordinates)
                                                                                                                                                                                       +compute_sph_avg_entropy_2_ISA(u,x_GL_0_R_scaled[u],
                                                                                                                                                                                                                      radial_Log_Int_grid,
                                                                                                                                                                                                                      Ng_theta,
                                                                                                                                                                                                                      a,
                                                                                                                                                                                                                      atomic_weights_current,
                                                                                                                                                                                                                      values_rho[a],
                                                                                                                                                                                                                      x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                      atomic_coordinates)) for u in range(Ng_radial)]
    
    
    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    integral = (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return integral
################################################################



################################################################
"""
Precomputes h_atom = [h_atom[a]]_{a=1..M} with 
h_atom[a] = { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
  
  where (r_j)_j = radial grid for ISA discretization (Ng_discretization_ISA=len(radial_Log_Int_grid) points), built as :
      
                  x_GL_0_R_discret_ISA, w_GL_0_R_discret_ISA = lgwt(Ng_discretization_ISA,0,1)
                  tab_discr_ISA = [0]

                    for k in range(Ng_discretization_ISA):
                        tab_discr_ISA.append(Rmax*x_GL_0_R_discret_ISA[k])
    
                    radial_Log_Int_grid=[tab_discr_ISA,tab_discr_ISA]
                    
                  potentially different from the radial grid for Gauss Legendre integration and
                  
        (theta_l)_l =  angular grid for angular integration (on theta in spherical coordinates)
        [ N_theta points ]

Uses the invariance along phi (i.e. along z axis) in the DIATOMIC case

- a = atom index

- Rmax >> |R2-R1| (integral over [0,+\infty] approximated by an integral over [0,Rmax])

- x_GL_0_R, w_GL_0_R : obtained previously as lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 (Angular Gauss-Legendre grid): obtained previously as : lgwt(Ng_theta,0,1)

"""
def precompute_molecular_density_DIATOMIC_ISA(Ng_discretization_ISA,
                                              radial_Log_Int_grid,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_theta,
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
                                              contraction_coefficients_pSHELLS_to_cSHELLS):

    
    # The 2 atoms are along the z axis
    grid_points_3D_atom_1 =[ [ [radial_Log_Int_grid[0][k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[0][2] + radial_Log_Int_grid[0][k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_discretization_ISA) ]
    
    grid_points_3D_atom_2 =[ [ [radial_Log_Int_grid[1][k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[1][2] + radial_Log_Int_grid[1][k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_discretization_ISA) ]
    
    # Molecular density rho_(j,l) evaluated at (R_a+r_j.(sin(theta_l),0,cos(theta_l))) :
        
    values_rho_around_atom_1_ISA = [ [ compute_density_grid_point(grid_points_3D_atom_1[k][l],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS)  for l in range(Ng_theta)] for k in range(Ng_discretization_ISA) ]
 
    
        
    values_rho_around_atom_2_ISA = [ [ compute_density_grid_point(grid_points_3D_atom_2[k][l],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS)  for l in range(Ng_theta)] for k in range(Ng_discretization_ISA) ]
 
    
    return values_rho_around_atom_1_ISA, values_rho_around_atom_2_ISA
################################################################

#################################################################################
#################################################################################
# GENERAL (not diatomic case) algorithm (very slow)


#################################################################################
"""
Implementation of the ISA-radial radial x Lebedev real-space discretization) 
grid-based algorithm (Lillestolen 2008 Chem. Phys. Letters)

NOT specific to the diatomic case (works whatever the molecule size)


-  discretization_pnts_atoms : radial x Lebedev angular grid for all atoms 
-  molecular_density_atomic_grids : precomputated [once and for all, before calling this function] 
   of the (total) molecular density on the atomic grids : i.e. (rho(r_(i,j)^a))_(i,j)
   ==> used at all the iterations of the ISA algorithm main loop 

- radial_Log_Int_grid[a] : radial grid (discretization) chosen around atom a
- Lbd_magical_number : number of Lebedev points 'Lebedev_grid_points' on the sphere S² chosen for the angular discretization / angular quadrature
"""
def ISA_radial_Lilestolen(discretization_pnts_atoms,
                          molecular_density_atomic_grids,
                          radial_Log_Int_grid,
                          Lbd_magical_number,
                          Lebedev_grid_points,
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
                          atomic_numbers):
    
    
    
    nb_atoms=len(atomic_coordinates)
    
    #################################################################################
    #INITIALIZATION OF ATOMIC WEIGHTS AND ATOMIC DENSITIES
    
    ##########################
    #Initialization : construction of rho_a^{(0)}(r_(n,j)^a)
    #in the case w_a^{(0)}(.) = 1  everywhere
    #==> no need of interpolating w_b(.) on a discretization point associated to atom a
    
    #Initialization of (w_a^{(0)})_a weight functions for all atoms a :
    #[[(w_1(r_n))_{n}],[(w_2(r_n))_{n}],[(w_3(r_n))_{n}],..]
    #i.e. w_i(r_n) = value of w_i(.) on {R_i+r_n.\sigma}, \sigma \in unit sphere
    atomic_weights_init=[[] for a in range(nb_atoms)]
    
    atomic_densities_init=[[] for a in range(nb_atoms)]
    
    #Initialization of (rho_a^{(0)})_a : initial atomic densities (discretized)
    for a in range(nb_atoms):
        
        for m in range(len(discretization_pnts_atoms[a])):
                
            point=discretization_pnts_atoms[a][m]
                                
            #rho_a^{(0)}(r_(n,j)^a)= (1/N_{atom}) * rho_{mol}(r_(n,j)^a) where 
            #rho_{mol} is the QM density

            #Use the PRECOMPUTED molecular density :
            atomic_densities_init[a].append((1/nb_atoms)*molecular_density_atomic_grids[a][m])
       
    
    #Initialization of (w_a^{(0)})_a : initial atomic weights (discretized) 
    for a in range(nb_atoms):
        for var in range(len(radial_Log_Int_grid)):   
            atomic_weights_init[a].append(1)
            
    #END INITIALIZATION
    #################################################################################
    
    
    #################################################################################            
    print('-----------------------------')     
    print('Iteration 0')       
    print('atomic_densities_init : rho_a^{0}=rho/N')
    #print(atomic_densities_init)
    
    print(' ')
    print('atomic_weights_init')
    #print(atomic_weights_init)
    print('-----------------------------')
    #################################################################################
    
    
    #################################################################################
    #MAIN LOOP naïve ISA grid-based algorithm
    
    convergence_criterion=1000000000
    
    iteration_nb=1
    
    #Initialization of 'atomic_densities_current' and 'atomic_weights_current'
    #[[(rho_1(r_(i,j)^{1}))_{i,j}],[(rho_2(r_(i,j)^{2}))_{i,j}],[(rho_3(r_(i,j)^{3}))_{i,j}],..]
    #where (rho_1(r_(i,j)^{1}))_{i,j} is the list of values of the atomic density of the 1st atom
    #on its associated grid
    
    
    atomic_densities_current=atomic_densities_init
    
    atomic_weights_current=atomic_weights_init
    
    
    while ((convergence_criterion > epsilon_convergence) and (iteration_nb<500)):
        
        
        atomic_weights_previous=atomic_weights_current
        
        atomic_weights_current=[[] for a in range(nb_atoms)]
        
        ##########################
        #Iteration k='iteration_nb'
        
        ######################################
        ##Construction of all w_a^{(k)}(.)
        for a in range(nb_atoms):
            
            ####################
            #Construction of  w_a^{(k)} ==> only on the radial grid
            #(w_a^{(k)}(.) have by construction spherical symmetry)
            for n in range(len(radial_Log_Int_grid)):
                
                #Radial distance from atom R_a :
                r_n=radial_Log_Int_grid[n]
                
                ##############
                #Construction of  w_a^{(k)}(r_n) : spherical average of rho_a^{(k-1)}(.) 
                #==> uses the values {rho_a^{(k-1)}(R_a+r_n.sigma_i)}_{i=1..N_{Lbdv}} ; sigma_i = Lebedev points 
                
                function_values_Lbdv_grid=np.zeros(Lbd_magical_number)
               
                for p in range(Lbd_magical_number):
    
                    function_values_Lbdv_grid[p]=atomic_densities_current[a][n*Lbd_magical_number+p]           
                
                atomic_weights_current[a].append(spherical_average_function(function_values_Lbdv_grid,r_n,atomic_coordinates[a],Lebedev_grid_points))
        #End construction of all w_a^{(k)}(.)
        ######################################
        
        
        #Saving values of rho_a^{(k-1)}(.) : not useful as the CV criterion 
        #can be tested on ||w_a^(k+1)-w_a^(k)|| only !
        
        ######################################
        #Construction (actualization) of all rho_a^{(k)}(.), 
        #knowing already all w_b^{(k)}(.) stored in 'atomic_weights_current'
        
        atomic_densities_current=[[] for a in range(nb_atoms)]
    
        for a in range(nb_atoms):
            
            ########################
            #Construction of  rho_a^{(k)}(.) => on radial grid x Lebedev grid
            for m in range(len(discretization_pnts_atoms[a])):       
            
                point=discretization_pnts_atoms[a][m]
                
                #Associated radial grid index :
                # r_(i,j)^a = R_a +r_i.sigma_j --> associated r_i radial value
                n_radial_grid=int(m/Lbd_magical_number)
                #Check n_radial_grid<= len(radial_Log_Int_grid)-1 = p-1
                
                ##############
                #Construction of  rho_a^{(k)}(.) from all atomic weight functions (w_b^{(k)}(.))_{b : atoms}
                
                #rho_a^{(k)}(r_(n,j)^a)= w_a^{(k)}(r_(n,j)^a) / (w_a^{(k)}(r_(n,j)^a)+ sum_b (w_b^{(k),interp.}(r_(n,j)^a))) * rho_{mol}(r_(n,j)^a)
                    
                #Sum of the interpolated weight functions of other atoms (different of a)
                #evaluated on this grid point associated to atom a :
                #sum_b (w_b^{(k),interp.}(r_(n,j)^a))
                sum_interpolated_weight_other_atoms=np.sum([interpolate_linearly_weight_other_atom_grid(b,point,atomic_weights_current[b],atomic_coordinates,radial_Log_Int_grid) for b in range(nb_atoms) if b!=a])
                           
                #We use the PRECOMPUTED molecular density :
                value_molecular_density=molecular_density_atomic_grids[a][m]
                
                rho_atom_a_point_m=(atomic_weights_current[a][n_radial_grid]/(atomic_weights_current[a][n_radial_grid]+sum_interpolated_weight_other_atoms))*value_molecular_density
                    
                atomic_densities_current[a].append(rho_atom_a_point_m)
                
        #End construction of all rho_a^{(k)}(.)
        ###################################################
        
        print('Iteration number '+str(iteration_nb))
        print('w_a : atomic_weights_current'+'  Iteration number '+str(iteration_nb))
        #print(atomic_weights_current)
        #print(len(atomic_weights_current))
        print(' ')
        print('rho_a : atomic_densities_current'+'  Iteration number '+str(iteration_nb))
        #print(atomic_densities_current)
        #print(len(atomic_densities_current[0]))
        print(' ')
        
        ######################
        # BEWARE : Works only for large enough discretization grids (cf. quadrature study)
        #Estimation of the constraint i.e. total charge conservation :
        #(measure of the discretization error)
        #sum_{a atom} sum_{i radial discr.} sum_{j Lbdv points} | rho^{mol}(r_(i,j)^a) - (rho_a(r_(i,j)^a)+sum_{b !=a} (rho_b^{interp.}(r_(i,j)^a)) ) |^2
        
        ##Convergence criterion :
        
        difference_weights_iter=[[(atomic_weights_current[a][i]-atomic_weights_previous[a][i]) for i in range(len(radial_Log_Int_grid))] for a in range(nb_atoms)]
      
        #L^2 norm without weigths :
        #convergence_criterion=np.linalg.norm(difference_weights_iter)
        
        convergence_criterion=compute_convergence_criterion_TRAPEZES(difference_weights_iter,radial_Log_Int_grid,nb_atoms)
        
        print('convergence criterion :')
        print(convergence_criterion)
        print('np.linalg.norm(difference_weights_iter)')
        print(np.linalg.norm(difference_weights_iter))
        print(' END ITERATION ')
        print(' ---------------------------- ')
        
        iteration_nb+=1
        
        if (convergence_criterion <= epsilon_convergence):
            print('CONVERGENCE ACHIEVED')
            print('atomic_weights_current')
            print(atomic_weights_current)
            print(' ')
            print('atomic_densities_current')
            #print(atomic_densities_current)
            print(' ')
            
    
            numerator_final=computes_constraint_total_charge_TRAPEZES(nb_atoms,molecular_density_atomic_grids,atomic_densities_current,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points)
        
            denominator_final=computes_constraint_total_charge_normalization(nb_atoms,molecular_density_atomic_grids,discretization_pnts_atoms,radial_Log_Int_grid,atomic_coordinates,Lebedev_grid_points)
        
            constraint_total_charge_final=numerator_final/denominator_final
        
            print('CONSTRAINT TOTAL CHARGE FINAL : rho = sum_a (rho_a)')
            print(constraint_total_charge_final)
            
        
            print(' END NAIVE ISA ALGORITHM')
     
    ########################
    # Plots the computed w_a(.) :
        
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" )")


    matplotlib.pyplot.plot(radial_Log_Int_grid,atomic_weights_current[0])
    
    matplotlib.pyplot.savefig(ISA_plots_dir+"w_1_ISA_RADIAL_"+str(Lbd_magical_number)+"_Lbdv_points_p_rad_"+str(len(radial_Log_Int_grid))+" Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" )")


    matplotlib.pyplot.plot(radial_Log_Int_grid,atomic_weights_current[1])
    
    matplotlib.pyplot.savefig(ISA_plots_dir+"w_2_ISA_RADIAL_"+str(Lbd_magical_number)+"_Lbdv_points_p_rad_"+str(len(radial_Log_Int_grid))+" Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    #END PLOTS
    ##########################
 
    # Returns atomic weights (w_a(r_i))_i at radial discretization points
    # and atomic densities (rho_a(r_(i,j)))_{i,j} at radial x angular (Lebedev) discretization (grid) points
    return atomic_weights_current, atomic_densities_current
    #END naïve ISA algorithm                   
    #################################################################################

#################################################################################
#################################################################################

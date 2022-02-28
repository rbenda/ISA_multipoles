#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu May 27 16:05:24 2021

@author: rbenda

Iterative algorithm to find the shape functions w_1(.) and w_2(.) of the ISA decomposition
in the DIATOMIC case, using functions F1_BIS(r,y) and F2_BIS(r,y) and the caracterization :
    
    - F1_BIS(r,w1(r))= 0 or w1(r)=0
    - F2_BIS(r,w2(r))=0 or w2(r)=0
    
and finding w_a(r) with a Newton algorithm to fund a zero of y --> Fa_BIS(r,y) 

Fa_BIS(.,.) : built with t = cos(theta) change of variables in the integral.

See explicative note or Thesis manuscript Robert Benda, 6.6 Appendix A : ISA-radial : an alternative scheme based on the Newton method (p. 257).

This method only serves as a sanity check of the ISA_radial (reference ISA method) and yields the same results (proatomic density profiles, and consequently atomic densities).
"""


import numpy as np
import math
import matplotlib

from ISA_auxiliary_functions import compute_density_grid_point
from ISA_auxiliary_functions import lgwt
from ISA_auxiliary_functions import interpolate_P1
from ISA_auxiliary_functions import sum_gaussians_TEST_DENSITY


ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'
ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

################################################################
################################################################
"""   
Gauss-Legendre nodes (points) on [-1,1] : computed previously e.g. with :
  x_GL, w_GL  = lgwt(Ng,-1,1) : GL grid on "t" variable for the computations of the integrals on t
  (replaces the angular GL grid on theta)
  
  - radial_grid = : usually = 0 + GL grid on [0,1] times R_max, with Ng points
"""
def ISA_iterative_diatomic_Cances_BIS(N,
                                      Ng, x_GL_minus1_plus1, w_GL_minus1_plus1,
                                      R,R1,R2,
                                      Ng_avg,
                                      radial_grid,
                                      atomic_numbers,
                                      QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                      density_matrix_coefficient_pGTOs,
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
                                      tab_alpha=[],
                                      density_type="Molecular"):

    
    ##s_atoms = compute_s_points_atoms(N,Ng,R,radial_grid,x_GL)
    grid_points_atoms = compute_grid_points_around_atoms(N,Ng,x_GL_minus1_plus1,
                                                         R,
                                                         R1,R2,
                                                         radial_grid)

    
    print('Molecular density precomputation for ISA diatomic-Newton (Cancès)')
    print('precomputation_function_h_a')
    
    #  Precomputation of the molecular density rho at the grid points in (Oxz) plane
    # Precomputed densities averaged along phi (in spherical coordinates)
    # due to invariance along z (diatomic) axis :
    
    ############
    # TRUE molecular density :
    if (density_type=="Molecular"):
        h_atom_1,h_atom_2, density_R1, density_R2 = precomputation_function_h_a_BIS(N,Ng,R,R1,R2,radial_grid,
                                                                                    x_GL_minus1_plus1,
                                                                                    density_matrix_coefficient_pGTOs,
                                                                                    position_nuclei_associated_basis_function_pGTO,
                                                                                    QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                                    nb_contracted_GTOs,
                                                                                    total_nb_primitive_GTOs,
                                                                                    correspondence_basis_pGTOs_atom_index,
                                                                                    angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                                    correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                                    tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                                    tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                                    contraction_coefficients_pSHELLS_to_cSHELLS,
                                                                                    atomic_coordinates)
        
    ############
    
    ############
    # Test molecular density (e.g. rho(r) = exp(-alpha_1*(r-R1)²) + exp(-alpha_2*(r-R2)²) )
    elif (density_type=="TEST"):
        h_atom_1,h_atom_2, density_R1, density_R2 = precomputation_function_h_a_TEST_2_GAUSSIANS(N,Ng,R,R1,R2,radial_grid,
                                                                                                 x_GL_minus1_plus1,
                                                                                                 atomic_coordinates,
                                                                                                 tab_alpha)
        
    ############
    
    print('density at atom 1 = '+str(density_R1))
    print('density at atom 2 = '+str(density_R2))
    
    #################################################
    # Initialization : w_2^(0)(.) : defined by its values at (r_(2,j))_{j=1..N} points
    # BEWARE : CHOICE OF INITIAL GUESS w_2^(0)(.)
    # IF BADLY CHOSEN : F1(1,j,k=0)(r_(1,j),y=0) will be < 0 for j >= j_max => yielding w_1(r_(1,j))=0
    # AND SUBSEQUENT PROBLEMS FOR F2(2,j,k=0)
    # Decreasing w_2^(0)(.)
    
    ##########
    
    #########
    # INITIAL GUESS
    
    ######
    # TRUE molecular density : taking half the spherical average of the molecular density :
    if (density_type=="Molecular"):
        
        print('Computation of initial guess w_2^(0)(.) from half the average molecular density ')
        
        # w_2^(0)(.) = 1/2 * <rho(.)>_2(.)  [cf. 1st iteration of ISA taking w_a(.) = 1 initially]
        atomic_weights_atom_2_current = [ average_rho_molecular_diatomic_atom(radial_grid[1][j],R1,Ng_avg,
                                                                              QM_data,boolean_cartesian_D_shells,
                                                                              boolean_cartesian_F_shells,
                                                                              density_matrix_coefficient_pGTOs,
                                                                              position_nuclei_associated_basis_function_pGTO,
                                                                              nb_contracted_GTOs,
                                                                              total_nb_primitive_GTOs,
                                                                              correspondence_basis_pGTOs_atom_index,
                                                                              angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                              correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                              tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                              tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                              contraction_coefficients_pSHELLS_to_cSHELLS) for j in range(len(radial_grid[1]))]
    
        print('atomic_weights_atom_2_current')
        print(atomic_weights_atom_2_current)
        print(' ')
        
        print('Computation of initial guess w_1^(0)(.) from half the average molecular density ')
    
        # w_1^(0)(.) = 1/2 * <rho(.)>_1(.) [cf. 1st iteration of ISA taking w_a(.) = 1 initially]
        atomic_weights_atom_1_current = [ average_rho_molecular_diatomic_atom(radial_grid[0][j],R2,Ng_avg,
                                                                              QM_data,boolean_cartesian_D_shells,
                                                                              boolean_cartesian_F_shells,
                                                                              density_matrix_coefficient_pGTOs,
                                                                              position_nuclei_associated_basis_function_pGTO,
                                                                              nb_contracted_GTOs,
                                                                              total_nb_primitive_GTOs,
                                                                              correspondence_basis_pGTOs_atom_index,
                                                                              angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                              correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                              tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                              tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                              contraction_coefficients_pSHELLS_to_cSHELLS) for j in range(len(radial_grid[0]))]
    
        print('atomic_weights_atom_1_current')
        print(atomic_weights_atom_1_current)
        print(' ')
        
    ############
    # Test molecular density (e.g. rho(r) = exp(-alpha_1*(r-R1)²) + exp(-alpha_2*(r-R2)²) )
    
    elif (density_type=="TEST"):
        
        atomic_weights_atom_2_current = [ average_rho_molecular_diatomic_atom_TEST(radial_grid[1][j],R1,0,
                                                                                   Ng_avg,
                                                                                   atomic_coordinates,
                                                                                   tab_alpha) for j in range(len(radial_grid[1]))]
    
    
        atomic_weights_atom_1_current = [ average_rho_molecular_diatomic_atom_TEST(radial_grid[0][j],R2,1,
                                                                                   Ng_avg,
                                                                                   atomic_coordinates,
                                                                                   tab_alpha) for j in range(len(radial_grid[0]))]
        
    #################################################
    
    
    test_loop = 1000
    
    k=0
    
    # 1e-5 seems too strict a criterion (stabilization of the
    #'error' between successive steps at test_loop ~ 1e-4)
    while (test_loop>1e-4) and (k <= 30):
        
        ###############################################
        # Computation of w_1^(k+1) from w_2^(k) :
        # w_(1,j)=w_1(r_(1,j)) for atom 1
        
        # Saving w_1^(k)(.) (for evaluating convergence / ||w_1^(k+1)-w_1^(k)||):
        atomic_weights_atom_1_previous = np.zeros(len(radial_grid[0]))
        
        for i in range(len(radial_grid[0])):
            atomic_weights_atom_1_previous[i] = atomic_weights_atom_1_current[i]
        
        
        # j=1 :
        # w_(1,1)^(k+1) = w_1(r_(1,1)=0) [at the (k+1)^{th} iteration] :
        # w_(1,1)^(k+1) = rho(\vec{R1})-w_2^k(R=|r2-r1|)
        
        # w_(1,1)^(k+1) = rho(R1)-w_2^(k)(R)  where R=|R2-R1|
 
        
        atomic_weights_atom_1_current[0]= density_R1 - interpolate_P1(atomic_weights_atom_2_current,radial_grid[1],R) 
    
        # For 2 ≤ j ≤ N , define the discretized F1 function at iteration k and point r_(1,j)
        #as well as its derivative
        # F1(r,y) = int_{|R-r| .. R+r} (h1(r,s)/(y+w2(s))) ds  - 1
        # Discretized at point r_(1,j) at iteration k by :
        # F1(r_(1,j),y)^{k} = -1 + L_(1,j) * sum_{l=1..Ng} (w_l * (h_(1,j,l)/(y+w_2^(k)(s_(1,j,l)))))
        # where L_(1,j) = R+r_(1,j) - |R-r_(1,j)|
        
        # w_2^(k)(s_(1,j,l)) is evaluated not at a natural grid point for w_2^(k)(.) in its P1 representation
        # => need for interpolation
        
        # F1(r_(1,j),y=0)^{k}
      
        #print('F1(r_(1,1),y=0) ; Ng = ')
        #print(discretized_F1(0,j,Ng,w_GL,h_atom_1,atomic_weights_atom_2_current,radial_grid,s_atoms))
        
        F1_current=[discretized_F1_BIS(0,j,R,R1,R2,
                                       N,
                                       Ng,w_GL_minus1_plus1,
                                       h_atom_1,
                                       atomic_weights_atom_2_current,
                                       radial_grid,
                                       grid_points_atoms) for j in range(1,N)]
        
        #print('[ F1^(k='+str(k)+')(r_(1,j),y=0) , j=2..N] ; Ng = '+str(Ng))
        #print(F1_current)
        #print(' ')
        
        # If F1(r_(1,j),0)^{k} <=  0 : set w_(1,j)^(k+1)  = 0
        for j in range(1,N):

            ###########################################################
            # NEWTON algorithm to find w_2^(k+1)(r_(2,j)) as a zero of y --> F2^(k)(r_(2,j),y)
            
            ###########   
            
            if (F1_current[j-1] < 0):
                
                print('F1(r_(1,'+str(j)+'),0)^{'+str(k)+'} < 0')
                print(' NO NEWTON algorithm : w1(r_(1,'+str(j)+') = 0 ')
                print(' ')
                
                atomic_weights_atom_1_current[j]=0
                
            # Else : if F1(r_(1,j),0)^{k} > 0 => find the unique root y* of y --> F1(r_(1,j),y)^{k} 
            # by the Newton algorithm :
            # y0 = w_(1,j)^(k) OR w_(2,j)^(k) >= 0 [TO CHECK !!]
            # y_(n+1) = y_n - F1(r_(1,j),y_n)^{k} / F1'(r_(1,j),y_n)^{k} 
            else :
                
                ############
                # To proceed by dichotomy in case of non-convergence : always keep trakc
                # of the largest iterate y_k such that F1(r_(1,j),y_k) > 0 
                # and of the smallest iterate y_h such that F1(r_(1,j),y_h) < 0 
                # As y --> F1(r_(1,j),y) is strictly decreasing : the solution is inside [y_k,y_l]
                # => draw a new guess in this interval
                # The following condition has always to be fulfilled : y_k < y_l
                # (otherwise, if reducing the Newton algorithm research inside [y_k,y_k] => infinite loop !)
                
                y_iterate_max_F1_pos = 0
            
                y_iterate_min_F1_neg = 1000
                
                
                # INITIAL GUESS FOR THE NEWTON ALGORITHM (looking for zero of F1(r_(1,j),.)):
                # y0 = w_(1,j)^(k) > 0
                y0 = atomic_weights_atom_1_current[j]
                y_current=y0
                
                #matplotlib.pyplot.scatter(y_current,0,c = 'red',marker='x')

                ##print('y_current=y0='+str(y_current))
                
                test=10000
                
                F1_current_n_j = 1000000
                
                iter_nb=0
                
                # Newton loop to find a zero of F1(r_(1,j),.)
                # Stop when the sequence of iterates has converged
                # or F1(r_(1,j),y_n)^{k} \approx 0
                
                while ((abs(F1_current_n_j)>1e-6) or (test>1e-4)):
                    
                #while (iter_nb<=5):
                    
                    #print('test = '+str(test))
                    
                    F1_current_n_j=discretized_F1_BIS(y_current,j,R,R1,R2,
                                                      N,
                                                      Ng,w_GL_minus1_plus1,
                                                      h_atom_1,
                                                      atomic_weights_atom_2_current,
                                                      radial_grid,
                                                      grid_points_atoms)
                                                                          
                    # Actualize the interval for dichotomy
                    if (F1_current_n_j >0):
                        if (y_current > y_iterate_max_F1_pos):
                            y_iterate_max_F1_pos = y_current
                            
                    elif (F1_current_n_j < 0):
                        if (y_current < y_iterate_min_F1_neg):
                            y_iterate_min_F1_neg = y_current                       
                         
                    if (abs(y_iterate_max_F1_pos-y_iterate_min_F1_neg)<1e-10):
                        print('ALERT ! y_iterate_max_F1_pos = y_iterate_min_F1_neg = '+str(y_iterate_min_F1_neg))
                        print(' ')
                        break
                        
                    # F1'(r_(1,j),y_n)^{k} [partial derivative with respect to y, evaluated at y=0]
                    # HAS TO BE NEGATIVE (y --> F1(r_(1,j),y) is a decreasing function, whatever the  value of r_(1,j))
                    F1_Tilde_current_n_j=discretized_F1_derivative_BIS(y_current,j,R,R1,R2,
                                                                       N,
                                                                       Ng,w_GL_minus1_plus1,
                                                                       h_atom_1,
                                                                       atomic_weights_atom_2_current,
                                                                       radial_grid,
                                                                       grid_points_atoms)
                                                                   
                    
                    if (F1_Tilde_current_n_j>=0):
                        print('ERROR : derivative of F1(r_(1,j)='+str(radial_grid[0][j])+',y_n='+str(y_current)+')^{k='+str(k)+'} = '+str(F1_Tilde_current_n_j)+ ' >=0 !!')
                        print(' ')
                        
                    #print('F1_derivative(r_(1,'+str(j)+'),y_n)^{'+str(k)+'} = F1_Tilde_current_n_j = '+str(F1_Tilde_current_n_j))
                    #print('F1_current_n_j  = '+str(F1_current_n_j))
                    
                    y_previous = y_current
                    
                    ## BEWARE : CHECK that y_current > 0 ! 
                    # (Ex : if y0 too large => y1 = intersection of the tangent to F(r_(1,j),.) and the x axis
                    # can be in the negative values => y1 < 0 ... (=> divergence of the Newton algorithm afterwards)
                    
                    # NEWTON ITERATION : y_(k+1)= y_k - F(y_k)/F'(y_k)
                    y_next_candidate = (y_current -  F1_current_n_j/F1_Tilde_current_n_j)
                    
                    if (y_next_candidate>0):
                        
                        y_current = y_next_candidate

                        test=abs(y_current-y_previous)
                        
                        iter_nb += 1
                    
                    # OTHERWISE : change the initial guess (decrease y0)
                    else:
                        
                        #print('Initial guess y0 (or y_current) too large : decrease y0 or y_current (divide by 2) ')
                        # We relaunch from a guess inside the interval :
                        y_current=(y_iterate_min_F1_neg+y_iterate_max_F1_pos)/2.
                       
                        
                    ####################
                    # Cases of non-convergence (or not quick enough)
                    
                    if ((iter_nb>=20) & (test>1e-4)):
                        # We relaunch from a guess inside the interval :
                        y_current=(y_iterate_min_F1_neg+y_iterate_max_F1_pos)/2.
                        
                        if (abs(y_iterate_max_F1_pos-y_iterate_min_F1_neg)<1e-10):
                            print('ALERT ! y_iterate_max_F1_pos = y_iterate_min_F1_neg = '+str(y_iterate_min_F1_neg))
                            print(' ')
                            break
                        
                        #break
                        
                    if (iter_nb>=100):
                        print('WARNING : NEWTON algo. (F1) UNCONVERGED for F1(r_(1,j='+str(j)+')=+'+str(radial_grid[0][j])+',.)^{k='+str(k)+'} in 100 iter !!!')
                        print('test = abs(y_current-y_previous) = '+str(test))
                        print('y_current = '+str(y_current))
                        print('y_previous = '+str(y_previous))
                        print('F1_current_n_j = '+str(F1_current_n_j))
                        print('---------------------------------')
                        break
                    
                ####################
                # Set w_(1,j)^(k+1)  = y* the root of F1 found by the Newton method
                atomic_weights_atom_1_current[j] = y_current
                # Mark the root found with the Newton algorithm with a special color :
                #matplotlib.pyplot.scatter(y_current,0,'bo')

                # END NEWTON algorithm
                ###########################################################
                
                #matplotlib.pyplot.show()
             
        print('atomic_weights_atom_1_current ITERATION k ='+str(k))
        print(atomic_weights_atom_1_current)
        print(' ')
        
        
        ##############################################
        # Computation of w_2^(k+1) from w_1^(k+1)

        # Saving w_2^(k)(.) (for evaluating convergence / ||w_2^(k+1)-w_2^(k)||):
        atomic_weights_atom_2_previous = np.zeros(len(radial_grid[1]))
        
        for i in range(len(radial_grid[1])):
            atomic_weights_atom_2_previous[i] = atomic_weights_atom_2_current[i]
        
        
        # w_(2,1)^(k+1) = rho(\vec{R2})-w_1^(k+1)(R=|r2-r1|)
        
        # w_(2,1)^(k+1) = rho(R2)-w_1^(k+1)(R)  where R=|R2-R1|
        atomic_weights_atom_2_current[0]= density_R2 - interpolate_P1(atomic_weights_atom_1_current,radial_grid[0],R) 

        # For 2 ≤ j ≤ N , define the discretized F2 function at iteration k and point r_(2,j)
        #as well as its derivative
        # F2(r,y) = int_{|R-r| .. R+r} (h1(r,s)/(w1(s)+y)) ds  - 1
        # Discretized at point r_(2,j) at iteration k by :
        # F2(r_(2,j),y)^{k} = -1 + L_(2,j) * sum_{l=1..Ng} (w_l * (h_(2,j,l)/(w_1^(k+1)(s_(2,j,l))+y)))
        # where L_(2,j) = R+r_(2,j) - |R-r_(2,j)|
        
        # w_1^(k+1)(s_(2,j,l)) is evaluated NOT
        #at a natural grid point for w_1^(k+1)(.) in its P1 representation
        # => need for interpolation
        
        # F2(r_(2,j),y=0)^{k}
      
        # atomic_weights_atom_1_current --> w_1^(k+1)
            
        F2_current=[discretized_F2_BIS(0,j,R,R1,R2,
                                       N,
                                       Ng,w_GL_minus1_plus1,
                                       h_atom_2,
                                       atomic_weights_atom_1_current,
                                       radial_grid,
                                       grid_points_atoms) for j in range(1,N)]

        # If F2(r_(1,j),0)^{k} <=  0 : set w_(2,j)^(k+1)  = 0
        for j in range(1,N):
           
            
            ###########################################################
            # NEWTON algorithm to find w_2^(k+1)(r_(2,j)) as a zero of y --> F2^(k)(r_(2,j),y)
            
            
            if (F2_current[j-1] <= 0):
                
                print('F2(r_(2,'+str(j)+'),0)^{'+str(k)+'} <=0')
                print(' ')
                atomic_weights_atom_2_current[j]=0
                
            # Else : if F2(r_(2,j),0)^{k} > 0 => find the unique root y* of y --> F2(r_(1,j),y)^{k} 
            # by the Newton algorithm :
            # y0 = w_(2,j)^(k)
            # y_(n+1) = y_n - F1(r_(1,j),y_n)^{k} / F1'(r_(1,j),y_n)^{k} 
            else :
                
                ############
                # To proceed by dichotomy in case of non-convergence : always keep trakc
                # of the largest iterate y_k such that F1(r_(1,j),y_k) > 0 
                # and of the smallest iterate y_h such that F1(r_(1,j),y_h) < 0 
                # As y --> F1(r_(1,j),y) is strictly decreasing : the solution is inside [y_k,y_l]
                # => draw a new guess in this interval
                
                y_iterate_max_F2_pos = 0
            
                y_iterate_min_F2_neg = 1000
                ###########   
            
                #INITIAL GUESS FOR THE NEWTON ALGORITHM (looking for zero of F2(r_(2,j),.)):
                # y0 > 0
                y0 = atomic_weights_atom_2_current[j]
                y_current=y0
                
                test=10000
                
                F2_current_n_j = 1000000

                iter_nb=0

                # Newton loop to find a zero of F2(r_(2,j),.)
                while ((abs(F2_current_n_j)>1e-6) or (test>1e-4)):
                    
                    F2_current_n_j=discretized_F2_BIS(y_current,j,R,R1,R2,
                                                      N,
                                                      Ng,w_GL_minus1_plus1,
                                                      h_atom_2,
                                                      atomic_weights_atom_1_current,
                                                      radial_grid,
                                                      grid_points_atoms)
                    
                    # Actualize the interval for dichotomy
                    if (F2_current_n_j >0):
                        if (y_current > y_iterate_max_F2_pos):
                            y_iterate_max_F2_pos = y_current
                            
                    elif (F2_current_n_j < 0):
                        if (y_current < y_iterate_min_F2_neg):
                            y_iterate_min_F2_neg = y_current      
                    
                    # F2'(r_(2,j),y_n)^{k} [partial derivative with respect to y, evaluated at y=0]
                    F2_Tilde_current_n_j=discretized_F2_derivative_BIS(y_current,j,R,R1,R2,
                                                                       N,
                                                                       Ng,w_GL_minus1_plus1,
                                                                       h_atom_2,
                                                                       atomic_weights_atom_1_current,
                                                                       radial_grid,
                                                                       grid_points_atoms)
                                                                       
            
                    if (F2_Tilde_current_n_j>=0):
                        print('ERROR : derivative of F2(r_(2,j)='+str(radial_grid[1][j])+',y_n='+str(y_current)+')^{k='+str(k)+'} = '+str(F2_Tilde_current_n_j)+ ' >=0 !!')
                    
                    #print('F2_derivative(r_(2,'+str(j)+'),y_n)^{'+str(k)+'} = F2_Tilde_current_n_j = '+str(F2_Tilde_current_n_j))
                    #print('F2_current_n_j  = '+str(F2_current_n_j))

                    iter_nb += 1
                    y_previous = y_current
                    
                    ## BEWARE : CHECK that y_current > 0 ! 
                    # (Ex : if y0 too large => y1 = intersection of the tangent to F(r_(1,j),.) and the x axis
                    # can be in the negative values => y1 < 0 ... (=> divergence of the Newton algorithm afterwards)
                    
                    if ((y_current -  F2_current_n_j/F2_Tilde_current_n_j)>0):
                     
                        y_current = y_current -  F2_current_n_j/F2_Tilde_current_n_j
                    
                        ## print('y_current= '+str(y_current))

                        test=abs(y_current-y_previous)
                    
                        iter_nb+=1
                        
                    # OTHERWISE : change the initial guess (decrease y0)
                    else:
                        #print('Initial guess y0 too large : decrease y0 (divide by 2) ')
                        
                        # We relaunch from a guess inside the interval :
                        y_current=(y_iterate_min_F2_neg+y_iterate_max_F2_pos)/2.
                        iter_nb+=1
                       
                                            
                    if ((iter_nb>=20) & (test>1e-4)):

                        # We relaunch from a guess inside the interval :
                        y_current=(y_iterate_min_F2_neg+y_iterate_max_F2_pos)/2.
                        iter_nb+=1
                        
                    if (iter_nb>=100):
                        print('WARNING : NEWTON algo. (F2) UNCONVERGED for F2(r_(2,j='+str(j)+')=+'+str(radial_grid[1][j])+',.)^{k='+str(k)+'} in 100 iter !!!')
                        print('test = abs(y_current-y_previous) = '+str(test))
                        print('y_current = '+str(y_current))
                        print('y_previous = '+str(y_previous))
                        print('F2_current_n_j = '+str(F2_current_n_j))
                        print('---------------------------------')
                        break
                        
                        #break
               
                # END NEWTON algorithm
                ###########################################################

                # Set w_(2,j)^(k+1)  = y* the root of F2 found by the Newton method
                atomic_weights_atom_2_current[j] = y_current
        
        print('atomic_weights_atom_2_current ITERATION k ='+str(k))
        print(atomic_weights_atom_2_current)
        print(' ')
        
        diff_atomic_weights_1 = [atomic_weights_atom_1_current[i]-atomic_weights_atom_1_previous[i] for i in range(len(radial_grid[0]))]

        diff_atomic_weights_2 = [atomic_weights_atom_2_current[i]-atomic_weights_atom_2_previous[i] for i in range(len(radial_grid[1]))]
        
        test_loop = np.linalg.norm(diff_atomic_weights_1)+np.linalg.norm(diff_atomic_weights_2)
        # [or a weighted norm with Gauss-Legendre weights]
        
        print('MAIN LOOP DIATOMIC-ISA iteration k = '+str(k))
        print(test_loop)
        print(' ---------------------------------------------  ')
        k+=1
       
    ##################################################
    
    print('CONVERGENCE ACHIEVED')
    print('test_loop = '+str(test_loop))
    
    print("atomic_weights_atom_1_current")
    print(atomic_weights_atom_1_current)
    print("atomic_weights_atom_2_current")
    print(atomic_weights_atom_2_current)
    
    # PLOT w_a(.) on a much finer grid than the grid used for the computation 
    # (thanks to interpolation of w_a(.) in-between)
    
    atomic_weights_atom_1_current_FINAL = [interpolate_P1(atomic_weights_atom_1_current, radial_grid[0], tab_r_plot_FINAL[k]) for k in range(len(tab_r_plot_FINAL))]
    atomic_weights_atom_2_current_FINAL = [interpolate_P1(atomic_weights_atom_2_current, radial_grid[1], tab_r_plot_FINAL[k]) for k in range(len(tab_r_plot_FINAL))]

    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) ISA-Newton NEW method")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,atomic_weights_atom_1_current_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,atomic_weights_atom_1_current_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"w_1_DIATOMIC_NEWTON_NEW_N_radial_"+str(N)+"_Ng_angular_"+str(Ng)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) ISA-Newton NEW method")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,atomic_weights_atom_2_current_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,atomic_weights_atom_2_current_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"w_2_DIATOMIC_NEWTON_NEW_N_radial_"+str(N)+"_Ng_angular_"+str(Ng)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')
    
    matplotlib.pyplot.show()

    # PLOT r--> 4*pi*r²*wa(r)
   
    w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * atomic_weights_atom_1_current_FINAL[i]) for i in range(len(tab_r_plot_FINAL))]
    w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * atomic_weights_atom_2_current_FINAL[i]) for i in range(len(tab_r_plot_FINAL))]
    
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_1(r) [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) ISA-Newton NEW method (interpolated) ")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_DIATOMIC_NEWTON_NEW_w_1_times_4pir2_N_radial_"+str(N)+"_Ng_angular_"+str(Ng)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",box_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_2(r)  [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) ISA-Newton NEW method (interpolated)")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"ISA_DIATOMIC_NEWTON_NEW_w_2_times_4pir2_N_radial_"+str(N)+"_Ng_angular_"+str(Ng)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",box_inches='tight')

    matplotlib.pyplot.show()
    
    return atomic_weights_atom_1_current, atomic_weights_atom_2_current, w_1_times_r2_values_FINAL, w_2_times_r2_values_FINAL
################################################################
################################################################


################################################################
################################################################
# AUXILIARY FUNCTIONS :

    

################################################################
"""
- x_GL = t_grid : Gauss-Legendre points in [-1,1]

Computes the points {\vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l) for the 2 atoms
"""
def compute_grid_points_around_atoms(N,Ng,t_grid,
                                     R,
                                     R1,R2,
                                     radial_grid):
                                     
    
    grid_points_around_atoms=[]
    
    # s[a] = (s_(a,j,l)_(j,l))
    
    ############################
    # Using Gauss-Legendre points in [-1,1] :
    # when reading GL weights from tables (up to n=64): https://pomax.github.io/bezierinfo/legendre-gauss.html

    
    # Points {\vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l)
    # j=0 : r_(a,j=0) = 0 => skip j=0 and start at j=1.
    grid_points_around_atom_1 =[ [ [ radial_grid[0][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R1 + radial_grid[0][j]*t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
       
    grid_points_around_atom_2 =[ [ [radial_grid[1][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R2 + radial_grid[1][j] *t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
    
    grid_points_around_atoms.append(grid_points_around_atom_1)
    
    grid_points_around_atoms.append(grid_points_around_atom_2)
    
    return grid_points_around_atoms
################################################################


################################################################
"""
# Precomputation of the density (functions h_a(.)) at the integration
# points (r_(a,j),s_(a,j,k)) :
    { (\vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z))}_(a \in Atoms,j,l)
Here \vec{R_a} = R_a * e_z in the diatomic case

- R : interatomic distance (R=|R2-R1| if the 2 atoms are at R1*e_z and R2*e_z along z axis)

- radial_grid =[[r_(a,1),...,r_(a,N),r_(a,N+1)]_{a \Atoms}]
- t_grid : GL legendre grid on [-1,1] of size Ng computed previously.

Requires 2(N-1)Ng evaluations of the molecular density rho(.)
"""
def precomputation_function_h_a_BIS(N,Ng,R,R1,R2,radial_grid,
                                    t_grid,
                                    density_matrix_coefficient_pGTOs,
                                    position_nuclei_associated_basis_function_pGTO,
                                    QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                    nb_contracted_GTOs,
                                    total_nb_primitive_GTOs,
                                    correspondence_basis_pGTOs_atom_index,
                                    angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                    correspondence_index_contracted_GTOs_primitive_GTOs,
                                    tab_correspondence_index_primitive_GTOs_primitive_shells,
                                    tab_correspondence_index_contracted_GTOs_contracted_shells,
                                    contraction_coefficients_pSHELLS_to_cSHELLS,
                                    atomic_coordinates):
    
    

    # ((rho_(a,j,l))_(j,l))_{a \in Atoms}

    # Points {\vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l)
    # j=0 : r_(a,j=0) = 0 => skip j=0 and start at j=1.
    grid_points_3D_atom_1 =[ [ [ radial_grid[0][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R1 + radial_grid[0][j]*t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
    grid_points_3D_atom_1_table = [ [radial_grid[0][j] * math.sqrt(1-t_grid[l]**2) for l in range(Ng) for j in range(1,N)],
                                    [R1 + radial_grid[0][j] * t_grid[l] for l in range(Ng) for j in range(1,N)] ]
    
       
       
    grid_points_3D_atom_2 =[ [ [radial_grid[1][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R2 + radial_grid[1][j] *t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
    grid_points_3D_atom_2_table =[ [radial_grid[1][j] *   math.sqrt(1-t_grid[l]**2) for l in range(Ng) for j in range(1,N)],
                                   [R2 + radial_grid[1][j] * t_grid[l] for l in range(Ng) for j in range(1,N)]]
    
    #####################
    # Plot of these grid points in the plane (Oxz) :
    # z as abscisses and x as ordonnees
    
    matplotlib.pyplot.figure(figsize=(10, 8), dpi=400)
    
    matplotlib.pyplot.xlim(-1,2.)

    matplotlib.pyplot.ylim(0,4)
    
    matplotlib.pyplot.xlabel('z (Å)')

    matplotlib.pyplot.ylabel(' x (Å) ')

    matplotlib.pyplot.title(" Discretized points for molecular density precomputation ")

    matplotlib.pyplot.scatter(atomic_coordinates[0][2],0, c = 'red',marker='x')

    matplotlib.pyplot.scatter(atomic_coordinates[1][2],0, c = 'red',marker='x')

    matplotlib.pyplot.scatter(grid_points_3D_atom_1_table[1],grid_points_3D_atom_1_table[0])
    
    matplotlib.pyplot.scatter(grid_points_3D_atom_2_table[1],grid_points_3D_atom_2_table[0])

    matplotlib.pyplot.show()

    
    h_atom_1 = [ [compute_density_grid_point(grid_points_3D_atom_1[j-1][l],
                                             QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                             density_matrix_coefficient_pGTOs,
                                             position_nuclei_associated_basis_function_pGTO,
                                             nb_contracted_GTOs,
                                             total_nb_primitive_GTOs,
                                             correspondence_basis_pGTOs_atom_index,
                                             angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                             correspondence_index_contracted_GTOs_primitive_GTOs,
                                             tab_correspondence_index_primitive_GTOs_primitive_shells,
                                             tab_correspondence_index_contracted_GTOs_contracted_shells,
                                             contraction_coefficients_pSHELLS_to_cSHELLS) for l in range(Ng)] for j in range(1,N)]
    
    
    h_atom_2 = [ [compute_density_grid_point(grid_points_3D_atom_2[j-1][l],
                                             QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                             density_matrix_coefficient_pGTOs,
                                             position_nuclei_associated_basis_function_pGTO,
                                             nb_contracted_GTOs,
                                             total_nb_primitive_GTOs,
                                             correspondence_basis_pGTOs_atom_index,
                                             angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                             correspondence_index_contracted_GTOs_primitive_GTOs,
                                             tab_correspondence_index_primitive_GTOs_primitive_shells,
                                             tab_correspondence_index_contracted_GTOs_contracted_shells,
                                             contraction_coefficients_pSHELLS_to_cSHELLS) for l in range(Ng)] for j in range(1,N)]
    
    # Precompute also rho(\vec(R1)) and rho(\vec(R2)) :
    density_R1=compute_density_grid_point([0,0,R1],
                                               QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                               density_matrix_coefficient_pGTOs,
                                               position_nuclei_associated_basis_function_pGTO,
                                               nb_contracted_GTOs,
                                               total_nb_primitive_GTOs,
                                               correspondence_basis_pGTOs_atom_index,
                                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                               correspondence_index_contracted_GTOs_primitive_GTOs,
                                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                                               contraction_coefficients_pSHELLS_to_cSHELLS)
     
    density_R2=compute_density_grid_point([0,0,R2],
                                               QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                               density_matrix_coefficient_pGTOs,
                                               position_nuclei_associated_basis_function_pGTO,
                                               nb_contracted_GTOs,
                                               total_nb_primitive_GTOs,
                                               correspondence_basis_pGTOs_atom_index,
                                               angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                               correspondence_index_contracted_GTOs_primitive_GTOs,
                                               tab_correspondence_index_primitive_GTOs_primitive_shells,
                                               tab_correspondence_index_contracted_GTOs_contracted_shells,
                                               contraction_coefficients_pSHELLS_to_cSHELLS)
    
    return h_atom_1, h_atom_2, density_R1, density_R2
################################################################


################################################################
"""
Used for test densities only (not originating from a quantum density as input)
"""
def precomputation_function_h_a_TEST_2_GAUSSIANS(N,Ng,R,R1,R2,radial_grid,
                                                t_grid,atomic_coordinates,
                                                tab_alpha):
    
    

    # ((rho_(a,j,l))_(j,l))_{a \in Atoms}

    # Points {\vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l)
    # j=0 : r_(a,j=0) = 0 => skip j=0 and start at j=1.
    grid_points_3D_atom_1 =[ [ [ radial_grid[0][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R1 + radial_grid[0][j]*t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
    grid_points_3D_atom_1_table = [ [radial_grid[0][j] * math.sqrt(1-t_grid[l]**2) for l in range(Ng) for j in range(1,N)],
                                    [R1 + radial_grid[0][j] * t_grid[l] for l in range(Ng) for j in range(1,N)] ]
    
       
       
    grid_points_3D_atom_2 =[ [ [radial_grid[1][j]*math.sqrt(1-t_grid[l]**2),
                                0,
                                R2 + radial_grid[1][j] *t_grid[l] ]  for l in range(Ng)] for j in range(1,N)]
    
    grid_points_3D_atom_2_table =[ [radial_grid[1][j] *   math.sqrt(1-t_grid[l]**2) for l in range(Ng) for j in range(1,N)],
                                   [R2 + radial_grid[1][j] * t_grid[l] for l in range(Ng) for j in range(1,N)]]
    
    #####################
    # Plot of these grid points in the plane (Oxz) :
    # z as abscisses and x as ordonnees
    
    matplotlib.pyplot.figure(figsize=(10, 8), dpi=400)
    
    matplotlib.pyplot.xlim(-1,2.)

    matplotlib.pyplot.ylim(0,4)
    
    matplotlib.pyplot.xlabel('z (Å)')

    matplotlib.pyplot.ylabel(' x (Å) ')

    matplotlib.pyplot.title(" Discretized points for molecular density precomputation ")

    matplotlib.pyplot.scatter(atomic_coordinates[0][2],0, c = 'red',marker='x')

    matplotlib.pyplot.scatter(atomic_coordinates[1][2],0, c = 'red',marker='x')

    matplotlib.pyplot.scatter(grid_points_3D_atom_1_table[1],grid_points_3D_atom_1_table[0])
    
    matplotlib.pyplot.scatter(grid_points_3D_atom_2_table[1],grid_points_3D_atom_2_table[0])

    matplotlib.pyplot.show()

    #################
    # TEST DENSITY 1 :
    """
    values_rho_around_atom_1 = [ [ exp_centered(grid_points_3D_atom_1[j-1][l],atomic_coordinates[0],atomic_coordinates[1],tab_alpha[0],tab_alpha[1]) for l in range(Ng)] for j in range(1,N)]
    
    values_rho_around_atom_2 = [ [ exp_centered(grid_points_3D_atom_2[j-1][l],atomic_coordinates[1],atomic_coordinates[0],tab_alpha[1],tab_alpha[0]) for l in range(Ng)] for j in range(1,N)]

    # Precompute also rho(\vec(R1)) and rho(\vec(R2)) :
    density_R1= exp_centered([0,0,R1],atomic_coordinates[0],atomic_coordinates[1],tab_alpha[0],tab_alpha[1])

    density_R2= exp_centered([0,0,R2],atomic_coordinates[1],atomic_coordinates[0],tab_alpha[1],tab_alpha[0])
    """
    #################
     
    #################
    # TEST DENSITY 2 :
    values_rho_around_atom_1 = [ [ sum_gaussians_TEST_DENSITY(grid_points_3D_atom_1[j-1][l],atomic_coordinates[0],atomic_coordinates[1],tab_alpha[0],tab_alpha[1]) for l in range(Ng)] for j in range(1,N)]
    
    values_rho_around_atom_2 = [ [ sum_gaussians_TEST_DENSITY(grid_points_3D_atom_2[j-1][l],atomic_coordinates[1],atomic_coordinates[0],tab_alpha[1],tab_alpha[0]) for l in range(Ng)] for j in range(1,N)]

    density_R1= sum_gaussians_TEST_DENSITY([0,0,R1],atomic_coordinates[0],atomic_coordinates[1],tab_alpha[0],tab_alpha[1])
    
    density_R2= sum_gaussians_TEST_DENSITY([0,0,R2],atomic_coordinates[1],atomic_coordinates[0],tab_alpha[1],tab_alpha[0])
    
    return values_rho_around_atom_1, values_rho_around_atom_2, density_R1, density_R2
################################################################



################################################################
"""
- x_GL, w_GL : Gauss-Legendre points / weights on [-1,1]
- r_(a,j,l) = |(Ra-Rb)²+r_j²+2*r_j*(Ra-Rb)*t_l   where t_l = x_GL[l]

- h_atom : precomputed rho(.) [molecular density] on 
  points {\vec{R_a} +r_(1,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l)
  [in (Oxz) plane]
  h_atom_1 = ((rho_(1,j,l))_(j,l))
  
- atomic_weights_current_atom = { w_a(r_(a,j))}_[j=0..N] 
 (atomic weights for atom a known on a radial grid 'radial_grid_atom' specific to atom a)

 - grid_points_atoms : obtained previously as :
     compute_grid_points_around_atoms(N,Ng,R,
                                     R1,R2,
                                     radial_grid,t_grid=x_GL)

 - Called with radial_grid_atom = radial_grid[0] (radial grid of 1st atom)

 - Called only for j>=1
"""
def discretized_F1_BIS(y,j,R,R1,R2,
                       N,
                       Ng,w_GL,
                       h_atom_1,
                       atomic_weights_atom_2_current,
                       radial_grid,
                       grid_points_atoms):

    # || \vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) - \vec{R_b} || = norm of grid_points_atoms_translated[j-1][l]
    grid_points_atom_1_translated=[ [grid_points_atoms[0][j-1][l][0],
                                     grid_points_atoms[0][j-1][l][1],
                                     grid_points_atoms[0][j-1][l][2]-R2] for l in range(Ng)] 
    
    # h_atom : of length (N-1) (1<=j<=(N-1))
    sum_integrand = np.sum([w_GL[l]*h_atom_1[j-1][l]/(y+interpolate_P1(atomic_weights_atom_2_current,radial_grid[1],np.linalg.norm(grid_points_atom_1_translated[l]))) for l in range(Ng)])

    return -1 + 0.5*sum_integrand
                                
################################################################           

################################################################
"""
 - Called only for j>=1
"""
def discretized_F2_BIS(y,j,R,R1,R2,
                       N,
                       Ng,w_GL,
                       h_atom_2,
                       atomic_weights_atom_1_current,
                       radial_grid,
                       grid_points_atoms):

    # || \vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) - \vec{R_b} || = norm of grid_points_atoms_translated[j-1][l]
    grid_points_atom_2_translated=[ [grid_points_atoms[1][j-1][l][0],
                                     grid_points_atoms[1][j-1][l][1],
                                     grid_points_atoms[1][j-1][l][2]-R1] for l in range(Ng)]
    
    # h_atom : of length (N-1) (1<=j<=(N-1))
    sum_integrand = np.sum([w_GL[l]*h_atom_2[j-1][l]/(y+interpolate_P1(atomic_weights_atom_1_current,radial_grid[0],np.linalg.norm(grid_points_atom_2_translated[l]))) for l in range(Ng)])

    return -1 + 0.5*sum_integrand
                                
################################################################  

################################################################
"""
 - Always called for j >=1 
- r_(a,j,l) = |(Ra-Rb)²+r_j²+2*r_j*(Ra-Rb)*t_l   where t_l = x_GL[l]

- h_atom : precomputed rho(.) [molecular density] on 
  points {\vec{R_a} +r_(1,j)*( sqrt(1-t_l²) e_x + t_l * e_z) }_(j,l)
  [in (Oxz) plane]
  h_atom_1 = ((rho_(1,j,l))_(j,l))
  
- atomic_weights_current_atom = { w_a(r_(a,j))}_[j=0..N] 
 (atomic weights for atom a known on a radial grid 'radial_grid_atom' specific to atom a)

 - grid_points_atoms : obtained previously as :
     compute_grid_points_around_atoms(N,Ng,R,
                                     R1,R2,
                                     radial_grid,t_grid=x_GL)

 - radial_grid[1] : where w_2^(k)(.) is known
  => Then : interpolated on r_(1,j,l), a grid point associated to atom 1

"""
def discretized_F1_derivative_BIS(y,j,R,R1,R2,
                                  N,
                                  Ng,w_GL,
                                  h_atom_1,
                                  atomic_weights_atom_2_current,
                                  radial_grid,
                                  grid_points_atoms):

    
    # || \vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) - \vec{R_b} || = norm of grid_points_atoms_translated[j-1][l]
    grid_points_atom_1_translated=[[ [grid_points_atoms[0][j-1][l][0],
                                     grid_points_atoms[0][j-1][l][1],
                                     grid_points_atoms[0][j-1][l][2]-R2] for l in range(Ng)] for j in range(1,N)]
    
    # h_atom : of length (N-1) (1<=j<=(N-1))
    sum_integrand = np.sum([w_GL[l]*h_atom_1[j-1][l]/(y+interpolate_P1(atomic_weights_atom_2_current,radial_grid[1],np.linalg.norm(grid_points_atom_1_translated[j-1][l])))**2 for l in range(Ng)])

    return -0.5*sum_integrand
                                
################################################################           


################################################################
"""
 - Always called for j >=1 
 - radial_grid[0] : where w_1^(k+1)(.) is known
  => Then : interpolated on s_(2,j,l), a grid point associated to atom 2
"""
def discretized_F2_derivative_BIS(y,j,R,R1,R2,
                                  N,
                                  Ng,w_GL,
                                  h_atom_2,
                                  atomic_weights_atom_1_current,
                                  radial_grid,grid_points_atoms):

    # || \vec{R_a} +r_(a,j)*( sqrt(1-t_l²) e_x + t_l * e_z) - \vec{R_b} || = norm of grid_points_atoms_translated[j-1][l]
    grid_points_atom_2_translated=[[ [grid_points_atoms[1][j-1][l][0],
                                     grid_points_atoms[1][j-1][l][1],
                                     grid_points_atoms[1][j-1][l][2]-R1] for l in range(Ng)] for j in range(1,N)]
    
    # h_atom : of length (N-1) (1<=j<=(N-1))
    sum_integrand = np.sum([w_GL[l]*h_atom_2[j-1][l]/(y+interpolate_P1(atomic_weights_atom_1_current,radial_grid[0],np.linalg.norm(grid_points_atom_2_translated[j-1][l])))**2 for l in range(Ng)])

    return -0.5*sum_integrand
 ################################################################           

                                
################################################################


################################################################
"""
 - Returns 0.5*<rho>_1(r) and 0.5*<rho>_2(r) for the initial guess 
   of w_1(.)^(0) and w_2(.)^(0) [half the average molecular density at distance R]
   => allows to have F1^(0)(r_(1,j),y=0) > 0 for all values of r_(1,j)
   [otherwise as y --> F1^(0)(r_(1,j),y) is strictly decreasing, we have w1(r_(1,j))=0 ...]
 - x_GL, w_GL :  Gauss-Legendre points and weights in [0,1]
 - R_atom = R1 or R2 depending on atom 1 or 2 (position of atom along z axis)
 - Ng_avg = number of Gauss-Legendre points used to compute the spherical average of
   the molecular density around this atom (1 or 2) : can be different from Ng
   = the number of GL points used elsewhere for other quadrature in the algorithm.
"""
def average_rho_molecular_diatomic_atom(r,R_atom,Ng_avg,
                                        QM_data,boolean_cartesian_D_shells,
                                        boolean_cartesian_F_shells,
                                        density_matrix_coefficient_pGTOs,
                                        position_nuclei_associated_basis_function_pGTO,
                                        nb_contracted_GTOs,
                                        total_nb_primitive_GTOs,
                                        correspondence_basis_pGTOs_atom_index,
                                        angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                        correspondence_index_contracted_GTOs_primitive_GTOs,
                                        tab_correspondence_index_primitive_GTOs_primitive_shells,
                                        tab_correspondence_index_contracted_GTOs_contracted_shells,
                                        contraction_coefficients_pSHELLS_to_cSHELLS):

    x_GL_0_1, w_GL_0_1 = lgwt(Ng_avg,0,1)
    
    grid_points_3D_atom =[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                            0,
                            R_atom + r * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_avg)]
    
    
    # Molecular density evaluated at the grid points at distance r from atom 1 (position R1)
    # (with theta [of spherical coordinates] following a Gauss-Legendre grid)
    density_grid_points_3D_atom = [ compute_density_grid_point(grid_points_3D_atom[l],
                                                                         QM_data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,
                                                                         density_matrix_coefficient_pGTOs,
                                                                         position_nuclei_associated_basis_function_pGTO,
                                                                         nb_contracted_GTOs,
                                                                         total_nb_primitive_GTOs,
                                                                         correspondence_basis_pGTOs_atom_index,
                                                                         angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,
                                                                         correspondence_index_contracted_GTOs_primitive_GTOs,
                                                                         tab_correspondence_index_primitive_GTOs_primitive_shells,
                                                                         tab_correspondence_index_contracted_GTOs_contracted_shells,
                                                                         contraction_coefficients_pSHELLS_to_cSHELLS) for l in range(Ng_avg)]
    
    integral_approx_average_atom = np.sum([w_GL_0_1[l]*density_grid_points_3D_atom[l]*math.sin(math.pi*x_GL_0_1[l]) for l in range(Ng_avg) ])

    return (math.pi/4.)*integral_approx_average_atom 
################################################################

################################################################
def average_rho_molecular_diatomic_atom_TEST(r,R_atom,index_atom,
                                             Ng_avg,
                                             atomic_coordinates,
                                             tab_alpha):

    x_GL_0_1, w_GL_0_1 = lgwt(Ng_avg,0,1)
    
    grid_points_3D_atom =[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                            0,
                            R_atom + r * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_avg)]
    
    index_other_atom=abs(1-index_atom)
    
    # Molecular density evaluated at the grid points at distance r from atom 1 (position R1)
    # (with theta [of spherical coordinates] following a Gauss-Legendre grid)
    density_grid_points_3D_atom = [ exp_centered(grid_points_3D_atom[l],atomic_coordinates[index_atom],atomic_coordinates[index_other_atom],tab_alpha[index_atom],tab_alpha[index_other_atom])  for l in range(Ng_avg)]
    
    integral_approx_average_atom = np.sum([w_GL_0_1[l]*density_grid_points_3D_atom[l]*math.sin(math.pi*x_GL_0_1[l]) for l in range(Ng_avg) ])

    return (math.pi/4.)*integral_approx_average_atom 
################################################################


################################################################
"""
Computes local atomic charges, dipoles and quadrupole from the converged w_a(.) functions
obtained by the ISA-Diatomic Newton procedure (endoded in 'value_atomic_weights_atoms').

      - values_rho[a] : values rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j=1..Ng_radial,l=1..Ng_theta} 
        where Rmax*x_GL_0_R[j] = r_j
        - x_GL_0_R, w_GL_0_R : Gauss Legendre points and weights on [0,1], then scaled by Rmax (--> \infty) to reach int_{0.. \infty} on the radial integral
"""
def compute_local_multipoles_ISA_DIATOMIC_NEWTON(value_atomic_weights_atoms,
                                                 values_rho,
                                                 Ng_theta,
                                                 x_GL_0_1, w_GL_0_1,
                                                 Ng_radial,
                                                 Rmax,
                                                 x_GL_0_R, w_GL_0_R,
                                                 radial_grid_DIATOMIC_Newton,
                                                 atomic_coordinates,
                                                 atomic_numbers,
                                                 logfile):
    
    partial_charges=[]

    atomic_dipoles_z = []

    atomic_quadrupoles = []
    
    print('--------------------')
    
    print('COMPUTATION ATOMIC MULTIPOLES ISA-DIATOMIC NEWTON or RADIAL in e.Ang^k (a.u.)')
    logfile.write('ATOMIC MULTIPOLES in e.Ang^k (a.u.)')
    logfile.write("\n")
    #print('ATOMIC CHARGES (k=0), DIPOLES (k=1) and QUADRUPOLES (k=2) in e.Ang^k (ATOMIC UNITS)')
    print(' ')
    
    for a in range(2):
        
        
        q_a = compute_atomic_charge_DIATOMIC_NEWTON(a,
                                                    radial_grid_DIATOMIC_Newton,
                                                    value_atomic_weights_atoms,
                                                    values_rho[a],
                                                    Ng_theta,
                                                    x_GL_0_1, w_GL_0_1,
                                                    Ng_radial,
                                                    Rmax,
                                                    x_GL_0_R, w_GL_0_R,
                                                    atomic_coordinates)
        
        partial_charges.append(q_a)
        
        print('q_'+str(a)+' = '+str(q_a))
        print('Partial charge atom '+str(a)+' = '+str(atomic_numbers[a]-q_a))
        logfile.write('q_'+str(a)+' = '+str(q_a))
        logfile.write("\n")
        logfile.write('Partial charge atom '+str(a)+' = '+str(atomic_numbers[a]-q_a))
        logfile.write("\n")
        
        d_z_a = compute_atomic_dipole_DIATOMIC_NEWTON(a,
                                                      radial_grid_DIATOMIC_Newton,
                                                      value_atomic_weights_atoms,
                                                      values_rho[a],
                                                      Ng_theta,
                                                      x_GL_0_1, w_GL_0_1,
                                                      Ng_radial,
                                                      Rmax,
                                                      x_GL_0_R, w_GL_0_R,
                                                      atomic_coordinates)
        
        atomic_dipoles_z.append(d_z_a)
        
        print('d_z_'+str(a)+' = '+str(d_z_a))
        logfile.write('d_z_'+str(a)+' = '+str(d_z_a))
        logfile.write("\n")
        
        Q_xx_a = compute_atomic_quadrupole_xx_DIATOMIC_NEWTON(a,
                                                              radial_grid_DIATOMIC_Newton,
                                                              value_atomic_weights_atoms,
                                                              values_rho[a],
                                                              Ng_theta,
                                                              x_GL_0_1, w_GL_0_1,
                                                              Ng_radial,
                                                              Rmax,
                                                              x_GL_0_R, w_GL_0_R,
                                                              atomic_coordinates)
        
        print('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write("\n")
     
        Q_zz_a = compute_atomic_quadrupole_zz_DIATOMIC_NEWTON(a,
                                                              radial_grid_DIATOMIC_Newton,
                                                              value_atomic_weights_atoms,
                                                              values_rho[a],
                                                              Ng_theta,
                                                              x_GL_0_1, w_GL_0_1,
                                                              Ng_radial,
                                                              Rmax,
                                                              x_GL_0_R, w_GL_0_R,
                                                              atomic_coordinates)
        
        print('Q_zz_'+str(a)+' = '+str(Q_zz_a))
        print(' ')
        logfile.write('Q_zz_'+str(a)+' = '+str(Q_zz_a))
        logfile.write("\n")
        logfile.write("\n") 
        
    print('Total charge : sum_a (q_a)')
    print(partial_charges[0]+partial_charges[1])
    print('--------------------')
    print(' ')
    logfile.write('Total charge : sum_a (q_a)')
    logfile.write("\n")
    logfile.write(str(partial_charges[0]+partial_charges[1]))
    logfile.write("\n")
    logfile.write('--------------------')
    
    return partial_charges, atomic_dipoles_z, atomic_quadrupoles
################################################################



################################################################
"""

Computes the atomic partial charge int_{R³} (rho_a(r) dr)

- x_GL_0_R, w_GL_0_R : GL grid obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : GL grid obtained previously as : lgwt(Ng_theta,0,1)

- a = atom index

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
 
Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid  radial_grid_DIATOMIC_Newton[a]
      
      Both values of w_1(.) and w_2(.) are needed to integrate rho_1 = (w_1/(w_1+w_2))*rho or rho_2 = (w_2/(w_1+w_2))*rho
      
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
      
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
     
    - values_rho_atom : values rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
       where Rmax*x_GL_0_R[j] = r_j (precomputed once and for all)
"""
def compute_atomic_charge_DIATOMIC_NEWTON(a,
                                          radial_grid_DIATOMIC_Newton,
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
    
    # 'compute_sph_avg_atomic_charge_DIATOMIC_NEWTON' returns a result in unit of rho(.) divided by unit of w_a(.)
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],x_GL_0_R_scaled[l]) * compute_sph_avg_atomic_charge_DIATOMIC_NEWTON(l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                a,
                                                                                                                                                                                                                radial_grid_DIATOMIC_Newton,
                                                                                                                                                                                                                values_w_a,
                                                                                                                                                                                                                values_rho_atom,
                                                                                                                                                                                                                x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                Ng_theta,
                                                                                                                                                                                                                atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom)* Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic dipole  int_{R³} (z * rho_a(r) dr) along z axis (DIATOMIC case : the 2 atoms are along z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_dipole_DIATOMIC_NEWTON(a,
                                          radial_grid_DIATOMIC_Newton,
                                          values_w_a,
                                          values_rho_atom,
                                          Ng_theta,
                                          x_GL_0_1, w_GL_0_1,
                                          Ng_radial,
                                          Rmax,
                                          x_GL_0_R, w_GL_0_R,
                                          atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**3 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**3 * x_GL_0_R_scaled[l]**3 * interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],x_GL_0_R_scaled[l])  * compute_sph_avg_atomic_dipole_DIATOMIC_NEWTON(l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                 a,
                                                                                                                                                                                                                 radial_grid_DIATOMIC_Newton,
                                                                                                                                                                                                                 values_w_a,
                                                                                                                                                                                                                 values_rho_atom,
                                                                                                                                                                                                                 x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                 Ng_theta,
                                                                                                                                                                                                                 atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic quadrupole  int_{R³} (xx * rho_a(r) dr)  =(Q_a)_{xx} = (Q_a)_{yy} by symmetry
(DIATOMIC case : the 2 atoms are along z ; invariance around z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_xx_DIATOMIC_NEWTON(a,
                                                 radial_grid_DIATOMIC_Newton,
                                                 values_w_a,
                                                 values_rho_atom,
                                                 Ng_theta,
                                                 x_GL_0_1, w_GL_0_1,
                                                 Ng_radial,
                                                 Rmax,
                                                 x_GL_0_R, w_GL_0_R,
                                                 atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)**3 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],x_GL_0_R_scaled[l]) * compute_sph_avg_atomic_quadrupole_xx_DIATOMIC_NEWTON(l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                       a,
                                                                                                                                                                                                                       radial_grid_DIATOMIC_Newton,
                                                                                                                                                                                                                       values_w_a,
                                                                                                                                                                                                                       values_rho_atom,
                                                                                                                                                                                                                       x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                       Ng_theta,
                                                                                                                                                                                                                       atomic_coordinates) for l in range(Ng_radial)]
    
    return math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic quadrupole  int_{R³} (xx * rho_a(r) dr)  =(Q_a)_{xx} = (Q_a)_{yy} by symmetry
(DIATOMIC case : the 2 atoms are along z ; invariance around z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_zz_DIATOMIC_NEWTON(a,
                                                 radial_grid_DIATOMIC_Newton,
                                                 values_w_a,
                                                 values_rho_atom,
                                                 Ng_theta,
                                                 x_GL_0_1, w_GL_0_1,
                                                 Ng_radial,
                                                 Rmax,
                                                 x_GL_0_R, w_GL_0_R,
                                                 atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)* cos(theta)**2 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],x_GL_0_R_scaled[l]) * compute_sph_avg_atomic_quadrupole_zz_DIATOMIC_NEWTON(l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                       a,
                                                                                                                                                                                                                       radial_grid_DIATOMIC_Newton,
                                                                                                                                                                                                                       values_w_a,
                                                                                                                                                                                                                       values_rho_atom,
                                                                                                                                                                                                                       x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                       Ng_theta,
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

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_charge_DIATOMIC_NEWTON(index_radial,r,
                                                  a,
                                                  radial_grid_DIATOMIC_Newton,
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
    
    # Coordinate of these points in Ang because we then interpolate 
    # with "interpolate_P1(values_w_a[b],radial_grid_DIATOMIC_Newton[b],x)"
    # with radial_grid_DIATOMIC_Newton[b] a radial grid in Angstrom, where the 
    # values of w_b(.) are known
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
    
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    
    # No need to convert the norm of grid_points_3D_atom_shifted_other_atom[l] in a.u. because
    # unit simplifies in the P1 interpolation
    
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],r)+interpolate_P1(values_w_a[b],radial_grid_DIATOMIC_Newton[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    # Result has unit of rho(.) divided by unit of w_a(.)
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_dipole_DIATOMIC_NEWTON(index_radial,r,
                                                  a,
                                                  radial_grid_DIATOMIC_Newton,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u]) /(interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],r)+interpolate_P1(values_w_a[b],radial_grid_DIATOMIC_Newton[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta)³ * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_xx_DIATOMIC_NEWTON(index_radial,r,
                                                         a,
                                                         radial_grid_DIATOMIC_Newton,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u])**3 /(interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],r)+interpolate_P1(values_w_a[b],radial_grid_DIATOMIC_Newton[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################



################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta)² * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- radial_grid_DIATOMIC_Newton[b] = radial grid on which w_b(.) was computed.
 DOES NOT coincide with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_zz_DIATOMIC_NEWTON(index_radial,r,
                                                         a,
                                                         radial_grid_DIATOMIC_Newton,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u])**2  /(interpolate_P1(values_w_a[a],radial_grid_DIATOMIC_Newton[a],r)+interpolate_P1(values_w_a[b],radial_grid_DIATOMIC_Newton[b],np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u])))  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################



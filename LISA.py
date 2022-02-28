#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:23:08 2021

@author: rbenda

Functions for the new GISA algorithm (L-ISA), minimizing the Kullback-Leibler entropy

"""
import numpy as np
import math
import matplotlib
from matplotlib import pyplot
import scipy.optimize

import cvxopt
#from cvxopt import div
#from cvxopt import matrix

from GISA import compute_basis_function_GISA
from GISA import compute_mass_rho_a_GISA
from GISA import compute_local_multipoles_GISA
from GISA import compute_sph_avg_integrand_total_mass
from GISA import compute_w_a_GISA
from GISA import convergence_criterion_GISA_LISA

ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'
ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'

############################
#CONVERSION CONSTANTS :
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
# For test densities (non-dimensonal, not originating from a QM code output)
#conversion_bohr_angstrom=1.
############################

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']


################################################################
################################################################
# FUNCTIONS FOR THE NEW GISA ALGORITHM (based on Kullback-Leibler entropy minimization) :
   
################################################################
"""
New mathematically-grounded GISA algorithm (L-ISA), based on :
    
    - Step 1 :
        minimization , for all a, of 
        Fa(c_a) = - int_{0..\infty} [ r² * wa(r) * log(sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) )] dr
        at fixed w_a(.) (w_a^(m)(.))
        yielding optimal coefficients (c_(a,k)^(m+1))_{a=1..nb_atoms, k=1..Ka}
    - Step 2 :
        Hirshfeld formula : rho_a(r) = rho(r) * (sum_{k=1..Ka} c_(a,k)^(m+1) *g_(a,k)(r) )/ (sum_b (sum_{k=1..Ka} c_(b,k)^(m+1) *g_(b,k)(r)))
        
- tab_K = {K_a}_{a=1..M} = number of shells per atom

- exponents_GISA=[([alpha_(a,1),...,alpha_(a,Ka)])_{a=1..M}] the exponents in the Gaussians
  defining the shape functions of the GISA method. These exponents are fixed in this method.
  These exponents are provided in the GISA original paper Verstraelen 2012.
  
- Rmax : maximal radial intergation limit in the integrals from 0 to \infty on r (radial)
"""
def LISA_algorithm(tab_K,
                   Ng_radial,
                   Ng_theta,
                   coeffs_c_init,
                   exponents_GISA,
                   Rmax,values_rho,
                   x_GL_0_1, w_GL_0_1,
                   x_GL_0_R, w_GL_0_R,
                   atomic_coordinates,
                   atomic_numbers,
                   logfile_LISA,
                   tab_r_plot_FINAL):
    
    nb_atoms = len(tab_K)
    
    coeffs_c=[]
    
    coeffs_c_a_list_steps=[]

    total_sum_Na_list_steps=[]

    mass_rho_a_list_steps = []

    ########################################
    # Initialisation : w_a^(0) : derived from the coefficients : (c_{a,k}^{(0)})_{a=1..M,k=1..K_a} 
    # For instance : all equal to 1
    
    for a in range(nb_atoms):
        
        matrix_coeffs_a = cvxopt.matrix([coeffs_c_init[a][k] for k in range(tab_K[a])])

        coeffs_c.append(matrix_coeffs_a)

    print('Initial coefficients (c_(a,k))_{a=1..M,k=1..Ka} :')
    print(coeffs_c)
    print(' ')
   
    # (N_a)_{a=1..M} : masses of the rho_a(.) 
    mass_rho_a = np.zeros(nb_atoms)
    
    ########################################
    
    
    ########################################
    print('Computing mass of (rho_a^(0))_a :')
    # N_a^(0) = int (rho_a^(0)) [mass of rho_a^(0)]:
    for a in range(nb_atoms):
        
        mass_rho_a[a] = compute_mass_rho_a_GISA(a,Rmax,Ng_radial,Ng_theta,
                                                coeffs_c_init,
                                                exponents_GISA,
                                                values_rho[a],
                                                x_GL_0_1, w_GL_0_1,
                                                x_GL_0_R, w_GL_0_R,
                                                atomic_coordinates)
                
        
        
    print('Initial mass of (rho_a^(0))_a ; using (w_a^(0))_a :')
    print(mass_rho_a)
    print(' ')
    print('Check total sum_a N_a :')
    total_sum_Na_init = np.sum(mass_rho_a)
    print(total_sum_Na_init)
    print(' ')
    
    total_sum_Na_list_steps.append(total_sum_Na_init)
    
    mass_rho_a_list_steps.append([])
    mass_rho_a_list_steps[0].append(mass_rho_a[0])
    mass_rho_a_list_steps[0].append(mass_rho_a[1])
    
    coeffs_c_a_list_steps.append(coeffs_c_init)
    ########################################
    
    ###########################
    # MAIN LOOP
    
    criterion_CV = 10000
    
    criterion_CV_list_step=[]
        
    total_entropy_list_step =[]
    
    nb_iter=1
    
    coeffs_c_previous=[]
        
    # Threshold for convergence :
    epsilon= 1e-10
    
    while ((criterion_CV>epsilon) and (nb_iter<150)):
        
        #############
        # Step m>=1 :
     
        # On sauvegarde la valeur de w_a^(m) (via les (c_(a,k)^(m))_k juste avant de calculer w_a^(m+1)
        # i.e. les (c_(a,k)^(m+1))_k

        coeffs_c_previous=[]
        
        for a in range(nb_atoms):
            
            coeffs_c_previous.append([])
            
            for k in range(len(coeffs_c[a])):
                
                coeffs_c_previous[a].append(coeffs_c[a][k])
                
        print('coeffs_c_previous')
        print(coeffs_c_previous)
        print(' ')
        
        # BEWARE : TO BE REDEFINED AT EACH LOOP !
        # At iteration m : spherical average (GL grid of size Ng_theta) of
        # rho(.)/ (sum_b w_b^(m-1)) at distance (r_l)_{l=1..Ng_radial) from atom a
        # (for the 2 atoms a)
        tab_sph_avg_rho_over_sum_rhob_atom=[]
        
        for a in range(nb_atoms):
            
            # Step 1 minimization 
            #(MINIMIZATION PROBLEM SOLVED SEPARATELY FOR EACH ATOM):
            #(c_{a,k}^{(m)})_{a=1..M,k=1..K_a} = argmin ( - int_{0..\infty} [ r² * wa(r) * log(sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) )] dr)
            # under the constraint sum_{k=1..K_a} (c_{a,k}^{(m)}) = N_a^(m-1) = int_{R³} rho_a^{(m)} 
        
            ###################################################
            ###################################################
            
            ###################################################
            # METHOD 1 :
            # USING CVXOPT (Python nonlinear convex optimization routines )
            # (linear equality or inequality constraints) 
            
            # K_a columns corresponding to the INEQUALITY constraints c_(a,k) >=0 
            # i.e. Gx <= h with x=(c_(a,k))_{k=1..Ka} and G[i,i]=-1 ; 0 otherwise
            
            # Conversion of the identity matrix into CVXOPT format :
            # G = matrix_constraint_ineq
            matrix_constraint_ineq = -cvxopt.matrix(np.identity(tab_K[a]))
            
            # h = vector_contraint_ineq
            vector_contraint_ineq = cvxopt.matrix(0.0,(tab_K[a],1))
                        
            ###########################
            # Linear equality constraints :
            # Ax = b with x=(c_(a,k))_{k=1..Ka} ; A = (1...1) and b = Na = (Na)
            ## matrix_constraint_eq = cvxopt.matrix(np.ones(tab_K[a]),(1,tab_K[a]))
            matrix_constraint_eq = cvxopt.matrix(1.0,(1,tab_K[a]))

            # N_a : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
            vector_contraint_eq = cvxopt.matrix(mass_rho_a[a],(1,1))

            # Spherical average (GL grid of size Ng_theta) of rho(.)/ (sum_b w_b^(m-1)) at distance (r_l)_{l=1..Ng_radial) from atom a
            # => computed once and for all AT CURRENT ITERATION and used for F_a, its gradient and Hessian :
            tab_sph_avg_rho_over_sum_rhob_atom_a = [ compute_sph_avg_integrand_total_mass(u,Rmax*x_GL_0_R[u],Ng_theta,
                                                                                          a,
                                                                                          coeffs_c_previous,
                                                                                          exponents_GISA,
                                                                                          values_rho[a],
                                                                                          x_GL_0_1, w_GL_0_1,
                                                                                          atomic_coordinates) for u in range(Ng_radial)]
            
            tab_sph_avg_rho_over_sum_rhob_atom.append(tab_sph_avg_rho_over_sum_rhob_atom_a)
            
            ##########################
            def F(x=None,z=None):
                      
                return F_LISA(a,
                              Rmax, Ng_radial,
                              x_GL_0_R,
                              w_GL_0_R,
                              Ng_theta,
                              x_GL_0_1, 
                              w_GL_0_1,
                              exponents_GISA,
                              coeffs_c_previous,
                              tab_sph_avg_rho_over_sum_rhob_atom_a,
                              mass_rho_a[a],
                              values_rho,
                              atomic_coordinates,
                              x, z)
            
            ##########################
            
            opt_CVX = cvxopt.solvers.cp(F,G = matrix_constraint_ineq, h = vector_contraint_ineq, A = matrix_constraint_eq, b = vector_contraint_eq,verbose=True)
            
            #print('Result Convex Optimization CVXOPT')
            #print(opt_CVX)
            
            for k in range(tab_K[a]):
                coeffs_c[a][k] = opt_CVX['x'][k]

            # END METHOD 1
            ###################################################
            
     
            ###################################################
            print('New coefficients (c_(a='+str(a)+',k))_{k=1..Ka} at iter '+str(nb_iter)+' : ')
            print(coeffs_c[a][:])
            print(' ')
            ###################################################
       
        # END LOOP ON ATOMS
        ###################
        print('Check constraint : sum_{k=1..Ka} (c_(a,k)^(m+1) ) =  N_a^(m) for all a ; ITERATION '+str(nb_iter))
        print('sum_{k=1..K1} (c_(1,k) ) = '+str(np.sum(coeffs_c[0]))+' = ? N1 = '+str(mass_rho_a_list_steps[nb_iter-1][0])+' ')
        print('sum_{k=1..K2} (c_(2,k) ) = '+str(np.sum(coeffs_c[1]))+' = ? N2 = '+str(mass_rho_a_list_steps[nb_iter-1][1])+' ')
        print(' ')
        
        coeffs_c_a_list_steps.append([coeffs_c[0][:],coeffs_c[1][:]])
        
        #########
        """
        # Compute the Hessian (to see if stable minimum or if minimum not reached enough)
        for a in range(nb_atoms):
            Hessian_a = computes_Hessian_Fa_LISA(a,coeffs_c[a],Rmax, 
                                                 x_GL_0_R,w_GL_0_R,Ng_radial,
                                                 x_GL_0_1, w_GL_0_1,Ng_theta,
                                                 exponents_GISA,
                                                 coeffs_c_previous,
                                                 values_rho,
                                                 atomic_coordinates)
            
            
            eigenv = np.linalg.eig(Hessian_a)[0]
            print('Atom a = '+str(a)+' eigenvalues Hessian_a :')
            print(eigenv)
            print(' ')
        """
        #########
        
        
        #print(' ')
        #print('tab_sph_avg_rho_over_sum_rhob_atom')
        #print(tab_sph_avg_rho_over_sum_rhob_atom)
        #print(' ')
        
        ###################################################
        # Total entropy S(rho^(m)|rho^{0,(m)}) at step m :
        # = int (rho_a^(m-1) * log(rho_a^(m-1))) - 4*pi*int_0^{+\infty} r²*<rho_a^(m-1)>_{s,a}(r)*log(sum_k c_(a,k)^(m) * Tilde(g)_{a,k}(r))

        total_entropy_tab = [ 4*math.pi*computes_integral_rhoa_log_rhoa_LISA(a,coeffs_c_previous,
                                                                             Rmax, 
                                                                             atomic_coordinates,
                                                                             x_GL_0_R,w_GL_0_R,Ng_radial,
                                                                             x_GL_0_1, w_GL_0_1,Ng_theta,
                                                                             exponents_GISA,
                                                                             values_rho,
                                                                             tab_sph_avg_rho_over_sum_rhob_atom[a]) 
                             +4*math.pi*computes_Fa_LISA(a,coeffs_c[a][:],Rmax, 
                                                         x_GL_0_R,w_GL_0_R,Ng_radial,
                                                         x_GL_0_1, w_GL_0_1,Ng_theta,
                                                         exponents_GISA,
                                                    	     coeffs_c_previous,tab_sph_avg_rho_over_sum_rhob_atom[a],
                                                         values_rho,
                                                         atomic_coordinates)  for a in range(2)]
        
        
        #########################
        total_entropy = np.sum(total_entropy_tab)

        print('Total_entropy L-ISA ITERATION '+str(nb_iter)+' = '+str(total_entropy))
        print(' ')

        
        print('Atom 1 : '+str(total_entropy_tab[0])+' ; atom 2 : '+str(total_entropy_tab[1]))
        print(' ')

        total_entropy_list_step.append(total_entropy)
        
      
        #########################
        
        ##########
        # Actualisation de la masse des rho_a(.) :
        # Computation of N_a^(m) = int (rho_a^(m)) [mass of rho_a^(m)]:
        # using the new w_a^(m)(.) functions i.e. the new (c_(a,k))_(a,k=1..Ka) coefficients
        
        print('Computing mass of (rho_a^('+str(nb_iter)+') )_a :')
        for a in range(nb_atoms):
                
            mass_rho_a[a] = compute_mass_rho_a_GISA(a,Rmax,Ng_radial,Ng_theta,
                                                    coeffs_c,
                                                    exponents_GISA,
                                                    values_rho[a],
                                                    x_GL_0_1, w_GL_0_1,
                                                    x_GL_0_R, w_GL_0_R,
                                                    atomic_coordinates)
                
   
                
        print('Current mass of (rho_a^(m='+str(nb_iter)+') )_a ; using w_a^(m='+str(nb_iter)+') :')
        print(mass_rho_a)
        print(' ')
        print('Check total sum_a N_a iteration '+str(nb_iter))
        total_sum_Na = np.sum(mass_rho_a)
        print(total_sum_Na)
        print(' ')
        

        total_sum_Na_list_steps.append(total_sum_Na)
        
        mass_rho_a_list_steps.append([])
        mass_rho_a_list_steps[nb_iter].append(mass_rho_a[0])
        mass_rho_a_list_steps[nb_iter].append(mass_rho_a[1])

        ##############
        # Convergence criterion
        #criterion_CV = 0
        
        
        criterion_CV = convergence_criterion_GISA_LISA(coeffs_c,
                                                       coeffs_c_previous,
                                                       exponents_GISA,
                                                       x_GL_0_R,w_GL_0_R,Ng_radial,
                                                       Rmax)
        criterion_CV_list_step.append(criterion_CV)
        
        print('Convergence criterion L-ISA sum_a ||w_a^(m+1)-w_a^(m)||_{L2,GL} iter. m='+str(nb_iter))
        print(criterion_CV)
        print('End iteration '+str(nb_iter))
        print('----------------------------------')
        ################
        
        nb_iter +=1
    
    print(' ')
    print('criterion_CV_list_step')
    print(criterion_CV_list_step)
    print(' ')
    print('TOTAL number of iterations to converge = '+str(nb_iter)+' for epsilon = '+str(epsilon))
    print(' ')
    print('total_entropy_list_step')
    print(total_entropy_list_step)
    print(' ')
    print('total_sum_Na_list_steps = [sum_a N_a^(m)]_{m=0..nb_iter='+str(nb_iter)+'} :')
    print(total_sum_Na_list_steps)
    print('mass_rho_a_list_steps')
    print(mass_rho_a_list_steps)
    print()
    
    print('coeffs_c_a_list_steps')
    print(coeffs_c_a_list_steps)
    print(' ')
    
    print('CONVERGENCE (OR MAX. NUMBER OF ITERATIONS) ACHIEVED')
    
    # w_a(.) : sont analytiques : évalués n'importe où

    print('w_1_values_FINAL_atom_0')
    w_1_values_FINAL = [compute_w_a_LISA(tab_r_plot_FINAL[i],coeffs_c[0],exponents_GISA[0]) for i in range(len(tab_r_plot_FINAL))]
    #print(w_1_values_FINAL)
    print(' ')
    print('w_2_values_FINAL_atom_1')
    w_2_values_FINAL = [compute_w_a_LISA(tab_r_plot_FINAL[i],coeffs_c[1],exponents_GISA[1]) for i in range(len(tab_r_plot_FINAL))]
    #print(w_2_values_FINAL)
    print(' ')
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ("+str(periodic_table[atomic_numbers[0]])+") GISA-NEW")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"LISA_w_1_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-Rtab_r_plot2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 2 ("+str(periodic_table[atomic_numbers[1]])+") GISA-NEW")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),linestyle='--',color='black')

    matplotlib.pyplot.savefig(ISA_plots_dir+"LISA_w_2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    # Print r--> 4*pi*r²*wa(r)
    w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_1_values_FINAL[i]) for i in range(len(tab_r_plot_FINAL))]

    w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_2_values_FINAL[i]) for i in range(len(tab_r_plot_FINAL))]
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)',fontsize=25)
    matplotlib.pyplot.ylabel('log(4πr²w_1(r))',fontsize=25)

    matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) GISA-NEW",fontsize=25)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"LISA_w_1_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),linestyle='--',color='black')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)',fontsize=25)
    matplotlib.pyplot.ylabel('log(4πr²w_2(r))',fontsize=25)

    matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) GISA-NEW",fontsize=25)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),linestyle='--',color='black')

    matplotlib.pyplot.savefig(ISA_plots_dir+"LISA_w_2_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    compute_local_multipoles_GISA(Rmax,
                                  coeffs_c,
                                  exponents_GISA,
                                  values_rho,
                                  Ng_theta,
                                  x_GL_0_1, w_GL_0_1,
                                  Ng_radial,
                                  x_GL_0_R, w_GL_0_R,
                                  atomic_coordinates,
                                  atomic_numbers,
                                  logfile_LISA)
        
    return coeffs_c, coeffs_c_a_list_steps, w_1_values_FINAL, w_2_values_FINAL, w_1_times_r2_values_FINAL, w_2_times_r2_values_FINAL
################################################################


################################################################
"""
- Ka = number of shells per atom a
- c_coeffs_previous : previous values (c_(a,k)^(m))_{k=1..Ka} derived at the previous iteration
- coeffs_c_init : guess values for the search of (c_(a,k)^(m+1))_{k=1..Ka}
- mass_rho_a_atom = N_a^(m)= int rho_a^(m) at previous step
"""
# NB : Default arguments are indicated (useful to evaluate
#      F_LISA(Ka,...,mass_rho_a_atom) i.e. when x=None, z=None 
#   or F_LISA(Ka,...,mass_rho_a_atom,x) when z=None)
def F_LISA(a,
           Rmax, Ng_radial,
           x_GL_0_R,
           w_GL_0_R,
           Ng_theta,
           x_GL_0_1, 
           w_GL_0_1,
           exponents_GISA,
           c_coeffs_previous,
           tab_sph_avg_rho_over_sum_rhob_atom_a,
           mass_rho_a_atom,
           values_rho,
           atomic_coordinates,
           x=None, z=None):
        
    
    if x is None: 
        
        # F_LISA() returns m,x0
        # where : m = number of non-linear constraints
        #         x0 = point in the domain definition of the function f to be minimized
        #              (here R^{Ka} as we minimize F({c_(a,k)}_{k=1..Ka}))
        #              BEWARE : do not take x0 = [0,..,0] here because the function Fa
        #              involves a log(.) ! 
        
        # Important : x0 used as first guess by the CVXOPT optimization algorithm
        # x0 = cvxopt.matrix(mass_rho_a_atom/Ka,(Ka,1))

        x0 = cvxopt.matrix(c_coeffs_previous[a][:])

        
        return 0, x0
    
    # Otherwise : we return (f(x),Df(x)) of z=None [function and gradient values]
    #                    or (f(x),Df(x),H(x)) otherwise [function, gradient and Hessian values]
        
    # x (the variable) = 'vector_c_a' = (c_(a,k))_{k=1..Ka} : vector of R^{Ka}

    ###########
    # Using normal definition of Fa (without factorization by exp(-alpha_a^{MAX}*r²) on the log(.))
    """
    f = computes_Fa_LISA(a,x,Rmax, 
                        x_GL_0_R,w_GL_0_R,Ng_radial,
                       x_GL_0_1, w_GL_0_1,Ng_theta,
                       exponents_GISA,
                       c_coeffs_previous,
                        tab_sph_avg_rho_over_sum_rhob_atom_a,
                        values_rho,
                        atomic_coordinates)
    """
    # Using the alternative definition of Fa 
    #(WITH factorization by exp(-alpha_a^{MAX}*r²) on the log(.))
    # (LogSumExp)
    
    f = computes_Fa_LISA_BIS(a,x,Rmax, 
                         x_GL_0_R,w_GL_0_R,Ng_radial,
                         x_GL_0_1, w_GL_0_1,Ng_theta,
                         exponents_GISA,
                         c_coeffs_previous,
                         tab_sph_avg_rho_over_sum_rhob_atom_a,
                         values_rho,
                         atomic_coordinates)
    ###########

    Df = cvxopt.matrix(computes_gradient_Fa_LISA(a,x,Rmax, 
                                                 x_GL_0_R,w_GL_0_R,Ng_radial,
                                                 x_GL_0_1, w_GL_0_1,Ng_theta,
                                                 exponents_GISA,
                                                 c_coeffs_previous,
                                                 tab_sph_avg_rho_over_sum_rhob_atom_a,
                                                 values_rho,
                                                 atomic_coordinates))
    
    if z is None: 
        
        return f, Df
            
    # Only one constraint : z0 * Hess(f)(x)
    H = z[0] * cvxopt.matrix(computes_Hessian_Fa_LISA(a,x,Rmax, 
                                                      x_GL_0_R,w_GL_0_R,Ng_radial,
                                                      x_GL_0_1, w_GL_0_1,Ng_theta,
                                                      exponents_GISA,
                                                      c_coeffs_previous,
                                                      tab_sph_avg_rho_over_sum_rhob_atom_a,
                                                      values_rho,
                                                      atomic_coordinates))
        
    # Both x and z are not 'None' : we return the function value, its gradient, and the Hessian matrix
    return f, Df, H
################################################################


################################################################
# Evaluate w_(a,GISA)(r) from coefficients (c_(a,k))_{k=1..Ka} and exponents
# (alpha_(a,k))_{k=1..Ka}, taking by convention w_(a,GISA)(.) centered in R_a
"""
- a = atom index
"""
def compute_w_a_LISA(r,coeffs_c_atom,exponents_GISA_atom):
    
    K_a = len(exponents_GISA_atom)
    
    return np.sum([coeffs_c_atom[k]*(exponents_GISA_atom[k]/math.pi)**(3./2.) * math.exp(-exponents_GISA_atom[k] * (r/conversion_bohr_angstrom)**2) for k in range(K_a)])
################################################################
 
"""
PROBLEM (TODO) with this function : the result changes when changing '100' in the np.min(.) to avoid math range error
"""
def compute_w_a_LISA_alpha_MAX_FACTOR(r,alpha_MAX,coeffs_c_atom,exponents_GISA_atom):
    
    K_a = len(exponents_GISA_atom)

    # To avoid having too large arguments in the exp() : min between 100 and the (positive) value -(exponents_GISA_atom[k]-alpha_MAX) * r²
    return np.sum([coeffs_c_atom[k]*(exponents_GISA_atom[k]/math.pi)**(3./2.) * math.exp(np.min([-(exponents_GISA_atom[k]-alpha_MAX) * (r/conversion_bohr_angstrom)**2,100])) for k in range(K_a)])

################################################################
   

################################################################
"""
Computes the atomic function 

Fa(c_a) = - int_{0..\infty} [ r² * rho_a^(0,(m))(r) * < rho/ (sum_b rho_b^(0,(m))) >_a(r) * log(sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) )] dr

where g_(a,k)(r) = (alpha_(a,k)/pi)^(3/2) * exp(-alpha_(a,k) * r²)

=> Returns a real number.

-  a : atom index
-  c_a = 'vector_c_a' = (c_(a,k))_(k=1..Ka)
-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.
-  c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
   of the iterative scheme (e.g. direct minimization)
- exponents_GISA_atom = exponents_GISA[a] = (alpha_(a,k))_{k=1..Ka}
"""
def computes_Fa_LISA(a,vector_c_a,Rmax, 
                     x_GL_0_R,w_GL_0_R,Ng_radial,
                     x_GL_0_1, w_GL_0_1,Ng_theta,
                     exponents_GISA,
                     c_coeffs_previous,
                     tab_sph_avg_rho_over_sum_rhob_atom_a,
                     values_rho,
                     atomic_coordinates):
        
   
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
        
    # Computes r**2 * rho_a^(0,(m))(r) * < rho/ (sum_b rho_b^(0,(m))) >_a(r) log(sum_{l=1..Ka} c_(a,l)*g_(a,l)(r) ) on the (Gauss-Legendre) radial grid  (R*x_l^(GL)):
    #  w_a(r) = sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) in our notations hence the use of the function 'compute_w_a_GISA'
    
    #  'compute_sph_avg_integrand_total_mass' returns the spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a])* tab_sph_avg_rho_over_sum_rhob_atom_a[u] * math.log(compute_w_a_LISA(x_GL_0_R_scaled[u],vector_c_a,exponents_GISA[a])) for u in range(Ng_radial)]
           

    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    Fa = - (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return Fa
################################################################

################################################################
"""
Computes the atomic function 

Fa(c_a) = - int_{0..\infty} [ r² * <rho_a>(r) * [ -alpha_(a0,k)^{MAX}*r² + log(sum_{k=1..Ka} c_(a,k)*(alpha_(a,k)/pi)^(3/2) * exp(-(alpha_(a,k)-alpha_(a0,k)^{MAX})* r²)) ] dr

        where <rho_a>(r) = rho_a^(0,(m))(r) * < rho/ (sum_b rho_b^(0,(m))) >_a(r) 
        
=> Returns a real number.

-  a : atom index
-  c_a = 'vector_c_a' = (c_(a,k))_(k=1..Ka)
-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.
-  c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
   of the iterative scheme (e.g. direct minimization)
- exponents_GISA_atom = exponents_GISA[a] = (alpha_(a,k))_{k=1..Ka}
"""
def computes_Fa_LISA_BIS(a,vector_c_a,Rmax, 
                         x_GL_0_R,w_GL_0_R,Ng_radial,
                         x_GL_0_1, w_GL_0_1,Ng_theta,
                         exponents_GISA,
                         c_coeffs_previous,
                         tab_sph_avg_rho_over_sum_rhob_atom_a,
                         values_rho,
                         atomic_coordinates):
        
   
    alpha_a_MAX = max(exponents_GISA[a])

    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]

    # Computes r**2 * rho_a^(0,(m))(r) * < rho/ (sum_b rho_b^(0,(m))) >_a(r) log(sum_{l=1..Ka} c_(a,l)*g_(a,l)(r) ) on the (Gauss-Legendre) radial grid  (R*x_l^(GL)):
    #  w_a(r) = sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) in our notations hence the use of the function 'compute_w_a_GISA'

    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a]) *  tab_sph_avg_rho_over_sum_rhob_atom_a[u] * (-alpha_a_MAX*(x_GL_0_R_scaled[u]/conversion_bohr_angstrom)**2 + math.log(compute_w_a_LISA_alpha_MAX_FACTOR(x_GL_0_R_scaled[u],alpha_a_MAX,vector_c_a,exponents_GISA[a]))) for u in range(Ng_radial)]

    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    Fa = - (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return Fa
################################################################

################################################################
"""
Returns int (rho_a^(m-1) * log(rho_a^(m-1))) 

        = 4*pi*int_0^{+\infty} r²*<rho_a^(m-1)>_{s,a}(r)*log(<rho_a^(m-1)>_{s,a}(r)) 
        
(ingredient of the total entropy S(rho^(m)|rho^{0,(m)}) at step m)
"""
def computes_integral_rhoa_log_rhoa_LISA(a,c_coeffs_previous,
                                          Rmax, 
                                          atomic_coordinates,
                                          x_GL_0_R,w_GL_0_R,Ng_radial,
                                          x_GL_0_1, w_GL_0_1,Ng_theta,
                                          exponents_GISA,
                                          values_rho,
                                          tab_sph_avg_rho_over_sum_rhob_atom_a):
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
        
    
    # Computes r**2 * rho_a^(0,(m))(r) * (< rho/ (sum_b rho_b^(0,(m))) >_a(r) * log (rho_a^(0,(m))(r)) - < rho/ (sum_b rho_b^(0,(m)))*log(sum_b rho_b^(0,(m))) >_a(r) + < rho*log(rho)/sum_b rho_b^(0,(m))>_a(r) )
    # on a GL radial grid (r_l)_l
    
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a]) * ( tab_sph_avg_rho_over_sum_rhob_atom_a[u] * math.log(compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a]))
                                                                                                                                                                                - compute_sph_avg_entropy_1(u,x_GL_0_R_scaled[u],Ng_theta,
                                                                                                                                                                                                             a,
                                                                                                                                                                                                             c_coeffs_previous,exponents_GISA,
                                                                                                                                                                                                             values_rho[a],
                                                                                                                                                                                                             x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                             atomic_coordinates)
                                                                                                                                                                                 +compute_sph_avg_entropy_2(u,x_GL_0_R_scaled[u],Ng_theta,
                                                                                                                                                                                                            a,
                                                                                                                                                                                                            c_coeffs_previous,exponents_GISA,
                                                                                                                                                                                                            values_rho[a],
                                                                                                                                                                                                            x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                            atomic_coordinates)) for u in range(Ng_radial)]
    
    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    integral = (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return integral
################################################################

################################################################
"""
Returns the  entropy s_{KL}(rho_a^(m)|rho_a^{0,(m)}) at step m)
"""
def computes_entropy_LISA(a,c_coeffs,
                          Rmax, 
                          atomic_coordinates,
                          x_GL_0_R,w_GL_0_R,Ng_radial,
                          x_GL_0_1, w_GL_0_1,Ng_theta,
                          exponents_GISA,
                          values_rho):
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
        
    

    #NEW
    # Computes r**2 * rho_a^(0,(m))(r) * (- < (rho/ (sum_b rho_b^(0,(m))))*log(sum_b rho_b^(0,(m))) >_a(r) + < rho*log(rho)/ (sum_b rho_b^(0,(m)))>_a(r) )
    # on a GL radial grid (r_l)_l
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs[a],exponents_GISA[a]) * (-compute_sph_avg_entropy_1(u,x_GL_0_R_scaled[u],Ng_theta,
                                                                                                                                                                                                 a,
                                                                                                                                                                                                 c_coeffs,
                                                                                                                                                                                                 exponents_GISA,
                                                                                                                                                                                                 values_rho[a],
                                                                                                                                                                                                 x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                 atomic_coordinates)                     
                                                                                                                                                                      +compute_sph_avg_entropy_2(u,x_GL_0_R_scaled[u],Ng_theta,
                                                                                                                                                                                                 a,
                                                                                                                                                                                                 c_coeffs,
                                                                                                                                                                                                 exponents_GISA,
                                                                                                                                                                                                 values_rho[a],
                                                                                                                                                                                                 x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                 atomic_coordinates)) for u in range(Ng_radial)]

    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    integral = (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return integral
################################################################

################################################################
"""
Computes the gradient of Fa(c_a) = - int_{0..\infty} [ r² * wa(r) * log(sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) )] dr

where g_(a,k)(r) = (alpha_(a,k)/pi)^(3/2) * exp(-alpha_(a,k) * r²)

=> Returns a vector of size Ka.

-  a : atom index
-  c_a = 'vector_c_a' = (c_(a,k))_(k=1..Ka)
-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.
- c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
  of the iterative scheme (e.g. direct minimization)
- exponents_GISA_atom = exponents_GISA[a] = (alpha_(a,k))_{k=1..Ka}
"""
def computes_gradient_Fa_LISA(a,vector_c_a,Rmax, 
                              x_GL_0_R,w_GL_0_R,Ng_radial,
                              x_GL_0_1, w_GL_0_1,Ng_theta,
                              exponents_GISA,
                              c_coeffs_previous,
                              tab_sph_avg_rho_over_sum_rhob_atom_a,
                              values_rho,
                              atomic_coordinates):
        
       
    # Number of  GISA shells for atom a :
    Ka = len(exponents_GISA[a])

    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
    
    gradient_Fa = np.zeros([1,Ka])
    
    for k in range(Ka):
        # Computes r**2 * w_a^(m)(r) * g_(a,k)(r) / (sum_{l=1..Ka} c_(a,l)*g_(a,l)(r) ) on the (Gauss-Legendre) radial grid  (R*x_l^(GL)):
        #  w_a(r) = sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) in our notations hence the use of the function 'compute_w_a_GISA'
        tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a]) * tab_sph_avg_rho_over_sum_rhob_atom_a[u]  * compute_basis_function_GISA(x_GL_0_R_scaled[u],k,exponents_GISA[a])/(compute_w_a_LISA(x_GL_0_R_scaled[u],vector_c_a,exponents_GISA[a])) for u in range(Ng_radial)]
           

        # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
        gradient_Fa[0][k] = - (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return gradient_Fa
################################################################


################################################################
"""
Computes the Hessian of Fa(c_a) = - int_{0..\infty} [ r² * wa(r) * log(sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) )] dr

where g_(a,k)(r) = (alpha_(a,k)/pi)^(3/2) * exp(-alpha_(a,k) * r²)

=> Returns a matrix of size Ka x Ka

-  a : atom index
-  c_a = 'vector_c_a' = (c_(a,k))_(k=1..Ka)
-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.
- c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
  of the iterative scheme (e.g. direct minimization)
- exponents_GISA_atom = exponents_GISA[a] = (alpha_(a,k))_{k=1..Ka}
"""
def computes_Hessian_Fa_LISA(a,vector_c_a,Rmax, 
                             x_GL_0_R,w_GL_0_R,Ng_radial,
                             x_GL_0_1, w_GL_0_1,Ng_theta,
                             exponents_GISA,
                             c_coeffs_previous,
                             tab_sph_avg_rho_over_sum_rhob_atom_a,
                             values_rho,
                             atomic_coordinates):
            
    # Number of  GISA shells for atom a :
    Ka = len(exponents_GISA[a])

    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
    
    Hessian_Fa = np.zeros([Ka,Ka])
    
    for j in range(Ka):
        
        for k in range(Ka):
            
            # Computes r**2 * w_a^(m)(r) * g_(a,j)(r) * g_(a,k)(r) / (sum_{l=1..Ka} c_(a,l)*g_(a,l)(r) ) on the (Gauss-Legendre) radial grid  (R*x_l^(GL)):
            #  w_a(r) = sum_{k=1..Ka} c_(a,k)*g_(a,k)(r) in our notations hence the use of the function 'compute_w_a_GISA'
            tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_LISA(x_GL_0_R_scaled[u],c_coeffs_previous[a],exponents_GISA[a]) * tab_sph_avg_rho_over_sum_rhob_atom_a[u] * compute_basis_function_GISA(x_GL_0_R_scaled[u],j,exponents_GISA[a]) * compute_basis_function_GISA(x_GL_0_R_scaled[u],k,exponents_GISA[a])/(compute_w_a_LISA(x_GL_0_R_scaled[u],vector_c_a,exponents_GISA[a]))**2 for u in range(Ng_radial)]
                               
               
    
            # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
            Hessian_Fa[j,k] = (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)
    
    return Hessian_Fa
################################################################


################################################################
"""
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of (rho(.)/ (sum_b w_b))*log((sum_b w_b)) at distance r from atom a :
# Used in the KL entropy calculation (int (rho_a * log(rho_a)))
def compute_sph_avg_entropy_1(index_radial,r,Ng_theta,
                              a,
                              coeffs_c,exponents_GISA,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * (math.sin(math.pi*x_GL_0_1[u])/(compute_w_a_GISA(r,coeffs_c[a],exponents_GISA[a])+compute_w_a_GISA(np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]),coeffs_c[abs(a-1)],exponents_GISA[abs(a-1)])))*math.log(compute_w_a_GISA(r,coeffs_c[a],exponents_GISA[a])+compute_w_a_GISA(np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]),coeffs_c[abs(a-1)],exponents_GISA[abs(a-1)]))  for u in range(Ng_theta)]
    
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
def compute_sph_avg_entropy_2(index_radial,r,Ng_theta,
                              a,
                              coeffs_c,exponents_GISA,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u]*math.log(values_rho_atom[index_radial][u]) * math.sin(math.pi*x_GL_0_1[u]) /(compute_w_a_GISA(r,coeffs_c[a],exponents_GISA[a])+compute_w_a_GISA(np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]),coeffs_c[abs(a-1)],exponents_GISA[abs(a-1)]))  for u in range(Ng_theta)]
    
    # Integral on theta only (not on phi)
    return math.pi * np.sum(values_integrand_theta) 
################################################################









################################################################
################################################################
# BELOW : for checking / debugging purposes only


################################################################
"""
For atom a
"""
def LISA_check_gradient_Hessian_finite_differences(a,tab_K,
                                                   Rmax,x_GL_0_R,w_GL_0_R,
                                                   Ng_radial,
                                                   exponents_GISA,
                                                   coeffs_c_previous):
    

    ###########################################
    # Check gradient by finite differences
            
    #########
    def func_equal(h,i):
        if h==i:
            return 1
        else:
            return 0
    ########
            
    h=0.001
            
    for i in range(tab_K[a]):
        # (1/(2h)) * (F_a(c_(a,1),...,c_(a,i)+h,...,c_(a,Ka)) - F_a(c_(a,1),...,c_(a,i)-h,...,c_(a,Ka)) )
        coeffs_c_plus = [coeffs_c_previous[a][m] + h*func_equal(m,i) for m in range(tab_K[a])]
        coeffs_c_minus = [coeffs_c_previous[a][m] - h*func_equal(m,i) for m in range(tab_K[a])]
        
        Fa_plus = computes_Fa_LISA(a,coeffs_c_plus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                   exponents_GISA[a],
                                   coeffs_c_previous)

        Fa_minus = computes_Fa_LISA(a,coeffs_c_minus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                    exponents_GISA[a],
                                    coeffs_c_previous)

        print('Gradient by finite differences :')
        print((Fa_plus-Fa_minus)/(2.*h))
                
        print(' ')
        print('Gradient by computes_gradient_Fa_LISA()')
        gradient = computes_gradient_Fa_LISA(a,coeffs_c_previous[a],Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                             exponents_GISA[a],
                                             coeffs_c_previous)

        print(gradient)
        print(' ')
        ###########################################
    
        ###########################################
        # Check Hessian by finite differences
            
        h=0.01
            
        for i in range(tab_K[a]):
            # (\partial² F_a /( \partial c_a,k  \partial c_a,i) \approx :
            # (1/(2h)) * ( (\partial F_a / \partial c_a,k)(c_(a,1),...,c_(a,i)+h,...,c_(a,Ka)) - (\partial F_a / \partial c_a,k)(c_(a,1),...,c_(a,i)-h,...,c_(a,Ka)) )
            coeffs_c_plus = [coeffs_c_previous[a][m] + h*func_equal(m,i) for m in range(tab_K[a])]
            coeffs_c_minus = [coeffs_c_previous[a][m] - h*func_equal(m,i) for m in range(tab_K[a])]
                                 
            #  Vector :
            grad_Fa_plus = computes_gradient_Fa_LISA(a,coeffs_c_plus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                                     exponents_GISA[a],
                                                     coeffs_c_previous)

            grad_Fa_minus = computes_gradient_Fa_LISA(a,coeffs_c_minus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                                      exponents_GISA[a],
                                                      coeffs_c_previous)
                
            print('Hessian i^{th} line (\partial² F_a /( \partial c_a,k  \partial c_a,i)_k ; i='+str(i)+'  by finite differences :')
            
            line_i_Hessian = [(grad_Fa_plus[0][k]-grad_Fa_minus[0][k])/(2*h)  for k in range(tab_K[a])]
                
            print(line_i_Hessian)
                
        print(' ')
        print('Hessian matrix by computes_Hessian_Fa_LISA()')
        Hessian = computes_Hessian_Fa_LISA(a,coeffs_c_previous[a],Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                           exponents_GISA[a],
                                           coeffs_c_previous)
        print(Hessian)
        ###########################################
            
        ###########################################
        # Check Hessian by finite differences (BIS)
            
        h=0.01
            
        #########
        def func_equal_BIS(h,i,k):
            if h==i:
                return 1
            elif h==k:
                return 1
            else:
                return 0
        ########
   
        def func_equal_BIS2(h,i,k):
            if h==i:
                return 1
            elif h==k:
                return -1
            else:
                return 0                
        ########
            
        def func_equal_BIS3(h,i,k):
            if h==i:
                return -1
            elif h==k:
                return 1
            else:
                return 0
        ########

        def func_equal_BIS4(h,i,k):
            if h==i:
                return -1
            elif h==k:
                return -1
            else:
                return 0
        ########
            
        for i in range(tab_K[a]):
            
            for k in range(tab_K[a]):
                    
                    
                if (i!=k):
                        
                    # (\partial² F_a /( \partial c_a,k  \partial c_a,i) :
    
                    coeffs_c_plus = [coeffs_c_previous[a][m] + h*func_equal_BIS(m,i,k) for m in range(tab_K[a])]
                    coeffs_c_minus = [coeffs_c_previous[a][m] + h*func_equal_BIS3(m,i,k) for m in range(tab_K[a])]
                                     
                    Fa_plus = computes_Fa_LISA(a,coeffs_c_plus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                               exponents_GISA[a],
                                               coeffs_c_previous)
    
                    Fa_minus = computes_Fa_LISA(a,coeffs_c_minus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                                exponents_GISA[a],
                                                coeffs_c_previous)
        
                    coeffs_c_plus_BIS = [coeffs_c_previous[a][m] + h*func_equal_BIS2(m,i,k) for m in range(tab_K[a])]
                    coeffs_c_minus_BIS = [coeffs_c_previous[a][m] + h*func_equal_BIS4(m,i,k) for m in range(tab_K[a])]
        
                    Fa_plus_BIS = computes_Fa_LISA(a,coeffs_c_plus_BIS,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                                   exponents_GISA[a],
                                                   coeffs_c_previous)
    
                    Fa_minus_BIS = computes_Fa_LISA(a,coeffs_c_minus_BIS,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                                    exponents_GISA[a],
                                                    coeffs_c_previous)
        
                        
                    print('Hessian coefficient (\partial² F_a /( \partial c_a,k  \partial c_a,i)_k ; k ='+str(k)+'; i='+str(i)+'  by finite differences from Fa :')
                
                    coeff_i_k_Hessian =((Fa_plus-Fa_minus)-(Fa_plus_BIS-Fa_minus_BIS))/(2*h)**2
                    
                    print(coeff_i_k_Hessian)
                        
            #if (i==i):
                
            # Hessian approximation :
            # (2/h²) * (F_a(c_(a,1),...,c_(a,i)+h,...,c_(a,Ka)) - 2*F_a(c_(a,1),...,c_(a,i),...,c_(a,Ka)) + F_a(c_(a,1),...,c_(a,i)-h,...,c_(a,Ka)) )
            coeffs_c_plus = [coeffs_c_previous[a][m] + h*func_equal(m,i) for m in range(tab_K[a])]
            coeffs_c_minus = [coeffs_c_previous[a][m] - h*func_equal(m,i) for m in range(tab_K[a])]
                   
            Fa_plus = computes_Fa_LISA(a,coeffs_c_plus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                       exponents_GISA[a],
                                       coeffs_c_previous)

            Fa_center = computes_Fa_LISA(a,coeffs_c_previous[a],Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                         exponents_GISA[a],
                                         coeffs_c_previous)
                        
            Fa_minus = computes_Fa_LISA(a,coeffs_c_minus,Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                        exponents_GISA[a],
                                        coeffs_c_previous)
                        
            coeff_i_i_Hessian = (Fa_plus-2*Fa_center+Fa_minus)/h**2
                        
            print('Hessian coefficient (\partial² F_a /( \partial² c_i,k)_i ; i='+str(i)+'  by finite differences from Fa :')
                                        
            print(coeff_i_i_Hessian)
                
        print(' ')
        print('Hessian matrix by computes_Hessian_Fa_LISA()')
        Hessian = computes_Hessian_Fa_LISA(a,coeffs_c_previous[a],Rmax, x_GL_0_R,w_GL_0_R,Ng_radial,
                                           exponents_GISA[a],
                                           coeffs_c_previous)
        print(Hessian)
        ###########################################
        
################################################################
            



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:01:47 2021

@author: rbenda

Functions for GISA algorithm (Verstraelen et al. 2012)
"""


import numpy as np
import math
import matplotlib

from ISA_auxiliary_functions import  lgwt

import quadprog
import cvxopt
from cvxopt import matrix

from matplotlib import pyplot

# Example : to solve a quadratic problem (under linear constraints) :
## cvxopt.cvxopt_solve_qp()
# or :
# quadprog.solve_qp()

# To solve non-linear optimization problems :
# cvxopt.solvers.cp()

from ISA_auxiliary_functions import compute_density_grid_point
from ISA_auxiliary_functions import compute_sph_avg_atomic_charge

############################
#CONVERSION CONSTANTS :
#1 Bohr in Angstroms
conversion_bohr_angstrom=0.529177209
# For test densities (non-dimensonal, not originating from a QM code output)
#conversion_bohr_angstrom=1.
############################

ISA_output_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_output_files_dir/'
ISA_plots_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/ISA_plots/'

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']

################################################################
################################################################
# FONCTIONS POUR GISA (classique, Verstraelen et al. 2012 Chem. Phys. Letters)


################################################################
"""
- tab_K = {K_a}_{a=1..M} = number of shells per atom
- exponents_GISA=[([alpha_(a,1),...,alpha_(a,Ka)])_{a=1..M}] the exponents in the Gaussians
  defining the shape functions of the GISA method. These exponents are fixed in this method.
- Rmax : maximal radial intergation limit in the integrals from 0 to \infty on r (radial)
"""
def GISA_classic_algorithm(tab_K,Ng_radial,Ng_theta,
                           coeffs_c_init,
                           exponents_GISA,
                           Rmax,values_rho,
                           x_GL_0_1, w_GL_0_1,
                           x_GL_0_R, w_GL_0_R,
                           atomic_coordinates,
                           atomic_numbers,
                           logfile_GISA,
                           tab_r_plot_FINAL):
    
    nb_atoms = len(tab_K)
    
    coeffs_c=[]
    
    total_sum_Na_list_steps=[]

    mass_rho_a_list_steps=[]

    #############
    # Initialisation : w_a^(0) : derived from the coefficients : (c_{a,k}^{(0)})_{a=1..M,k=1..K_a} 
    # For instance : all equal to 1
    
    for a in range(nb_atoms):
        
        coeffs_c.append([coeffs_c_init[a][k] for k in range(tab_K[a])])
        
    print('Initial coefficients (c_(a,k))_{a=1..M,k=1..Ka} :')
    print(coeffs_c)
    print(' ')
   
    # (N_a)_{a=1..M} : masses of the rho_a(.) 
    mass_rho_a = np.zeros(nb_atoms)
    
    
    print('Computing mass of (rho_a^(0))_a :')
    # OR SIMPLY FIXING THEM TO Z_a (atomic numbers) ??
    
    # N_a^(0) = int (rho_a^(0)) [mass of rho_a^(0)]:
    for a in range(nb_atoms):
        
        mass_rho_a[a] = compute_mass_rho_a_GISA(a,Rmax,
                                                Ng_radial,Ng_theta,
                                                coeffs_c,
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

    print('mass_rho_a_list_steps')
    print(mass_rho_a_list_steps)
    
    
    print('Check "constraint iter. 0" : sum_{k=1..Ka} (c_(a,k) ) =  Na for all a ; ITERATION 0')
    print('sum_{k=1..K1} (c_(1,k)^(0) ) = '+str(np.sum(coeffs_c[0]))+' = ? N1 = '+str(mass_rho_a[0])+' ')
    print('sum_{k=1..K2} (c_(2,k)^(0) ) = '+str(np.sum(coeffs_c[1]))+' = ? N2 = '+str(mass_rho_a[1])+' ')
    print(' ')
    
    #############
    
    ###########################
    # MAIN LOOP
    
    total_entropy_list_step=[]
    
    criterion_CV = 10000
    
    criterion_CV_list_step=[]
        
    nb_iter=1
    
    coeffs_c_previous=[]

    # Threshold for convergence :
    epsilon= 1e-10
    
    while ((criterion_CV> epsilon) and (nb_iter<20)):

        #############
        # Step m>=1 :
        # (c_{a,k}^{(m)})_{a=1..M,k=1..K_a} = argmin || sum_{k=1..K_a} c_{a,k}.zeta_{a,GISA}^k(.+R_a) - < rho_a^{(m)} >_a(|.-R_a|)  ||_{L_2(R+)}
        # under the constraint sum_{k=1..K_a} (c_{a,k}^{(m)}) = N_a^(m-1) = int_{R³} rho_a^{(m)} 
            
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
        
        for a in range(nb_atoms):
            
            # Can be computed one and for all outside of this function 
            A_a = compute_matrix_A_atom_GISA(exponents_GISA[a])
            #print('A_a')
            #print(A_a)
            #print('A_{a='+str(a)+'} invertible ?')
            #print('det(A_a) = '+str(np.linalg.det(A_a)))
            #print(' ')
            
            #v,w = np.linalg.eig(A_a)
            #print('Eigenvalues :')
            #print(v)
            #print('  ')
            
            B_a = compute_vector_B_atom_GISA(a,Rmax,Ng_radial,Ng_theta,coeffs_c,exponents_GISA,
                                             atomic_coordinates,
                                             values_rho[a],
                                             x_GL_0_1, w_GL_0_1,
                                             x_GL_0_R, w_GL_0_R)
            

            #print('B_{a='+str(a)+'} vector iteration '+str(nb_iter))
            #print(B_a)
            #print(' ')
            
            # Constant :
            T_a = compute_constant_cost_function_GISA(a,Rmax,Ng_radial,
                                                      coeffs_c,
                                                      exponents_GISA,
                                                      x_GL_0_R, w_GL_0_R)

        
            
            #GISA_cost_function_atom_a = np.dot(coeffs_c[a],np.dot(A_a,np.transpose(coeffs_c[a])))-2*np.dot(B_a,np.transpose(coeffs_c[a]))+T_a
            #print('GISA COST function value = '+str(GISA_cost_function_atom_a))
            
            ###################################################
            ###################################################
            # METHOD 2 : using Python quadratic programming optimization routines 
            # (linear equality or inequality constraints) 
            
            matrix_constraint = np.zeros([tab_K[a],tab_K[a]+1])
            # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
            matrix_constraint[:,0] = np.ones(tab_K[a])
            # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0 
            matrix_constraint[0:tab_K[a],1:(tab_K[a]+1)] = np.identity(tab_K[a])
            
            vector_contraint = np.zeros(tab_K[a]+1)
            
            # First coefficient : N_a : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
            vector_contraint[0] = mass_rho_a[a]
            
            coeffs_c[a] = quadprog.solve_qp(G=2*A_a, a = 2*B_a, C=matrix_constraint, b=vector_contraint, meq=1)[0]

            print('New coefficients (c_(a,k))_{a=1..M,k=1..Ka} at iter '+str(nb_iter)+' : ')
            print(coeffs_c)
            print(' ')
            ############
            
            
        ########################  
        # CALCUL de l'entropie S(rho^(m+1)|rho^{0,(m)}) :
        total_entropy_tab = [ 4*math.pi*computes_entropy_GISA(a,coeffs_c,
                                                              Rmax, 
                                                              atomic_coordinates,
                                                              x_GL_0_R,w_GL_0_R,Ng_radial,
                                                              x_GL_0_1, w_GL_0_1,Ng_theta,
                                                              exponents_GISA,
                                                              values_rho) for a in range(2)]
        
        
        
        total_entropy = np.sum(total_entropy_tab)

        print('Total_entropy GISA ITERATION '+str(nb_iter)+' = '+str(total_entropy))
        print(' ')
                
        print('Atom 1 : '+str(total_entropy_tab[0])+' ; atom 2 : '+str(total_entropy_tab[1]))
        print(' ')

        total_entropy_list_step.append(total_entropy)
        
        ########################
        
        ##########
        # Actualisation de la masse des rho_a(.) :
        # Computation of N_a^(m) = int (rho_a^(m)) [mass of rho_a^(m)]:
        # using the new w_a^(m)(.) functions i.e. the new (c_(a,k))_(a,k=1..Ka)
        # coefficients
        print('Computing mass of (rho_a^('+str(nb_iter)+') )_a :')
        for a in range(nb_atoms):
                
            mass_rho_a[a] = compute_mass_rho_a_GISA(a,Rmax,Ng_radial,Ng_theta,coeffs_c,
                                                    exponents_GISA,
                                                    values_rho[a],
                                                    x_GL_0_1, w_GL_0_1,
                                                    x_GL_0_R, w_GL_0_R,
                                                    atomic_coordinates)
                
   
        mass_rho_a_list_steps.append([])
        mass_rho_a_list_steps[nb_iter].append(mass_rho_a[0])
        mass_rho_a_list_steps[nb_iter].append(mass_rho_a[1])
                
        print('Current mass of (rho_a^(m='+str(nb_iter)+') )_a ; using w_a^(m='+str(nb_iter)+') :')
        print(mass_rho_a)
        print(' ')
        print('Check total sum_a N_a iteration '+str(nb_iter))
        total_sum_Na = np.sum(mass_rho_a)
        print(total_sum_Na)
        print(' ')
        total_sum_Na_list_steps.append(total_sum_Na)

        # BEWARE : 'mass_rho_a[:]' was just recomputed with the new values c_(a,k)^(m+1), 
        # while the constraint in the optimization on (c_(a,k))_k was
        # sum_{k=1..Ka} c_(a,k) = N_a^(m) [i.e. mass_rho_a at the previous iteration]
        
        print('Check constraint : sum_{k=1..Ka} (c_(a,k) ) =  Na for all a ; ITERATION '+str(nb_iter))
        print('sum_{k=1..K1} (c_(1,k) ) = '+str(np.sum(coeffs_c[0]))+' = ? N1 = '+str(mass_rho_a_list_steps[nb_iter-1][0])+' ')
        print('sum_{k=1..K2} (c_(2,k) ) = '+str(np.sum(coeffs_c[1]))+' = ? N2 = '+str(mass_rho_a_list_steps[nb_iter-1][1])+' ')
        print(' ')

        ##############
        # Check convergence
        
        #criterion_CV = 0
        
        ############################
        # METHOD 2 (NEW CRITERION simply MEASURING the deviation between 
        # (c_{a,k}^{(m)} and (c_{a,k}^{(m+1)})
        """
        for a in range(nb_atoms):
            
            criterion_CV += np.sum([(coeffs_c[a][k]-coeffs_c_previous[a][k])**2  for k in range(tab_K[a])])
        """
        ############################
        # METHOD 3 : L2 norm (computed with GL weights) of w_a^(m+1)-w_a^(m)
        criterion_CV = convergence_criterion_GISA_LISA(coeffs_c,
                                                       coeffs_c_previous,
                                                       exponents_GISA,
                                                       x_GL_0_R,w_GL_0_R,Ng_radial,
                                                       Rmax)
        
        criterion_CV_list_step.append(criterion_CV)
        
        print('Convergence criterion GISA sum_a ||w_a^(m+1)-w_a^(m)||_{L2,GL} iter. m='+str(nb_iter))
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
    print(' ')
    print('mass_rho_a_list_steps')
    print(mass_rho_a_list_steps)
    print(' ')
    print(' ')
    
    print('CONVERGENCE ACHIEVED')
    print('----------------------------')
    print('Converged values of coefficients (c_(a,k))_{a=1..M,k=1..Ka} :')
    print(coeffs_c)
    print('----------------------------')
    print(' ')

    print('w_1_values_FINAL_atom_0')
    
    w_1_values_FINAL = [compute_w_a_GISA(Rmax*x_GL_0_R[i],coeffs_c[0],exponents_GISA[0]) for i in range(len(x_GL_0_R))]
    
    print(w_1_values_FINAL)
    print(' ')
    
    w_1_values_FINAL_PLOT = [compute_w_a_GISA(tab_r_plot_FINAL[i],coeffs_c[0],exponents_GISA[0]) for i in range(len(tab_r_plot_FINAL))]

    
    print('w_2_values_FINAL_atom_1')
    
    w_2_values_FINAL = [compute_w_a_GISA(Rmax*x_GL_0_R[i],coeffs_c[1],exponents_GISA[1]) for i in range(len(x_GL_0_R))]
    print(w_2_values_FINAL)
    
    print(' ')
    
    w_2_values_FINAL_PLOT = [compute_w_a_GISA(tab_r_plot_FINAL[i],coeffs_c[1],exponents_GISA[1]) for i in range(len(tab_r_plot_FINAL))]

    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) GISA")

    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_1_values_FINAL)
    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_1_values_FINAL,'bo')

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_values_FINAL_PLOT)
    #matplotlib.pyplot.scatter(tab_r_plot,w_1_values_FINAL_PLOT)
    
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"GISA_w_1_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()

    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) GISA")

    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_2_values_FINAL)
    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_2_values_FINAL,'bo')

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_values_FINAL_PLOT)
    #matplotlib.pyplot.scatter(tab_r_plot,w_2_values_FINAL_PLOT)
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"GISA_w_2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')
    
    matplotlib.pyplot.show()
    
    #########################
    # PLOT r --> 4*pi*r²*wa(r)
    #w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (Rmax*x_GL_0_R[i])**2 * compute_w_a_GISA(Rmax*x_GL_0_R[i],coeffs_c[0],exponents_GISA[0])) for i in range(len(x_GL_0_R))]
    #w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (Rmax*x_GL_0_R[i])**2 * compute_w_a_GISA(Rmax*x_GL_0_R[i],coeffs_c[1],exponents_GISA[1])) for i in range(len(x_GL_0_R))]
    
    w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_1_values_FINAL_PLOT[i]) for i in range(len(tab_r_plot_FINAL))]
    w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_2_values_FINAL_PLOT[i]) for i in range(len(tab_r_plot_FINAL))]
       
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_1(r) [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) GISA-classic")

    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_1_times_r2_values_FINAL)
    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_1_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL,'bo')
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')
    
    matplotlib.pyplot.savefig(ISA_plots_dir+"GISA_w_1_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_2(r)  [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) GISA-classic")

    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_2_times_r2_values_FINAL)
    #matplotlib.pyplot.plot(np.dot(Rmax,x_GL_0_R),w_2_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL)
    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"GISA_w_2_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    #########################
    ## Compute local atomic charges, dipoles and quadrupolar tensors AT CONVERGENCE
    ## using the obtained converged w_a(.) [here, obtained by the GISA method]
    
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
                              logfile_GISA)
    
    return  coeffs_c, w_1_times_r2_values_FINAL, w_2_times_r2_values_FINAL
################################################################


################################################################
"""
- exponents_GISA_atom =  exponents_GISA[a]
"""
def compute_matrix_A_atom_GISA(exponents_GISA_atom):
    
    K_a = len(exponents_GISA_atom)
    
    A = np.zeros([K_a,K_a])

    A = [[((exponents_GISA_atom[k]*exponents_GISA_atom[l])**(3./2.))/math.sqrt(exponents_GISA_atom[k]+exponents_GISA_atom[l])**3 for l in range(K_a)] for k in range(K_a)]

    return np.dot(math.sqrt(math.pi)/(4.*math.pi**3), A)
################################################################

################################################################
# Evaluate w_(a,GISA)(r) from coefficients (c_(a,k))_{k=1..Ka} and exponents
# (alpha_(a,k))_{k=1..Ka}, taking by convention w_(a,GISA)(.) centered in R_a
"""
- a = atom index
- 'exponents_GISA_atom' provided in ATOMIC units
=> (alpha/pi)**(3./2.) result in Bohr^(-3) [unit of the density : e/Bohr³ (atomic units)] 
"""
def compute_w_a_GISA(r,coeffs_c_atom,exponents_GISA_atom):
    
    K_a = len(exponents_GISA_atom)
    
    return np.sum([coeffs_c_atom[k]*(exponents_GISA_atom[k]/math.pi)**(3./2.) * math.exp(-exponents_GISA_atom[k] * (r/conversion_bohr_angstrom)**2) for k in range(K_a)])
################################################################




################################################################
# Evaluate a basis function of the GISA method :
# g_(a,k)(r) = (alpha_(a,k)/pi)^(3/2) * exp(-alpha_(a,k) * r²) at r 
#(distance with respect to atom a at stake)
"""
- 'exponents_GISA_atom' provided in ATOMIC units
"""
def compute_basis_function_GISA(r,k,exponents_GISA_atom):
        
    return (exponents_GISA_atom[k]/math.pi)**(3./2.) * math.exp(-exponents_GISA_atom[k] * (r/conversion_bohr_angstrom)**2) 
################################################################

"""
- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
"""
################################################################
def compute_vector_B_atom_GISA(a,R_max,Ng_radial,Ng_theta,coeffs_c,exponents_GISA,
                               atomic_coordinates,
                               values_rho_atom,
                               x_GL_0_1, w_GL_0_1,
                               x_GL_0_R, w_GL_0_R):
        
    K_a = len(exponents_GISA[a])

    vector_B = np.zeros(K_a)

    x_GL_0_R_scaled = [R_max*x_GL_0_R[l] for l in range(Ng_radial)]

    tab_sph_avg = [ compute_sph_avg_integrand_total_mass(l,x_GL_0_R_scaled[l],
                                                         Ng_theta,a,
                                                         coeffs_c,exponents_GISA,
                                                         values_rho_atom,
                                                         x_GL_0_1, w_GL_0_1,
                                                         atomic_coordinates) for l in range(Ng_radial)]
    ########
    for k in range(K_a):
                    
        # Computes r² * w_a^(m)(r) * exp(-alpha_(a,k)*r²) * <rho(.)/ (sum_b w_b)>_a(r) on the (GL) radial grid  (R_max*x_l^(GL)):
        # r² added 08/06/21

        tab_integrand = [ w_GL_0_R[l] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * math.exp(-exponents_GISA[a][k] * (x_GL_0_R_scaled[l]/conversion_bohr_angstrom)**2 ) * compute_w_a_GISA(x_GL_0_R_scaled[l],coeffs_c[a],exponents_GISA[a]) * tab_sph_avg[l] for l in range(Ng_radial)]

        # 2*math.pi takes into account the invariance around z axis (along phi) in the Diatomic case
        vector_B[k] = 2*math.pi * (1/(4*math.pi)) * (exponents_GISA[a][k]/math.pi)**(3./2.) * (1/conversion_bohr_angstrom) * R_max * np.sum(tab_integrand)
    ########
    
    return vector_B
################################################################

################################################################
def compute_constant_cost_function_GISA(a,R_max,Ng_radial,
                                        coeffs_c,
                                        exponents_GISA,
                                        x_GL_0_R, w_GL_0_R):

    T_a_constant = 0

    x_GL_0_R_scaled = [R_max*x_GL_0_R[l] for l in range(Ng_radial)]

                    
    # Computes r² * w_a^(m)(r)  on the (GL) radial grid  (R_max*x_l^(GL)):

    tab_integrand = [ w_GL_0_R[l] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_w_a_GISA(x_GL_0_R_scaled[l],coeffs_c[a],exponents_GISA[a])  for l in range(Ng_radial)]

    T_a_constant =  (1/conversion_bohr_angstrom) * R_max * np.sum(tab_integrand)
    
    
    return T_a_constant

################################################################

################################################################
"""
- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- a = atom index

Parameters :
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - R--> infinity (R >> |R2-R1| the distance between the two atoms)
     (general case : R >> typical extension of the electronic cloud of the molecule)
"""
def compute_mass_rho_a_GISA(a,R_max,
                            Ng_radial,
                            Ng_theta,
                            coeffs_c,
                            exponents_GISA,
                            values_rho_atom,
                            x_GL_0_1, w_GL_0_1,
                            x_GL_0_R, w_GL_0_R,
                            atomic_coordinates):
    
    
    x_GL_0_R_scaled = [R_max*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * <rho(.)/ (sum_b w_b)>_a(r) on the (GL) radial grid  (R*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_w_a_GISA(x_GL_0_R_scaled[l],coeffs_c[a],exponents_GISA[a]) * compute_sph_avg_integrand_total_mass(l,x_GL_0_R_scaled[l],
                                                                                                                                                                                          Ng_theta,a,
                                                                                                                                                                                          coeffs_c,exponents_GISA,
                                                                                                                                                                                          values_rho_atom,
                                                                                                                                                                                          x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                          atomic_coordinates) for l in range(Ng_radial)]

    # Take into account Gauss-Legendre weights :
    return 2*math.pi * (1/conversion_bohr_angstrom) * R_max * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_integrand_total_mass(index_radial,r,Ng_theta,
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
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(compute_w_a_GISA(r,coeffs_c[a],exponents_GISA[a])+compute_w_a_GISA(np.linalg.norm(grid_points_3D_atom_shifted_other_atom[u]),coeffs_c[abs(a-1)],exponents_GISA[abs(a-1)]))  for u in range(Ng_theta)]
    
    # Integral on theta only (not on phi)
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
[VERIFICATION PURPOSES ONLY]

Computes the integral over R³ of rho(x) dx as :
    
    int_{0..\infty} 4*pi*r² * <rho>_a(r) dr
    = 2*pi * int_{0..\infty}  r² * int_{0..pi} [sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))] d theta
    
- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- a = atom index

- values_rho_atom = values of rho(.) evaluated at the points around atom a :
   (R_a+r_l.(sin(theta_j),0,cos(theta_j)))_{l=1..,Nb_radial,j=1..Ng_theta} 
   
(the index of atom a is implicit : this function is called for values_rho_atom_BIS = values_rho_BIS[a])

Parameters :
    - Ng_theta : number of Gauss-Legendre points in theta

    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - R--> infinity (R >> |R2-R1| the distance between the two atoms)
     (general case : R >> typical extension of the electronic cloud of the molecule)
"""
def compute_mass_rho_MOLECULAR_GISA(R,
                                    Ng_radial,
                                    Ng_theta,
                                    values_rho_atom,
                                    x_GL_0_1, w_GL_0_1,
                                    x_GL_0_R, w_GL_0_R):
    
    
    x_GL_0_R_scaled = [R*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * <rho(.)/ (sum_b w_b)>_a(r) on the (GL) radial grid  (R*x_l^(GL)):
    # BEWARE !! Molecular density in atomic units ==> conversion of distances in Bohrs !!
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_sph_avg_integrand_mass_rho_MOLECULAR(l,
                                                                                                                             Ng_theta,
                                                                                                                             values_rho_atom,
                                                                                                                             x_GL_0_1, 
                                                                                                                             w_GL_0_1) for l in range(Ng_radial)]
    
    # Take into account Gauss-Legendre weights :
    # BEWARE !! Conversion of distances in Bohrs (atomic units) !!
    return 2*math.pi * (1/conversion_bohr_angstrom) * R * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################


################################################################
"""
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- Ng_theta : number of Gauss-Legendre points in theta

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.) at distance r from atom a :
def compute_sph_avg_integrand_mass_rho_MOLECULAR(index_radial,
                                                 Ng_theta,
                                                 values_rho_atom,
                                                 x_GL_0_1, w_GL_0_1):
        
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    values_integrand_theta = [ (w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) ) for u in range(Ng_theta)]
    
    # Returns a result in atomic units (unit of the density rho(.))
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
################################################################
################################################################
"""
[VERIFICATION PURPOSES ONLY]

Diatomic case but WITHOUT assuming invariance along phi (in spherical coordinate)
=> To solve problem with the total charge of O2 not recovered by
'compute_mass_rho_MOLECULAR_GISA()' function (Gauss-Legendre radial and angular integrations)

Computes the integral over R³ of rho(x) dx as :
    
    int_{0..\infty} 4*pi*r² * <rho>_a(r) dr
    = 2*pi * int_{0..\infty}  r² * int_{0..2*pi} int_{0..pi} [sin(theta) * rho(R_a+r.(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)))] d theta
    
- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- a = atom index

- values_rho_atom_BIS = values of rho(.) evaluated at the points around atom a :
   (R_a+r_l.(sin(theta_j)*cos(phi_i),sin(theta_j)*sin(phi_i),cos(theta_j)))_{l=1..,Nb_radial,j=1..Ng_theta,i=1..Ng_phi} 
   
(the index of atom a is implicit : this function is called for values_rho_atom_BIS = values_rho_BIS[a])

Parameters :
    - Ng_theta : number of Gauss-Legendre points in theta
    - Ng_phi : number of Gauss-Legendre points in phi
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - R--> infinity (R >> |R2-R1| the distance between the two atoms)
     (general case : R >> typical extension of the electronic cloud of the molecule)
"""
def compute_mass_rho_MOLECULAR_GISA_BIS(R,
                                        Ng_radial,
                                        Ng_theta,
                                        Ng_phi,
                                        values_rho_atom_BIS,
                                        x_GL_0_1, w_GL_0_1,
                                        x_GL_0_R, w_GL_0_R):
    
    
    x_GL_0_R_scaled = [R*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * <rho(.)/ (sum_b w_b)>_a(r) on the (GL) radial grid  (R*x_l^(GL)):
    # BEWARE !! Molecular density in atomic units ==> conversion of distances in Bohrs !!
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_sph_avg_integrand_mass_rho_MOLECULAR_BIS(l,
                                                                                                                                 Ng_theta,
                                                                                                                                 Ng_phi,
                                                                                                                                 values_rho_atom_BIS,
                                                                                                                                 x_GL_0_1, w_GL_0_1) for l in range(Ng_radial)]
    
    # Take into account Gauss-Legendre weights :
    # BEWARE !! Conversion of distances in Bohrs (atomic units) !!
    return (1/conversion_bohr_angstrom) * R * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################


################################################################
"""
Diatomic case but WITHOUT assuming invariance along phi (in spherical coordinate)

-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
- Ng_theta : number of Gauss-Legendre points in theta
- Ng_phi : number of Gauss-Legendre points in phi

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index

- values_rho_atom_BIS = values of rho(.) evaluated at the points around atom a :
   (R_a+r_l.(sin(theta_j)*cos(phi_i),sin(theta_j)*sin(phi_i),cos(theta_j)))_{l=1..,Nb_radial,j=1..Ng_theta,i=1..Ng_phi} 
   
"""
# Spherical average of rho(.) at distance r=r_l=tab_r[index_radial] = x_GL_0_R[index_radial] from atom a :
def compute_sph_avg_integrand_mass_rho_MOLECULAR_BIS(index_radial,
                                                     Ng_theta,
                                                     Ng_phi,
                                                     values_rho_atom_BIS,
                                                     x_GL_0_1, w_GL_0_1):
        
    # Molecular density rho(.) evaluated at (R_a+r_l.(sin(theta_j)*cos(phi_i),sin(theta_j)*sin(phi_i),cos(theta_j)))_{l=1..,Nb_radial,j=1..Ng_theta,i=1..Ng_phi}  : 
    
    sum_values_integrand_theta = np.sum([ np.sum([(w_GL_0_1[u]  * w_GL_0_1[v] * values_rho_atom_BIS[index_radial][u][v] * math.sin(math.pi*x_GL_0_1[u]) ) for v in range(Ng_phi)]) for u in range(Ng_theta) ])
    
    # Returns a result in atomic units (unit of the density rho(.))
    return 2* math.pi * math.pi * sum_values_integrand_theta
################################################################


################################################################
"""
Precomputes h_atom = [h_atom[a]]_{a=1..M} with 
h_atom[a] = { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
  
  where (r_j)_j = radial grid for radial integration (N_radial points) and
        (theta_l)_l =  angular grid for angular integration (on theta in spherical coordinates)
        [ N_theta points ]

Uses the invariance along phi (i.e. along z axis) in the DIATOMIC case

- a = atom index

- Rmax >> |R2-R1| (integral over [0,+\infty] approximated by an integral over [0,Rmax])

- x_GL_0_R, w_GL_0_R : obtained previously as lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 (Angular Gauss-Legendre grid): obtained previously as : lgwt(Ng_theta,0,1)

"""
def precompute_molecular_density_DIATOMIC(Ng_radial,Ng_theta,Rmax,
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
                                          contraction_coefficients_pSHELLS_to_cSHELLS):
    
    # Scaled radial Gauss-Legendre grid :
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]

    # The 2 atoms are along the z axis
    grid_points_3D_atom_1 =[ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[0][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    grid_points_3D_atom_2 =[ [ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l]) ,
                              0,
                              atomic_coordinates[1][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    # Molecular density rho_(j,l) evaluated at (R_a+r_j.(sin(theta_l),0,cos(theta_l))) :
        
    values_rho_around_atom_1 = [ [ compute_density_grid_point(grid_points_3D_atom_1[k][l],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS)  for l in range(Ng_theta)] for k in range(Ng_radial) ]
 
    
        
    values_rho_around_atom_2 = [ [ compute_density_grid_point(grid_points_3D_atom_2[k][l],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS)  for l in range(Ng_theta)] for k in range(Ng_radial) ]
 
    
    return values_rho_around_atom_1, values_rho_around_atom_2
################################################################



################################################################
"""
Diatomic case but WITHOUT assuming invariance along phi (in spherical coordinate)

Precomputes h_atom = [h_atom[a]]_{a=1..M} with 
h_atom[a] = { rho(R_a+r_l.(sin(theta_j)*cos(phi_i),sin(theta_j)*sin(phi_i),cos(theta_j)))_{l=1..,Nb_radial,j=1..Ng_theta,i=1..Ng_phi} 
  
  where (r_j)_j = radial grid for radial integration (N_radial points) and
        (theta_l)_l =  angular grid for angular integration (on theta in spherical coordinates)
        [ N_theta points ]

Uses the invariance along phi (i.e. along z axis) in the DIATOMIC case

- a = atom index

- Rmax >> |R2-R1| (integral over [0,+\infty] approximated by an integral over [0,Rmax])

- x_GL_0_R, w_GL_0_R : obtained previously as lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 (Angular Gauss-Legendre grid for theta AND phi) : 
    obtained previously as : lgwt(Ng_theta,0,1)

"""
def precompute_molecular_density_DIATOMIC_BIS(Ng_radial,Ng_theta,Ng_phi,Rmax,
                                              x_GL_0_1, w_GL_0_1,
                                              x_GL_0_R, w_GL_0_R,
                                              atomic_coordinates,
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
                                              contraction_coefficients_pSHELLS_to_cSHELLS):
        
    # Scaled radial Gauss-Legendre grid :
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]

    # The 2 atoms are along the z axis
    grid_points_3D_atom_1 =[ [[ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l])*math.cos(2*math.pi*x_GL_0_1[v]) ,
                              x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l])*math.sin(2*math.pi*x_GL_0_1[v]),
                              atomic_coordinates[0][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ] for v in range(Ng_phi)]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    grid_points_3D_atom_2 =[ [[ [x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l])*math.cos(2*math.pi*x_GL_0_1[v]) ,
                              x_GL_0_R_scaled[k] * math.sin(math.pi*x_GL_0_1[l])*math.sin(2*math.pi*x_GL_0_1[v]),
                              atomic_coordinates[1][2] + x_GL_0_R_scaled[k] * math.cos(math.pi*x_GL_0_1[l]) ] for v in range(Ng_phi)]  for l in range(Ng_theta)] for k in range(Ng_radial) ]
    
    # Molecular density rho_(j,l) evaluated at (R_a+r_j.(sin(theta_l),0,cos(theta_l))) :
        
    values_rho_around_atom_1_BIS = [ [ [compute_density_grid_point(grid_points_3D_atom_1[k][l][v],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS) for v in range(Ng_phi)] for l in range(Ng_theta)] for k in range(Ng_radial) ]
 
    
        
    values_rho_around_atom_2_BIS = [ [ [ compute_density_grid_point(grid_points_3D_atom_2[k][l][v],
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
                                                              contraction_coefficients_pSHELLS_to_cSHELLS) for v in range(Ng_phi)] for l in range(Ng_theta)] for k in range(Ng_radial) ]
 
    
    return values_rho_around_atom_1_BIS, values_rho_around_atom_2_BIS
################################################################


################################################################
"""
Returns the values of w_b(.) at the angular grid points centered on atom a (Ra) :
    
    {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)}_{1..Ng_theta} at r_l point of a radial grid :

  where (theta_j)_j is an angular Gauss-Legendre grid on theta (x_GL_0_1).
  
  
  
- If b = a : we are simply computing w_b(r) = w_a(r) : no need to compute the values on all the
  angular GL grid : we simply return one value w_a(r)
  
- Otherwise (if b= abs(a-1) in the diatomic case) : we return 
 {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)}_{1..Ng_theta}
 the values of w_b(.) at the grid points centered on the other atom a
"""
def compute_w_b_GISA_tab(b,a,r,
                         x_GL_0_1,Ng_theta,
                         coeffs_c,
                         exponents_GISA,
                         atomic_coordinates):
    
    K_b = len(exponents_GISA[b])

    if (b==a):
        return np.sum([coeffs_c[b][k]*(exponents_GISA[b][k]/math.pi)**(3./2.) * math.exp(-exponents_GISA[b][k] * (r/conversion_bohr_angstrom)**2) for k in range(K_b)])
    
    else:
        # Works only for two atoms on the same z axis :
        grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                                    0,
                                                    atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
        
        
        norm_grid_point_around_a = [ np.linalg.norm(grid_points_3D_atom_shifted_other_atom[l]) for l in range(Ng_theta)]
                        
        # w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) = sum_{k=1..Kb} c_(b,k) * (alpha_(b,k))³ / (8*pi) * exp(-alpha_(b,k)*|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)
        tab_w_b_values=[ np.sum([coeffs_c[b][k]*(exponents_GISA[b][k]/math.pi)**(3./2.) * math.exp(-exponents_GISA[b][k] * (norm_grid_point_around_a[j]/conversion_bohr_angstrom)**2) for k in range(K_b)]) for j in range(Ng_theta)]
        
        return tab_w_b_values
################################################################



################################################################
"""

Computes the atomic partial charge int_{R³} (rho_a(r) dr)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)


- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_charge_GISA(a,Rmax,coeffs_c,
                               exponents_GISA,
                               values_rho_atom,
                               Ng_theta,
                               x_GL_0_1, w_GL_0_1,
                               Ng_radial,
                               x_GL_0_R, w_GL_0_R,
                               atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**2 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
        
    
    # values_w_a[a](x_GL_0_R_scaled[l]) not defined
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_w_b_GISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_GISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_charge_GISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                              coeffs_c,
                                                                                                                                                              exponents_GISA,
                                                                                                                                                              values_rho_atom,
                                                                                                                                                              x_GL_0_1, w_GL_0_1,
                                                                                                                                                              Ng_theta,
                                                                                                                                                              atomic_coordinates) for l in range(Ng_radial)]
    
    return 2*math.pi * (1/conversion_bohr_angstrom) * Rmax * np.sum([w_GL_0_R[l]*tab_integrand[l] for l in range(Ng_radial)])
################################################################

################################################################
"""

Computes the atomic dipole  int_{R³} (z * rho_a(r) dr) along z axis (DIATOMIC case : the 2 atoms are along z)

- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_dipole_GISA(a,Rmax,
                               coeffs_c,
                               exponents_GISA,
                               values_rho_atom,
                               Ng_theta,
                               x_GL_0_1, w_GL_0_1,
                               Ng_radial,
                               x_GL_0_R, w_GL_0_R,
                               atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**3 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**3 * x_GL_0_R_scaled[l]**3 * compute_w_b_GISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_GISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_dipole_GISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                              coeffs_c,
                                                                                                                                                              exponents_GISA,
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


- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_xx_GISA(a,Rmax,
                                      coeffs_c,
                                      exponents_GISA,
                                      values_rho_atom,
                                      Ng_theta,
                                      x_GL_0_1, w_GL_0_1,
                                      Ng_radial,
                                      x_GL_0_R, w_GL_0_R,
                                      atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)**3 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * compute_w_b_GISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_GISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_quadrupole_xx_GISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                                     coeffs_c,
                                                                                                                                                                     exponents_GISA,
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


- a = atom index

Parameters :
    - values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
      radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
    - Ng_radial = number of Gauss-Legendre points in the radial direction 
      (to compute the radial integration)
    - Rmax--> infinity (Rmax >> |R2-R1| the distance between the two atoms)
     (general case : Rmax >> typical extension of the electronic cloud of the molecule)
"""
def compute_atomic_quadrupole_zz_GISA(a,Rmax,
                                      coeffs_c,
                                      exponents_GISA,
                                      values_rho_atom,
                                      Ng_theta,
                                      x_GL_0_1, w_GL_0_1,
                                      Ng_radial,
                                      x_GL_0_R, w_GL_0_R,
                                      atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)* cos(theta)**2 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * compute_w_b_GISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_GISA,
                                                                                                     atomic_coordinates)  * compute_sph_avg_atomic_quadrupole_zz_GISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                                      coeffs_c,
                                                                                                                                                                      exponents_GISA,
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

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)

- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_charge_GISA(index_radial,r,a,
                                       coeffs_c,
                                       exponents_GISA,
                                       values_rho_atom,
                                       x_GL_0_1, w_GL_0_1,
                                       Ng_theta,
                                       atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)
    
    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_GISA_tab(a,a,r,
                                        x_GL_0_1,Ng_theta,
                                        coeffs_c,
                                        exponents_GISA,
                                        atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_GISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_GISA,
                                                     atomic_coordinates)
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
        
    # values_w_a[a][index_radial] = w_a(r)
    
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)


- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_dipole_GISA(index_radial,r,a,
                                       coeffs_c,
                                       exponents_GISA,
                                       values_rho_atom,
                                       x_GL_0_1, w_GL_0_1,
                                       Ng_theta,
                                       atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)


    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_GISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_GISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_GISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_GISA,
                                                     atomic_coordinates)
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u]) /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes :
    int_{0..pi} [ sin(theta)³ * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)


- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_xx_GISA(index_radial,r,a,
                                              coeffs_c,
                                              exponents_GISA,
                                              values_rho_atom,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_theta,
                                              atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)


    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_GISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_GISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_GISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_GISA,
                                                     atomic_coordinates)
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u])**3 /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################



################################################################
"""
Computes :
    int_{0..pi} [ sin(theta) * cos(theta)² * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))

- values_w_a[a] = values of the atomic shape function w_a(.) on the GL 
  radial grid ( Rmax*x^{GL}_l )_{l=1..Nb_radial}
-  x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)


- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_zz_GISA(index_radial,r,a,
                                              coeffs_c,
                                              exponents_GISA,
                                              values_rho_atom,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_theta,
                                              atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_GISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_GISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_GISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_GISA,
                                                     atomic_coordinates)
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u])**2  /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes and prints atomic local charges, dipoles, and quadrupolar tensor components
from the final w_a(.) of the GISA and L-ISA procedure (i.e. simply  from the optimal 
                                                      (c_(a,k)^{opt})_{a,k} coefficients)

Used both to compute the local (atomic) multipole moments at the end of the GISA (classic)
and of the L-ISA (GISA- NEW formulation) procedure
- 'values_rho' : values of the molecular density at radial x angular (in theta) grid points
  previously computed thanks to the function 'precompute_molecular_density_DIATOMIC()'
"""
def compute_local_multipoles_GISA(Rmax,
                                  coeffs_c_GISA_FINAL,
                                  exponents_GISA,
                                  values_rho,
                                  Ng_theta,
                                  x_GL_0_1, w_GL_0_1,
                                  Ng_radial,
                                  x_GL_0_R, w_GL_0_R,
                                  atomic_coordinates,
                                  atomic_numbers,
                                  logfile):

    
    partial_charges=[]

    atomic_dipoles_z = []

    atomic_quadrupoles = []

    print('--------------------')
    print('COMPUTATION ATOMIC MULTIPOLES GISA in e.Ang^k (a.u.)')
    logfile.write('ATOMIC MULTIPOLES in e.Ang^k (a.u.)')
    logfile.write("\n")
    #print('ATOMIC CHARGES (k=0), DIPOLES (k=1) and QUADRUPOLES (k=2) in e.Ang^k (ATOMIC UNITS)')
    print(' ')
    
    for a in range(2):
        
        q_a = compute_atomic_charge_GISA(a,Rmax,
                                         coeffs_c_GISA_FINAL,
                                         exponents_GISA,
                                         values_rho[a],
                                         Ng_theta,
                                         x_GL_0_1, w_GL_0_1,
                                         Ng_radial,
                                         x_GL_0_R, w_GL_0_R,
                                         atomic_coordinates)
        
        partial_charges.append(q_a)
        
        print('q_'+str(a)+' = '+str(q_a))
        print('Partial charge atom '+str(a)+' = '+str(atomic_numbers[a]-q_a))
        logfile.write('q_'+str(a)+' = '+str(q_a))
        logfile.write("\n")
        logfile.write('Partial charge atom '+str(a)+' = '+str(atomic_numbers[a]-q_a))
        logfile.write("\n")
        
        d_z_a = compute_atomic_dipole_GISA(a,Rmax,
                                           coeffs_c_GISA_FINAL,
                                           exponents_GISA,
                                           values_rho[a],
                                           Ng_theta,
                                           x_GL_0_1, w_GL_0_1,
                                           Ng_radial,
                                           x_GL_0_R, w_GL_0_R,
                                           atomic_coordinates)
        
        atomic_dipoles_z.append(d_z_a)
        
        print('d_z_'+str(a)+' = '+str(d_z_a))
        print(' ')
        logfile.write('d_z_'+str(a)+' = '+str(d_z_a))
        logfile.write("\n")
        
        
        Q_xx_a = compute_atomic_quadrupole_xx_GISA(a,Rmax,
                                                   coeffs_c_GISA_FINAL,
                                                   exponents_GISA,
                                                   values_rho[a],
                                                   Ng_theta,
                                                   x_GL_0_1, w_GL_0_1,
                                                   Ng_radial,
                                                   x_GL_0_R, w_GL_0_R,
                                                   atomic_coordinates)
        
        print('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write("\n")
                
     
        Q_zz_a = compute_atomic_quadrupole_zz_GISA(a,Rmax,
                                                   coeffs_c_GISA_FINAL,
                                                   exponents_GISA,
                                                   values_rho[a],
                                                   Ng_theta,
                                                   x_GL_0_1, w_GL_0_1,
                                                   Ng_radial,
                                                   x_GL_0_R, w_GL_0_R,
                                                   atomic_coordinates)
        
        print('Q_zz_'+str(a)+' = '+str(Q_zz_a))
        print(' ')
        logfile.write('Q_zz_'+str(a)+' = '+str(Q_zz_a))
        logfile.write("\n")
        logfile.write("\n")    
     
        atomic_quadrupoles.append(Q_xx_a)
        
        atomic_quadrupoles.append(Q_zz_a)
        
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
def generate_random_guess_GISA(N_a_target,tab_K_GISA):
    
    coeffs_c_init = []
    
    nb_atoms = len(tab_K_GISA)
    
    for a in range(nb_atoms):
        
        temp_random_c_a =[]
        
        for k in range(tab_K_GISA[a]):
            
            c_a_k = np.random.rand()
            
            temp_random_c_a.append(c_a_k)
            
        # Renormalize by ( N_a^{target} / sum_(k=1..Ka) (c_(a,k)) ) where c_(a,k) have been randomly generated between 0 and 1
        sum_tot = np.sum(temp_random_c_a)
        
        coeffs_c_init.append([(N_a_target[a]/sum_tot)*temp_random_c_a[k] for k in range(tab_K_GISA[a])])
                                                                                                                                                                      
    return coeffs_c_init
################################################################
                                                                                                                                                                        
                                                                                                                                                                          
                                                                                                                                                                         
################################################################  
"""
- tab_w_a_ISA_radial[a] = (w_a^{ISA-radial}(r_l)) where (r_l)_{l=1..N^{GL}}
are Gauss-Legendre points used for the radial discretization (and for the quadrature)

- mass_rho_a_ISA_radial[a] = total mass associated to w_a^{ISA-radial}(.) profile
"""  
def generate_GISA_guess_from_ISA_radial_profile(tab_w_a_ISA_radial,
                                                mass_rho_a_ISA_radial,
                                                exponents_GISA,
                                                Ng_radial,
                                                R_max,
                                                x_GL_0_R, w_GL_0_R,
                                                atomic_coordinates,
                                                tab_K):
         
    nb_atoms = len(exponents_GISA)

    coeffs_c=[]
    
    for a in range(nb_atoms):
                     
        coeffs_c.append([])
                                                                                                                                   
        A_a = compute_matrix_A_atom_GISA(exponents_GISA[a])
            
        B_a = compute_vector_B_atom_GISA_fit_ISA_radial(a,tab_w_a_ISA_radial[a],
                                                        R_max,Ng_radial,
                                                        exponents_GISA,
                                                        atomic_coordinates,
                                                        x_GL_0_R, w_GL_0_R)
            

            
            
        ###################################################
        ###################################################
        # Using Python quadratic programming optimization routines 
        # (linear equality or inequality constraints) 
        # to generate coefficients (c_(a,k))_k for GISA fitting to the ISA-radial solution
            
        matrix_constraint = np.zeros([tab_K[a],tab_K[a]+1])
        # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
        matrix_constraint[:,0] = np.ones(tab_K[a])
        # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0 
        matrix_constraint[0:tab_K[a],1:(tab_K[a]+1)] = np.identity(tab_K[a])
            
        vector_contraint = np.zeros(tab_K[a]+1)
            
        # First coefficient : N_a : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
        vector_contraint[0] = mass_rho_a_ISA_radial[a]
            
        coeffs_c[a] = quadprog.solve_qp(G=2*A_a, a = 2*B_a, C=matrix_constraint, b=vector_contraint, meq=1)[0]

        print('Guess coefficients GISA (c_(a,k))_{a=1..M,k=1..Ka} from ISA-radial :')
        print(coeffs_c)
        print(' ')
        
        ############
     
        
    print('Check constraint : sum_{k=1..Ka} (c_(a,k) )')
    print('sum_{k=1..K1} (c_(1,k) ) = '+str(np.sum(coeffs_c[0])))
    print('sum_{k=1..K2} (c_(2,k) ) = '+str(np.sum(coeffs_c[1])))
    print(' ')
        
    return coeffs_c
################################################################            

"""
- x_GL_0_R, w_GL_0_R : obtained previously as : lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : obtained previously as : lgwt(Ng_theta,0,1)
"""
################################################################
def compute_vector_B_atom_GISA_fit_ISA_radial(a,tab_w_a_ISA_radial_atom,
                                              R_max,Ng_radial,
                                              exponents_GISA,
                                              atomic_coordinates,
                                              x_GL_0_R, w_GL_0_R):
        
    K_a = len(exponents_GISA[a])

    vector_B = np.zeros(K_a)

    x_GL_0_R_scaled = [R_max*x_GL_0_R[l] for l in range(Ng_radial)]

    ########
    for k in range(K_a):
                    
        # Computes r² * exp(-alpha_(a,k)*r²) * w_a^{ISA-radial}(r) on the (GL) radial grid  (R_max*x_l^(GL)):
        
        # Use interpolation to compute w_a(r_l) if needed (if the quadrature discretization differs
        # from the radial grid points where w_a^{ISA-radial} was computed)
        tab_integrand = [ w_GL_0_R[l] * (x_GL_0_R_scaled[l]/conversion_bohr_angstrom)**2 * math.exp(-exponents_GISA[a][k] * (x_GL_0_R_scaled[l]/conversion_bohr_angstrom)**2 ) * tab_w_a_ISA_radial_atom[l]  for l in range(Ng_radial)]

        # 2*math.pi takes into account the invariance around z axis (along phi) in the Diatomic case
        vector_B[k] =  (exponents_GISA[a][k]/math.pi)**(3./2.) * (1/conversion_bohr_angstrom) * R_max * np.sum(tab_integrand)
    ########
    
    return vector_B
################################################################

###############################################################
# FUNCTIONS FOR ENTROPY CALCULATIONS :

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
"""
Returns the  entropy s_{KL}(rho_a^(m)|rho_a^{0,(m)}) at step m
"""
def computes_entropy_GISA(a,c_coeffs,
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
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_a_GISA(x_GL_0_R_scaled[u],c_coeffs[a],exponents_GISA[a]) * (-compute_sph_avg_entropy_1(u,x_GL_0_R_scaled[u],Ng_theta,
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
coeffs_c_previous = (c_(a,k)^(m))_{k=1..m_a}
coeffs_c          = (c_(a,k)^(m+1))_{k=1..m_a}

Computes sqrt(4*pi*int_0^{+\infty} r² * |sum_{k=1..m_a} (c_(a,k)^(m+1)-c_(a,k)^(m))*(alpha_(a,k)/pi)**(3/2)*exp(-alpha_(a,k)*r²) |² dr) 
         = || w_a^(m+1)(.)-w_a^(m)(.) ||_{L²,GL}

the convergence criterion of GISA and LISA

"""
def convergence_criterion_GISA_LISA(coeffs_c,
                                    coeffs_c_previous,
                                    exponents_GISA,
                                    x_GL_0_R,w_GL_0_R,Ng_radial,
                                    Rmax):

    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]

    CV_criterion = 0
    
    nb_atoms = len(exponents_GISA)

    for a in range(nb_atoms):
            
        tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * np.sum([(coeffs_c[a][k]-coeffs_c_previous[a][k])*(exponents_GISA[a][k]/math.pi)**(3./2.) * math.exp(-exponents_GISA[a][k] * (x_GL_0_R_scaled[u]/conversion_bohr_angstrom)**2) for k in range(len(exponents_GISA[a]))])**2 for u in range(Ng_radial)]            

        CV_criterion += np.sum(tab_integrand_a)

    return math.sqrt(4*math.pi*CV_criterion)
################################################################




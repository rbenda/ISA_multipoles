#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:32:05 2021

Functions for MB-ISA algorithm (Verstraelen et al. 2016, JCTC)
"""


import numpy as np
import math
import matplotlib
from matplotlib import pyplot

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
# MB-ISA method functions :

    
################################################################
"""
- tab_K = {K_a}_{a=1..M} = number of shells per atom

- Rmax >> |R2-R1| (integral over [0,+\infty] approximated by an integral over [0,Rmax])

- x_GL_0_R, w_GL_0_R : radial Gauss-Legendre grid obtained previously as lgwt(Ng_radial,0,1)
- x_GL_0_1, w_GL_0_1 : angular Gauss-Legendre grid obtained previously as : lgwt(Ng_theta,0,1)

"""
def MBISA_classic_algorithm(tab_K,coeffs_c_init,
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
                            tab_r_plot_FINAL):
    
    
    nb_atoms = len(tab_K)
    
    # List of list of coeffs. (c_(a,k))_{a=1..M,k=1..Ka} at all successive iterations of the MB-ISA algorithm :
    coeffs_c_iter=[]
    
    # List of list of exponents (alpha_(a,k))_{a=1..M,k=1..Ka} at all successive iterations of the MB-ISA algorithm :
    exponents_alpha_iter = []
    
    # (N_a)_{a=1..M} : masses of the rho_a(.) 
    mass_rho_a = np.zeros(nb_atoms)
    
    total_sum_Na_list_steps=[]

    #############
    # Initialisation : w_a^(0) : derived from the coefficients : 
    # (c_{a,k}^{(0)})_{a=1..M,k=1..K_a}  and (alpha_{a,k}^{(0)})_{a=1..M,k=1..K_a}
    
    print('---------------------------------')
    print('MB-ISA method : initialization of coefficients and exponents')
    coeffs_c_iter.append([])
    
    print('coeffs_c_init')
    print(coeffs_c_init)
    
    for a in range(nb_atoms):
    
        coeffs_c_iter[0].append([coeffs_c_init[a][k] for k in range(tab_K[a])])

    exponents_alpha_iter.append([])
    
    for a in range(nb_atoms):
        
        exponents_alpha_iter[0].append([exponents_MBISA_init[a][k] for k in range(tab_K[a])])

    print('Initial coefficients (c_(a,k))_{a=1..M,k=1..Ka} MB-ISA :')
    print(coeffs_c_iter[0])
    print(' ')
   
    print('Initial exponents (alpha_(a,k))_{a=1..M,k=1..Ka} MB-ISA :')
    print(exponents_alpha_iter[0])
    print(' ')
    
    
    print('coeffs_c_iter')
    print(coeffs_c_iter)
    print(' ')
    print('exponents_alpha_iter')
    print(exponents_alpha_iter)
    print(' ')
    # (N_a)_{a=1..M} : masses of the rho_a(.) = sum_{k=1..Ka} (c_(a,k))
    #mass_rho_a = np.zeros(nb_atoms)
    
    nb_iter=0
    
    criterion_CV=1000
    
    criterion_CV_list_step=[]
    
    coeffs_c_current = coeffs_c_iter[0]

    print('Initial coeffs_c_current')
    print(coeffs_c_current)
    print(' ')
    
    exponents_alpha_current = exponents_alpha_iter[0]
    
    print('Initial exponents_alpha_current')
    print(exponents_alpha_current)
    print(' ')
    
    total_entropy_list_step=[]
    
    # Threshold for convergence :
    epsilon = 1e-10
    
    ################
    # MB-ISA loop
    while ((criterion_CV > epsilon) and (nb_iter<300)):
        
        print('MB-ISA ITERATION n°'+str(nb_iter))
        #print('coeffs_c_current')
        #print(coeffs_c_current)
        #print('exponents_alpha_current')
        #print(exponents_alpha_current)
        print(' ')
        
        coeffs_c_iter.append([])

        exponents_alpha_iter.append([])

        coeffs_c_atoms_new_iter = []
        
        ##########################
        for a in range(nb_atoms):
                            
            # Compute (c_(a,k)^(m+1))_{k=1..Ka} 
            # from (c_{b,l}^{(m)})_{b=1..M,l=1..Kb} and from
            # (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb} 
            #[coefficients and exponents of all atoms
            # and all shells at the previous iteration (loop)]
                
            # coeffs_c_current = coeffs_c_iter[nb_iter]
            # exponents_current = exponents_alpha_iter[nb_iter]
            coeffs_c_atom_a_NEW = computes_new_coeffs_MB_ISA(a,coeffs_c_current,
                                                             exponents_alpha_current,
                                                             values_rho[a],
                                                             Ng_theta,
                                                             x_GL_0_1, w_GL_0_1,
                                                             Ng_radial,
                                                             x_GL_0_R, w_GL_0_R,
                                                             Rmax,
                                                             atomic_coordinates)
            
            coeffs_c_atoms_new_iter.append(coeffs_c_atom_a_NEW)
        
        ##########################
        
        coeffs_c_iter[nb_iter+1].append(coeffs_c_atoms_new_iter)
        
        #Store the previous coefficients (c_(a,k)^(m))_{a=1..M,k=1..Ka}
        coeffs_c_previous = coeffs_c_current
        
        coeffs_c_current = coeffs_c_atoms_new_iter
        print(' ')
        #print('coeffs_c_iter')
        #print(coeffs_c_iter)
        #print(' ')
        print('New coefficients (c_(a,k)^(m+1))_{a=1..M,k=1..Ka} MB-ISA : iteration m ='+str(nb_iter))
        print(coeffs_c_iter[nb_iter+1])
        print(' ')
   
        # CHECK 'constraint' of conserved total mass (sum_a int rho_a = N = nb of electrons)
        # even if this constraint is not explicitly imposed (implicit via Hirshfled formula)
        mass_rho_a = [np.sum(coeffs_c_iter[nb_iter+1][0][0]),np.sum(coeffs_c_iter[nb_iter+1][0][1])]
        print('mass_rho_a MB-ISA iteration m ='+str(nb_iter))
        print(mass_rho_a)
        print(' ')
        print('Check total sum_a( sum_{k=1..Ka} c_(a,k)^(m+1) ) = N ; MB-ISA  :')
        total_sum_Na = np.sum(mass_rho_a)
        total_sum_Na_list_steps.append(total_sum_Na)
        print(total_sum_Na)
        print(' ')

        # Once c_{a,k}^{(m+1)} are computed for all atoms a and all associated shells k=1..Ka :
        # we can deduce the (alpha_{b,l}^{(m+1)})_{b=1..M,l=1..Kb} from :
        # (c_{b,l}^{(m+1)})_{b=1..M,l=1..Kb}
        # (c_{b,l}^{(m)})_{b=1..M,l=1..Kb}
        # (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb}
        
        exponents_alpha_atoms_new_iter = []

        ##########################
        for a in range(nb_atoms):
                            
            # Compute (alpha_(a,k)^(m+1))_{k=1..Ka} 
            # from (c_{a,k}^{(m+1)})_{k=1..Ka} = coeffs_c_iter[nb_iter+1][a]
            # (c_{b,l}^{(m)})_{b=1..M,l=1..Kb} = coeffs_c_iter[nb_iter] 
            # (coefficients c_(b,k) at the previous iteration) and from
            # (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb} 
            # [coeffs. and exponents of all atoms and all shells at the previous iteration (loop)]
            
            # coeffs_c_current[a] = coeffs_c_iter[nb_iter+1][0][a] = (c(a,k)^(m+1))_{k=1..Ka} 
            # coeffs_c_previous = coeffs_c_iter[nb_iter][0] 
            # exponents_alpha_current = exponents_alpha_iter[nb_iter][0]
            exponents_atom_a_NEW = computes_new_exponents_MB_ISA(a,coeffs_c_current[a],
                                                                 coeffs_c_previous,
                                                                 exponents_alpha_current,
                                                                 values_rho[a],
                                                                 Ng_theta,
                                                                 x_GL_0_1, w_GL_0_1,
                                                                 Ng_radial,
                                                                 x_GL_0_R, w_GL_0_R,
                                                                 Rmax,
                                                                 atomic_coordinates)
            
            

    
            exponents_alpha_atoms_new_iter.append(exponents_atom_a_NEW)
        
        ##########################
        
        exponents_alpha_iter[nb_iter+1].append(exponents_alpha_atoms_new_iter)
        
        #Store the previous coefficients (c_(a,k)^(m))_{a=1..M,k=1..Ka}
        exponents_alpha_previous = exponents_alpha_current
        
        exponents_alpha_current = exponents_alpha_atoms_new_iter
        
        #print('exponents_alpha_iter')
        #print(exponents_alpha_iter)
        #print(' ')
        print('New exponents (alpha_(a,k)^(m+1))_{a=1..M,k=1..Ka} MB-ISA : iteration m ='+str(nb_iter))
        print(exponents_alpha_iter[nb_iter+1])
        print(' ')
        
        # CHECK CONSTRAINT sum_{k=1..Ka} (c_(a,k)) = Na ??
        
        ########################  
        # CALCUL de l'entropie S(rho^(m+1)|rho^{0,(m)}) :
        total_entropy_tab = [ 4*math.pi*computes_entropy_MB_ISA(a,coeffs_c_current,
                                                                Rmax, 
                                                                atomic_coordinates,
                                                                x_GL_0_R,w_GL_0_R,Ng_radial,
                                                                x_GL_0_1, w_GL_0_1,Ng_theta,
                                                                exponents_alpha_current,
                                                                values_rho) for a in range(2)]
        
        
        
        total_entropy = np.sum(total_entropy_tab)

        print('Total_entropy MB-ISA ITERATION '+str(nb_iter)+' = '+str(total_entropy))
        print(' ')
                
        print('Atom 1 : '+str(total_entropy_tab[0])+' ; atom 2 : '+str(total_entropy_tab[1]))
        print(' ')

        total_entropy_list_step.append(total_entropy)
        
        ############
        # OLD convergence criterion :
        # Test the convergence of the values of (c_(a,k))_{a=1..M,l=1..Ka} 
        # and (alpha_(a,k))_{a=1..M,k=1..Ka}      
        
        """
        # sum_{a=1..M,k=1..Ka} | alpha_(a,k)^(m+1) - alpha_(a,k)^(m) |² :
        ## criterion_CV_exponents = np.sum( [np.sum([abs(exponents_alpha_iter[nb_iter+1][a][k]-exponents_alpha_iter[nb_iter][a][k])**2 for k in range(tab_K[a])]) for a in range(nb_atoms)])
        criterion_CV_exponents = np.sum([np.sum([abs(exponents_alpha_current[a][k]-exponents_alpha_previous[a][k])**2 for k in range(tab_K[a])]) for a in range(nb_atoms)])
        
        # sum_{a=1..M,k=1..Ka} | c(a,k)^(m+1) - c(a,k)^(m) |² :
        ## criterion_CV_coeffs= np.sum([np.sum([abs(coeffs_c_iter[nb_iter+1][a][k]-coeffs_c_iter[nb_iter][a][k])**2 for k in range(tab_K[a])] )for a in range(nb_atoms)])
        criterion_CV_coeffs = np.sum([np.sum([abs(coeffs_c_current[a][k]-coeffs_c_previous[a][k])**2 for k in range(tab_K[a])]) for a in range(nb_atoms)])

        test = criterion_CV_exponents + criterion_CV_coeffs
        """
        
        # NEW convergence criterion (L² norm of the difference of successive iterates)
        criterion_CV =convergence_criterion_MB_ISA(coeffs_c_current,
                                                   coeffs_c_previous,
                                                   exponents_alpha_current,
                                                   exponents_alpha_previous,
                                                   x_GL_0_R,w_GL_0_R,Ng_radial,
                                                   Rmax)
        
        criterion_CV_list_step.append(criterion_CV)
        
        nb_iter += 1
        
        print('Convergence criterion MB-ISA sum_(a,k) ( | alpha_(a,k)^(m+1) - alpha_(a,k)^(m) |² + | c(a,k)^(m+1) - c(a,k)^(m) |² ) iteration n°'+str(nb_iter))
        #print(test)
        print(criterion_CV)
        print('End iteration '+str(nb_iter))
        print('----------------------------------')
    #################

    #total_iter = len(coeffs_c_iter)
    
    print('CONVERGENCE ACHIEVED')
    print('criterion_CV = '+str(criterion_CV))
    print(' ')
    print('criterion_CV_list_step')
    print(criterion_CV_list_step)
    print(' ')
    print('TOTAL number of iterations to converge = '+str(nb_iter)+' for epsilon = '+str(epsilon))
    print(' ')
    print('--------------')
    print('MB-ISA final coefficients c_(a,k) :')
    print(coeffs_c_current)
    print('MB-ISA final exponents alpha_(a,k) :')
    print(exponents_alpha_current)
    print(' ')
    print('total_sum_Na_list_steps')
    print(total_sum_Na_list_steps)
    print(' ')
    print(' ')
    print('total_entropy_list_step')
    print(total_entropy_list_step)
    print(' ')
    
    print('w_1_values_FINAL_atom_0')
    

    # ANALYTIC functions (w_1(.) and w_2(.)) => we can plot them continuously
    # coeffs_c_iter[total_iter] = coeffs_c_current
    # exponents_alpha_iter[total_iter] = exponents_alpha_current
    w_1_values_FINAL = [compute_w_b_MBISA_tab(0,0,tab_r_plot_FINAL[i],
                                              x_GL_0_1,Ng_theta,
                                              coeffs_c_current,
                                              exponents_alpha_current,
                                              atomic_coordinates) for i in range(len(tab_r_plot_FINAL))]
    
   # print(w_1_values_FINAL)
    print(' ')
    
    print('w_2_values_FINAL_atom_1')
    
    w_2_values_FINAL = [compute_w_b_MBISA_tab(1,1,tab_r_plot_FINAL[i],
                                              x_GL_0_1,Ng_theta,
                                              coeffs_c_current,
                                              exponents_alpha_current,
                                              atomic_coordinates) for i in range(len(tab_r_plot_FINAL))]
    
    #print(w_2_values_FINAL)
    print(' ')
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('w_1(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) MB-ISA")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_values_FINAL)

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    #matplotlib.pyplot.plot(tab_r_plot,w_1_values_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"MB-ISA_"+str(len(exponents_MBISA_init[0]))+"_shells_w_1_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)
    
    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('w_2(.) weight factor')

    matplotlib.pyplot.title("Weight factor around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) MB-ISA")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_values_FINAL)

    #matplotlib.pyplot.plot(tab_r_plot,w_2_values_FINAL,'bo')

    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"MB-ISA_"+str(len(exponents_MBISA_init[0]))+"_shells_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()
    
    #############################
    # PLOT r--> 4*pi*r²*wa(r)
    w_1_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_1_values_FINAL[i]) for i in range(len(tab_r_plot_FINAL)) ]
    
    w_2_times_r2_values_FINAL = [np.log10(4*math.pi * (tab_r_plot_FINAL[i])**2 * w_2_values_FINAL[i]) for i in range(len(tab_r_plot_FINAL)) ]
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R1| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_1(r) [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 1 ( "+str(periodic_table[atomic_numbers[0]])+" ) MB-ISA")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_1_times_r2_values_FINAL,'bo')

    matplotlib.pyplot.savefig(ISA_plots_dir+"MB-ISA_"+str(len(exponents_MBISA_init[0]))+"_shells_w_1_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')
   
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.show()
    
    matplotlib.pyplot.figure(figsize=(15,12),dpi=400)

    matplotlib.pyplot.xlabel('|r-R2| (Å)')
    matplotlib.pyplot.ylabel('4*pi*r²*w_2(r)  [LOG SCALE]')

    matplotlib.pyplot.title("Weight factor times r² around atom n° 2 ( "+str(periodic_table[atomic_numbers[1]])+" ) MB-ISA")

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL)

    matplotlib.pyplot.plot(tab_r_plot_FINAL,w_2_times_r2_values_FINAL,'bo')
    
    pyplot.axvline(x=abs(atomic_coordinates[0][2]-atomic_coordinates[1][2]),color='red')

    matplotlib.pyplot.savefig(ISA_plots_dir+"MB-ISA_"+str(len(exponents_MBISA_init[0]))+"_shells_w_2_times_4pir2_Ng_radial_"+str(Ng_radial)+"_Ng_theta_"+str(Ng_theta)+"_Z1_"+str(atomic_numbers[0])+"_Z2_"+str(atomic_numbers[1])+".png",bbox_inches='tight')

    matplotlib.pyplot.show()

    ###############################
    # Compute atomic local multipoles (up to quadrupoles):
        
    compute_local_multipoles_MBISA(Rmax,
                                   coeffs_c_current,
                                   exponents_alpha_current,
                                   values_rho,
                                   Ng_theta,
                                   x_GL_0_1, w_GL_0_1,
                                   Ng_radial,
                                   x_GL_0_R, w_GL_0_R,
                                   atomic_coordinates,
                                   atomic_numbers,
                                   logfile_MBISA)
                              
    
    return coeffs_c_current, exponents_alpha_current, w_1_times_r2_values_FINAL, w_2_times_r2_values_FINAL
    
################################################################    
    
################################################################
"""
Computes and returns the new coefficients (c_(a,k)^(m+1))_{k=1..Ka} 
associated to all shells of atom a, in the MB-ISA scheme  using :
    - (c_{b,l}^{(m)})_{b=1..M,l=1..Kb} coefficients of the previous iteration
    - (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb} MB-ISA exponents of the previous iteration.

c_(a,k)^(m+1) = c_(a,k)^(m) * (alpha_(a,k)^(m))³ * int_{R³} [ rho(x) * exp(-alpha_(a,k)^(m)*|x-R_a|) /( sum_b sum_{l=1..Kb} c_(b,l)^(m) * (alpha_(b,l)^(m))³ * exp(-alpha_(b,l)^(m)*|x-R_b|))  dx


g_(a,k)(r) = (alpha_(a,k)/(8*pi))³ * exp(-alpha_(a,k) * |r|)

-  a : atom index

-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.
- x_GL_0_R, w_GL_0_R : Gauss Legendre radial grid

--x_GL_1, w_GL_0_1 : Gauss Legendre angular grid

- c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
  of the iterative scheme (e.g. direct minimization)
  
- exponents_MBISA[a] =  (alpha_(a,k))_{k=1..Ka}

exponents_MBISA and vector_c allow to define the values of all paritionning functions w_b(.)
"""
def computes_new_coeffs_MB_ISA(a,coeffs_c_previous,
                               exponents_MBISA_previous,
                               values_rho_atom,
                               Ng_theta,
                               x_GL_0_1, w_GL_0_1,
                               Ng_radial,
                               x_GL_0_R, w_GL_0_R,
                               Rmax,
                               atomic_coordinates):
        
          
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Number of shells associated to atom a (fixed parameter)
    Ka = len(coeffs_c_previous[a])
    
    coeffs_c_atom_a_NEW=np.zeros(Ka)
    
    
    # Computes r**2 * exp(-alpha_(a,k)^(m)*r) * int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|)) ] d theta
    # on the (GL) radial grid  (Rmax*x_l^(GL)):
    # (where w_a(r) and {w_b(|R_a+r.(sin(theta_j),0,cos(theta_j))-R_b|)}_{j=1..Ng_theta} are computed inside the function :
    # compute_sph_avg_charge_MBISA() which computes the spherical average on theta (from 0 to pi)
        
    for k in range(Ka):
    
        # Don't forget Gauss-Legendre weights (an radial grid)
        tab_integrand_a_k = [ w_GL_0_R[l] * (x_GL_0_R_scaled[l]/conversion_bohr_angstrom)**2 * math.exp(-exponents_MBISA_previous[a][k]*abs(x_GL_0_R_scaled[l]/conversion_bohr_angstrom)) * compute_sph_avg_charge_MBISA(a,l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                         exponents_MBISA_previous,
                                                                                                                                                                                                                         coeffs_c_previous,
                                                                                                                                                                                                                         values_rho_atom,
                                                                                                                                                                                                                         x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                         Ng_theta,
                                                                                                                                                                                                                         atomic_coordinates) for l in range(Ng_radial)]
        
        
        # c_(a,k)^(m+1) = c_(a,k)^(m) * (alpha_(a,k)^(m))³ / (8*math.pi) * int_{R³} [ rho(x) * exp(-alpha_(a,k)^(m)*|x-R_a|) /( sum_b sum_{l=1..Kb} c_(b,l)^(m) * (alpha_(b,l)^(m))³/(8*pi) * exp(-alpha_(b,l)^(m)*|x-R_b|))  dx

        coeffs_c_atom_a_NEW[k] = coeffs_c_previous[a][k] * ( (exponents_MBISA_previous[a][k]**3)/(8*math.pi) ) * (Rmax/conversion_bohr_angstrom) * 2*math.pi * np.sum(tab_integrand_a_k)

    # Returns the new coefficients (c_(a,k)^(m+1))_(k=1..Ka) associated to atom a (and its shell) :
    return coeffs_c_atom_a_NEW
################################################################


################################################################
"""
Computes and returns the new MB-ISA EXPONENTS (alpha_(a,k)^(m+1))_{k=1..Ka} 
associated to all shells of atom a, in the MB-ISA scheme  using :
    - (c_{a,k}^{(m+1)})_{k=1..Ka} = coeffs_c_a_NEW
      computed thanks to computes_new_coeffs_MB_ISA() applied to atom a
    
    - (c_{b,l}^{(m)})_{b=1..M,l=1..Kb} = coeffs_c_a_NEW 
      coefficients of the previous iteration
    
    - (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb} = exponents_MBISA_previous
      MB-ISA exponents of the previous iteration.

Iteration formula to update the MB-ISA exponent :
    
alpha_(a,k)^(m+1) = 3* ( c_(a,k)^(m+1) / c_(a,k)^(m) ) * (8*pi/(alpha_(a,k)^(m))^3) 
                     / int_{R³} [ rho(x) * |x-Ra| * exp(-alpha_(a,k)^(m)*|x-R_a|) /( sum_b sum_{l=1..Kb} c_(b,l)^(m) * (alpha_(b,l)^(m))³ * exp(-alpha_(b,l)^(m)*|x-R_b|))  dx


g_(a,k)(r) = (alpha_(a,k)/(8*pi))³ * exp(-alpha_(a,k) * |r|) normalized (Slater)

-  a : atom index
-  c_a = 'vector_c[a]' = (c_(a,k))_(k=1..Ka)

-  Rmax : radial point up to which we integrate on r (radially) from 0 to +\infty.

- c_coeffs_previous = (c_(a,k)^(m))_{k=1..Ka} obtained at the previous iteration
  of the iterative scheme (e.g. direct minimization)
  
- exponents_MBISA[a] =  (alpha_(a,k))_{k=1..Ka}

exponents_MBISA and vector_c allow to define the values of all paritionning functions w_b(.)
"""
def computes_new_exponents_MB_ISA(a,coeffs_c_a_NEW,
                                  coeffs_c_previous,
                                  exponents_MBISA_previous,
                                  values_rho_atom,
                                  Ng_theta,
                                  x_GL_0_1, w_GL_0_1,
                                  Ng_radial,
                                  x_GL_0_R, w_GL_0_R,
                                  Rmax,
                                  atomic_coordinates):
        
          
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Number of shells associated to atom a (fixed parameter)
    Ka = len(coeffs_c_previous[a])
    
    exponents_MBISA_atom_a_NEW=np.zeros(Ka)

    
    # Computes r**3 * exp(-alpha_(a,k)^(m)*r) * int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|)) ] d theta
    # on the (GL) radial grid  (Rmax*x_l^(GL)):
    # (where w_a(r) and {w_b(|R_a+r.(sin(theta_j),0,cos(theta_j))-R_b|)}_{j=1..Ng_theta} are computed inside the function :
    # compute_sph_avg_charge_MBISA() which computes the spherical average on theta (from 0 to pi)
        
    for k in range(Ka):
    
        # Don't forget Gauss-Legendre weights (an radial grid)
        # Only change with respect to (c_(a,k)^(m+1)) calculation in 'computes_new_coeffs_MB_ISA()' function : 
        # r³ instead of r² (the spherical average on theta is identical)
        # The spherical average on theta uses (c_{b,l}^{(m)})_{b=1..M,l=1..Kb}
        # and (alpha_{b,l}^{(m)})_{b=1..M,l=1..Kb} ONLY, i.e. the coefficients and exponents 
        # AT THE PREVIOUS ITERATION
        tab_integrand_a_k = [ w_GL_0_R[l] * (x_GL_0_R_scaled[l]/conversion_bohr_angstrom)**3 * math.exp(-exponents_MBISA_previous[a][k]*abs(x_GL_0_R_scaled[l]/conversion_bohr_angstrom)) * compute_sph_avg_charge_MBISA(a,l,x_GL_0_R_scaled[l],
                                                                                                                                                                                                                         exponents_MBISA_previous,
                                                                                                                                                                                                                         coeffs_c_previous,
                                                                                                                                                                                                                         values_rho_atom,
                                                                                                                                                                                                                         x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                                         Ng_theta,
                                                                                                                                                                                                                         atomic_coordinates) for l in range(Ng_radial)]
        
        
        # alpha_(a,k)^(m+1) = 3* ( c_(a,k)^(m+1) / c_(a,k)^(m) ) * (8*pi/(alpha_(a,k)^(m))³) 
        #                       / int_{R³} [ rho(x) * |x-Ra| * exp(-alpha_(a,k)^(m)*|x-R_a|) /( sum_b sum_{l=1..Kb} c_(b,l)^(m) * (alpha_(b,l)^(m))³/(8*pi) * exp(-alpha_(b,l)^(m)*|x-R_b|))  dx

        integral_denominator = (1/conversion_bohr_angstrom) * Rmax * 2*math.pi * np.sum(tab_integrand_a_k)
        
        exponents_MBISA_atom_a_NEW[k] = 3*(coeffs_c_a_NEW[k]/coeffs_c_previous[a][k]) * (8*math.pi/exponents_MBISA_previous[a][k]**3) / integral_denominator

    # Returns the new coefficients (alpha_(a,k)^(m+1))_(k=1..Ka) associated to atom a (and its shell) :
    return exponents_MBISA_atom_a_NEW
################################################################


################################################################
"""
Returns the tab of values of w_b(.) at the angular grid points centered on atom a (Ra) :
    
    {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)}_{1..Ng_theta}, with r_l the point of a radial grid :

  where (theta_j)_j is an angular Gauss-Legendre grid on theta (x_GL_0_1).
  
  
- If b = a : we are simply computing w_b(r) = w_a(r) : no need to compute the values on all the
  angular GL grid : we simply return one value w_a(r)
  
- Otherwise (if b= abs(a-1) in the diatomic case) : we return 
 {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)}_{1..Ng_theta}
 the values of w_b(.) at the grid points centered on the other atom a
"""
def compute_w_b_MBISA_tab(b,a,r,
                          x_GL_0_1,Ng_theta,
                          coeffs_c,
                          exponents_MBISA,
                          atomic_coordinates):
    
    K_b = len(exponents_MBISA[b])

    if (b==a):
        return np.sum([coeffs_c[b][k]*(exponents_MBISA[b][k])**(3.)/(8*math.pi) * math.exp(-exponents_MBISA[b][k] * abs(r/conversion_bohr_angstrom)) for k in range(K_b)])
    
    else:
        # Works only for two atoms on the same z axis :
        grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                                    0,
                                                    atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[b][2]]  for l in range(Ng_theta)]
        
        
        norm_grid_point_around_a = [ np.linalg.norm(grid_points_3D_atom_shifted_other_atom[l]) for l in range(Ng_theta)]
                        
        # w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) = sum_{k=1..Kb} c_(b,k) * (alpha_(b,k))³ / (8*pi) * exp(-alpha_(b,k)*|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|)
        tab_w_b_values=[ np.sum([coeffs_c[b][k]*(exponents_MBISA[b][k])**(3.)/(8*math.pi) * math.exp(-exponents_MBISA[b][k] * norm_grid_point_around_a[j]/conversion_bohr_angstrom) for k in range(K_b)]) for j in range(Ng_theta)]
        
        return tab_w_b_values
################################################################


################################################################
"""
Computes the angular integral in theta arising in the diatomic case : (invariance around z axis i.e. along phi) :
    
    int_{0..pi} [ sin(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|)) ] d theta
    
where :
    - a = atom index around which this angular (spherical, by invariance) average is computed
    - rho(R_a+r.(sin(theta),0,cos(theta)) = molecular density, precomputed at these
          points with a radial discretization r_l (x_GL_0_R)
          and the GL angular discretization theta_l  (x_GL_0_1)
          
    - w_a(r) = sum_{k=1..Ka} c_(a,k) * (alpha_(a,k))³ / (8*pi) * exp(-alpha_(a,k)*r)
    - w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|) = sum_{k=1..Kb} c_(b,k) * (alpha_(b,k))³ / (8*pi) * exp(-alpha_(b,k)*|R_a+r.(sin(theta),0,cos(theta))-R_b|)
    
    are MB-ISA atomic weight functions.
    
    - exponents_MBISA = {alpha_(b,l)}_{b=1..M,l=1..Kb}
    - coeffs_c = {c_(b,l)}_{b=1..M,l=1..Kb}

"r" is evaluated here at r_l where l='index_radial' denotes the index of the radial
discretization point in the radial grid x_GL_0_R (present in the MB-ISA main algorithm function)
"""
def compute_sph_avg_charge_MBISA(a,index_radial,r,
                                 exponents_MBISA,
                                 coeffs_c,
                                 values_rho_atom,
                                 x_GL_0_1, w_GL_0_1,
                                 Ng_theta,
                                 atomic_coordinates):
        

    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_MBISA_tab(a,a,r,
                                        x_GL_0_1,Ng_theta,
                                        coeffs_c,
                                        exponents_MBISA,
                                        atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_MBISA_tab(b,a,r,
                                                      x_GL_0_1,Ng_theta,
                                                      coeffs_c,
                                                      exponents_MBISA,
                                                      atomic_coordinates)
    
    # Molecular density rho(.) evaluated at (R_a+r_l.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l] = values_rho_atom[index_radial]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):

    
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    # Returns a result in ATOMIC UNITS
    return math.pi * np.sum(values_integrand_theta) 
################################################################


################################################################
"""
Computes and prints atomic local charges, dipoles, and quadrupolar tensor components
from the final w_a(.) of the GISA procedure (i.e. simply  from the optimal 
                                             (c_(a,k)^{opt})_{a,k} coefficients)

- 'values_rho' : values of the molecular density at radial x angular (in theta) grid points
  previously computed thanks to the function 'precompute_molecular_density_DIATOMIC()'
"""
def compute_local_multipoles_MBISA(Rmax,
                                   coeffs_c_MBISA_FINAL,
                                   exponents_MBISA_FINAL,
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
    print('COMPUTATION ATOMIC MULTIPOLES MB-ISA in e.Ang^k (a.u.)')
    logfile.write('ATOMIC MULTIPOLES in e.Ang^k (a.u.)')
    logfile.write("\n")
    #print('ATOMIC CHARGES (k=0), DIPOLES (k=1) and QUADRUPOLES (k=2) in e.Ang^k (ATOMIC UNITS)')
    print(' ')
    
    for a in range(2):
        
        q_a = compute_atomic_charge_MBISA(a,Rmax,
                                         coeffs_c_MBISA_FINAL,
                                         exponents_MBISA_FINAL,
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
        
        d_z_a = compute_atomic_dipole_MBISA(a,Rmax,
                                           coeffs_c_MBISA_FINAL,
                                           exponents_MBISA_FINAL,
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
        
        Q_xx_a = compute_atomic_quadrupole_xx_MBISA(a,Rmax,
                                                   coeffs_c_MBISA_FINAL,
                                                   exponents_MBISA_FINAL,
                                                   values_rho[a],
                                                   Ng_theta,
                                                   x_GL_0_1, w_GL_0_1,
                                                   Ng_radial,
                                                   x_GL_0_R, w_GL_0_R,
                                                   atomic_coordinates)
        
        print('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write('Q_xx_'+str(a)+' = '+str(Q_xx_a)+'  = Q_yy_'+str(a))
        logfile.write("\n")
     
        Q_zz_a = compute_atomic_quadrupole_zz_MBISA(a,Rmax,
                                                   coeffs_c_MBISA_FINAL,
                                                   exponents_MBISA_FINAL,
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
def compute_atomic_charge_MBISA(a,Rmax,coeffs_c,
                               exponents_MBISA,
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
    tab_integrand = [ (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[l]**2 * compute_w_b_MBISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_MBISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_charge_MBISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                              coeffs_c,
                                                                                                                                                              exponents_MBISA,
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
def compute_atomic_dipole_MBISA(a,Rmax,
                               coeffs_c,
                               exponents_MBISA,
                               values_rho_atom,
                               Ng_theta,
                               x_GL_0_1, w_GL_0_1,
                               Ng_radial,
                               x_GL_0_R, w_GL_0_R,
                               atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**3 * w_a^(m)(r) * int_{0..pi} [ sin(theta) * cos(theta) * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**3 * x_GL_0_R_scaled[l]**3 * compute_w_b_MBISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_MBISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_dipole_MBISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                              coeffs_c,
                                                                                                                                                              exponents_MBISA,
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
def compute_atomic_quadrupole_xx_MBISA(a,Rmax,
                                      coeffs_c,
                                      exponents_MBISA,
                                      values_rho_atom,
                                      Ng_theta,
                                      x_GL_0_1, w_GL_0_1,
                                      Ng_radial,
                                      x_GL_0_R, w_GL_0_R,
                                      atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)**3 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * compute_w_b_MBISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_MBISA,
                                                                                                     atomic_coordinates) * compute_sph_avg_atomic_quadrupole_xx_MBISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                                     coeffs_c,
                                                                                                                                                                     exponents_MBISA,
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
def compute_atomic_quadrupole_zz_MBISA(a,Rmax,
                                      coeffs_c,
                                      exponents_MBISA,
                                      values_rho_atom,
                                      Ng_theta,
                                      x_GL_0_1, w_GL_0_1,
                                      Ng_radial,
                                      x_GL_0_R, w_GL_0_R,
                                      atomic_coordinates):
    
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[k] for k in range(Ng_radial)]
    
    # Computes r**4 * w_a^(m)(r) * int_{0..pi} [ sin(theta)* cos(theta)**2 * rho(R_a+r.(sin(theta),0,cos(theta)))]/ (w_a(r) + sum_{b \neq a} w_b(|R_a+r.(sin(theta),0,cos(theta))-R_b|))
    #on the (GL) radial grid  (Rmax*x_l^(GL)):
    tab_integrand = [ (1/conversion_bohr_angstrom)**4 * x_GL_0_R_scaled[l]**4 * compute_w_b_MBISA_tab(a,a,x_GL_0_R_scaled[l],
                                                                                                     x_GL_0_1,Ng_theta,
                                                                                                     coeffs_c,
                                                                                                     exponents_MBISA,
                                                                                                     atomic_coordinates)  * compute_sph_avg_atomic_quadrupole_zz_MBISA(l,x_GL_0_R_scaled[l],a,
                                                                                                                                                                      coeffs_c,
                                                                                                                                                                      exponents_MBISA,
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

 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
 
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_charge_MBISA(index_radial,r,a,
                                       coeffs_c,
                                       exponents_MBISA,
                                       values_rho_atom,
                                       x_GL_0_1, w_GL_0_1,
                                       Ng_theta,
                                       atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)
    
    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_MBISA_tab(a,a,r,
                                        x_GL_0_1,Ng_theta,
                                        coeffs_c,
                                        exponents_MBISA,
                                        atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_MBISA_tab(b,a,r,
                                                      x_GL_0_1,Ng_theta,
                                                      coeffs_c,
                                                      exponents_MBISA,
                                                      atomic_coordinates)
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho_atom[index_radial][u] (r = r_l = tab_r[index_radial])
    
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

 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_dipole_MBISA(index_radial,r,a,
                                       coeffs_c,
                                       exponents_MBISA,
                                       values_rho_atom,
                                       x_GL_0_1, w_GL_0_1,
                                       Ng_theta,
                                       atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)


    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_MBISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_MBISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_MBISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_MBISA,
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

 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_xx_MBISA(index_radial,r,a,
                                              coeffs_c,
                                              exponents_MBISA,
                                              values_rho_atom,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_theta,
                                              atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)


    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_MBISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_MBISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_MBISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_MBISA,
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

 TODO => coincides with ( Rmax*x^{GL}_l )_{l=1..Nb_radial} ??
- index_radial : index of r in the radial grid (r_1,...r_M=R) where M = Ng_radial
- a = atom index
- values_rho_atom = precomputed molecular density values around atom a :
    { rho(\vec{R_a} +r_j*sin(theta_l) e_x + r_j*cos(theta_l) e_z ) }_{j,l} 
"""
# Spherical average of rho(.)/ (sum_b w_b) at distance r from atom a :
def compute_sph_avg_atomic_quadrupole_zz_MBISA(index_radial,r,a,
                                              coeffs_c,
                                              exponents_MBISA,
                                              values_rho_atom,
                                              x_GL_0_1, w_GL_0_1,
                                              Ng_theta,
                                              atomic_coordinates):
        
    # Index of the other atom (in the diatomic case)
    b = abs(a-1)

    # Compute the values of w_a(r_l)
    value_w_a_r = compute_w_b_MBISA_tab(a,a,r,
                                       x_GL_0_1,Ng_theta,
                                       coeffs_c,
                                       exponents_MBISA,
                                       atomic_coordinates)
    
    # Index of the other atom (in the diatomic case) :
    b = abs(a-1)
    
    # Compute the values of w_b(.) on the grid points of atom a :
    # {w_b(|R_a+r_l.(sin(theta_j),0,cos(theta_j))-R_b|) }_{j=1..Ng_theta} :
    value_w_b_GL_angular_grid = compute_w_b_MBISA_tab(b,a,r,
                                                     x_GL_0_1,Ng_theta,
                                                     coeffs_c,
                                                     exponents_MBISA,
                                                     atomic_coordinates)
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * math.sin(math.pi*x_GL_0_1[u]) * math.cos(math.pi*x_GL_0_1[u])**2  /(value_w_a_r+value_w_b_GL_angular_grid[u])  for u in range(Ng_theta)]
    
    return math.pi * np.sum(values_integrand_theta) 
################################################################

################################################################
"""
coeffs_c_previous = (c_(a,k)^(m))_{k=1..m_a}
coeffs_c          = (c_(a,k)^(m+1))_{k=1..m_a}

exponents_alpha_previous = (alpha_(a,k)^(m))_{k=1..m_a}
exponents_alpha_current  = (alpha_(a,k)^(m+1))_{k=1..m_a}

Computes sqrt(4*pi*int_0^{+\infty} r² * |sum_{k=1..m_a} (c_(a,k)^(m+1)*(alpha_(a,k)^(m+1))³*exp(-alpha_(a,k)^(m+1)*r)-c_(a,k)^(m)*(alpha_(a,k)^(m))³*exp(-alpha_(a,k)^(m)*r))/(8*pi)|² dr) 
         = || w_a^(m+1)(.)-w_a^(m)(.) ||_{L²,GL}

the convergence criterion of MB-ISA

"""
def convergence_criterion_MB_ISA(coeffs_c,
                                 coeffs_c_previous,
                                 exponents_alpha_current,
                                 exponents_alpha_previous,
                                 x_GL_0_R,w_GL_0_R,Ng_radial,
                                 Rmax):

    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]

    CV_criterion = 0
    
    nb_atoms = len(exponents_alpha_current)

    for a in range(nb_atoms):
            
        tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * np.sum([coeffs_c[a][k] * (exponents_alpha_current[a][k])**(3.) * math.exp(-exponents_alpha_current[a][k] * abs(x_GL_0_R_scaled[u]/conversion_bohr_angstrom)) - coeffs_c_previous[a][k] * (exponents_alpha_previous[a][k])**(3.) * math.exp(-exponents_alpha_previous[a][k] * abs(x_GL_0_R_scaled[u]/conversion_bohr_angstrom))  for k in range(len(exponents_alpha_current[a]))])**2 for u in range(Ng_radial)]            

        CV_criterion += np.sum(tab_integrand_a)

    return math.sqrt((1/(16*math.pi))*CV_criterion)
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
def compute_sph_avg_entropy_1_MB_ISA(index_radial,r,
                                     Ng_theta,
                                     a,
                                     coeffs_c,exponents_MBISA,
                                     values_rho_atom,
                                     x_GL_0_1, w_GL_0_1,
                                     atomic_coordinates):
        
        
    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    """
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[abs(a-1)][2]]  for l in range(Ng_theta)]
    """
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Scalar :
    w_a_r_value = compute_w_b_MBISA_tab(a,a,r,
                                        x_GL_0_1,Ng_theta,
                                        coeffs_c,
                                        exponents_MBISA,
                                        atomic_coordinates)
    
    # Table [ w_b(|R_a+r.(sin(theta_l),0,cos(theta_l))-Rb|) ]_{l=1..Ng_theta} with b=abs(a-1) [diatomic case]:
    tab_w_b_angular = compute_w_b_MBISA_tab(abs(a-1),a,r,
                                            x_GL_0_1,Ng_theta,
                                            coeffs_c,
                                            exponents_MBISA,
                                            atomic_coordinates) 
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u] * (math.sin(math.pi*x_GL_0_1[u])/(w_a_r_value+tab_w_b_angular[u]))*math.log(w_a_r_value+tab_w_b_angular[u])  for u in range(Ng_theta)]
                                                                                                                                                                                                                                                                               
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
def compute_sph_avg_entropy_2_MB_ISA(index_radial,r,
                                     Ng_theta,
                                     a,
                                     coeffs_c,exponents_MBISA,
                                     values_rho_atom,
                                     x_GL_0_1, w_GL_0_1,
                                     atomic_coordinates):
        
        
    # Vectors {R_a+r.(sin(theta_l),0,cos(theta_l))-Rb }_{l=1..Ng_theta}
    # (the 2 atoms being along z axis)
    # => useful to evaluate w_b(.) on the grid points of atom a
    """
    grid_points_3D_atom_shifted_other_atom=[ [r * math.sin(math.pi*x_GL_0_1[l]) ,
                                              0,
                                              atomic_coordinates[a][2] + r * math.cos(math.pi*x_GL_0_1[l]) - atomic_coordinates[abs(a-1)][2]]  for l in range(Ng_theta)]
    
    """
    
    # Scalar :
    w_a_r_value = compute_w_b_MBISA_tab(a,a,r,
                                        x_GL_0_1,Ng_theta,
                                        coeffs_c,
                                        exponents_MBISA,
                                        atomic_coordinates)
    
    # Table [ w_b(|R_a+r.(sin(theta_l),0,cos(theta_l))-Rb|) ]_{l=1..Ng_theta} with b=abs(a-1) [diatomic case]:
    tab_w_b_angular = compute_w_b_MBISA_tab(abs(a-1),a,r,
                                            x_GL_0_1,Ng_theta,
                                            coeffs_c,
                                            exponents_MBISA,
                                            atomic_coordinates) 
    
    
    # Molecular density rho(.) evaluated at (R_a+r.(sin(theta_j),0,cos(theta_j)))_{j=1..Ng_theta} : 
    # values_rho[l]
    
    # Works only in the diatomic case (b = abs(a-1) = 1 if a=0 and equals 0 if a=1):
    values_integrand_theta = [ w_GL_0_1[u] * values_rho_atom[index_radial][u]*math.log(values_rho_atom[index_radial][u]) * math.sin(math.pi*x_GL_0_1[u]) /(w_a_r_value+tab_w_b_angular[u])  for u in range(Ng_theta)]
    
    # Integral on theta only (not on phi)
    return math.pi * np.sum(values_integrand_theta) 
################################################################

################################################################
"""
Returns the  entropy s_{KL}(rho_a^(m)|rho_a^{0,(m)}) at step m
"""
def computes_entropy_MB_ISA(a,c_coeffs,
                            Rmax, 
                            atomic_coordinates,
                            x_GL_0_R,w_GL_0_R,Ng_radial,
                            x_GL_0_1, w_GL_0_1,Ng_theta,
                            exponents_MBISA,
                            values_rho):
    
    x_GL_0_R_scaled = [Rmax*x_GL_0_R[u] for u in range(Ng_radial)]
        


    #NEW
    # Computes r**2 * rho_a^(0,(m))(r) * (- < (rho/ (sum_b rho_b^(0,(m))))*log(sum_b rho_b^(0,(m))) >_a(r) + < rho*log(rho)/ (sum_b rho_b^(0,(m)))>_a(r) )
    # on a GL radial grid (r_l)_l
    tab_integrand_a = [ w_GL_0_R[u] * (1/conversion_bohr_angstrom)**2 * x_GL_0_R_scaled[u]**2 * compute_w_b_MBISA_tab(a,a,x_GL_0_R_scaled[u],
                                                                                                                      x_GL_0_1,Ng_theta,
                                                                                                                      c_coeffs,
                                                                                                                      exponents_MBISA,
                                                                                                                      atomic_coordinates) * (-compute_sph_avg_entropy_1_MB_ISA(u,x_GL_0_R_scaled[u],
                                                                                                                                                                               Ng_theta,
                                                                                                                                                                               a,
                                                                                                                                                                               c_coeffs,
                                                                                                                                                                               exponents_MBISA,
                                                                                                                                                                               values_rho[a],
                                                                                                                                                                               x_GL_0_1, w_GL_0_1,
                                                                                                                                                                               atomic_coordinates)                     
                                                                                                                                                                      +compute_sph_avg_entropy_2_MB_ISA(u,x_GL_0_R_scaled[u],
                                                                                                                                                                                                        Ng_theta,
                                                                                                                                                                                                        a,
                                                                                                                                                                                                        c_coeffs,
                                                                                                                                                                                                        exponents_MBISA,
                                                                                                                                                                                                        values_rho[a],
                                                                                                                                                                                                        x_GL_0_1, w_GL_0_1,
                                                                                                                                                                                                        atomic_coordinates)) for u in range(Ng_radial)]

    # Approximation of the integral defining the gradient component by a Gauss-Legendre summation :
    integral = (Rmax/conversion_bohr_angstrom) * np.sum(tab_integrand_a)

    return integral
################################################################



# END MB-ISA method functions
################################################################
################################################################

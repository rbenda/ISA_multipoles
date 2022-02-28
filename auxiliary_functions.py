#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:13:48 2020

@author: rbenda
"""


import numpy as np
#import matplotlib
import math
import scipy.special

import scipy.spatial.distance

import openbabel

#from multipoles_DMA import computes_total_quadrupole_from_local_multipoles_DMA_sites
#from multipoles_DMA import computes_total_dipole_from_local_multipoles_DMA_sites


Open_babel_files_dir='/home/rbenda/Desktop/THESE/CODE_MULTIPOLES/FINAL_CODE/Open_babel_files_dir/'


################
"""
#Cf. code M. Herbst
#https://github.com/molsturm/gint/blob/master/src/interface/python/gint/gaussian/_gaussian_shells.py
#Function cartesian_xyz_expontents(l, ordering):
##Produce the list of cartesian exponents.
##This determines the order inside the cartesian exponents, which is used.
##The precise ordering can be adapted using the ordering parameter.
##The following orderings are currently implemented:
##- standard: The ordering of the Common Component Architecture (CCA) standard
##          as described in the paper DOI 10.1002/jcc
for l in range(6):
    print('l = '+str(l))
    print([ (a, l-a-c, c) for a in range(l, -1, -1) for c in range(0, l-a+1) ])
"""


################################################
#GLOBAL VARIABLES :
#Distance tolerance threshold 
epsilon=1e-8
#Tolerance threshold on the multipole redistribution weight from P_{alpha,beta} to S
#(below this threshold, nothing is transfered from P_{alpha,beta} to S)
epsilon_weight=1e-5
############################
#CONSTANTS :
#1 Bohr in Angstroms (Gaussian QM output given in atomic units 
#i.e. bohr and Hartrees ==> e.g. 'zeta' gaussian exponents
#provided in Bohr^{-2}
conversion_bohr_angstrom=0.529177209
#1 Debye in e.Ang :
conversion_Debye_in_e_Ang=0.20819434
##=> 1 Debye = 0.20819434/0.529177209 e.Bohr (a.u.) = 0.3934302847120538 e.Bohr
conversion_Debye_atomic_units=conversion_Debye_in_e_Ang/conversion_bohr_angstrom

###############################################

periodic_table = ['','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','P','Si','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','IR','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']



#############################################################
#Returns integral \int (u**q * exp(-u**2) du) on the domain [-\infty,+\infty]
def gaussian_polynom_integral_all_R(q):
    
    if (q%2 ==1):
        #Odd function integrated over an even domain => integral=0
        return 0
    
    else:
        #Euler Gamma function of Python :
        #=2* (\int_{0..+\infty} (u**q * exp(-u**2) du))=2*(\Gamma((q+1)/2)/2)
        return scipy.special.gamma((q+1)/2) 
#############################################################

#############################################################
#For a point S = (x_S,y_S,z_S) [final DMA expansion site] : returns True 
#if S is an atom, and returns the associated index atom.
#Useful in the main function n°2 'compute_DMA_multipole_moment_final_expansion_center_one_site()'
#to correct the monopoles (of the ELECTRONIC charge) by adding the positive charge of the nuclei
def boolean_final_expansion_site_is_atom(x_S,y_S,z_S,atomic_coordinates):
    
    #Loop over the number of atoms :
    for k in range(len(atomic_coordinates)):
       
        if (scipy.spatial.distance.pdist([[x_S,y_S,z_S],atomic_coordinates[k]])[0]< epsilon):
            
            return [True,k]

    #If the final expansion sute 'S' did not coincide with any nuclei : we return 'False' :
    return [False]
#############################################################
    


    
##########################################################################################################################
#Auxiliary function for MAIN FUNCTION N°1 giving one of the coefficients 
#in front of each homogeneous polynom : x**i * y**j * z**k * exp(-zeta*(r^2)
#(useful primarily for 'compute_DMA_multipole_moment_natural_expansion_center_primitive_GTOs()' function
#but also for 'compute_scalar_product_two_primitive_GTOs()' function in extraction_QM_info.py code))
#################
#Input parameters "tab_indexes_alpha,tab_indexes_beta,R_alpha,R_beta,zeta_alpha,zeta_beta" : available (computed) before calling this function,
#in the main function "compute_DMA_multipole_moment_natural_expansion_center_primitive_GTOs()"
#Input provided : "tab_indexes_alpha_k=tab_indexes_alpha[k]" 
#where tab_indexes_alpha=conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l_alpha,m_alpha)
#BEWARE of units for nuclei position 'R_alpha' and 'R_beta' : 
#have to be in the same units (Bohrs or Angstroms) as the 'zeta' gaussian exponents (usually given in atomic units i.e. in Bohrs^{-2}  !
def coeff_W_binomial_expansion_polynoms(l_alpha,l_beta,tab_indexes_alpha_k,tab_indexes_beta_q,i_1,j_1,i_2,j_2,h_1,h_2,R_alpha,R_beta,zeta_alpha,zeta_beta):

    #Coordinates of overlap center P_alpha_beta :
    #BEWARE of units : 'zeta' gaussian exponents have to be in same units as the atom coordinates (Angstroms)
    #==> conversion from Angstroms^{-2} to atomic units (Bohrs^{-2}) at the end of this function
    #'R_alpha', 'R_beta' provided in Angstroms
    overlap_center_alpha_beta=[((zeta_alpha/(zeta_alpha+zeta_beta))*R_alpha[i]+(zeta_beta/(zeta_alpha+zeta_beta))*R_beta[i]) for i in range(3)]
    ##overlap_center_alpha_beta=[((zeta_alpha/(zeta_alpha+zeta_beta))*R_alpha[i]+(zeta_beta/(zeta_alpha+zeta_beta))*R_beta[i]) for i in range(3)]

    coeff_W=1
    
    ##Binomial coefficients :
    coeff_W=coeff_W*scipy.special.binom(tab_indexes_alpha_k[0],i_1)*scipy.special.binom(tab_indexes_alpha_k[1],j_1)*scipy.special.binom(tab_indexes_alpha_k[2],h_1)
    
    coeff_W=coeff_W*scipy.special.binom(tab_indexes_beta_q[0],i_2)*scipy.special.binom(tab_indexes_beta_q[1],j_2)*scipy.special.binom(tab_indexes_beta_q[2],h_2)

    ##########
    coeff_W=coeff_W*(overlap_center_alpha_beta[0]-R_alpha[0])**(tab_indexes_alpha_k[0]-i_1)
    
    coeff_W=coeff_W*(overlap_center_alpha_beta[1]-R_alpha[1])**(tab_indexes_alpha_k[1]-j_1)
    
    coeff_W=coeff_W*(overlap_center_alpha_beta[2]-R_alpha[2])**(tab_indexes_alpha_k[2]-h_1)
    
    ##########
    coeff_W=coeff_W*(overlap_center_alpha_beta[0]-R_beta[0])**(tab_indexes_beta_q[0]-i_2)
    
    coeff_W=coeff_W*(overlap_center_alpha_beta[1]-R_beta[1])**(tab_indexes_beta_q[1]-j_2)
    
    coeff_W=coeff_W*(overlap_center_alpha_beta[2]-R_beta[2])**(tab_indexes_beta_q[2]-h_2)
    
    #Converted in Bohr**(l_alpha+l_beta-i_1-i_2-j_1-j_2-h_1-h_2)
    #('overlap_center_alpha_beta' components : in Angstroms)
    return coeff_W*(1/conversion_bohr_angstrom)**(l_alpha+l_beta-i_1-i_2-j_1-j_2-h_1-h_2)
##########################################################################################################################
    
########################################################################################################
#Computes maximum angular order l_max=l_{alpha}+l_{beta} for a product of two primitive GTOs
#\chi_{alpha}^{pGTO}*\chi_{beta}^{pGTO}(.)
##Plays in "compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders()" function
#when 'natural' multipole moments (/overlap centers) (Q_(l,m)^{alpha,beta})_{P_{alpha,beta}) are computed
#=> defines the order to which they are computed
####################################################
#Must be consistent with l_max_user provided as the final multipole moments
#(Q_(l,m))_{S} are derived from the multipole moments computed with respect to 'natural' overlap centers
#i.e. (Q_(l,m)^{alpha,beta})_{P_{alpha,beta}} => to get (Q_(l,m))_{S} up to order L => (Q_(l,m)^{alpha,beta})_{P_{alpha,beta}}
#have to be available up to order L (and below, cf. redistribution formula)
####################################################
#'alpha', 'beta' : indexes of primitive GTOs
def l_max(alpha,beta,angular_momentum_numbers_primitive_GTOs):
        
    #General case :
    #return angular_momentum_numbers_primitive_GTOs[alpha][0]+angular_momentum_numbers_primitive_GTOs[beta][0]
    
    ##So far :'conversion_spherical_harmonics_cartesian_homogeneous_polynoms()' function not coded for l>= 5
    return min(angular_momentum_numbers_primitive_GTOs[alpha][0]+angular_momentum_numbers_primitive_GTOs[beta][0],4)

    #In theory : for example for a product  of two 'F' functions (i.e. l=3)
    #=> necessary to go up to l=6 ... 

##############################################################################################

##############################################################################################
#First loads a structure as an OBMol structure (uses OPEN BABEL python library)
#returns an OBMol object with information loaded from the structure 'filename'   
def list_bond_centers(atomic_coordinates,filename):
  
    list_bond_centers=[]
    
    #Use OpenBabel function to derive the topology of the molecule 
    #and the list of bonds
    tmpconv = openbabel.OBConversion()
    
    ##########################
    ##Beware : if .log / .dat / .out (etc..) i.e. output file of a QM calculation
    ##is not readable by Open Babel ==> provide the .xyz file of the optimized,
    ##final geometry
    
    #(Temporary fix)
    
    #filename=Open_babel_files_dir+'O2_PBE0_6-31Gd_psi4_opt.xyz'
    ##########################
    
    inFormat = openbabel.OBConversion.FormatFromExt(filename)
    tmpconv.SetInFormat(inFormat)
    
    mol = openbabel.OBMol()
    tmpconv.ReadFile(mol, filename)
    
    #mol: OBMol object
    iterbond = openbabel.OBMolBondIter(mol)
    
    #We iterate over the list of bonds 'iterbond' of the molecule
    for bond in iterbond:
        #Creation of OBAtom objects (representing atoms at the bond extremities)
        atom_1 = bond.GetBeginAtom()
        atom_2 = bond.GetEndAtom()
        
        #Coordinates of the atoms of this bond
        #(length of the bond =bond.GetLength())
        x_1=atom_1.GetX() 
        y_1=atom_1.GetY() 
        z_1=atom_1.GetZ() 
        
        x_2=atom_2.GetX() 
        y_2=atom_2.GetY() 
        z_2=atom_2.GetZ() 

        coordinates_center_bond=[(x_1+x_2)/2.,(y_1+y_2)/2.,(z_1+z_2)/2.]
        
        list_bond_centers.append(coordinates_center_bond)
        
    return list_bond_centers
##############################################################################################
            
            
##############################################################################
#'index_philosophy' =1 : only nuclei positions are retained as final sites (AMOEBA philosophy)
#'index_philosophy' =2 : nuclei positions AND bond centers are retained as final sites
#'index_philosophy' =3 : nuclei positions AND user-defined additional sites 
#(possibly including all bond centers) are retained as final sites
#'index_philosophy' =4 : user-defined sites (possibly not including any nuclei, 
#e.g. for 'Total' multipolar moments with respect to the origin of coordinates)
#'index_philosophy' =5 : all overlap centers are retained as final DMA expansion sites (e.g. for small molecules)
#'atomic_coordinates' / 'basis_set_exponents' : assumed to have been read before in the input QM file
##########################
#Uses np.argmin() and np.min() functions
#min([distance(P_{alpha,beta},C_m) | m=1..N]) 
###############
#Returns the LIST of one or several equally closest (to the overlap center
#P_{alpha,beta}) final DMA expansion stites
def compute_position_closest_final_site(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,atomic_coordinates,index_philosophy,positions_final_expansion_sites):
    
    #############################
    if ((index_philosophy>=1) and (index_philosophy<=4)):
        
        overlap_center=coordinates_natural_center(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO)
        
        #We look the ATOM the closest to the natural overlap center
        #If two (or several) atoms are equally distant from this overlap
        #center P_{alpha,beta} => we return all of them

        #On arrondit à 10^{-6} toutes les distances 
        #(so that we don't miss any minimum via 'np.min()' function, because of precision (e.g. 10^{-10}))
        ##In the case 'index_philosophy==1' : positions_final_expansion_sites=atomic_coordinates
        distances_overlap_center_atoms=[np.int(10**6*scipy.spatial.distance.pdist([positions_final_expansion_sites[k],overlap_center])[0])/10**6 for k in range(len(positions_final_expansion_sites))]
        
        #Index of the closest site (minimum distance) : first occurence 
        #(definition of Python 'np.argmin() function)
        #BEWARE : in the case of multiple sites (atoms) equally close to the overlap 
        #center P_{alpha,beta} =>  delete from the list of distances
        #'distances_overlap_center_atoms' the distance to the first closest site,
        #and find the next occurence of the minimum distance (second equally closest final site) and so on.
        
        #Example (for index_philosophy==1 i.e. only atomic sites retained ): 
        #Product of two identical gaussian functions [same zeta exponents], 
        #- each one centered on a H atom of NH3 
        #- each one centered on two carbon atoms of benzene bound to each other
        
        index_first_closest_site=np.argmin(distances_overlap_center_atoms)
        
        first_closest_distance=np.min(distances_overlap_center_atoms)
        
        ##############################################
        #METHOD 1:
        #We construct the list of all final DMA sites equally close to
        #the overlap center P_{alpha,beta} (we stop as soon as the 
        #minimum of all distances, computed after having taken out the 
        #previously found minimum distances [occurences of this minimum distance
        #in the table 'distances_overlap_center_atoms'], is strictly larger than 
        #the minimum distance common to all equally closest final expansion sites)
        list_closest_site=[positions_final_expansion_sites[index_first_closest_site]]
        
        #Case of a diatomic molecule : two identical atoms => for overlap
        #centers at the middle of the bond, the two atomic positions
        #have to be returned (case of index_philosophy==1)
        #Index of the last encountered closest site :
        index_last_closest_site=index_first_closest_site
        
        nb_other_equally_close_sites=0
        
        while (len(distances_overlap_center_atoms)>=2):
            
            #We delete the minimum distance previously encountered :
            del distances_overlap_center_atoms[index_last_closest_site]
 
            #If there was in fact a second (or, next) occurrence of the minimum distance
            #encountered before :
            if (np.min(distances_overlap_center_atoms)<first_closest_distance+epsilon):
                #We add the two positions (x_0,y_0,z_0), (x_1,y_1,z_1) ...
                #of the atoms equally close to this overlap center, to the 
                #list which will be returned :
                
                #There is another EQUALLY CLOSE (to the pverlap center P_{alpha,beta}
                #DMA final site
                nb_other_equally_close_sites+=1
                
                index_argmin_bis=np.argmin(distances_overlap_center_atoms)
                
                #This index 'index_last_closest_site' does not correspond to the
                #atom index due to the previous deletion of one or several elements
                #of the table 'distances_overlap_center_atoms' with the Python function "del ...[i]"
                index_last_closest_site=index_argmin_bis
                
                #Compute the correct atom index (corresponding to the 
                #next closest -- to this overlap center -- final site)
                #taking into account the 'deletion' of one or several distances from 
                #the table of distances 'distances_overlap_center_atoms'
                #(always in increasing order as np.min() or np.argmin() find the 
                #FIRST OCCURENCE of the minimum in a table)
                #(décalage d'indice dans 'distances_overlap_center_atoms' après 
                #suppression d'un ou plusieurs de ses éléments = au nombre d'éléments supprimés
                #précédemment)
                
                index_next_closest_atom=index_argmin_bis+nb_other_equally_close_sites
                ##We have always index_argmin_bis>=index_last_closest_site
                #as the function np.argmin() returns THE FIRST OCCURENCE
                #of the minimum
                   
                list_closest_site.append(positions_final_expansion_sites[index_next_closest_atom])
            
            #In the case of no other equally close final site : we break the loop :
            else:
                break
            
        #If there was only one atom (= final site) the closest to the overlap center P_{alpha,beta}
        #=> list_closest_site = [positions_final_expansion_sites[index_first_closest_site]]
        
        return list_closest_site
        ##############################################
   
    #If all sites are retained for the final distributed expansion 
    #=> OK : the overlap center P_{alpha,beta} is already an expansion center
    if (index_philosophy==5):
        
        return [coordinates_natural_center(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO)]
##############################################################################
        
    
##############################################################################
#'alpha' and 'beta' indexes span the whole basis set functions 
#=> are each associated to a given nuclei
#This function simply computes the barycenter (with dzeta gaussian exponents as coefficients)
#of the positions R_{alpha} and R_{beta} where chi_{alpha}(.) and chi_{beta}(.) are respectively centered.
def coordinates_natural_center(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO):
    
    R_alpha=position_nuclei_associated_basis_function_pGTO[alpha]
    R_beta=position_nuclei_associated_basis_function_pGTO[beta]
    
    #The function "import_exponents_GTO_basis_sets(QM_output_file)"
    #is assumed to have been applied previously, once and for all
    #yielding the output 'basis_set_exponents'
    zeta_alpha=basis_set_exponents_primitive_GTOs[alpha]
    zeta_beta=basis_set_exponents_primitive_GTOs[beta]
    
    ##BEWARE of the UNIT  : cf. result if this function used in the function
    #'binomial_coefficient_real_spherical_harmonics()' => substraction done with
    #position [x_S,y_S,z_S] of the final expansion site ==> have to be in same physical unit
    #Here we perform the conversion in Ang^{-2}
    coordinates_center=[((zeta_alpha/(zeta_alpha+zeta_beta))*R_alpha[i]+(zeta_beta/(zeta_alpha+zeta_beta))*R_beta[i]) for i in range(3)]
    #Returns a TABLE (dimension 3) as R_alpha and R_beta are tables
    ##return (dzeta_alpha/(dzeta_alpha+dzeta_beta))*R_alpha+(dzeta_beta/(dzeta_alpha+dzeta_beta))*R_beta

    return coordinates_center
##############################################################################
    


##############################################################################
#''boolean_all_atomic_sites' : boolean variable : whether to keep all nuclei positions as DMA final sites or not.
#'atomic_sites_to_omit' : list of atom indexes to omit as final DMA expansion sites
#Will be of size >0 only if 'boolean_all_atomic_sites'==False
def number_final_expansion_centers(atomic_coordinates,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,nb_primitive_GTOs,correspondence_basis_pGTOs_atom_index):
    
    
    #Case of the EXACT multipolar expansion, i.e. all overlap centers are kept as DMA final sites
    if (index_philosophy==5):
        
        number_final_DMA_sites=len(atomic_coordinates)
        
        #We then count the 'P_{alpha,beta}' overlap centers not centered at nuclei
        #i.e. originating from products of pGTOs NOT centered at the same point
        for alpha in range(nb_primitive_GTOs):
            
            for beta in range(alpha+1,nb_primitive_GTOs):
                
                #In the case of two primitive GTOs centered on two different atoms (of different indexes)
                #==> the center of the product of the two gaussians is not a nuclei => we add it
                #in the list of overlap centers
                if (correspondence_basis_pGTOs_atom_index[alpha]!=correspondence_basis_pGTOs_atom_index[beta]):
                    
                    number_final_DMA_sites+=1
           
        return number_final_DMA_sites
    
    #Case of approximate multipolar expansion i.e. with redistribution of multipole moments
    #to a reduced number of final sites
    else :
        
        ##'atomic_sites_to_omit' should be empty in the case of boolean_all_atomic_sites=True
        return len(atomic_coordinates)-len(atomic_sites_to_omit)+len(user_extra_final_sites)
    
    ##Or :
    ##return len(coordinates_final_expansion_centers(atomic_coordinates,correspondence_basis_pGTOs_atom_index,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,nb_primitive_GTOs))
 
##############################################################################
    
##############################################################################
#Computes the coordinates of the final multipole expansion sites (returns a list [[x_1,y_1,z_1],..,[x_N,y_N,z_N]])
#where [x_i,y_i,z_i] is the coordinate of the i^{th} DMA final expansion center
#=> appeals to the function 'coordinates_natural_center()' for all the (alpha,beta) such that
#the natural expansion center P_{alpha,beta} is kept as final expansion center
##'atomic_coordinates' already obtained (befire calling this function) 
#as data.atomcoords (cclib attribute) where 'data' is parsed with cclib from Gaussian output (e.g. .log file)
##'correspondence_basis_pGTOs_atom_index' = tab of indexes of the atoms to which each primitive GTO is associated
def coordinates_final_expansion_centers(atomic_coordinates,correspondence_basis_pGTOs_atom_index,boolean_all_atomic_sites,atomic_sites_to_omit,user_extra_final_sites,index_philosophy,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,nb_primitive_GTOs):
    
    #Case of exact multipolar expansion ==> all overlap centers are kept
    #==> nuclei + overlap centers which are not nuclei
    #==> BEWARE : count the nuclei (all the 'P_{alpha,alpha}' expansion centers
    #coincide with nuclei, and also P_{alpha,beta} with \chi_{alpha}^{pGTO} and \chi_{beta}^{pGTO}
    #both centered at the same nuclei) and then all the 'natural' overlap centers 
    #not centered on the nuclei
    if (index_philosophy==5):
        
        list_final_DMA_sites=[]
        
        #We first count the nuclei
        for i in range(len(atomic_coordinates)):
            
            list_final_DMA_sites.append(atomic_coordinates[i])
        
        #Overlap centers P_{alpha,beta} with beta > alpha (as P_{alpha,alpha} coincide with nuclei ==> already counted)
        list_overlap_centers=[[coordinates_natural_center(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO) for beta in range(alpha+1,nb_primitive_GTOs)] for alpha in range(nb_primitive_GTOs)]
        
        #We then count the 'P_{alpha,beta}' overlap centers not centered at nuclei
        #i.e. originating from products of pGTOs NOT centered at the same point
        for alpha in range(nb_primitive_GTOs):
            
            for beta in range(alpha+1,nb_primitive_GTOs):
                
                #In the case of two primitive GTOs centered on two different atoms (of different indexes)
                #==> the center of the product of the two gaussians is not a nuclei => we add it
                #in the list of overlap centers
                if (correspondence_basis_pGTOs_atom_index[alpha]!=correspondence_basis_pGTOs_atom_index[beta]):
                    
                    ##If same 'zeta' exponents for two couples of pGTOs (alpha,beta)
                    ##==> same position of the overlap center P_{alpha,beta} ==> double-counting !
                    ##=> To avoid this : check that this overlap center is not already in the list
                    ##(not the most efficient/ quick way to test this...)
                    
                    test_overlap_center_already_encountered=False
                    
                    for k in range(len(list_final_DMA_sites)):
                        
                        if (scipy.spatial.distance.pdist([list_final_DMA_sites[k],list_overlap_centers[alpha][beta-alpha-1]])[0]< epsilon):
                            
                            test_overlap_center_already_encountered=True
                            
                    #Add this final overlap center only if 'test_overlap_center_already_encountered' is False
                    #(i.e. we count this newly encountered overlap center)
                    if (test_overlap_center_already_encountered==False):
                        
                        list_final_DMA_sites.append(list_overlap_centers[alpha][beta-alpha-1])
                
                
        return list_final_DMA_sites
    
    else:
        
        if (boolean_all_atomic_sites==True):
        
            if (len(user_extra_final_sites)==0):
    
                return atomic_coordinates
        
            else:
            
                list_final_DMA_sites=[]
            
                for i in range(len(atomic_coordinates)):
                
                    list_final_DMA_sites.append(atomic_coordinates[i])
            
                for k in range(len(user_extra_final_sites)):
                
                    list_final_DMA_sites.append(user_extra_final_sites[k])
                
                return list_final_DMA_sites
        #Case where we omit some of the nuclei as final expansion centers    
        else:
        
            list_final_DMA_sites=[]
        
            for k in range(len(atomic_coordinates)):
            
                if ((k in atomic_sites_to_omit)==False):
                
                    list_final_DMA_sites.append(atomic_coordinates[k])
        
            #We also count the additional DMA expansion sites (if any)
            #provided by the user
            for k in range(len(user_extra_final_sites)):
                
                list_final_DMA_sites.append(user_extra_final_sites[k])
        
            return list_final_DMA_sites
##############################################################################


##############################################################################
#Checks if an overlap center P_{alpha,beta} is or not a final expansion site
#for the DMA analysis (if yes ; returns the index of this DMA final site)
def overlap_center_is_DMA_final_expansion_site(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,positions_final_expansion_sites):
    
    overlap_center=coordinates_natural_center(alpha,beta,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO) 
    
    for k in range(len(positions_final_expansion_sites)):
      
        if (scipy.spatial.distance.pdist([overlap_center,positions_final_expansion_sites[k]])[0] < epsilon):
            
            #We return 'True' and the index of the DMA final expansion site
            #which coincides with this 'natural' overlap center P_{alpha,beta}
            return [True,k]
        
    #No final DMA expansion site is epsilon-close to this overlap center
    return [False]
##############################################################################                

##############################################################################
#Returns the LIST of tuples (alpha_k,beta_k) associated to the p^th final expansion center 
#i.e. such that P_{alpha_k,beta_k}=S_p (p^th final expansion center)
#Indeed : several elementary distributions [i.e. for different couple of indexes (alpha_k,beta_k)] (chi_{alpha_k}*chi_{beta_k})(.) can be centered at the same point
#(ONLY POSSIBLE IF ALL FINAL EXPANSION CENTERS ARE NATURAL EXPANSION CENTERS) => i.e. only for some (distributed) multipole 'philosophies'
def list_product_pGTOs_associated_final_expansion_center(p):
    
    return [0,0]
##############################################################################


##############################################################################
##############################################################################
##############################################################################

##############################################################################
#Function that computes the coefficients {c^(l)_{i1,i2,i3}}_{i1,i2,i3} of :
#r^l * Y_l^0(theta,phi) = R_l^0(\vec{r}) = \sum_{i1,i2,i3 / i1+i2+i3=n} [c^(l)_{i1,i2,i3} * x^{i1}*y^{i2}*z^{i3}]
#with NORMALIZED definition of the spherical harmonics Y_l^0(theta,phi) [||Y_l^0(.)||_{S²} = 1]
def computes_coefficient_solid_harmonic_l_0_recurrence_formula(l):
    
    if (l==0):
        
        tab_coefficients_0=np.zeros([1,1,1])
        
        #Normalized spherical harmonic Y_0^0(.)=1/sqrt(4*pi)
        tab_coefficients_0[0,0,0]=math.sqrt(1./(4*math.pi))
        
        return tab_coefficients_0
    
    elif (l==1):
        
        #8 coefficients (in front of the polynoms 1, x, y, z, x*y, x*z, y*z, x*y*z)
        #i.e. all the product of polynoms with each variable of degree <= l (=1 here)
        tab_coefficients_1=np.zeros([2,2,2])
        #R_l^0(\vec{r}) = z = 1* (x^0 * y^0 * z^1)
        tab_coefficients_1[0,0,1]=math.sqrt(3./(4*math.pi))
        
        return tab_coefficients_1
        
    elif (l>=2):
        
        #tab_coefficients[i,1,i2,i3]=c^(l)_{i1,i2,i3}
        #(l+1)³ coefficients, but only those such that i1+i2+i3 are POSSIBLY not
        #equal to 0  have a contribution
        #over the homogeneous polynom of degree l : x^{i1}*y^{i2}*z^{i3}
        #(of the solid harmonic of order (l,0))
        tab_coefficients_l=np.zeros([l+1,l+1,l+1])
        
        #{c^(l-1)_{i1,i2,i3}}_{i1,i2,i3} :
        tab_coefficients_l_minus_1=computes_coefficient_solid_harmonic_l_0_recurrence_formula(l-1)
        
        #{c^(l-2)_{i1,i2,i3}}_{i1,i2,i3} :
        tab_coefficients_l_minus_2=computes_coefficient_solid_harmonic_l_0_recurrence_formula(l-2)

        
        for i1 in range(l+1):
            for i2 in range(l+1):
                for i3 in range(l+1):
                    
                    #The only non-zero coefficients c^(l)_{i1,i2,i3}
                    #are those associated to homogeneous polynoms : x^{i1}*y^{i2}*z^{i3}
                    #of degree l (=i1+i2+i3)
                    if ((i1+i2+i3)==l):
                        
                        if (i3>=1):

                            tab_coefficients_l[i1,i2,i3]+=math.sqrt((2*l+1)/(2*l-1))*((2*l-1)/l)*tab_coefficients_l_minus_1[i1,i2,i3-1]
                        
                        if (i1>=2):

                            tab_coefficients_l[i1,i2,i3]+= -((l-1)/l)*math.sqrt((2*l+1)/(2*l-3))*tab_coefficients_l_minus_2[i1-2,i2,i3]
      
                        if (i2>=2):

                            tab_coefficients_l[i1,i2,i3]+= -((l-1)/l)*math.sqrt((2*l+1)/(2*l-3))*tab_coefficients_l_minus_2[i1,i2-2,i3]
                            
                        if (i3>=2):

                            tab_coefficients_l[i1,i2,i3]+= -((l-1)/l)*math.sqrt((2*l+1)/(2*l-3))*tab_coefficients_l_minus_2[i1,i2,i3-2]       
                                                    
        return tab_coefficients_l
##############################################################################

##############################################################################
#Function that returns the solid harmonic of order l : 
#r^l * Y_l^0(theta,phi) = R_l^0(\vec{r}) 
#                       = \sum_{i1,i2,i3 / i1+i2+i3=n} [c^(l)_{i1,i2,i3} * x^{i1}*y^{i2}*z^{i3}]
#with NORMALIZED definition of the spherical harmonics Y_l^0(theta,phi) [||Y_l^0(.)||_{S²} = 1]
#in the format of the function 'conversion_spherical_harmonics_cartesian_homogeneous_polynoms()':
#[nb,[[i1,i2,i3],[.,.,.],..],[c^(l)_{i1,i2,i3},...]] where 'nb' is the total
#number of non-zero c^(l)_{i1,i2,i3} coefficients (in any order) associated to the 'monomer'
#of degree exponents [i1,i2,i3]
def computes_coefficient_solid_harmonic_l_0_recurrence_formula_list_format(l):
    #Three dimensional array of dimensions (l+1)**3 :
    tab_coefficients_l=computes_coefficient_solid_harmonic_l_0_recurrence_formula(l)
    
    
    #BEWARE : np.nonzero() function seems to work only for l=2 !
    ##non_zero_coeffs=np.nonzero(computes_coefficient_solid_harmonic_l_0_recurrence_formula(2))
    
    #Number of homogeneous polynoms x^{i1}*y^{i2}*z^{i3} (of degree l i.e. such that i1+i2+i3=l)
    #composing R_l^0(\vec{r}) ; the solid harmonic of rank l
    nb_polynoms=np.count_nonzero(tab_coefficients_l)
    #(nb_polynoms=len(non_zero_coeffs) in the specific case l=2)
    
    #List of exponents [i1,i2,i3] of homogeneous polynoms x^{i1}*y^{i2}*z^{i3} 
    #(of degree l i.e. such that i1+i2+i3=l) composing the real 
    #solid harmonic R_l^0(\vec{r}) 
    list_exponents_polynoms=[]
    
    #List of (non-zero) coefficients c^(l)_{i1,i2,i3} in factor of the 
    #homogeneous polynoms x^{i1}*y^{i2}*z^{i3} 
    #(of degree l i.e. such that i1+i2+i3=l) composing the real 
    #solid harmonic R_l^0(\vec{r})     
    list_coefficients_polynoms=[]
    
    for i1 in range(l+1):
        for i2 in range(l+1):
            for i3 in range(l+1):
                if (abs(tab_coefficients_l[i1,i2,i3])>0.):
                    
                    list_exponents_polynoms.append([i1,i2,i3])
                    list_coefficients_polynoms.append(tab_coefficients_l[i1,i2,i3])
                    
                    
    solid_harmonic_list=[nb_polynoms,list_exponents_polynoms,list_coefficients_polynoms]
    #In case the function np.non_zero() works :
    ##solid_harmonic_list=[nb_polynoms,[non_zero_coeffs[i] for i in range(nb_polynoms)],[tab_coefficients_l[non_zero_coeffs[i][0],non_zero_coeffs[i][1],non_zero_coeffs[i][2]] for i in range(nb_polynoms)]]
    
    return solid_harmonic_list
##############################################################################


##############################################################################
#Function that computes the coefficients {c^(l,m)_{i1,i2,i3}}_{i1,i2,i3} of the REAL solid harmonic :
#r^l * Y_l^m(theta,phi) = R_l^m(\vec{r}) 
#                       = \sum_{i1,i2,i3 / i1+i2+i3=n} [c^(l,m)_{i1,i2,i3} * x^{i1}*y^{i2}*z^{i3}]
#with NORMALIZED definition of the spherical harmonics Y_l^m(theta,phi) [||Y_l^m(.)||_{S²} = 1]
#for m \neq 0 (otherwise, for m=0, use functions computes_coefficient_solid_harmonic_l_0_recurrence_formula() 
#and computes_coefficient_solid_harmonic_l_0_recurrence_formula_list_format())
####################
#For m > 0 :
#c^(l,m)_{i1,i2,i3} = K_l^m * 
#                       \sum_{p+2*(k-q) = i1, 2r+m-p=i2, 2*(q-r-k)+l-m=i3} 
#                                   [ (-1)^{(m-p)/2} . \delta_{(m-p) even} * c_{m,p}^{l,k} * binom(q,k)*binom(r,q)]
####################
#For m < 0 :
#c^(l,m)_{i1,i2,i3} = K_l^m * 
#                       \sum_{p+2*(k-q) = i1, 2r+m-p=i2, 2*(q-r-k)+l-m=i3} 
#                                   [ (-1)^{(m-p-1)/2} . \delta_{(m-p) odd} * c_{m,p}^{l,k} * binom(q,k)*binom(r,q)]
###########
#where K_l_m=math.sqrt((2*l+1)/(2*math.pi))*math.sqrt((l-|m|)!/(l+|m|)!)
#and   c_{m,p}^{l,k}=((-1)**k/2**l) * binom(m,p) * binom(l,k) *  binom(2*(l-k),l) * ((l-2*k)!/(l-2*k-|m|)!)
def computes_coefficient_solid_harmonic_l_m(l,m):
    
    if (m==0):
        
        return computes_coefficient_solid_harmonic_l_0_recurrence_formula(l)
    
    #tab_coefficients[i,1,i2,i3]=c^(l,m)_{i1,i2,i3}
    #(l+1)³ coefficients, but only those such that i1+i2+i3 are POSSIBLY not
    #equal to 0 have a contribution
    #over the homogeneous polynom of degree l : x^{i1}*y^{i2}*z^{i3})
    #(of the solid harmonic of order (l,m))
    tab_coefficients_l_m=np.zeros([l+1,l+1,l+1])
    
    #K_l^m constant :
    K_l_m=math.sqrt((2*l+1)/(2*math.pi))*math.sqrt(math.factorial(l-abs(m))/math.factorial(l+abs(m)))
    
    
    for i1 in range(l+1):
        for i2 in range(l+1):
            for i3 in range(l+1):

                for k in range(int(np.floor((l-abs(m))/2))+1):
                    for p in range(abs(m)+1):
                            
                        #Factor c_{m,p}^{l,k}:
                        c_m_p_l_k=(1/2**l)*(-1)**k * scipy.special.binom(abs(m),p) * scipy.special.binom(l,k) *  scipy.special.binom(2*(l-k),l) * (math.factorial(l-2*k)/math.factorial(l-2*k-abs(m)))
                            
                        for q in range(k+1):
                            for r in range(q+1):

                                if ((p+2*(k-q))==i1) and ((2*r+abs(m)-p)==i2) and ((2*(q-r-k)+l-abs(m))==i3): 

                                    if (m>0):
                                        
                                        if ((m-p)%2==0):

                                            tab_coefficients_l_m[i1,i2,i3]+=K_l_m*(-1)**((m-p)/2) * c_m_p_l_k * scipy.special.binom(k,q) * scipy.special.binom(q,r)
                                                                          
                                    elif (m<0):
                                        
                                        if ((abs(m)-p-1)%2==0):
                                            tab_coefficients_l_m[i1,i2,i3]+=K_l_m*(-1)**((abs(m)-p-1)/2) * c_m_p_l_k * scipy.special.binom(k,q) * scipy.special.binom(q,r)
                                    
                                
    return tab_coefficients_l_m
##############################################################################


##############################################################################
#Function that returns the solid harmonic of order l,m  : 
#r^l * Y_l^m(theta,phi) = R_l^m(\vec{r}) 
#                       = \sum_{i1,i2,i3 / i1+i2+i3=n} [c^(l,m)_{i1,i2,i3} * x^{i1}*y^{i2}*z^{i3}]
#with NORMALIZED definition of the spherical harmonics Y_l^m(theta,phi) [||Y_l^m(.)||_{S²} = 1]
#in the format of the function 'conversion_spherical_harmonics_cartesian_homogeneous_polynoms()':
#[nb,[[i1,i2,i3],[.,.,.],..],[c^(l)_{i1,i2,i3},...]] where 'nb' is the total
#number of non-zero c^(l,m)_{i1,i2,i3} coefficients (in any order) associated to the 'monomer'
#of degree exponents [i1,i2,i3]
##BEWARE of normalization coefficient (due to normalization of Y_l^m(.)) as
#far as local distributed multipoles are concerned.
#Example : Q_0^0 = Q_{tot} =  \sqrt(4*pi) * \int r^0 * Y_0^0 * \rho(r) dr
#with the NORMALIZED definition of Y_0^0 !
def computes_coefficient_solid_harmonic_l_m_list_format(l,m):
    #Three dimensional array of dimensions (l+1)**3 :
    tab_coefficients_l_m=computes_coefficient_solid_harmonic_l_m(l,m)
    
    
    #BEWARE : np.nonzero() function seems to work only for l=2 !
    ##non_zero_coeffs=np.nonzero(computes_coefficient_solid_harmonic_l_0_recurrence_formula(2))
    
    #Number of homogeneous polynoms x^{i1}*y^{i2}*z^{i3} (of degree l i.e. such that i1+i2+i3=l)
    #composing R_l^0(\vec{r}) ; the solid harmonic of rank l
    nb_polynoms=np.count_nonzero(tab_coefficients_l_m)
    #(nb_polynoms=len(non_zero_coeffs) in the specific case l=2)
    
    #List of exponents [i1,i2,i3] of homogeneous polynoms x^{i1}*y^{i2}*z^{i3} 
    #(of degree l i.e. such that i1+i2+i3=l) composing the real 
    #solid harmonic R_l^0(\vec{r}) 
    list_exponents_polynoms=[]
    
    #List of (non-zero) coefficients c^(l)_{i1,i2,i3} in factor of the 
    #homogeneous polynoms x^{i1}*y^{i2}*z^{i3} 
    #(of degree l i.e. such that i1+i2+i3=l) composing the real 
    #solid harmonic R_l^0(\vec{r})     
    list_coefficients_polynoms=[]
    
    for i1 in range(l+1):
        for i2 in range(l+1):
            for i3 in range(l+1):
                if (abs(tab_coefficients_l_m[i1,i2,i3])>0.):
                    
                    list_exponents_polynoms.append([i1,i2,i3])
                    list_coefficients_polynoms.append(tab_coefficients_l_m[i1,i2,i3])
                    
                    
    solid_harmonic_list=[nb_polynoms,list_exponents_polynoms,list_coefficients_polynoms]
    #In case the function np.non_zero() works :
    ##solid_harmonic_list=[nb_polynoms,[non_zero_coeffs[i] for i in range(nb_polynoms)],[tab_coefficients_l[non_zero_coeffs[i][0],non_zero_coeffs[i][1],non_zero_coeffs[i][2]] for i in range(nb_polynoms)]]
    
    return solid_harmonic_list
##############################################################################

"""
l=5
for m in range(-l,l+1):
   print('l=5, m = '+str(m))
   print(computes_coefficient_solid_harmonic_l_m_list_format(l,m))
   print(' ')
"""


##############################################################################
##############################################################################
##############################################################################

##############################################################################
#ALTERNATIVE (manual) definitions of the real solid harmonics, encoding the exceptions
#of CARTESIAN shells (e.g. cartesian d-shells or f-shells) used in the basis sets
#of some Quantum Chemistry codes.


##############################################################################  
#Encodes the definition of a pure spherical harmonic K_l^m * r^l * Y_l^m(\vec{r}/|\vec{r}|)
#where K_l^m is the normalization coefficient OF THE SPHERICAL HARMONIC Y_l^m(\vec{r}/|\vec{r}|)=Y_l^m(theta,phi) ALONE
#and not of the whole SOLID HARMONIC (r^l * Y_l^m(\vec{r}/|\vec{r}|))
#Solid hamonics times gaussians (exp(-zeta*r²)) are defined as the basic primitive GTOs and are thus normalized
#(elsewhere, see function 'compute_normalization_coefficient_primitive_GTO()')
#i.e. r^l * Y_l^m(\vec{r}/|\vec{r}|) * exp(-zeta*r²) is normalized (but not the solid harmonic)
#Computes :
# - number p(l,m) of terms (homogeneous cartesian polynoms in x, y, z of degree l) defining r^l * Y_l^m(\vec{r}/|\vec{r}|)
# - exponents {n_k(l_{\alpha},m_{\alpha}),m_k(l_{\alpha},m_{\alpha}),t_k(l_{\alpha},m_{\alpha})}_k describing each component [each of the homogeneous polynom of degree l]
# - coefficients {w_k^(l_{\alpha},m_{\alpha})} [integers, as they do not include the normalization constant taken into account elsewhere]
#Example : r^2 * Y_2^{-2}(\theta,\phi)=xy [l=2,m=-2] : p(2,-2)=1 ; n_1(2,-2)=1, m_1(2,-2)=1, t_1(2,-2)=0 ; w_1(2,-2)=1
##Returns [p(l,m),[[n_1(l,m),m_1(l,m),t_1(l,m)],..[n_{p(l,m)}(l,m),m_{p(l,m)}(l,m),t_{p(l,m)}(l,m)]],[w_1^(l,m),w_2^(l,m),..,w_{p(l,m)}(l,m)]]
def conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l,m):
    
    if (l==0):
        #Polynom 1/sqrt(4*pi)
        #return [1,[[0,0,0]],[1/math.sqrt(4*math.pi)]]
        return [1,[[0,0,0]],[1]]
    
    if (l==1):
        #Spherical harmonics Y_1^m ; where m \in {-1,0,1}
        #have normalizing factor \sqrt(3/(4*pi))
        if (m==-1):
            #Polynom \sqrt(3/(4*pi)) * y 
            #return [1,[[0,1,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : y
            return [1,[[0,1,0]],[1]]
        
        elif (m==0):
            #Polynom \sqrt(3/(4*pi)) * z  
            #return [1,[[0,0,1]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : z
            return [1,[[0,0,1]],[1]]
        
        elif (m==1):
            #Polynom \sqrt(3/(4*pi)) * x 
            #return [1,[[1,0,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : x
            return [1,[[1,0,0]],[1]]
        
    """
    ################
    #Cf. code M. Herbst
    #https://github.com/molsturm/gint/blob/master/src/interface/python/gint/gaussian/_gaussian_shells.py
    Function cartesian_xyz_expontents(l, ordering):
    
    ##Produce the list of cartesian exponents.
    ##This determines the order inside the cartesian exponents, which is used.
    ##The precise ordering can be adapted using the ordering parameter.
    ##The following orderings are currently implemented:
    ##- standard: The ordering of the Common Component Architecture (CCA) standard
    ##          as described in the paper DOI 10.1002/jcc
    ######################################
    #if ordering == "standard":
    #Standard ordering as described in DOI 10.1002/jcc
    #return [ (a, l-a-c, c) for a in range(l, -1, -1) for c in range(0, l-a+1) ]
    ######################################
    """
            
    #####################################################
    #Case of regular spherical d-harmonics (5 harmonics):
    
    if (l==2):
        if(m==-2):
            #Polynom \sqrt(15/(4*pi))*x*y (normalized real spherical harmonic)
            #return [1,[[1,1,0]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[1,1,0]],np.dot(math.sqrt(3),[1])]
        
        elif (m==-1):
            #Polynom \sqrt(15/(4*pi))*y*z(normalized real spherical harmonic)
            #return [1,[[0,1,1]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[0,1,1]],np.dot(math.sqrt(3),[1])]
        
        elif (m==0):
            #Polynom (1/4)* sqrt(5/pi) *[2*(z^2)-x^2-y^2] (normalized real spherical harmonic)
            #return [3,[[0,0,2],[2,0,0],[0,2,0]],np.dot(math.sqrt(5/(16*math.pi)),[2,-1,-1])]
            #Stone's definition :
            return [3,[[0,0,2],[2,0,0],[0,2,0]],np.dot(0.5,[2,-1,-1])]
        
        elif (m==1):
            #Polynom \sqrt(15/(4*pi))*x*z (normalized real spherical harmonic)
            #return [1,[[1,0,1]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[1,0,1]],np.dot(math.sqrt(3),[1])]
        
        elif (m==2):
            #Polynom \sqrt(15/(16*pi))*[x^2-y^2] (normalized real spherical harmonic)
            #return [2,[[2,0,0],[0,2,0]],np.dot(math.sqrt(15./(16.*math.pi)),[1,-1])]
            #Stone's definition :
            return [2,[[2,0,0],[0,2,0]],np.dot(math.sqrt(3./4.),[1,-1])]
    #################################################"
    
    ####################
    ##l=3 :
    #In Gaussian : regular spherical harmonics f-functions
    #Stone's non-normalized spherical harmonics
    if (l==3):
        
        if (m==-3):
            #Polynom sqrt(35/(32*math.pi))  * [3*x^2*y-y^3] (normalized spherical harmonic)
            #return [2,[[2,1,0],[0,3,0]],np.dot(math.sqrt(35/(32*math.pi)),[3,-1])]
            #Stone's definition :
            return [2,[[2,1,0],[0,3,0]],np.dot(math.sqrt(5/8),[3,-1])]
        
        elif (m==-2):
            #Polynom sqrt(105/(4*math.pi)) * xyz (normalized spherical harmonic)
            #return [1,[[1,1,1]],np.dot(math.sqrt(105/(4*math.pi)),[1])]
            #Stone's definition :
            return [1,[[1,1,1]],np.dot(math.sqrt(15),[1])]
        
        elif (m==-1):
            #Polynom math.sqrt(21/(32*math.pi))* [y*(4*z^2-x^2-y^2)] (normalized spherical harmonic)
            #return [3,[[0,1,2],[2,1,0],[0,3,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
            #Stone's definition :
            return [3,[[0,1,2],[2,1,0],[0,3,0]],np.dot(math.sqrt(3/8),[4,-1,-1])]        
        
        elif (m==0):
            #Polynom math.sqrt(7/(16*math.pi)) * [2*z^3-3*x^2*z-3*y^2*z]=math.sqrt(7/(16*math.pi)) * [5*z^3-3*z*r^2]
            #(normalized spherical harmonic)
            #return [3,[[0,0,3],[2,0,1],[0,2,1]],np.dot(0.5,[2,-3,-3])]
            #Stone's definition :
            return [3,[[0,0,3],[2,0,1],[0,2,1]],np.dot(0.5,[2,-3,-3])]        
        
        elif (m==1):
            #Polynom math.sqrt(21/(32*math.pi)) * [x*(4*z^2-x^2-y^2)]
            #return [3,[[1,0,2],[3,0,0],[1,2,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
            #Stone's definition :
            return [3,[[1,0,2],[3,0,0],[1,2,0]],np.dot(math.sqrt(3/8),[4,-1,-1])]        
        
        elif (m==2):
            #Polynom sqrt(105/(16*math.pi)) * [z*(x^2-y^2)] (normalized spherical harmonic)
            #return [2,[[2,0,1],[0,2,1]],np.dot(math.sqrt(105/(16*math.pi)),[1,-1])]
            #Stone's definition :
            return [2,[[2,0,1],[0,2,1]],np.dot(math.sqrt(15/4),[1,-1])]        
        
        elif (m==3):
            #Polynom sqrt(35/(32*math.pi)) * [x^3-3*x*y^2] (normalized spherical harmonic)
            #return [2,[[3,0,0],[1,2,0]],np.dot(math.sqrt(35/32*math.pi),[1,-3])]
            #Stone's definition :
            return [2,[[3,0,0],[1,2,0]],np.dot(math.sqrt(5/8),[1,-3])]        
        
    ########################
    ##l=4 : (normalization coeffs. taken from Stone GDMA manual)
    if (l==4):
        if (m==-4):
            
            #Polynom (3./4.)*sqrt(35/pi) * [x^3*y-x*y^3] (normalized spherical harmonic)
            #return [2,[[3,1,0],[1,3,0]],np.dot((3./4.)*math.sqrt(35/math.pi),[1,-1])]
            
            #Stone's definition :
            return [2,[[3,1,0],[1,3,0]],np.dot(math.sqrt(35/4),[1,-1])]
        
        elif (m==-3):
            #Stone's definition : 
            #Polynom sqrt(35/8) * [3*x^2*y*z-y^3*z]
            return [2,[[2,1,1],[0,3,1]],np.dot(math.sqrt(35/8),[3,-1])]
        
        elif (m==-2):
            #Stone's definition : 
            #Polynom sqrt(5/4) * [6*x*y*z^2-x^3*y-x*y^3]=sqrt(5/4) * [x*y*(7*z^2-r^2)]
            return [3,[[1,1,2],[3,1,0],[1,3,0]],np.dot(math.sqrt(5/4),[6,-1,-1])]
        
        elif (m==-1):
            ##ERROR corrected (Robert) 05/08/20
            #Stone's definition : polynom sqrt(5/8) * [4*y*z^3-3*x^2*y*z-3*y^3*z]= sqrt(5/8) * [7*y*z^3-3*y*z*r^2]
            return [3,[[0,1,3],[2,1,1],[0,3,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
        
        elif (m==0):
            #Stone's definition :
            #Polynom (1/8) * [8*z^4+3*x^4+3*y^4+6*x^2*y^2-24*x^2*z^2-24*y^2*z^2]=(1/8) * [35*z^4-30*z^2*r^2+3*r^4]
            return [6,[[0,0,4],[4,0,0],[0,4,0],[2,2,0],[2,0,2],[0,2,2]],np.dot(1./8.,[8,3,3,6,-24,-24])]
        
        elif (m==1):
            #Stone's definition :
            #Polynom sqrt(5/8) * [4*x*z^3-3*x*y^2*z-3*x^3*z]= sqrt(5/8) * [7*x*z^3-3*x*z*r^2]
            return [3,[[1,0,3],[1,2,1],[3,0,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
      
        elif (m==2):
            #Stone's definition :
            ##ERROR corrected (Robert) 05/08/20 (y^4 polynom represented with [0,0,4] instead of [0,4,0] !)
            #Polynom sqrt(5/16) * [-x^4+y^4+6*x^2*z^2-6*y^2*z^2]= sqrt(5/16) * [(x^2-y^2)*(7*z^2-r^2)] where r^2=x^2+y^2+z^2
            return [4,[[4,0,0],[0,4,0],[2,0,2],[0,2,2]],np.dot(math.sqrt(5/16),[-1,1,6,-6])]
        
        elif (m==3):
            #Stone's definition :
            #Polynom sqrt(35/8) * [z*(x^3-3*x*y^2)]
            return [2,[[3,0,1],[1,2,1]],np.dot(math.sqrt(35/8),[1,-3])]
        
        elif (m==4):
            #Stone's definition :
            #Polynom sqrt(35/64) * [x^4-6*x^2*y^2+y^4]
            return [3,[[4,0,0],[2,2,0],[0,4,0]],np.dot(math.sqrt(35/64),[1,-6,1])]
        
##############################################################################

##############################################################################
#Encodes the definition of a pure spherical harmonic r^l * Y_l^m(\vec{r}/|\vec{r}|
#with NORMALIZED real spherical harmonic Y_l^m(.)
#EXCEPT when l==2 (d-type function) : encodes directly cartesian type functions 
# x² / y² / z² / x*y / x*z / y*z
def conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells(l,m):
 
    if (l==0):
        #Polynom 1/sqrt(4*pi)
        #return [1,[[0,0,0]],[1/math.sqrt(4*math.pi)]]
        return [1,[[0,0,0]],[1]]
    
    if (l==1):
        #Spherical harmonics Y_1^m ; where m \in {-1,0,1}
        #have normalizing factor \sqrt(3/(4*pi))
        if (m==-1):
            #Polynom \sqrt(3/(4*pi)) * y 
            #return [1,[[0,1,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : y
            return [1,[[0,1,0]],[1]]
        
        elif (m==0):
            #Polynom \sqrt(3/(4*pi)) * z  
            #return [1,[[0,0,1]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : z
            return [1,[[0,0,1]],[1]]
        
        elif (m==1):
            #Polynom \sqrt(3/(4*pi)) * x 
            #return [1,[[1,0,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : x
            return [1,[[1,0,0]],[1]]
    
    ###################################
    #Case of d-type shells : in Gaussian  : no need to convert from spherical to cartesian harmonics
    #as cartesian functions are used for the d-type shells  (XX, YY, ZZ, XY, XZ, YZ) i.e. 
    #are = only one homogeneous polynom of degree 2
    #(and transformed internally by Gaussian into spherical harmonics ??)
    #############################
    #Pure conventions :
    #(l=2,m=100) : XX ; (l=2,m=101) : YY ; (l=2,m=102) : ZZ ; 
    #(l=2,m=103) : XY ; (l=2,m=104) : XZ ; (l=2,m=105) : YZ
    
    ##NORMALIZATION factors ??
    ##(TO DO)
    
    if (l==2):
        
        if(m==100):
            #Polynom x*x
            return [1,[[2,0,0]],[1]]
        
        elif (m==101):
            #Polynom y*y
            return [1,[[0,2,0]],[1]]
        
        elif (m==102):
            #Polynom z*z
            return [1,[[0,0,2]],[1]]  
        
        elif (m==103):
            #Polynom x*y
            return [1,[[1,1,0]],[1]]
        
        elif (m==104):
            #Polynom x*z
            return [1,[[1,0,1]],[1]]
        
        elif (m==105):
            #Polynom y*z
            return [1,[[0,1,1]],[1]]
            
    ####################
    ##l=3 :
    #In Gaussian : regular spherical harmonics f-functions
    """
    if (l==3):
        
        if (m==-3):
            #Polynom sqrt(35/(32*math.pi))  * [3*x^2*y-y^3]
            return [2,[[2,1,0],[0,3,0]],np.dot(math.sqrt(35/(32*math.pi)),[3,-1])]
        
        elif (m==-2):
            #Polynom sqrt(105/(4*math.pi)) * xyz
            return [1,[[1,1,1]],np.dot(math.sqrt(105/(4*math.pi)),[1])]
        
        elif (m==-1):
            #Polynom math.sqrt(21/(32*math.pi))* [y*(4*z^2-x^2-y^2)]
            return [3,[[0,1,2],[2,1,0],[0,3,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
        
        elif (m==0):
            #Polynom math.sqrt(7/(16*math.pi)) * [2*z^3-3*x^2*z-3*y^2*z]=math.sqrt(7/(16*math.pi)) * [5*z^3-3*z*r^2]
            return [3,[[0,0,3],[2,0,1],[0,2,1]],np.dot(0.5,[2,-3,-3])]
        
        elif (m==1):
            #Polynom math.sqrt(21/(32*math.pi)) * [x*(4*z^2-x^2-y^2)]
            return [3,[[1,0,2],[3,0,0],[1,2,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
        
        elif (m==2):
            #Polynom sqrt(105/(16*math.pi)) * [z*(x^2-y^2)]
            return [2,[[2,0,1],[0,2,1]],np.dot(math.sqrt(105/(16*math.pi)),[1,-1])]
        
        elif (m==3):
            #Polynom sqrt(35/(32*math.pi)) * [x^3-3*x*y^2]
            return [2,[[3,0,0],[1,2,0]],np.dot(math.sqrt(35/32*math.pi),[1,-3])]
    """
    #Stone's non-normalized spherical harmonics (CORRECTION 15/09/20)
    if (l==3):
        
        if (m==-3):
            #Polynom sqrt(35/(32*math.pi))  * [3*x^2*y-y^3] (normalized spherical harmonic)
            #return [2,[[2,1,0],[0,3,0]],np.dot(math.sqrt(35/(32*math.pi)),[3,-1])]
            #Stone's definition :
            return [2,[[2,1,0],[0,3,0]],np.dot(math.sqrt(5/8),[3,-1])]
        
        elif (m==-2):
            #Polynom sqrt(105/(4*math.pi)) * xyz (normalized spherical harmonic)
            #return [1,[[1,1,1]],np.dot(math.sqrt(105/(4*math.pi)),[1])]
            #Stone's definition :
            return [1,[[1,1,1]],np.dot(math.sqrt(15),[1])]
        
        elif (m==-1):
            #Polynom math.sqrt(21/(32*math.pi))* [y*(4*z^2-x^2-y^2)] (normalized spherical harmonic)
            #return [3,[[0,1,2],[2,1,0],[0,3,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
            #Stone's definition :
            return [3,[[0,1,2],[2,1,0],[0,3,0]],np.dot(math.sqrt(3/8),[4,-1,-1])]        
        
        elif (m==0):
            #Polynom math.sqrt(7/(16*math.pi)) * [2*z^3-3*x^2*z-3*y^2*z]=math.sqrt(7/(16*math.pi)) * [5*z^3-3*z*r^2]
            #(normalized spherical harmonic)
            #return [3,[[0,0,3],[2,0,1],[0,2,1]],np.dot(0.5,[2,-3,-3])]
            #Stone's definition :
            return [3,[[0,0,3],[2,0,1],[0,2,1]],np.dot(0.5,[2,-3,-3])]        
        
        elif (m==1):
            #Polynom math.sqrt(21/(32*math.pi)) * [x*(4*z^2-x^2-y^2)]
            #return [3,[[1,0,2],[3,0,0],[1,2,0]],np.dot(math.sqrt(21/(32*math.pi)),[4,-1,-1])]
            #Stone's definition :
            return [3,[[1,0,2],[3,0,0],[1,2,0]],np.dot(math.sqrt(3/8),[4,-1,-1])]        
        
        elif (m==2):
            #Polynom sqrt(105/(16*math.pi)) * [z*(x^2-y^2)] (normalized spherical harmonic)
            #return [2,[[2,0,1],[0,2,1]],np.dot(math.sqrt(105/(16*math.pi)),[1,-1])]
            #Stone's definition :
            return [2,[[2,0,1],[0,2,1]],np.dot(math.sqrt(15/4),[1,-1])]        
        
        elif (m==3):
            #Polynom sqrt(35/(32*math.pi)) * [x^3-3*x*y^2] (normalized spherical harmonic)
            #return [2,[[3,0,0],[1,2,0]],np.dot(math.sqrt(35/32*math.pi),[1,-3])]
            #Stone's definition :
            return [2,[[3,0,0],[1,2,0]],np.dot(math.sqrt(5/8),[1,-3])]   
        
    ################
    ##l=4 : (normalization coeffs. taken from Stone GDMA manual)
    
    if (l==4):
        if (m==-4):
            
            #Polynom (3./4.)*sqrt(35/pi) * [x^3*y-x*y^3] (normalized spherical harmonic)
            #return [2,[[3,1,0],[1,3,0]],np.dot((3./4.)*math.sqrt(35/math.pi),[1,-1])]
            
            #Stone's definition :
            return [2,[[3,1,0],[1,3,0]],np.dot(math.sqrt(35/4),[1,-1])]
        
        elif (m==-3):
            #Stone's definition : 
            #Polynom sqrt(35/8) * [3*x^2*y*z-y^3*z]
            return [2,[[2,1,1],[0,3,1]],np.dot(math.sqrt(35/8),[3,-1])]
        
        elif (m==-2):
            #Stone's definition : 
            #Polynom sqrt(5/4) * [6*x*y*z^2-x^3*y-x*y^3]=sqrt(5/4) * [x*y*(7*z^2-r^2)]
            return [3,[[1,1,2],[3,1,0],[1,3,0]],np.dot(math.sqrt(5/4),[6,-1,-1])]
        
        elif (m==-1):
            ##ERROR corrected (Robert) 05/08/20
            #Stone's definition : polynom sqrt(5/8) * [4*y*z^3-3*x^2*y*z-3*y^3*z]= sqrt(5/8) * [7*y*z^3-3*y*z*r^2]
            return [3,[[0,1,3],[2,1,1],[0,3,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
        
        elif (m==0):
            #Stone's definition :
            #Polynom (1/8) * [8*z^4+3*x^4+3*y^4+6*x^2*y^2-24*x^2*z^2-24*y^2*z^2]=(1/8) * [35*z^4-30*z^2*r^2+3*r^4]
            return [6,[[0,0,4],[4,0,0],[0,4,0],[2,2,0],[2,0,2],[0,2,2]],np.dot(1./8.,[8,3,3,6,-24,-24])]
        
        elif (m==1):
            #Stone's definition :
            #Polynom sqrt(5/8) * [4*x*z^3-3*x*y^2*z-3*x^3*z]= sqrt(5/8) * [7*x*z^3-3*x*z*r^2]
            return [3,[[1,0,3],[1,2,1],[3,0,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
      
        elif (m==2):
            #Stone's definition :
            ##ERROR corrected (Robert) 05/08/20 (y^4 polynom represented with [0,0,4] instead of [0,4,0] !)
            #Polynom sqrt(5/16) * [-x^4+y^4+6*x^2*z^2-6*y^2*z^2]= sqrt(5/16) * [(x^2-y^2)*(7*z^2-r^2)] where r^2=x^2+y^2+z^2
            return [4,[[4,0,0],[0,4,0],[2,0,2],[0,2,2]],np.dot(math.sqrt(5/16),[-1,1,6,-6])]
        
        elif (m==3):
            #Stone's definition :
            #Polynom sqrt(35/8) * [z*(x^3-3*x*y^2)]
            return [2,[[3,0,1],[1,2,1]],np.dot(math.sqrt(35/8),[1,-3])]
        
        elif (m==4):
            #Stone's definition :
            #Polynom sqrt(35/64) * [x^4-6*x^2*y^2+y^4]
            return [3,[[4,0,0],[2,2,0],[0,4,0]],np.dot(math.sqrt(35/64),[1,-6,1])]

##############################################################################


##############################################################################
#Encodes the definition of a pure spherical harmonic r^l * Y_l^m(\vec{r}/|\vec{r}|
#EXCEPT when l==3 (f-type function) : encodes directly cartesian type functions :
# XXX, YYY, ZZZ; XXY, XXZ, YYX, YYZ, ZZX, ZZY, XYZ
def conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells(l,m):
    
    if (l==0):
        #Polynom 1/sqrt(4*pi)
        #return [1,[[0,0,0]],[1/math.sqrt(4*math.pi)]]
        return [1,[[0,0,0]],[1]]
    
    if (l==1):
        #Spherical harmonics Y_1^m ; where m \in {-1,0,1}
        #have normalizing factor \sqrt(3/(4*pi))
        if (m==-1):
            #Polynom \sqrt(3/(4*pi)) * y 
            #return [1,[[0,1,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : y
            return [1,[[0,1,0]],[1]]
        
        elif (m==0):
            #Polynom \sqrt(3/(4*pi)) * z  
            #return [1,[[0,0,1]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : z
            return [1,[[0,0,1]],[1]]
        
        elif (m==1):
            #Polynom \sqrt(3/(4*pi)) * x 
            #return [1,[[1,0,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : x
            return [1,[[1,0,0]],[1]]
        
    #####################################################
    #Case of regular spherical d-harmonics (5 harmonics):
    
    if (l==2):
        if(m==-2):
            #Polynom \sqrt(15/(4*pi))*x*y (normalized real spherical harmonic)
            #return [1,[[1,1,0]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[1,1,0]],np.dot(math.sqrt(3),[1])]
        
        elif (m==-1):
            #Polynom \sqrt(15/(4*pi))*y*z(normalized real spherical harmonic)
            #return [1,[[0,1,1]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[0,1,1]],np.dot(math.sqrt(3),[1])]
        
        elif (m==0):
            #Polynom (1/4)* sqrt(5/pi) *[2*(z^2)-x^2-y^2] (normalized real spherical harmonic)
            #return [3,[[0,0,2],[2,0,0],[0,2,0]],np.dot(math.sqrt(5/(16*math.pi)),[2,-1,-1])]
            #Stone's definition :
            return [3,[[0,0,2],[2,0,0],[0,2,0]],np.dot(0.5,[2,-1,-1])]
        
        elif (m==1):
            #Polynom \sqrt(15/(4*pi))*x*z (normalized real spherical harmonic)
            #return [1,[[1,0,1]],[math.sqrt(15/(4*math.pi))]]
            #Stone's definition :
            return [1,[[1,0,1]],np.dot(math.sqrt(3),[1])]
        
        elif (m==2):
            #Polynom \sqrt(15/(16*pi))*[x^2-y^2] (normalized real spherical harmonic)
            #return [2,[[2,0,0],[0,2,0]],np.dot(math.sqrt(15./(16.*math.pi)),[1,-1])]
            #Stone's definition :
            return [2,[[2,0,0],[0,2,0]],np.dot(math.sqrt(3./4.),[1,-1])]
    #################################################"
        
    ####################
    ##l=3 : cartesian form
    if (l==3):
        
        if (m==300):
            #Polynom x*x*x
            return [1,[[3,0,0]],[1]]
                
        elif (m==301):
            #Polynom y*y*y
            return [1,[[0,3,0]],[1]]
        
        elif (m==302):
            #Polynom z*z*z
            return [1,[[0,0,3]],[1]]  
        
        elif (m==303):
            #Polynom x*x*y
            return [1,[[2,1,0]],[1]]
        
        elif (m==304):
            #Polynom x*x*z
            return [1,[[2,0,1]],[1]]
        elif (m==305):
            #Polynom y*y*x
            return [1,[[1,2,0]],[1]]
        
        elif (m==306):
            #Polynom y*y*z
            return [1,[[0,2,1]],[1]]
        
        elif (m==307):
            #Polynom z*z*x
            return [1,[[1,0,2]],[1]]        
        
        elif (m==308):
            #Polynom z*z*y
            return [1,[[0,1,2]],[1]]         

        elif (m==309):
            #Polynom x*y*z
            return [1,[[1,1,1]],[1]]  
        
    ########################
    ##l=4 : (normalization coeffs. taken from Stone GDMA manual)
    if (l==4):
        if (m==-4):
            
            #Polynom (3./4.)*sqrt(35/pi) * [x^3*y-x*y^3] (normalized spherical harmonic)
            #return [2,[[3,1,0],[1,3,0]],np.dot((3./4.)*math.sqrt(35/math.pi),[1,-1])]
            
            #Stone's definition :
            return [2,[[3,1,0],[1,3,0]],np.dot(math.sqrt(35/4),[1,-1])]
        
        elif (m==-3):
            #Stone's definition : 
            #Polynom sqrt(35/8) * [3*x^2*y*z-y^3*z]
            return [2,[[2,1,1],[0,3,1]],np.dot(math.sqrt(35/8),[3,-1])]
        
        elif (m==-2):
            #Stone's definition : 
            #Polynom sqrt(5/4) * [6*x*y*z^2-x^3*y-x*y^3]=sqrt(5/4) * [x*y*(7*z^2-r^2)]
            return [3,[[1,1,2],[3,1,0],[1,3,0]],np.dot(math.sqrt(5/4),[6,-1,-1])]
        
        elif (m==-1):
            ##ERROR corrected (Robert) 05/08/20
            #Stone's definition : polynom sqrt(5/8) * [4*y*z^3-3*x^2*y*z-3*y^3*z]= sqrt(5/8) * [7*y*z^3-3*y*z*r^2]
            return [3,[[0,1,3],[2,1,1],[0,3,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
        
        elif (m==0):
            #Stone's definition :
            #Polynom (1/8) * [8*z^4+3*x^4+3*y^4+6*x^2*y^2-24*x^2*z^2-24*y^2*z^2]=(1/8) * [35*z^4-30*z^2*r^2+3*r^4]
            return [6,[[0,0,4],[4,0,0],[0,4,0],[2,2,0],[2,0,2],[0,2,2]],np.dot(1./8.,[8,3,3,6,-24,-24])]
        
        elif (m==1):
            #Stone's definition :
            #Polynom sqrt(5/8) * [4*x*z^3-3*x*y^2*z-3*x^3*z]= sqrt(5/8) * [7*x*z^3-3*x*z*r^2]
            return [3,[[1,0,3],[1,2,1],[3,0,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
      
        elif (m==2):
            #Stone's definition :
            ##ERROR corrected (Robert) 05/08/20 (y^4 polynom represented with [0,0,4] instead of [0,4,0] !)
            #Polynom sqrt(5/16) * [-x^4+y^4+6*x^2*z^2-6*y^2*z^2]= sqrt(5/16) * [(x^2-y^2)*(7*z^2-r^2)] where r^2=x^2+y^2+z^2
            return [4,[[4,0,0],[0,4,0],[2,0,2],[0,2,2]],np.dot(math.sqrt(5/16),[-1,1,6,-6])]
        
        elif (m==3):
            #Stone's definition :
            #Polynom sqrt(35/8) * [z*(x^3-3*x*y^2)]
            return [2,[[3,0,1],[1,2,1]],np.dot(math.sqrt(35/8),[1,-3])]
        
        elif (m==4):
            #Stone's definition :
            #Polynom sqrt(35/64) * [x^4-6*x^2*y^2+y^4]
            return [3,[[4,0,0],[2,2,0],[0,4,0]],np.dot(math.sqrt(35/64),[1,-6,1])]
     
##############################################################################



##############################################################################
#Encodes the definition of a pure spherical harmonic r^l * Y_l^m(\vec{r}/|\vec{r}|
#EXCEPT when l==2 (d-type function) : encodes directly cartesian type functions 
# x² / y² / z² / x*y / x*z / y*z
#EXCEPT when l==3 (f-type function) : encodes directly cartesian type functions :
# XXX, YYY, ZZZ; XXY, XXZ, YYX, YYZ, ZZX, ZZY, XYZ
def conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells(l,m):
    
    if (l==0):
        #Polynom 1/sqrt(4*pi)
        #return [1,[[0,0,0]],[1/math.sqrt(4*math.pi)]]
        return [1,[[0,0,0]],[1]]
    
    if (l==1):
        #Spherical harmonics Y_1^m ; where m \in {-1,0,1}
        #have normalizing factor \sqrt(3/(4*pi))
        if (m==-1):
            #Polynom \sqrt(3/(4*pi)) * y 
            #return [1,[[0,1,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : y
            return [1,[[0,1,0]],[1]]
        
        elif (m==0):
            #Polynom \sqrt(3/(4*pi)) * z  
            #return [1,[[0,0,1]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : z
            return [1,[[0,0,1]],[1]]
        
        elif (m==1):
            #Polynom \sqrt(3/(4*pi)) * x 
            #return [1,[[1,0,0]],[math.sqrt(3/(4*math.pi))]]
            #Stone definition : x
            return [1,[[1,0,0]],[1]]
    
    ###################################
    #Case of d-type shells : in Gaussian  : no need to convert from spherical to cartesian harmonics
    #as cartesian functions are used for the d-type shells  (XX, YY, ZZ, XY, XZ, YZ) i.e. 
    #are = only one homogeneous polynom of degree 2
    #(and transformed internally by Gaussian into spherical harmonics ??)
    #############################
    #Pure conventions :
    #(l=2,m=100) : XX ; (l=2,m=101) : YY ; (l=2,m=102) : ZZ ; 
    #(l=2,m=103) : XY ; (l=2,m=104) : XZ ; (l=2,m=105) : YZ
    
    ##NORMALIZATION factors ??
    
    if (l==2):
        
        if(m==100):
            #Polynom x*x
            return [1,[[2,0,0]],[1]]
        
        elif (m==101):
            #Polynom y*y
            return [1,[[0,2,0]],[1]]
        
        elif (m==102):
            #Polynom z*z
            return [1,[[0,0,2]],[1]]  
        
        elif (m==103):
            #Polynom x*y
            return [1,[[1,1,0]],[1]]
        
        elif (m==104):
            #Polynom x*z
            return [1,[[1,0,1]],[1]]
        
        elif (m==105):
            #Polynom y*z
            return [1,[[0,1,1]],[1]]
        
    ####################
    ##l=3 : cartesian form
    if (l==3):
        
        if (m==300):
            #Polynom x*x*x
            return [1,[[3,0,0]],[1]]
                
        elif (m==301):
            #Polynom y*y*y
            return [1,[[0,3,0]],[1]]
        
        elif (m==302):
            #Polynom z*z*z
            return [1,[[0,0,3]],[1]]  
        
        elif (m==303):
            #Polynom x*x*y
            return [1,[[2,1,0]],[1]]
        
        elif (m==304):
            #Polynom x*x*z
            return [1,[[2,0,1]],[1]]
        
        elif (m==305):
            #Polynom y*y*x
            return [1,[[1,2,0]],[1]]
        
        elif (m==306):
            #Polynom y*y*z
            return [1,[[0,2,1]],[1]]
        
        elif (m==307):
            #Polynom z*z*x
            return [1,[[1,0,2]],[1]]        
        
        elif (m==308):
            #Polynom z*z*y
            return [1,[[0,1,2]],[1]]         

        elif (m==309):
            #Polynom x*y*z
            return [1,[[1,1,1]],[1]]  
        
    ##l=4 : check (taken from Stone GDMA manual)
    if (l==4):
        if (m==-4):
            #Polynom sqrt(35/4) * x^3*y-x*y^3
            return [2,[[3,1,0],[1,3,0]],np.dot(math.sqrt(35/4),[1,-1])]
        elif (m==-3):
            #Polynom sqrt(35/8) * 3*x^2*y*z-y^3*z
            return [2,[[2,1,1],[0,3,1]],np.dot(math.sqrt(35/8),[3,-1])]
        
        elif (m==-2):
            #Polynom sqrt(5/4) * [6*x*y*z^2-x^3*y-x*y^3]=sqrt(5/4) * [x*y*(7*z^2-r^2)]
            return [3,[[1,1,2],[3,1,0],[1,3,0]],np.dot(math.sqrt(5/4),[6,-1,-1])]
        
        elif (m==-1):
            ##ERROR corrected (Robert) 05/08/20
            #Polynom sqrt(5/8) * [4*y*z^3-3*x^2*y*z-3*y^3*z]= sqrt(5/8) * [7*y*z^3-3*y*z*r^2]
            return [3,[[0,1,3],[2,1,1],[0,3,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
        
        elif (m==0):
            #Polynom (1/8) * [8*z^4+3*x^4+3*y^4+6*x^2*y^2-24*x^2*z^2-24*y^2*z^2]=(1/8) * [35*z^4-30*z^2*r^2+3*r^4]
            return [6,[[0,0,4],[4,0,0],[0,4,0],[2,2,0],[2,0,2],[0,2,2]],np.dot(1./8.,[8,3,3,6,-24,-24])]
        
        elif (m==1):
            #Polynom sqrt(5/8) * [4*x*z^3-3*x*y^2*z-3*x^3*z]= sqrt(5/8) * [7*x*z^3-3*x*z*r^2]
            return [3,[[1,0,3],[1,2,1],[3,0,1]],np.dot(math.sqrt(5/8),[4,-3,-3])]
      
        elif (m==2):
            ##ERROR corrected (Robert) 05/08/20 (y^4 polynom represented with [0,0,4] instead of [0,4,0] !)
            #Polynom sqrt(5/16) * [-x^4+y^4+6*x^2*z^2-6*y^2*z^2]= sqrt(5/16) * [(x^2-y^2)*(7*z^2-r^2)] where r^2=x^2+y^2+z^2
            return [4,[[4,0,0],[0,4,0],[2,0,2],[0,2,2]],np.dot(math.sqrt(5/16),[-1,1,6,-6])]
               
        elif (m==3):
            #Polynom sqrt(35/8) * [z*(x^3-3*x*y^2)]
            return [2,[[3,0,1],[1,2,1]],np.dot(math.sqrt(35/8),[1,-3])]
        
        elif (m==4):
            #Polynom sqrt(35/64) * [x^4-6*x^2*y^2+y^4]
            return [3,[[4,0,0],[2,2,0],[0,4,0]],np.dot(math.sqrt(35/64),[1,-6,1])]
        
##############################################################################



##############################################################################
#Computes |r-S|^l * Y_l^m(r-S/|r-S|)  [REAL complex solid harmonic]
#vec_r=[x,y,z] the point at which we will evaluate the potential
#in atomic units (Bohrs**l)
def real_solid_harmonics_real_multipoles(vec_r,l,m,x_S,y_S,z_S):
    
    tab_position_center_S=[x_S,y_S,z_S]

    value_solid_harmonic=0
        
   
    #\vec{a}=\vec{r}-\vec{S} in Angstroms
    vector_a=[(vec_r[j]-tab_position_center_S[j]) for j in range(0,3)]
    
    

    #(m-q) can be > (in absolute value) to (l-k) (\in [0,l]), cf. redistribution formula
    #==> by convention Y_{l-k}^{m-q} = 0 in that case
    #[spherical harmonic not defined in that case]
    if (abs(m)<=l):
        
        #'Solid harmonic' |a|^(l) * Y_{l}^{m}(\vec{a}/|\vec{a})
        #=sum of homogeneous polynoms of degree (l) real_solid_harmonics_real_multipolesin a_x, a_y, a_z
        coeff_harmonics_cartesian=conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l,m)
 
        number_homogeneous_polynoms=coeff_harmonics_cartesian[0]
        tab_indexes_polynoms=coeff_harmonics_cartesian[1]
        weights_spherical_harmonics_to_cartesian_alpha=coeff_harmonics_cartesian[2]

        for i in range(number_homogeneous_polynoms):
        
            # += w_i(l,m) * [a_x**n_i(l,m)*a_y**m_i(l,m)*a_z**t_i(l,m)]
            value_solid_harmonic+=weights_spherical_harmonics_to_cartesian_alpha[i]*(vector_a[0]**tab_indexes_polynoms[i][0])*(vector_a[1]**tab_indexes_polynoms[i][1])*(vector_a[2]**tab_indexes_polynoms[i][2])
                   
        #Conversion from Ang**(l) to Bohr**l  (atomic units)
        return ((1/conversion_bohr_angstrom)**l)*value_solid_harmonic
        
    else:
        #Case |m-q| > |l-k|
        return 0
##############################################################################


#########################################################################################################
#Conversion of quadrupole moments from spherical form into cartesian quadrupolar tensor :
#The input variable 'local_quadrupole_tensor_spherical_form' will be 
#typically taken as 'result_multipoles[0][2]'={(Q_(2,-2))_{R_k},(Q_(2,-1))_{R_k},(Q_(2,0))_{R_k},(Q_(2,1))_{R_k},(Q_(2,2))_{R_k}}
#i.e. the output (multipole moments in spherical form) of the DMA analysis (at order 2) for the point R_k
def conversion_quadrupole_tensor_spherical_to_cartesian(local_quadrupole_tensor_spherical_form):
    
    """
    matrix_cartesian_to_spherical_multipoles=[[0,-1,0,0,0,0],
                                         [0,0,0,0,1,0],
                                         [-1/(2*math.sqrt(3)),0,0,-1/(2*math.sqrt(3)),0,1/math.sqrt(3)],
                                         [0,0,1,0,0,0],
                                         [1,0,0,-1,0,0]]

    """
    ###################
    #Definition compatible with Stone's definition :
    matrix_cartesian_to_spherical_multipoles=[[0,math.sqrt(3),0,0,0,0],
                                              [0,0,0,0,math.sqrt(3),0],
                                              [-0.5,0,0,-0.5,0,1],
                                              [0,0,math.sqrt(3),0,0,0],
                                              [math.sqrt(3/4.),0,0,-math.sqrt(3/4.),0,0]]
    ##Remark :
    ##This matrix (with all coefficients multiplied by 3/2=1.5)
    #is the (pseudo-)inverse of the matrix given by Stone :
    #M=[[0,0,-1/2,0,sqrt(3)/2],
    #  [sqrt(3)/2,0,0,0,0],
    #  [0,0,0,sqrt(3)/2,0],
    #  [0,0,-1/2,0,-sqrt(3)/2],
    #  [0,sqrt(3)/2,0,0,0],
    #  [0,0,1,0,0]]
    #such that [Q_xx,Q_xy,Q_xz,Q_yy,Q_yz,Q_zz]=M*[Q_22s,Q21s,Q20,Q21c,Q22C]
    
    #Facteur math.sqrt(4*math.pi/15.) ??
    Q_xx,Q_xy,Q_xz,Q_yy,Q_yz,Q_zz=np.dot(np.linalg.pinv(matrix_cartesian_to_spherical_multipoles),local_quadrupole_tensor_spherical_form)

    quadrupolar_tensor=[[Q_xx,Q_xy,Q_xz],
                        [Q_xy,Q_yy,Q_yz],
                        [Q_xz,Q_yz,Q_zz]]
    
    return quadrupolar_tensor
    ##Traceless quadrupole ? (make it traceless ? oranother function ?)
#########################################################################################################    

#########################################################################################################
#Inverse transformation of 'conversion_quadrupole_tensor_spherical_to_cartesian()'
#Components of input 'local_quadrupole_tensor_cartesian_form' provided in atomic units (e*Borh^2)
def conversion_quadrupole_tensor_cartesian_to_spherical(local_quadrupole_tensor_cartesian_form):

    Q_xx=local_quadrupole_tensor_cartesian_form[0][0]
    Q_xy=local_quadrupole_tensor_cartesian_form[0][1]
    Q_xz=local_quadrupole_tensor_cartesian_form[0][2]
    Q_yy=local_quadrupole_tensor_cartesian_form[1][1]
    Q_yz=local_quadrupole_tensor_cartesian_form[1][2]
    Q_zz=local_quadrupole_tensor_cartesian_form[2][2]
    
    vec_cartesian_components_quadrupole=[Q_xx,Q_xy,Q_xz,Q_yy,Q_yz,Q_zz]
    
    """
    matrix_cartesian_to_spherical_multipoles=[[0,-1,0,0,0,0],
                                         [0,0,0,0,1,0],
                                         [-1/(2*math.sqrt(3)),0,0,-1/(2*math.sqrt(3)),0,1/math.sqrt(3)],
                                         [0,0,1,0,0,0],
                                         [1,0,0,-1,0,0]]
    """
    
    
    """
    Correct matrix with good normalizing constants ??
    Matrix not consistent with Stone's manual... 
    """
    
    #With proportionality factors consistent with the definition of the spherical harmonics 
    #(and with their (normalizing?) factors, taken as in Stone's GDMA manual)
    matrix_cartesian_to_spherical_multipoles=[[0,-math.sqrt(3),0,0,0,0],
                                         [0,0,0,0,math.sqrt(3),0],
                                         [-0.5,0,0,-0.5,0,1.],
                                         [0,0,math.sqrt(3),0,0,0],
                                         [math.sqrt(3/4.),0,0,-math.sqrt(3/4.),0,0]]
    
    #constant=math.sqrt(15./4*math.pi)
    constant=1
    
    #Results in atomic units (e*Borh^2)
    return constant*np.dot(matrix_cartesian_to_spherical_multipoles,vec_cartesian_components_quadrupole)
#########################################################################################################


################################################################################################
#Writes the results of the Distributed Multipole Analysis in an output file
#as well as the redistribution "philosophy" (number of final DMA expansion sites) 
#and "strategy" (redistribution weights)
#Input arguments :
# - 'DMA_output_files' : name of the directory to store the DMA output files
# - 'filename_DMA_output_values' : filename chosen for the DMA output file
# - 'result_multipoles' : resul of the applying the function "compute_DMA_multipole_moment_final_expansion_center_all_sites_all_orders.()"
# - 'positions_final_expansion_sites' : positions of the final DMA expansion sites chosen for the DMA 
# -  'atomic_sites_to_omit' : atoms to omit as final DMA sites in the expansion 
# - 'index_philosophy' : philosophy of the DMA expansion (which final sites are retained for the expansion)
# - 'l_max_user' = tables of the angular momentum order for local multipole moments at each site
# - 'QM_data' : QM ouput (input to this DMA code) parsed by cclib
#########
#Rq : total dipole and quadrupole moments : computed previously as :
#total_dipole_moment=computes_total_dipole_from_local_multipoles_DMA_sites(0,0,0,result_multipoles,positions_final_expansion_sites)
#(in cartesian form : d_x, d_y, d_z components with d_x <--> Q(1,1), d_y <--> Q(1,-1), d_z <--> Q(1,0))
#total_quadrupolar_tensor_cartesian=computes_total_quadrupole_from_local_multipoles_DMA_sites(0.,0.,0.,result_multipoles,positions_final_expansion_sites)
#Identity=np.diag([1,1,1],0)
#trace_quadrupolar_tensor_cartesian_traceless=sum([total_quadrupolar_tensor_cartesian_Debye_Ang[i][i] for i in range(3)])
#total_quadrupolar_tensor_spherical_traceless=total_quadrupolar_tensor_cartesian_Debye_Ang-np.dot((1/3.)*trace_quadrupolar_tensor_cartesian_traceless,Identity)
#########
def writes_DMA_output_values(file_DMA_output_values,result_multipoles,total_dipole_moment,total_quadrupolar_tensor_spherical_traceless,total_quadrupolar_tensor_spherical,positions_final_expansion_sites,atomic_sites_to_omit,index_philosophy,l_max_user,QM_data,atomic_coordinates,filename_QM_output):
    
    #As first line, we write the total number of grid points :
    file_DMA_output_values.write('----------------------------------------')
    file_DMA_output_values.write('\n')
    file_DMA_output_values.write('DMA Analysis (output)')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("Robert Benda, Eric Cancès 2020")
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write('----------------------------------------')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("Input file (output of QM calculation) = "+str(filename_QM_output))
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    ###################################
    #Writing the values of the local DMA multipole moments in the output file :
       
        
    #######################################################################
    #PRINTING OF THE RESULTS of LOCAL multipole moments
    
    for i in range(len(result_multipoles)):
        
        x_S=positions_final_expansion_sites[i][0]
        y_S=positions_final_expansion_sites[i][1]
        z_S=positions_final_expansion_sites[i][2]
        
        test_center_S_atom=boolean_final_expansion_site_is_atom(x_S,y_S,z_S,atomic_coordinates)
        
        #################
        #Case of a redistribution center (final DMA site)
        #which is an atom
        if (test_center_S_atom[0]==True):
    
            ##print('SITE = ATOM  '+' n° '+str(i)+' (mass '+str(QM_data.atommasses[test_center_S_atom[1]])+'), at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
            file_DMA_output_values.write('SITE = ATOM  '+str(periodic_table[QM_data.atomnos[i]])+' n° '+str(i)+', at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
            file_DMA_output_values.write('           Maximum rank =  '+str(l_max_user[i]))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
            
        #Case of a redistribution center (final DMA site)
        #which is NOT an atom        
        else:
            file_DMA_output_values.write('SITE '+' n° '+str(i)+'), at POSITION : ['+str(x_S)+','+str(y_S)+','+str(z_S)+']')
            file_DMA_output_values.write('           Maximum rank =  '+str(l_max_user[i]))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
        #################
        
        #################
        #Printing the local distributed multipoles at this DMA final site :
            
        #Stone : "                   Q00  =  -0.788156"
        file_DMA_output_values.write('Q00 =   '+str(int(10**6 *result_multipoles[i][0][0])/10**6))
        file_DMA_output_values.write("\n")
        file_DMA_output_values.write("\n")
        if (l_max_user[i]>=1):
            file_DMA_output_values.write('|Q1| =   '+str(int(10**6 * np.linalg.norm(result_multipoles[i][1]))/10**6))
            #file_DMA_output_values.write("\n")
            file_DMA_output_values.write('  Q10 =  '+str(int(10**6 *result_multipoles[i][1][1])/10**6)+'  '+'Q11c =  '+str(int(10**6 *result_multipoles[i][1][2])/10**6)+'  Q11s =  '+str(int(10**6 *result_multipoles[i][1][0])/10**6))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
        if (l_max_user[i]>=2):
            file_DMA_output_values.write('|Q2| =   '+str(int(10**6 *np.linalg.norm(result_multipoles[i][2]))/10**6))
            #file_DMA_output_values.write("\n")
            file_DMA_output_values.write('  Q20 =  '+str(int(10**6 *result_multipoles[i][2][2])/10**6)+'  '+'Q21c =  '+str(int(10**6 *result_multipoles[i][2][3])/10**6)+'  Q21s =  '+str(int(10**6 *result_multipoles[i][2][1])/10**6)+'  Q22c =  '+str(int(10**6 *result_multipoles[i][2][4])/10**6)+'  Q22s =  '+str(int(10**6 *result_multipoles[i][2][0])/10**6))
            file_DMA_output_values.write("\n")
            ##Conversion of the quadrupolar moments from spherical to cartesian tensor :
            local_quadrupole_cartesian_form=conversion_quadrupole_tensor_spherical_to_cartesian(result_multipoles[i][2])
            file_DMA_output_values.write('Q_xx =  '+str(int(10**6 *local_quadrupole_cartesian_form[0][0])/10**6)+', Q_yy =  '+str(int(10**6 *local_quadrupole_cartesian_form[1][1])/10**6)+', Q_zz =  '+str(int(10**6 *local_quadrupole_cartesian_form[2][2])/10**6)+', Q_xy =  '+str(int(10**6 *local_quadrupole_cartesian_form[0][1])/10**6)+', Q_xz =  '+str(int(10**6 *local_quadrupole_cartesian_form[0][2])/10**6)+', Q_yz =  '+str(int(10**6 *local_quadrupole_cartesian_form[1][2])/10**6))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
        if (l_max_user[i]>=3):
            file_DMA_output_values.write('|Q3| =   '+str(int(10**6 *np.linalg.norm(result_multipoles[i][3]))/10**6))
            #file_DMA_output_values.write("\n")
            file_DMA_output_values.write('  Q30 =  '+str(int(10**6 *result_multipoles[i][3][3])/10**6)+'  '+'Q31c =  '+str(int(10**6 *result_multipoles[i][3][4])/10**6)+'  Q31s =  '+str(int(10**6 *result_multipoles[i][3][2])/10**6)+'  Q32c =  '+str(int(10**6 *result_multipoles[i][3][5])/10**6)+'  Q32s =  '+str(int(10**6 *result_multipoles[i][3][1])/10**6)+'  Q33c =  '+str(int(10**6 *result_multipoles[i][3][6])/10**6)+'  Q33s =  '+str(int(10**6 *result_multipoles[i][3][0])/10**6))
            ##Conversion of the octopolar moments from spherical to cartesian tensor :
            #Formula ??

            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
                
        if (l_max_user[i]>=4):
            file_DMA_output_values.write('|Q4| =   '+str(int(10**6 *np.linalg.norm(result_multipoles[i][4]))/10**6))
            #file_DMA_output_values.write("\n")
            file_DMA_output_values.write('  Q40 =  '+str(int(10**6 *result_multipoles[i][4][4])/10**6)+'  '+'Q41c =  '+str(int(10**6 *result_multipoles[i][4][5])/10**6)+'  Q41s =  '+str(int(10**6 *result_multipoles[i][4][3])/10**6)+'  Q42c =  '+str(int(10**6 *result_multipoles[i][4][6])/10**6)+'  Q42s =  '+str(int(10**6 *result_multipoles[i][4][2])/10**6)+'  Q43c =  '+str(int(10**6 *result_multipoles[i][4][7])/10**6)+'  Q43s =  '+str(int(10**6 *result_multipoles[i][4][1])/10**6)+'  Q44c =  '+str(int(10**6 *result_multipoles[i][4][8])/10**6)+'  Q44s =  '+str(int(10**6 *result_multipoles[i][4][0]/10**6)))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
             
        if (l_max_user[i]>=5):
            file_DMA_output_values.write('|Q5| =   '+str(int(10**6 *np.linalg.norm(result_multipoles[i][5]))/10**6))
            file_DMA_output_values.write('  Q50 =  '+str(int(10**6 *result_multipoles[i][5][5])/10**6)+'  '+'Q51c =  '+str(int(10**6 *result_multipoles[i][5][6])/10**6)+'  Q51s =  '+str(int(10**6 *result_multipoles[i][5][4])/10**6)+'  Q52c =  '+str(int(10**6 *result_multipoles[i][5][7])/10**6)+'  Q52s =  '+str(int(10**6 *result_multipoles[i][5][3])/10**6)+'  Q53c =  '+str(int(10**6 *result_multipoles[i][5][8])/10**6)+'  Q53s =  '+str(int(10**6 *result_multipoles[i][5][2])/10**6)+'  Q54c =  '+str(int(10**6 *result_multipoles[i][5][9])/10**6)+'  Q54s =  '+str(int(10**6 *result_multipoles[i][5][1]/10**6))+'  Q55c =  '+str(int(10**6 *result_multipoles[i][5][10]/10**6))+'  Q55s =  '+str(int(10**6 *result_multipoles[i][5][0]/10**6)))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
            
        if (l_max_user[i]>=6):
            file_DMA_output_values.write('|Q6| =   '+str(int(10**6 *np.linalg.norm(result_multipoles[i][6]))/10**6))
            file_DMA_output_values.write('  Q60 =  '+str(int(10**6 *result_multipoles[i][6][6])/10**6)+'  '+'Q61c =  '+str(int(10**6 *result_multipoles[i][6][7])/10**6)+'  Q61s =  '+str(int(10**6 *result_multipoles[i][6][5])/10**6)+'  Q62c =  '+str(int(10**6 *result_multipoles[i][6][8])/10**6)+'  Q62s =  '+str(int(10**6 *result_multipoles[i][6][4])/10**6)+'  Q63c =  '+str(int(10**6 *result_multipoles[i][6][9])/10**6)+'  Q63s =  '+str(int(10**6 *result_multipoles[i][6][3])/10**6)+'  Q64c =  '+str(int(10**6 *result_multipoles[i][6][10])/10**6)+'  Q64s =  '+str(int(10**6 *result_multipoles[i][6][2]/10**6))+'  Q65c =  '+str(int(10**6 *result_multipoles[i][6][11]/10**6))+'  Q65s =  '+str(int(10**6 *result_multipoles[i][6][1]/10**6))+'  Q66c =  '+str(int(10**6 *result_multipoles[i][6][12]/10**6))+'  Q66s =  '+str(int(10**6 *result_multipoles[i][6][0]/10**6)))
            file_DMA_output_values.write("\n")
            file_DMA_output_values.write("\n")
                
        file_DMA_output_values.write("\n")
    
    ####################################################################### 
        
    file_DMA_output_values.write('----------------------------------------------------')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write('TOTAL MOLECULAR MOMENTS (FROM REDISTRIBUTED LOCAL MULTIPOLES): ')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    
    nb_DMA_final_sites=len(result_multipoles)
    
    ############################################
    #Checking total sum of monopoles (= total charge) :
    ##In the case of omitting nuclei ==> add to the final total monopole the sum 
    ##of (positive) charge (Z index) of omitted nuclei   
    total_charge=0
    
    if (index_philosophy==4):
        total_charge=np.sum([result_multipoles[i][0] for i in range(nb_DMA_final_sites)])
        for k in atomic_sites_to_omit:
            total_charge+=QM_data.atomnos[k]

    else:   
        total_charge=np.sum([result_multipoles[i][0] for i in range(nb_DMA_final_sites)])

    
    #######################################################################
    #Printing the results of TOTAL multipole moments
    file_DMA_output_values.write('Total charge : '+str(int(10**6 *total_charge)/10**6))
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    
    file_DMA_output_values.write('TOTAL DIPOLE MOMENT COMPUTED FROM REDISTRIBUTED LOCAL MULTIPOLES (a.u.): ')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write('|Q1| =   '+str(int(10**6 * np.linalg.norm(total_dipole_moment))/10**6))
    file_DMA_output_values.write('  Q10 =  '+str(int(10**6 *total_dipole_moment[2])/10**6)+'  '+'Q11c =  '+str(int(10**6 *total_dipole_moment[0])/10**6)+'  Q11s =  '+str(int(10**6 *total_dipole_moment[1])/10**6))
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    
    file_DMA_output_values.write('TOTAL (TRACELESS) QUADRUPOLE MOMENT COMPUTED FROM REDISTRIBUTED LOCAL MULTIPOLES (in Debye.Ang=Buckinghams) :')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write('Q_xx =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[0][0])/10**6)+', Q_yy =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[1][1])/10**6)+', Q_zz =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[2][2])/10**6)+', Q_xy =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[0][1])/10**6)+', Q_xz =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[0][2])/10**6)+', Q_yz =  '+str(int(10**6 *total_quadrupolar_tensor_spherical_traceless[1][2])/10**6))
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
     
    #Computed as :
    ##total_quadrupolar_tensor_spherical=conversion_quadrupole_tensor_cartesian_to_spherical(total_quadrupolar_tensor_cartesian)
       
    file_DMA_output_values.write('Total spherical quadrupolar tensor (a.u. e.Bohr²) -- for comparison with Stone GDMA :')
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write('|Q2| =   '+str(int(10**6 *np.linalg.norm(total_quadrupolar_tensor_spherical))/10**6))
    file_DMA_output_values.write('  Q20 =  '+str(int(10**6 *total_quadrupolar_tensor_spherical[2])/10**6)+'  '+'Q21c =  '+str(int(10**6 *total_quadrupolar_tensor_spherical[3])/10**6)+'  Q21s =  '+str(int(10**6 *total_quadrupolar_tensor_spherical[1])/10**6)+'  Q22c =  '+str(int(10**6 *total_quadrupolar_tensor_spherical[4])/10**6)+'  Q22s =  '+str(int(10**6 *total_quadrupolar_tensor_spherical[0])/10**6))
    file_DMA_output_values.write("\n")
    file_DMA_output_values.write("\n")
    
################################################################################################

################################################################################################
#Reads a text file with the Lebedev points coordinates and weights and store them in an
#object readable by Python (list of arrays [[x_1,y_1,z_1,w_1],...[x_N,y_N,z_N,w_N]])
def reads_Lebedev_grid(Lebedev_grid_points_file,Lebedev_grid_points_dir):
    
    file_object_Lebedev_grid  = open(Lebedev_grid_points_dir+Lebedev_grid_points_file,"r")
        
    Lebedev_grid_file_readlines=file_object_Lebedev_grid.readlines()
    
    nb_grid_points=len(Lebedev_grid_file_readlines)-1

    #List of the list of the Lebedev N grid points (coordinates and weights for integration)
    #[[x_1,y_1,z_1,w_1],...[x_N,y_N,z_N,w_N]]
    Lebedev_grid_points=[]    
    
    #We skip the first line 
    #'#    Index      x_grid      y_grid   z_grid   w_grid (weight)'
    #BEWARE : there must not be a blank final line after the last Lebedev point in the Lebedev points file
    for i in range(1,nb_grid_points+1):
        
        line=Lebedev_grid_file_readlines[i].split(" ")

        line_cleaned=[]

        for k in range(len(line)):
            
            if ((line[k] != '') and (line[k] !='\n')):

                line_cleaned.append(float(line[k]))

        #We add the point [x_i,y_i,z_i,w_i] (coordinates and weight w_i for integration)
        #to the list of already encountered Lebedev points
        Lebedev_grid_points.append([line_cleaned[1],line_cleaned[2],line_cleaned[3],line_cleaned[4]])
                
    return Lebedev_grid_points
################################################################################################

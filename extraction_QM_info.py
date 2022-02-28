#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 12:08:08 2020

@author: rbenda

Auxiliary functions to :
    
    - read outputs of Gaussian 
    (as input for this code, to compute (DMA) multipole moments)
    
    - convert it in suitable form for DMA multipole calculations
    (e.g. conversion of the density matrix in terms of contracted GTOs
    to density matrix in terms of primitive GTOs 
    in the function 'compute_density_matrix_coefficient_pGTOs()'
    
    - compute normalization constants of cGTOs and pGTOs, and scalar products

Uses cclib library parsers or attributes.

So far : for g09 : for an input '.log' file to be parsed ==> 'pop=full' and 'iop(3/33=1)'
needed as Gaussian input keywords, so that 'mocoeffs' attribute available from the log file
(as well as 'nbasis' and 'homos' attributes, so that the density matrix can be computed by density.py
cclib code)

"""


import numpy as np
import time
import math
#import sys

import cclib

from auxiliary_functions import gaussian_polynom_integral_all_R
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells
from auxiliary_functions import conversion_spherical_harmonics_cartesian_homogeneous_polynoms
from auxiliary_functions import computes_coefficient_solid_harmonic_l_m_list_format

from auxiliary_functions import coeff_W_binomial_expansion_polynoms


############################
#CONSTANTS :
#1 Bohr in Angstroms (Gaussian QM output given in atomic units 
#i.e. bohr and Hartrees ==> e.g. 'zeta' gaussian exponents
#provided in Bohr^{-2}
conversion_bohr_angstrom=0.529177209
############################

################################################################################################
################################################################################################
#Functions to import data from output QM file :

############################################################################
#Input SCF density matrix e.g. from Gaussian output (using cclib functions)
#=> coefficients (D_{alpha,beta})_{\alpha,\beta cGTOs} of the whole charge density
#IN THE cGTOs basis (i.e. n(r) = \sum_{\alpha, \beta \in cGTOs} D_{alpha,beta})_{\alpha,\beta cGTOs} \chi_{\alpha}^{cGTO}*(r) * \chi_{\beta}^{cGTO}(r)
###################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
#This argument 'data' is modified by computing and adding the density matrix ('density' attribute)
#as available argument after calling to this function.
def import_density_matrix_contracted_GTOs(data,logfile_DMA):

    d = cclib.method.Density(data)
    
    #Function to calculate the density matrix from info parsed by cclib :
    ##d.calculate(fupdate=0.05)
    d.calculate()
    #After calling this function, the density matrix will be available as "d.density"
    #First axis = spin contributions
    #Second and third axes : for the density matrix
    
    
    logfile_DMA.write('Density matrix cGTOs :')
    logfile_DMA.write('\n')
    logfile_DMA.write('\n')
    if (len(d.density)==2):
        logfile_DMA.write('alpha AND beta density matrices (open-shell UHF or ROHF calculation)')
    elif (len(d.density)==1):
        logfile_DMA.write('alpha density matrix only (closed-shell calculation)')
    
    logfile_DMA.write('\n')   
    logfile_DMA.write('Size density matrix : ')
    logfile_DMA.write(str(len(d.density[0])))
    
    logfile_DMA.write('\n')
    logfile_DMA.write('\n')
    
    """
    #Probleme de matrice densité pour filename = "output_O2_PBE0_6-31Gd_y_axis.dat"
    #(Psi4 calculation)
    Comparer avec calcul direct à partir de data.mocoeffs[0] ??
    TO DO
    """
    
    #################################################################################
    #Alternative method : computing directly the density matrix from mocoeffs cclib attribute
    #(molecular orbitals coefficients) :
        
    nb_basis_functions=len(d.density[0])
    density_matrix=[np.zeros([nb_basis_functions,nb_basis_functions]),np.zeros([nb_basis_functions,nb_basis_functions])]
    

    #To find the number of OCCUPIED molecular orbitals : we count the total number
    #of electrons (= total charge of the nuclei FOR A NEUTRAL MOLECULE)
    # minus the charge of the molecule (positive in case of an ion, negative otherwise)
    total_nb_electrons=np.sum(data.atomnos)-data.charge

    #For a closed-shell system :
    nb_occupied_MO=int(total_nb_electrons/2)

    ##TODO CHECK : for an open-shell system : find and define the number of doubly occupied, and of singly occupied orbitals :
    # number of sinlgy occupied orbitals :
    Ns=total_nb_electrons-2*nb_occupied_MO
        
    #D_ij = 2* \sum_{k occ. MO} (c_(ki)*c_(kj))
    for i in range(nb_basis_functions):
        for j in range(nb_basis_functions):
            #There as as many occupied Molecular Orbitals as nb of basis functions
            #(contracted GTOs) : the SCF procedure consists in a simple rotation of orbitals

            #*** Sum on DOUBLY OCCUPIED molecular orbitals
            for k in range(nb_occupied_MO):
                #k^{th} line of data.mocoeffs[0] = coefficients of the M.O. phi_k
                #in the basis of contracted GTOs
                density_matrix[0][i][j]+=2*data.mocoeffs[0][k][i]*data.mocoeffs[0][k][j]

            #*** Sum on SINGLY OCCUPIED molecular orbitals (=> no factor 2)
            for k in range(nb_occupied_MO,nb_occupied_MO+Ns+1):     
                density_matrix[1][i][j]+=data.mocoeffs[0][k][i]*data.mocoeffs[0][k][j]
    
    
    #################################################################################
    

    
    #BEWARE : For an UHF calculation : alpha-density matrix / beta-density matrix [each one = matrix nb_contracted_GTOs*nb_contracted_GTOs]
    # (two density matrices, each one constructed from the set of alpha (respectively beta) MO coefficients)
    #=> returns the sum of the 2 matrices (i.e. the sum of alpha+beta density matrices)
    
    #print('Size density matrix : ')
    #print(len(d.density[0]))
    #print(len(density_matrix[0]))

    return d.density
    #return density_matrix
##############################################################
    

# - contraction coefficients (used to go from PGTOs [Primitive Gaussian Type Orbitals] to CGTOs [Contracted Gaussian Type Orbitals])


##############################################################################
#"pop=full" input keyword necessary in Gaussian
#Function called in case of an UHF calculation only (separation between alpha and beta MO coefficients)
def import_alpha_MO_coefficients(data):
    
    return data.mocoeffs[0]
##############################################################################
    
##############################################################################
#"pop=full" input keyword necessary in Gaussian
#Function called in case of an UHF calculation only (separation between alpha and beta MO coefficients)
def import_beta_MO_coefficients(data):
    
    return data.mocoeffs[1]
##############################################################################

    
##############################################################################
#"pop=full" input keyword necessary in Gaussian
#Function called in case of an ROHF calculation only (only one set of MO coefficients)
def import_MO_coefficients(data):
    
    return data.mocoeffs[0]
##############################################################################
    

##############################################################################
#Returns the number of contracted GTOs basis functions (cclib attribute)
def import_number_contracted_GTOs(data):
    
    return data.nbasis
##############################################################################
    
##############################################################################
#Returns the number of Molecular Orbitals (cclib attribute)
#nb_MOs always equal to nb_contracted_GTOs (i.e. 'nb_basis_functions') ??
def import_number_MOs(data):
    
    return data.nmo
##############################################################################
    
    
#################################################################################
#Import information on the underlying QM basis used (using cclib functions) : 
#(case of SCF density matrix obtained from DFT, using localized Gaussian Type Orbitals)
#Extracts gaussian exponents {zeta_{alpha}}_{alpha} of PRIMITIVE SHELLS
#(one primitive SHELL of angular momentum l = sum of 
#(2l+1) primitive GTOs [sherical functions of indexes (l,m)] all sharing the same Gaussian exponent)
#Uses 'gbasis' attribute of cclib 
#[i.e. same result as using "ccget gbasis (..).log" 
#(BEWARE : 'SP' contracted shells are separated into their 'S' and 'P' components in this output)
#[can also be read from 'Primitive exponents' section in (..).fchk file]
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
##############
#'gbasis' attribute => for Gaussian (g09) : 'gfinput' must have been 
#used as input keyword so that this attribute is available
def import_basis_set_exponents_primitive_shells(data):
    
    gbasis_output_cclib=data.gbasis
    
    basis_set_exponents_primitive_shells=[]

    #Loop on the number of atoms
    for k in range(len(gbasis_output_cclib)):
        #Loop on the number of contracted shells per atom
        for l in range(len(gbasis_output_cclib[k])):
            #Loop on all the primitive shells constituting this contracted shell
            #(loop on both 'S' and 'P' in the case of a 'SP' contracted shell)
            for h in range(len(gbasis_output_cclib[k][l][1])):
                
                basis_set_exponents_primitive_shells.append(gbasis_output_cclib[k][l][1][h][0])
    
    return basis_set_exponents_primitive_shells
############################################################################## 
    
##############################################################################
#List of basis set exponents of primitive GTOs (in the order of increasing atom index)
##BEWARE : must be consisten with the order of affection of angular momentum number '(l_alpha,m_alpha)' of pGTOs
##done by the functions 'computes_angular_momentum_numbers_primitive_GTOs()'
##and with the function 'compute_correspondence_index_contracted_GTOs_primitive_GTOs()'
##giving the correspondence between contracted GTOs (indexes) and primitive GTOs (indexes)
##(and incidentally with the fresults of the function 'compute_correspondence_index_primitive_GTOs_contracted_GTOs()'
##which is deduced from the function 'compute_correspondence_index_contracted_GTOs_primitive_GTOs()'
def import_basis_set_exponents_primitive_GTOs(data,boolean_cartesian_D_shells,boolean_cartesian_F_shells):
    
    gbasis_output_cclib=data.gbasis
    
    basis_set_exponents_primitive_GTOs=[]

    #Loop on the number of atoms
    for k in range(len(gbasis_output_cclib)):
        #Loop on the number of contracted shells per atom
        for l in range(len(gbasis_output_cclib[k])):
            
            if (gbasis_output_cclib[k][l][0]=='S'):
                
                #Loop on all the primitive shells constituting this contracted shell
                #(loop on both 'S' and 'P' in the case of a 'SP' contracted shell)
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    #1 S primitive shell = 1 S primitive GTO 
                    basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                
            elif (gbasis_output_cclib[k][l][0]=='P'):
                
                #in 'gbasis' info, 'SP' shells are splitted into 'S' and 'P' shells
                #1 P primitive shell = full shell = 2*1+1 = 3 primitive GTOs : (l=1,m=-1..1)
                #of same zeta exponents
                    
                #Loop on all the primitive shells (of different zeta exponents) constituting this contracted shell
                #PX, PY and PZ - type primitive GTOs (in increasing order)
                for i in range(3):
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                        
                       
            elif (gbasis_output_cclib[k][l][0]=='D'):
                #1 D primitive shell = full shell = 2*2+1 = 5 primitive spherical GTOs : (l=2,m=-2..2) => in Gaussian 6 cartesian GTOs
                #of same zeta exponents
                    
                if (boolean_cartesian_D_shells==True):

                    #Loop on all the primitive shells constituting this contracted shell
                    #XX, YY, ZZ, XY, XZ and YZ -type primitive GTOs 
                    #(for Gaussian / GAMESS cartesian d-shells)
                    
                    for i in range(6):
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                        
                  
                elif (boolean_cartesian_D_shells==False):
                    
                    #(l=2,m=0), (l=2,m=1), (l=2,m=-1), (l=2,m=2), (l=2,m=-2) type
                    #primitive GTOs are treated in increasing order 
                    for i in range(5):
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                   

            elif (gbasis_output_cclib[k][l][0]=='F'):
                
                #1 F primitive shell = full shell = 2*3+1 = 7 primitive GTOs 
                #in spherical form (l=2,m=-2..2) ; 10 primitive GTOs in cartesian form
                #==> of same zeta exponents.
                
                #Loop on all the primitive shells constituting this contracted shell

                #Case of cartesian representation (exple : GAMESS with keyword 'isphr=-1' 
                #in section $contrl ; Psi4 [default ?])
                if (boolean_cartesian_F_shells==True):
                    
                    ##GAMESS order for cartesian f-functions : 
                    ##XXX, YYY, ZZZ; XXY, XXZ, YYX, YYZ, ZZX, ZZY, XYZ f-type functions are 
                    #treated in increasing order.
                    ##Order of indexation =>> see 'aonames' cclib attribute.
                    
                    for i in range(10):
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                    

                #Case of spherical f-functions representation :
                elif (boolean_cartesian_F_shells==False):
                    
                    #'F 0','F+1','F-1','F+2','F-2' -type primitive GTOs  (for Gaussian output) 
                    for i in range(7):
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
                            
            elif (gbasis_output_cclib[k][l][0]=='G'):
                #Spherical G-functions :
                for i in range(9):
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
            
            elif (gbasis_output_cclib[k][l][0]=='H'):
                #Shperical H-functions :   
                for i in range(11):
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
            
            elif (gbasis_output_cclib[k][l][0]=='I'):
                #Shperical I-functions :   
                for i in range(13):
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        basis_set_exponents_primitive_GTOs.append(gbasis_output_cclib[k][l][1][h][0])
            
                
                   
    #The final list 'basis_set_exponents_primitive_GTOs' has to be of length 'nb_primitive_GTOs=computes_nb_primitive_GTOs()'
    return basis_set_exponents_primitive_GTOs
##############################################################################
    
    
############################################################################################################
#Associates to the 'linear' index alpha of a pGTO, its angular momentum number (l_alpha,m_alpha)
#==> list of lists [(l_0,m_0),(l_1,m_1),...] where (l_i,m_i) are the angular momentum quantum number associated to
#the i^{th} primitive (in the indexation order defined for instance by "import_basis_set_exponents_primitive_GTOs()")
#By convention : 
#- FOR Gaussian (g09, g19) : 
#If l_i =1 : the order (1,1), (1,-1), (1,0) is chosen cf. r Y_1^m(theta,phi) \varpropto x, y ,z when m=1,0,-1
#and order PX, PY, PZ in the indexation of cGTOs in Gaussian
#For Psi4 : 
def computes_angular_momentum_numbers_primitive_GTOs(data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,input_type,psi4_exception_P_GTOs_indexation_order,psi4_exception_D_GTOs_indexation_order):
    
    gbasis_output_cclib=data.gbasis
    
    ##Beware : 'angular_momentum_numbers_primitive_GTOs' has to  be filled in
    #correct order (corresponding to the order of indexation of the contracted GTOs,
    #the density matrix being written in the basis of contracted GTOs)
    angular_momentum_numbers_primitive_GTOs=[]
    
    #Loop on the number of atoms
    for k in range(len(gbasis_output_cclib)):
        #Loop on the number of contracted shells per atom
        for l in range(len(gbasis_output_cclib[k])):
            
            if (gbasis_output_cclib[k][l][0]=='S'):
                    
                #Loop on all the primitive shells constituting this contracted shell
                #(loop on both 'S' and 'P' for an 'SP' contracted shell)
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    #1 S primitive shell = 1 S primitive GTO 
                    angular_momentum_numbers_primitive_GTOs.append([0,0])
                
            elif (gbasis_output_cclib[k][l][0]=='P'):
                    
                #in 'gbasis' info, 'SP' shells are splitted into 'S' and 'P' shells
                #1 P primitive shell = full shell = 2*1+1 = 3 primitive GTOs : (l=1,m=-1..1)
                #of same zeta exponents
                    
                #We first treat PX, then PY, then PZ components (following the indexation order in 
                #Gaussian, cf. "ccget aonames (..).log : displays order of indexation of the A.O. of the basis)
                #Loop on all the primitive shells constituting this contracted shell
                
                #BEWARE ! (Psi4 exception)
                #*** In Psi4, when Spherical Harmonics =TRUE
                #(e.g. for (aug)-cc-pVXZ Dunning's type basis sets):
                #the indexation order for the primitive GTOs of each 
                #P primitive shell is PZ, PX, PY : ("p0, p+1, p-1")
                #######
                #When "Spherical Harmonics    = FALSE" (e.g. for 6-31G(d)
                #Pople's type basis sets) => the order is px, py, pz
                
                """
                Temporary fix ?
                """
                if ((input_type=='Psi4') and (psi4_exception_P_GTOs_indexation_order==True)):
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,0])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,1])
                            
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,-1])
                
                #Regular order px, py, pz (e.g. for Gaussian or GAMESS output) :
                else:
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,1])
                            
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,-1])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([1,0])
                    
            elif (gbasis_output_cclib[k][l][0]=='D'):
                
                #1 D primitive shell = full shell = 2*2+1 = 5 primitive spherical GTOs : (l=2,m=-2..2) of same zeta exponents
                
                if (boolean_cartesian_D_shells==True):
                    
                    #############
                    #In Gaussian / GAMESS and with basis sets such as 6-31G(d) :
                    #6 cartesian GTOs (of same zeta exponents) : XX, YY, ZZ, XY, XZ, YZ
                    #Pure conventions :
                    #(l=2,m=100) : XX ; (l=2,m=101) : YY ; (l=2,m=102) : ZZ ; (l=2,m=103) : XY ;
                    #(l=2,m=104) : XZ ; (l=2,m=105) : YZ
                    #treated in the order above
                    #Loop on all the primitive shells constituting this contracted shell                    
                    
                    #BEWARE ! (Psi4 exception)
                    #*** In Psi4, when Spherical Harmonics =FALSE
                    #(e.g. for 6-31G(d) Pople's type basis sets)
                    #the indexation order for the primitive GTOs of each 
                    #D primitive shell is dxx, dxy, dxz, dyy, dyz, dzz
                    #==> not the same indexation order as for Gaussian / GAMESS 
                    #in the cartesian D-shells case (XX, YY, ZZ, XY, XZ, YZ) !
                    #######
                    #When "Spherical Harmonics    = TRUE" 
                    #(e.g. for (aug)-cc-pVXZ Dunning's type basis sets)
                    #=> the order is d0, d+1, d-1, d+2, d-2 ==> OK
                    #(see case boolean_cartesian_D_shells==False)
                   
                    
                    """
                    Temporary fix ?
                    """
                    #############
                    if ((input_type=='Psi4') and (psi4_exception_D_GTOs_indexation_order==True)):
                        #Psi4 cartesian D-shells order in the case psi4_exception_P_GTOs_indexation_order=True :
                        #XX, XY, XZ, YY, YZ, ZZ :
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,100])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,103])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,104])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,101])
                           
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,105])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,102])
                        
                    else:
                        #XX, YY, ZZ, XY, XZ, YZ
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,100])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,101])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,102])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,103])
                           
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,104])
                            
                        for h in range(len(gbasis_output_cclib[k][l][1])):
                            angular_momentum_numbers_primitive_GTOs.append([2,105])
                        
                #Case of 5 spherical D GTOs
                #Order of indexation : cf. 'aonames' cclib attribute :
                #'D 0', 'D+1', 'D-1', 'D+2', 'D-2'
                elif (boolean_cartesian_D_shells==False):
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([2,0])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([2,1])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([2,-1])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([2,2])
                       
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([2,-2])
                    

            elif (gbasis_output_cclib[k][l][0]=='F'):
                #1 F primitive shell = full shell = 2*3+1 = 7 primitive GTOs 
                #in spherical form (l=2,m=-2..2) ; 10 primitive GTOs in cartesian form
                #==> of same zeta exponents.
                
                #Loop on all the primitive shells constituting this contracted shell

                #Case of cartesian representation 
                #Pure conventions :
                #(l=2,m=300) : XXX ; (l=2,m=301) : YYY ; (l=2,m=302) : ZZZ ; (l=2,m=303) : XXY ;
                #(l=2,m=304) : XXZ ; (l=2,m=305) : YYX ; (l=2,m=306) : YYZ ; (l=2,m=307) : ZZX ;
                #(l=2,m=308) : ZZY ; (l=2,m=309) : XYZ
                if (boolean_cartesian_F_shells==True):
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,300])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,301])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,302])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,303])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,304])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,305])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,306])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,307])
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,308])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,309])
                
                elif (boolean_cartesian_F_shells==False):
                    #Order : cf. order of indexation of spherical f-functions composing
                    #a F shell, in Gaussian (F-0, F 1, F-1, F 2, F-2, F 3, F-3)
                    
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,0])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,1])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,-1])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,2])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,-2])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,3])
                        
                    for h in range(len(gbasis_output_cclib[k][l][1])):
                        angular_momentum_numbers_primitive_GTOs.append([3,-3])
                        
            elif (gbasis_output_cclib[k][l][0]=='G'):
                #Case of 9 spherical G GTOs
                #Order of indexation : cf. 'aonames' cclib attribute :
                #'G 0', 'G+1', 'G-1', 'G+2', 'G-2','G+3', 'G-3', 'G+4', 'G-4'
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,0])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,-1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,-2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,3])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,-3])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,4])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([4,-4])
            
            elif (gbasis_output_cclib[k][l][0]=='H'):
                #Case of 11 spherical H GTOs (e.g. one 'H' function on O for cc-pV5Z basis)
                #Order of indexation : cf. 'aonames' cclib attribute :
                #'H 0', 'H+1', 'H-1', 'H+2', 'H-2','H+3', 'H-3', 'H+4', 'H-4', 'H+5', 'H-5'
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,0])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,-1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,-2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,3])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,-3])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,4])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,-4])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,5])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([5,-5])
                    
            elif (gbasis_output_cclib[k][l][0]=='I'):
                #Case of 11 spherical H GTOs (e.g. one 'H' function on O for cc-pV5Z basis)
                #Order of indexation : cf. 'aonames' cclib attribute :
                #'I 0', 'I+1', 'I-1', 'I+2', 'I-2','I+3', 'I-3', 'I+4', 'I-4', 'I+5', 'I-5', 'I+6', 'I-6'
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,0])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-1])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-2])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,3])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-3])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,4])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-4])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,5])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-5])
                    
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,6])
                        
                for h in range(len(gbasis_output_cclib[k][l][1])):
                    angular_momentum_numbers_primitive_GTOs.append([6,-6])

    #Order of indexation of primitive GTOs DEFINED by this function (choice of order of different values of 'm' at a given 'l'
    #contrary to function 'import_basis_set_exponents_primitive_GTOs()' [indistinguishability of the different 'm' values of primitive GTOs
    #as shared basis set exponents between the different 'm' at a given 'l' of a primitive shell]))
    return angular_momentum_numbers_primitive_GTOs
############################################################################################################


    
########################################################################
#Counts the number of PRIMITIVE SHELLS (using cclib "gbasis" attribute) per atom
#For 'SP' shells : counts both S and P primitive shells
#Returns a table of length = nb_atoms AND the total number of primitive shells 
#the the whole system/molecule (i.e. sum of all numbers of this table)
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_nb_primitive_shells(data):
    
    gbasis_output_cclib=data.gbasis

    nb_atom=data.natom
    
    nb_primitive_shells=np.zeros(nb_atom)
    
    #Loop on atom index (len(gbasis_output_cclib)=nb_atom) 
    for k in range(len(gbasis_output_cclib)):
        #Loop on contracted shells associated to these atoms
        for l in range(len(gbasis_output_cclib[k])):
            #In case of an 'SP' shell : the 'gbasis' info correcly lists
            #all the primitive S and P shells
            nb_primitive_shells[k]+=len(gbasis_output_cclib[k][l][1])
             
    return [nb_primitive_shells,int(sum(nb_primitive_shells))]
##############################################################################
    

##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_types_primitive_shells(data):
    
    gbasis_output_cclib=data.gbasis
    

    #List of lists of primitive shells types for each atom
    #"list_types_primitive_shells[k]" = [S,S,S,S,S,S,S,S,P,P] in case of an atom
    #with a contracted S shell of 6 primitive shells and a contracted SP shell
    #of 2 primitive (S and P) shells
    list_types_primitive_shells=[[]]
    
    
    #Loop on atom index (len(gbasis_output_cclib)=nb_atom) 
    for k in range(len(gbasis_output_cclib)):
        
        #Loop on contracted shells associated to these atoms
        for l in range(len(gbasis_output_cclib[k])):
            #In case of an 'SP' shell : the 'gbasis' info correcly lists
            #all the primitive S and P shells
            
            #Loop on the primitive shells (of same type, e.g. 'S' or 'P')
            for h in range(len(gbasis_output_cclib[k][l][1])):
                
                list_types_primitive_shells[k].append(gbasis_output_cclib[k][l][0])
        
        #At the end of this loop on "l", list_types_primitive_shells[k] must
        #have been appended nb_primitive_shells[k]=import_nb_primitive_shells(data)[k] times
        if (k< (len(gbasis_output_cclib)-1)):
            list_types_primitive_shells.append([])
 
    return list_types_primitive_shells
##############################################################################
    

##############################################################################
#Returns a table tab of length of nb_atoms
#where tab[i]= number of contracted shells on atom i 
#(ex : S / SP / SP / D /F => 5 contracted shells ['SP' shell = 1 contracted shell])
#AND the total number of contracted shells, summing on all atoms
###############################################################
#Counts the number of CONTRACTED SHELLS (using cclib "gbasis" attribute) per atom
#Returns a table of length = nb_atoms
#More clever way by 'reading' / parsing the .fchk file  ??
#(where nb_contracted_shells per atom is apparent [one block of all contracted shells per atom])
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_nb_contracted_shells(data,output_type):
    
    gbasis_output_cclib=data.gbasis
    
    nb_atoms=data.natom

    nb_contracted_shells=np.zeros(nb_atoms)
    
    #Loop on atom index     
    for k in range(len(gbasis_output_cclib)):
    
        #Loop on contracted shells associated to these atoms
        l=0
        while (l<len(gbasis_output_cclib[k])):
            
            #In case of an 'SP' contracted shell : the 'gbasis' info lists all the primitive S and P shells
            #but there is only one contracted shell => check if it is an 'SP' shell by
            #looking at the gaussian exponents (should be equal for the 'S' and 'P' functions
            #of the SP shell by definition of the SP shell)
            
            
            if (l<len(gbasis_output_cclib[k])-1):

                if ((gbasis_output_cclib[k][l][0]=='S') and (gbasis_output_cclib[k][l+1][0]=='P')):
        
                    #We check if the first exponent of ('S', ((zeta_1,contraction_1),...)) is the same
                    #as the following first exponent of ('P', (zeta_1', contraction_1'),..) [i.e. definition of 'SP' shell]
                    #(should be the case by definition of an 'SP' shell)
                    if (gbasis_output_cclib[k][l][1][0][0]==gbasis_output_cclib[k][l+1][1][0][0]):
                        
                        if (output_type=='Gaussian' or output_type=='GAMESS'):
                            #In this case : for Gaussian / GAMESS count only one contracted shell (the 'SP' shell)
                            #and 'jump' two steps forward
                            #==> *** One SP shell is thus counted as one contracted shell.
                            nb_contracted_shells[k]+=1
                            l+=2
                            
                        elif (output_type=='Psi4'):
                            #Case of Psi4 : count 2 contracted shells ('S' and 'P' contracted shell)
                            #instead of counting only one contracted 'SP' shell
                            nb_contracted_shells[k]+=2
                            l+=2
                        
                    else:
                        #If the exponents of the first 'S' and 'P' shell of the 'SP' shell are not equal
                        #=> this is not an 'SP' shell : this is a regular S shell AND a regular P shell
                        #(i.e. 2 contracted shells)
                        nb_contracted_shells[k]+=2
                        l+=2
                else:
                    nb_contracted_shells[k]+=1
                    l+=1
            else:
                nb_contracted_shells[k]+=1
                l+=1
             
    return [nb_contracted_shells,int(sum(nb_contracted_shells))]
##############################################################################

##############################################################################
#Function which returns a list of lists :
#The list of index k is for instance ['S','SP','SP',..,'D','F'] 
#(table of length nb_contracted_shells_atom_k)
#giving the types (angular momentum of contracted shells) of contracted shells on this atom k.
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_types_contracted_shells(data,output_type):
    
    gbasis_output_cclib=data.gbasis
    
    list_contracted_shells=[[]]
    
    compteur_nb_atoms=0
    #Loop on atom index     
    for k in range(len(gbasis_output_cclib)):
    
        #Loop on contracted shells associated to these atoms
        l=0
        while (l<len(gbasis_output_cclib[k])):
            
            #In case of an 'SP' contracted shell (e.g. for 6-31G(d) basis) :
            #the 'gbasis' info lists all the primitive S and P shells successively
            #but there is only one contracted shell => check if it is an 'SP' shell by
            #looking at the gaussian exponents (should be equal for the 'S' and 'P' functions
            #of the SP shell by definition of the SP shell)
            
            if (l<len(gbasis_output_cclib[k])-1):

                if ((gbasis_output_cclib[k][l][0]=='S') and (gbasis_output_cclib[k][l+1][0]=='P')):
        
                    #We check if the first exponent of ('S', ((zeta_1,contraction_1),...)) is the same
                    #as the following first exponent of ('P', (zeta_1', contraction_1'),..) [i.e. definition of 'SP' shell]
            
                    if (gbasis_output_cclib[k][l][1][0][0]==gbasis_output_cclib[k][l+1][1][0][0]):
                        
                        if (output_type=='Gaussian' or output_type=='GAMESS'):
                            #In this case : for Gaussian / GAMESS count only one contracted shell (the 'SP' shell)
                            #In this case : count the 'SP' contracted shell
                            #and 'jump' two steps forward
                            list_contracted_shells[compteur_nb_atoms].append('SP')
                            l+=2
                            
                        elif (output_type=='Psi4'):
                            #Case of Psi4 : count 2 contracted shells ('S' and 'P' contracted shell)
                            #instead of counting only one contracted 'SP' shell
                            list_contracted_shells[compteur_nb_atoms].append('S')
                            list_contracted_shells[compteur_nb_atoms].append('P')
                            l+=2
                            
                    else:
                        #If the exponents of the first 'S' and 'P' shell of the 'SP' shell are not equal
                        #=> this is not an 'SP' shell : this is a regular S shell AND a regular P shell
                        list_contracted_shells[compteur_nb_atoms].append('S')
                        list_contracted_shells[compteur_nb_atoms].append('P')
                        l+=2

                else:
                    list_contracted_shells[compteur_nb_atoms].append(gbasis_output_cclib[k][l][0])
                    l+=1
            else:
                list_contracted_shells[compteur_nb_atoms].append(gbasis_output_cclib[k][l][0])
                l+=1
                
        compteur_nb_atoms+=1
        
        if (k<(len(gbasis_output_cclib)-1)):
            list_contracted_shells.append([])
    
    #"list_contracted_shells[k]" list has to be of size "nb_contracted_shells[k]" (of atom k) in the end
    #==> len(list_contracted_shells[k]) can be used as = nb_contracted_shells[k]
    return list_contracted_shells
##############################################################################
    
                
################################################################################################
#Function who SHOULD provide the same info as given by 'data.nbasis' cclib attribute
#i.e. the total number of contracted GTOs b(actual basis functions used in the calculations)
#=> useful only for verification purposes
################################################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def computes_nb_contracted_GTOs(data,output_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells):
    
    nb_atoms=data.natom
    
    nb_contracted_GTOs=np.zeros(nb_atoms)
        
    list_contracted_shells=import_types_contracted_shells(data,output_type)
    
    for k in range(nb_atoms):
        
        #"nb_contracted_GTOs[k]"=sum_{mu contracted Shell on atom k} (sum_{l \in L_{mu}^{cShell}} (2*l+1) )
        #where L_{mu}^{cShell} = all angular momentum numbers of the primitive Shells entering the cShell
        #={0} for a S ccontracted Shell
        #={0,1} for an SP ccontracted Shell
        #={2} for a D contracted Shell
        #etc...
        for l in range(len(list_contracted_shells[k])):
        
            if (list_contracted_shells[k][l]=='S'):
                
                nb_contracted_GTOs[k]+=1
                
            elif (list_contracted_shells[k][l]=='P'):
                #3 primitive GTOs for the 'P' shell
                #Index order of the associated cGTOs : PX, PY, PZ 
                nb_contracted_GTOs[k]+=3
            
            elif (list_contracted_shells[k][l]=='SP'):
                #one primitive GTO the the 'S' shell part, 3 primitive GTOs for the 'P' shell part
                #Index order of the associated cGTOs : S, PX, PY, PZ 
                nb_contracted_GTOs[k]+=4
            
            elif (list_contracted_shells[k][l]=='D'):
                
                #A 'D' shell in Gaussian is by convention written in terms
                #either of cartesian coordinates (6 components : by index order on XX, YY, ZZ, XY, XZ, YZ)
                #(cf. 'aonames' cclib attribute)
                if (boolean_cartesian_D_shells==True):
                    nb_contracted_GTOs[k]+=6
                    
                #or in terms of the five 'd' spherical harmonics
                elif (boolean_cartesian_D_shells==False):
                    nb_contracted_GTOs[k]+=5
            
            elif (list_contracted_shells[k][l]=='F'):
                #An 'F' shell in Gaussian is by convention written in terms of 
                #spherical f-functions (m=-3..3) i.e. 7 components in spherical form
                #Index order of the cGTOs : m=0, m=1, m=-1,m=2,m=-2, m=3,m=-3
                #10 components of a f-shell in cartesian form
                #Order of indexation : cf. 'aonames' attribute parsed by cclib.
                
                if (boolean_cartesian_F_shells==True):
                    nb_contracted_GTOs[k]+=10
                    
                elif (boolean_cartesian_F_shells==False):
                    nb_contracted_GTOs[k]+=7
            
            elif (list_contracted_shells[k][l]=='G'):
                #Case of spherical g-functions in the basis
                nb_contracted_GTOs[k]+=9
            
            elif (list_contracted_shells[k][l]=='H'):
                #Case of spherical h-functions in the basis
                nb_contracted_GTOs[k]+=11
                
            elif (list_contracted_shells[k][l]=='I'):
                #Case of spherical i-functions in the basis
                nb_contracted_GTOs[k]+=13

            
    
    ##Sum of all contracted GTOs centered over all atoms
    #sum(nb_contracted_GTOs[k] , k=0..(nb_atom-1))
    return int(sum(nb_contracted_GTOs)) 
############################################################################################################
    
############################################################################################################
#For each contracted shell, there is sum{alpha=1..k_{mu}} (2*l_{alpha}+1) pGTOs (2*l+1)
#where k_{mu} is the number of primitive shells contributing to this contracted shell
#Computes the table "nb_primitive_GTOs" such that "nb_primitive_GTOs[k]" is the 
#number of primitive GTOs centered on atom k
#and returns both this table and the sum of all these values 'nb_primitive_GTOs[k]'
#Example : = 89 primitive GTOs for Cu(2+) 6-31G(d) basis set in Gaussian g09
def computes_nb_primitive_GTOs(data,output_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells):

    nb_atoms=data.natom
    
    #matrix of size nb_atoms*nb_tot_contracted_shells
    #number_primitive_shells_per_contracted_shells[k][lambda]=0 for the contracted shells of index lambda
    #not associated to (i.e. centered at) atom k
    number_primitive_shells_per_contracted_shells=import_number_primitive_shells_per_contracted_shell(data,output_type)
   
    #List such that list[k]=['S','SP,..,'D',...] is the list of types (i.e. angular momentum)
    #of contracted shells associated to (i.e. centered on) atom k
    #=> len(list[k]) = nb of contracted shell associated to atom k
    list_contracted_shells=import_types_contracted_shells(data,output_type)
    
    nb_primitive_GTOs=np.zeros(nb_atoms)
    
    #Loop on the number of atoms
    for k in range(nb_atoms):
        
        #Loop on (all) the contracted shells of atom k                        
        for l in range(len(list_contracted_shells[k])):
                                       
            #sum_{alpha=1..k_{l}} (2*l_{alpha}+1)
            #i.e. sum on the number of primitive shells constitutive of the l^{th}
            #contracted shell centered on atom k [which is of type 'list_contracted_shells[k][l]']
            #times number of primitive GTOs for each primitive shell 
            #(i.e. (2L+1) for a primitive shell of angular momentum L)
                 
            if (list_contracted_shells[k][l]=='S'):
                    
                nb_primitive_GTOs[k]+=1*number_primitive_shells_per_contracted_shells[k][l]
            
            elif (list_contracted_shells[k][l]=='P'):
                    
                nb_primitive_GTOs[k]+=3*number_primitive_shells_per_contracted_shells[k][l]
                        
            elif (list_contracted_shells[k][l]=='SP'):
                #(2*0+1)+(2*1+1)=4 primitive GTOs associated to a couple of one S and one P primitive shell
                #[of same gaussian exponent] (definition of an SP shell)
                #If the l^{th} contracted shell centered on atom k is of 
                #SP type : number_primitive_shells_per_contracted_shells[k][mu+l]=number of 'SP primitive shells'
                #i.e. of couples (S,P) constitutive of the contracted SP shell
                nb_primitive_GTOs[k]+=4*number_primitive_shells_per_contracted_shells[k][l]
                
            elif (list_contracted_shells[k][l]=='D'):
                
                #Cartesian d-functions GTOs are used by convention in Gaussian
                #(transformed internally into 5 spherical d-harmonics + one s-function)
                if (boolean_cartesian_D_shells==True):
                    nb_primitive_GTOs[k]+=6*number_primitive_shells_per_contracted_shells[k][l]
                    
                elif (boolean_cartesian_D_shells==False):
                    nb_primitive_GTOs[k]+=5*number_primitive_shells_per_contracted_shells[k][l]
                    
            elif (list_contracted_shells[k][l]=='F'):
                #Spherical f-functions are used in Gaussian, 
                #cartesian (by default) in GAMESS/Psi4
                
                if (boolean_cartesian_F_shells==True):
                    nb_primitive_GTOs[k]+=10*number_primitive_shells_per_contracted_shells[k][l]
                
                elif (boolean_cartesian_F_shells==False):
                    nb_primitive_GTOs[k]+=7*number_primitive_shells_per_contracted_shells[k][l]
            
            elif (list_contracted_shells[k][l]=='G'):
                #Spherical g-functions
                nb_primitive_GTOs[k]+=9*number_primitive_shells_per_contracted_shells[k][l]
                
            elif (list_contracted_shells[k][l]=='H'):
                #Spherical g-functions
                nb_primitive_GTOs[k]+=11*number_primitive_shells_per_contracted_shells[k][l]

            elif (list_contracted_shells[k][l]=='I'):
                #Spherical g-functions
                nb_primitive_GTOs[k]+=13*number_primitive_shells_per_contracted_shells[k][l]

            
        #The primitive GTOs associated to all the contracted shells 
        #centered on atom k have been treated (i.e. counted)
        
    return [nb_primitive_GTOs,int(sum(nb_primitive_GTOs))]
############################################################################################################    


############################################################################################################
#For a given index i of a cGTO ==> gives the list of indexes of pGTOs associated to this cGTO (i.e. entering the definition of this cGTO)
#\chi^{cGTO}_i(r) = \chi^{cGTO}_{l,m} = \sum_{k} c_i^k * \chi^{pGTO}_{l,m,k} where 
#c_i^k is the contraction coefficient (from the 'mother' primitive shell of the pGTO
#to the 'mother' contracted shell of the cGTO) and 
#\chi^{pGTO}_{l,m,k}(r)=r^l Y_l^m(theta,phi) * exp(-zeta_k*r^2) with 'zeta_k' the 
#gaussian exponent of the 'mother' primitive shell of the pGTO.
def compute_correspondence_index_contracted_GTOs_primitive_GTOs(data,output_type,boolean_cartesian_D_shells,boolean_cartesian_F_shells):
    
    #matrix of size nb_atoms*nb_tot_contracted_shells
    #number_primitive_shells_per_contracted_shells[k][lambda]=0 for the contracted shells of index lambda
    #not associated to (i.e. centered at) atom k
    number_primitive_shells_per_contracted_shells=import_number_primitive_shells_per_contracted_shell(data,output_type)
    
    #List such that list[k]=['S','SP,..,'D',...] is the list of types (i.e. angular momentum)
    #of contracted shells associated to (i.e. centered on) atom k
    #=> len(list[k]) = nb of contracted shell associated to atom k
    list_contracted_shells=import_types_contracted_shells(data,output_type)
    
    #List which will be a list of "nb_contracted_GTOs" lists (each being the 
    #list of indexes of pGTOs associated to this cGTO)
    correspondence_index_contracted_GTOs_primitive_GTOs=[]
    
    #Number of primitive GTOs encountered up-to-date :
    compteur_nb_primitive_GTOs=0
    
    #Number of contracted GTOs encountered up-to-date :
    ##compteur_nb_contracted_GTOs=0
    
    #Loop on the number of atoms
    for k in range(data.natom):
        
        #Loop on (all) the contracted shells of atom k                        
        for l in range(len(list_contracted_shells[k])):
                                       
            #sum_{alpha=1..k_{l}} (2*l_{alpha}+1)
            #i.e. sum on the number of primitive shells constitutive of the l^{th}
            #contracted shell centered on atom k [which is of type 'list_contracted_shells[k][l]']
            #times number of primitive GTOs for each primitive shell 
            #(i.e. (2L+1) for a primitive shell of angular momentum L)
                 
            if (list_contracted_shells[k][l]=='S'):
                
                list_indexes_pGTOs_in_contracted_GTO=[]
                #Sum on contraction terms
                for h in range(number_primitive_shells_per_contracted_shells[k][l]):

                    list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)                     
                    compteur_nb_primitive_GTOs+=1
                    
                correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
                
            elif (list_contracted_shells[k][l]=='P'):

                ##We treat these 'P' contracted GTOs (PX, PY, PZ) and
                ##the 'P' (PX, PY, PZ) associated primitive GTOs (as many as the contraction degree 
                ##for the 'mother' 'P' contracted shell ; each having a different gaussian 
                ##exponent of the associated primitive shell) 
                   
                #We treat first PX, then PY, and then PZ contracted GTOs :
                for i in range(3):
                    
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
    
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
               
           
            elif (list_contracted_shells[k][l]=='SP'):
                #(2*0+1)+(2*1+1)=4 primitive GTOs associated to a couple of one S and one P primitive shell
                #[of same gaussian exponent] (definition of an SP shell)
                #If the l^{th} contracted shell centered on atom k is of 
                #SP type : number_primitive_shells_per_contracted_shells[k][mu+l]=number of 'SP primitive shells'
                #i.e. of couples (S,P) constitutive of the contracted SP shell
                
                ##We first treat the link between the 'S' contracted GTO and
                ##the 'S' associated primitive GTOs (as many as the contraction degree 
                ##for the 'mother' 'SP' contracted shell)
                ##We then treat the link between the 'P' contracted GTOs (PX, PY, PZ) and
                ##the 'P' (PX, PY, PZ) associated primitive GTOs (as many as the contraction degree 
                ##for the 'mother' 'SP' contracted shell ; each having a different gaussian 
                ##exponent of the associated primitive shell) 
                
                for i in range(4):
                    
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
    
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)                     
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
                
            elif (list_contracted_shells[k][l]=='D'):
                #Cartesian d-functions GTOs are used by convention in Gaussian
                #(transformed internally into 5 spherical d-harmonics + one s-function)
                
                #Number of f functions per f-shell
                #In the case of a CARTESIAN representation of the f-shells
                #there are 10 f basis functions, only 7 in the case of a SPHERICAL
                #representation
                if (boolean_cartesian_D_shells==True):
                    nb_d_functions_shell=6
                else:
                    nb_d_functions_shell=5
                
                for i in range(nb_d_functions_shell):
                    
                    #We treat i^{th}-type (d-type) contracted GTO :
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                            
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
                    
               
            elif (list_contracted_shells[k][l]=='F'):
                #Spherical f-functions are used in Gaussian
                
                ##We treat the link between the 'F' contracted GTOs and
                ##the 'F' associated primitive GTOs (as many as the contraction degree 
                ##for the 'mother' 'F' contracted shell ; each having a different gaussian 
                ##exponent of the associated primitive shell) 
                    
                #Number of f functions per f-shell
                #In the case of a CARTESIAN representation of the f-shells
                #there are 10 f basis functions, only 7 in the case of a SPHERICAL
                #representation
                if (boolean_cartesian_F_shells==True):
                    nb_f_functions_shell=10
                else:
                    nb_f_functions_shell=7
                    
                for i in range(nb_f_functions_shell):
                    
                    #We treat the i^{th} f-type contracted GTO :
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
    
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
            
            elif (list_contracted_shells[k][l]=='G'):
                #9 spherical g-functions
                for i in range(9):
                    
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
            
            elif (list_contracted_shells[k][l]=='H'):
                #9 spherical h-functions
                for i in range(11):
                    
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
            
            elif (list_contracted_shells[k][l]=='I'):
                #9 spherical i-functions
                for i in range(13):
                    
                    list_indexes_pGTOs_in_contracted_GTO=[]
                    #Sum on contraction terms
                    for h in range(number_primitive_shells_per_contracted_shells[k][l]):
                        list_indexes_pGTOs_in_contracted_GTO.append(compteur_nb_primitive_GTOs)
                        compteur_nb_primitive_GTOs+=1
                        
                    correspondence_index_contracted_GTOs_primitive_GTOs.append(list_indexes_pGTOs_in_contracted_GTO)
            


            
    #'correspondence_index_contracted_GTOs_primitive_GTOs' should be a list
    #of nb_contracted_GTOs lists, the concatenation of all the lists
    #giving exactly all pGTOs indexes (from 0 to (nb_pGTOs-1))
    return correspondence_index_contracted_GTOs_primitive_GTOs
############################################################################################################

############################################################################################################
#For the index alpha of a primitive GTO --> returns the (list of) index(es) "i"
#of the (one or several) associated contracted GTO 
#in which definition this primitive GTO enters
#BEWARE : unique index "i" of cGTO associated to each pGTO in the case of a SEGMENTED contraction scheme only
#In the general case of a GENERAL contraction scheme a primitive shell can play in the definition of 
#several contracted shells and thus a pGTO can play in the definition of several contracted GTOs !
#Thus two contracted GTOs are made up from two disjoint sets of primitive GTOs
#ONLY in the case of a SEGMENTED contraction scheme
def compute_correspondence_index_primitive_GTOs_contracted_GTOs(correspondence_index_contracted_GTOs_primitive_GTOs,nb_primitive_GTOs):
    
    correspondence_index_primitive_GTOs_contracted_GTOs=[]
    
    for k in range(nb_primitive_GTOs):
        correspondence_index_primitive_GTOs_contracted_GTOs.append([])
        
    #len(correspondence_index_contracted_GTOs_primitive_GTOs)=nb_contracted_GTOs
    for k in range(len(correspondence_index_contracted_GTOs_primitive_GTOs)):
        
        for l in range(len(correspondence_index_contracted_GTOs_primitive_GTOs[k])):
            #By definition, 'correspondence_index_contracted_GTOs_primitive_GTOs[k][l]'
            #span all the indexes of primitive GTOs associated to the contracted GTO of index k
            correspondence_index_primitive_GTOs_contracted_GTOs[correspondence_index_contracted_GTOs_primitive_GTOs[k][l]].append(k)
            
    #The list 'correspondence_index_primitive_GTOs_contracted_GTOs' is 
    #of length 'nb_primitive_GTOs' 
    return correspondence_index_primitive_GTOs_contracted_GTOs
############################################################################################################
            
            
#############################################################################################################
#Return the number of primitive shells per [contracted] shell for all atoms
#Cf. "Number of primitives per shell" section of .fchk file 
#Example :  the result "number_primitive_shells_per_contracted_shells" has to be a
#list of lists [[6,6],[3,1,1]]
#for a system of two atoms, with S, SP contracted shells (made up of 6 primitive shells each)
#centered on the first atom and 1 P, 1 D, 1 F primtive shell (made up of 3, 1 and 1 primitive shells 
#respectively) centered on the second atom
######################
#cf. Table of length "nb_contracted_shells[index_atom]" (cf. syntax "gfinput" info in (...).log g09 file)
#Result is [[6,6,6,3,1,3,1,1]] for one Cu(2+) atom, 6-31G(d), g09 :
#Basis made up of contracted shells : S / SP / SP / SP / SP / D / D / F centered on the single ion.
##BEWARE : the two second "6" refer to 6 S SHELLS and 6 P SHELLS (SP shell)
#=> 12 primitive SHELLS (6 of S type, 6 of P type) to construct the second contracted shell...
def import_number_primitive_shells_per_contracted_shell(data,output_type):
    
    gbasis_output_cclib=data.gbasis
    nb_atoms=data.natom


    number_primitive_shells_per_contracted_shells=[[]]
          
    #Loop on atom index     
    for k in range(nb_atoms):
                
        #Loop on contracted shells associated to these atoms
        l=0
        while (l<len(gbasis_output_cclib[k])):
            #In case of an 'SP' contracted shell : the 'gbasis' info lists all the primitive S and P shells
            #but there is only one contracted shell => check if it is an 'SP' shell by
            #looking at the gaussian exponents (should be equal for the 'S' and 'P' functions
            #of the SP shell by definition of the SP shell)
            
            if (l<len(gbasis_output_cclib[k])-1):

                if ((gbasis_output_cclib[k][l][0]=='S') and (gbasis_output_cclib[k][l+1][0]=='P')):
        
                    #We check if the first exponent of ('S', ((zeta_1,contraction_1),...)) is the same
                    #as the following first exponent of ('P', (zeta_1', contraction_1'),..) [i.e. definition of 'SP' shell]
            
                    if (gbasis_output_cclib[k][l][1][0][0]==gbasis_output_cclib[k][l+1][1][0][0]):
                        
                        if (output_type=='Gaussian' or output_type=='GAMESS'):
                            #In this case : 'SP' contracted shell => sum of x 'SP' primitive shells 
                            #Ex: of SP contracted shell = sum of 6 shells (6 S shells and 6 P shells, with different contraction coefficients) 
                            #=> count only 6
                            #and 'jump' two steps forward
                            number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l][1]))
                            l+=2
                            
                        elif (output_type=='Psi4'):
                        
                            #In Psi4 : no 'SP' contracted shell => 'S' shell followed by a 'P' shell
                            number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l][1]))
                            number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l+1][1]))
                            l+=2
                            
                    else:
                        #Case of a 'S' shell followed by a 'P' shell
                        number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l][1]))
                        number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l+1][1]))
                        l+=2
                else:
                    
                    number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l][1]))
                    l+=1
            else:
                number_primitive_shells_per_contracted_shells[k].append(len(gbasis_output_cclib[k][l][1]))
                l+=1
          
        #We prepare a new list for the next atom (EXCEPT if we already reached the last atom)
        if (k < (nb_atoms-1)):
            number_primitive_shells_per_contracted_shells.append([])
        
    #Returns a list of nb_atom lists (each of size nb_contracted_shells[k] : 
    #nb of contracted shells centered on atom k)
    #Total number of coefficients of the list : nb_contracted_shells
    return number_primitive_shells_per_contracted_shells
#######################################################################################################
    
    
#######################################################################
#Returns the unique correspondence : l (index of basis function) --> i (index of the atom where it is centered)
#We use cclib attribute "atombasis"
#"atombasis[i]" : list of indexes of basis functions (cGTOs) associated / centered at atom i
##Ex : atombasis[1] will contain the indices of atomic orbitals on the second atom of the molecule.
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_correspondence_basis_function_cGTOs_atom_index(data):
    
    #"data.nbasis" : cclib attribute = number of basis functions (cGTOs)
    correspondence_basis_cGTOs_atom_index=np.zeros(data.nbasis)
    
    #Loop on the number of atoms
    #"data.natom" : cclib attribute = number of atoms
    for i in range(data.natom):
        
        #Loop on the number of basis functions (i.e. contrated GTOs) per atom
        for l in range(data.atombasis[i]):
            #"data.atombasis" : cclib attribute (only if g09 "pop=full" input keyword ?)
            correspondence_basis_cGTOs_atom_index[data.atombasis[i][l]]=i
            ##'data.atombasis[i][l]' should span all possible cGTO basis functions indexes 
            #(i.e. from 0 to [data.nbasis-1])
        
    return correspondence_basis_cGTOs_atom_index
##############################################################################
   
   
############################################################################## 
#Returns a table 'correspondence_basis_primitive_shell_atom_index' : 
#alpha (index of the (pSHELL) basis function) --> index of the atom where this basis function is centered
#Exactly similar to "Shell to atom map" section of (..).fchk file 
##but for CONTRACTED SHELLS in the case of the (...).fchk file section.   
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_correspondence_basis_primitive_shells_atom_index(data):
    
    gbasis_output_cclib=data.gbasis
    
    nb_primitive_shells=import_nb_primitive_shells(data)
    
    correspondence_basis_primitive_shell_atom_index=np.zeros(nb_primitive_shells)

    #Counts the number of (PRIMITIVE) SHELLS encountered up to date (in the loop below)
    #in the list of basis functions (primitive SHELLS) output of 
    #"ccget gbasis (...).log" [info from 'gfinput' option of Gaussian] 
    #or available from data.gbasis attribute
    index_courant=0

    #Loop on the number of atoms
    for k in range(len(gbasis_output_cclib)):
        
        #Loop on the number of contracted shells
        for l in range(len(gbasis_output_cclib[k])):
            #All these "len(gbasis_output_cclib[k][l][1])" basis function are centered on the k^{th} atom
            #(by definition of the ccget "gbasis" output)
            correspondence_basis_primitive_shell_atom_index[index_courant:index_courant+len(gbasis_output_cclib[k][l][1])]=k
            index_courant+=len(gbasis_output_cclib[k][l][1])

    return correspondence_basis_primitive_shell_atom_index
##############################################################################
    
#######################################################################
#Returns the unique correspondence : 
#l (index of PRIMITIVE GTO) --> i (index of the atom where it is centered)
#We use cclib attribute "atombasis"
#"atombasis[i]" : list of indexes of basis functions (cGTOs) associated / centered at atom i
##Ex : atombasis[1] will contain the indices of atomic orbitals on the second atom of the molecule.
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
###################
#'nb_primitive_GTOs' obtained previously as 
#'computes_nb_primitive_GTOs(data,boolean_cartesian_D_shells)'
def import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms):
    
    nb_primitive_GTOs_per_atom=nb_primitive_GTOs[0]

    total_nb_primitive_GTOs=nb_primitive_GTOs[1]
        
    correspondence_basis_pGTOs_atom_index=np.zeros(total_nb_primitive_GTOs)
    
    compteur_nb_primitive_GTOs=0
    
    #Loop on the number of atoms
    #"data.natom" : cclib attribute = number of atoms
    for k in range(nb_atoms):
        
        for h in range(int(nb_primitive_GTOs_per_atom[k])):
            
            correspondence_basis_pGTOs_atom_index[compteur_nb_primitive_GTOs]=k
            compteur_nb_primitive_GTOs+=1
            
    #At the end 'compteur' has been incremented of 
    #sum(nb_primitive_GTOs_per_atom[k],k=0..(nb_atom-1)) = total_nb_primitive_GTOs
    #'correspondence_basis_pGTOs_atom_index[]' is thus completely filled
        
    return correspondence_basis_pGTOs_atom_index
##############################################################################

#############################################################################
#'alpha' : index of QM basis set function (cGTO)
##cf. "Shell to atom map" in .fchk file ( alpha [index of the contracted GTO] --> i [index of the atom])
#Position of atom index i : read in "Current cartesian coordinates" section of .fchk file
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
##QUESTION : better to call this function once and for all for all values of 'alpha'
##'alpha' : index of the cGTO
""" UNUSED function ???
"""
def import_position_nuclei_associated_basis_function_cGTO(alpha,atomic_coordinates,data):
    
    index_atom_associated_basis_function_cGTO=import_correspondence_basis_function_cGTOs_atom_index(data)

    return atomic_coordinates[int(index_atom_associated_basis_function_cGTO[alpha])]
##############################################################################    
    

#############################################################################
## mu [index of the primitive GTO] --> R_i [coordinates of the atom where it is centered]
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
##########
#'nb_primitive_GTOs' obtained previously as 
#'computes_nb_primitive_GTOs(data,boolean_cartesian_D_shells)'
def import_position_nuclei_associated_basis_function_pGTO(atomic_coordinates,nb_primitive_GTOs,nb_atoms,data):
    
    index_atom_associated_basis_function_pGTO=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms)
    
    position_nuclei_associated_basis_function_pGTO=[]

    total_nb_pGTOs=nb_primitive_GTOs[1]
    
    for mu in range(total_nb_pGTOs):
        #'mu' : index of the pGTO
        #Evaluation of 'atomcoords' attribute at the last index 
        #(case of a geometry optimization => last [optimal] geometry)
        position_nuclei_associated_basis_function_pGTO.append(atomic_coordinates[int(index_atom_associated_basis_function_pGTO[mu])])

    #At the end 'position_nuclei_associated_basis_function_pGTO' must be a list of length 'nb_primitive_GTOs'
    return position_nuclei_associated_basis_function_pGTO
##############################################################################    
    


##############################################################################
#Returns {c_{alpha}}_{alpha=1..nb(pSHELLS)} : all contraction coefficients ,
#each one being associated uniquely to a given primitive SHELL (in the order of primitive shells,
#i.e. in the increasing number of atom index, in the increasing order of listed contracted shells,
#and for 'SP' shells : first list of contraction coefficients associated to the 'S' primitive shell part of the 'SP' shell
#and then list of contraction coefficients associated to the 'P' primitive shell part of the 'SP' shell)
#=> the goal of the other function 'import_contraction_scheme_matrix_pSHELLS_to_cSHELLS()'
#will be precisely to recover the index of the contracted SHELL to which each primitive 
#shell (and thus each contraction coefficient) is uniquely associated
################
#BEWARE : this linear (one-dimensional array) representation of the contraction scheme
#works only for a SEGMENTED contraction scheme ??
#[e.g. for Pople's type basis sets, but not for Dunning's type basis set ?] 
#i.e. every contracted SHELL is built upon contraction from primitive SHELLS 
#which are used only to build this specific contracted shell. 
#(the other contracted shells are built by linear combination of different primitive shells)
#shells are built by linear combination of different primitive shells)
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
def import_contraction_coefficients_pSHELLS_to_cSHELLS_one_array(data):
    
    gbasis_output_cclib=data.gbasis
    
    contraction_coefficients_pSHELLS_to_cSHELLS_one_array=[]

    #Loop on atom index
    for k in range(len(gbasis_output_cclib)):
        
        #Loop on contracted shells associated to these atoms
        for l in range(len(gbasis_output_cclib[k])):
            
            #Loop on primitive shells associated to each of these contracted shell
            for h in range(len(gbasis_output_cclib[k][l][1])):
                
                contraction_coefficients_pSHELLS_to_cSHELLS_one_array.append(gbasis_output_cclib[k][l][1][h][1])
    
    return contraction_coefficients_pSHELLS_to_cSHELLS_one_array
##############################################################################    

##############################################################################
#Reads "Number of primitives per shell" section
#cf. par exemple "6    6     6    3     1      3      1     1"
##########################
#Returns the contraction matrix of {\chi_{alpha}^{cSHELL}}_{alpha} in the basis of {\chi_{mu}^{pSHELL}}_{mu}
#i.e. [[0,0,0.02,...0.01,0]
#      [..................]
#      ....
#      [..0.1,...0.3,....]]
#where (horizontally) primitive SHELLS are ordered following the order in the (..).log file ("gfinput" keyword)
#and 'S' functions first, then 'P' functions ; for 'SP' shells. (i.e. equivalently, the order provided by 'gbasis' cclib attribute)
#Returns a matrix of size nb_contracted_shells*nb_primitive_shells
##############################################################################
#The input argument 'data' is supposed to be computed previously as :
#parser = ccopen(QM_output_file)
#data = parser.parse()
#########################
#TO RECODE / ADAPT for a more general contraction scheme ? 
#(i.e. not a SEGMENTED contraction schem as for Pople's family of basis sets like 6-31G(d))
def import_contraction_scheme_matrix_pSHELLS_to_cSHELLS(data,output_type,nb_contracted_shells,input_type):
    
    gbasis_output_cclib=data.gbasis
    
    #"number_primitive_shells_per_contracted_shell[k]" =[n_1,..,n_d] where n_i is 
    #the number of primitive shells used to construct the i^th contracted SHELL centered on atom k
    #(and d = len(number_primitive_shells_per_contracted_shell[k]) is the number of contracted shell in the atom k)
    #BEWARE : in the case of 'SP' shells, n_i=number of primitive S (or equivalently P) shells playing 
    #in the contraction (we do not count both primitive S and P shells of an SP shell ; 
    #in other word we count one 'primitive SP shell' => difference with the "nb_primitive_shells" from "import_nb_primitive_shells()" function)
    number_primitive_shells_per_contracted_shell=import_number_primitive_shells_per_contracted_shell(data,output_type)
    
    nb_atoms=data.natom
        
    nb_primitive_shells=import_nb_primitive_shells(data)[1]

    contraction_coefficients_one_array=import_contraction_coefficients_pSHELLS_to_cSHELLS_one_array(data)
    
    contraction_matrix_cSHELLS_basis_pSHELLS=np.zeros([nb_contracted_shells,nb_primitive_shells])
    
    #Counts the number of lines already treated (filled) in the matrix of contraction coefficients
    #'contraction_matrix_cSHELLS_basis_pSHELLS'
    #i.e. counts the number of contracted shells already treated (over all atoms).
    compteur_nb_encountered_contracted_SHELLS=0
        
    #Counts the number of primitive shells already encountered
    compteur=0

    #Loop on the number of atoms :
    for nb in range(nb_atoms):
        #We span the contracted shells associated to the atom of index 'nb'
        #(Indicated by the two indexes "nb 0" at the beginning of the section [or by " **** "])
    
        #Counts the number of contracted shells already encountered for this atom (of index 'nb')
        compteur_nb_encountered_contracted_SHELLS_per_atom=0
    
        ##BEWARE : as "gbasis_output_cclib[nb]" can have more lines than "nb_contracted_shells"
        #as 'SP' shells are separated in two lines ('S',[zeta_1,c_1],..) and ('P',[zeta_1,(c_1)'],..)
        #(case of Pople's type basis sets such as 6-31G(d) => presence of 'SP' shells)
        #***Beware : in Psi4 'SP' shells are well separated in a S and a P shell 
        #(of same zeta exponents, and different contraction coefficients)
        mu=0
        ##Loop on the contracted shells associated to the atom of index 'nb'
        while (mu<len(gbasis_output_cclib[nb])):
        
            #We check if the first exponent of ('S', ((zeta_1,contraction_1),...)) is the same
            #as the following first exponent of ('P', (zeta_1', contraction_1'),..) 
            ## ==> i.e. case of an 'SP' shell (cf. e.g. Pople's type basis sets such as 6-31G(d))
            if (mu<len(gbasis_output_cclib[nb])-1):

                if ((gbasis_output_cclib[nb][mu][0]=='S') and (gbasis_output_cclib[nb][mu+1][0]=='P')):
        
                    #If the first exponent 'zeta_1' associated to these S and P functions is the same :
                    #(SHOULD ALWAYS be the case for an 'SP' shell)
                    #('SP' shell => means that the first "zeta_1" gaussian exponent is the same for both S and P)
                    if (gbasis_output_cclib[nb][mu][1][0][0]==gbasis_output_cclib[nb][mu+1][1][0][0]):
                        ##or at a given precision threshold (real numbers...) ?
                        #=> case of an 'SP' shell : contraction of S + P primitive SHELLS to give one contracted SHELL
                        #***(for Gaussian or GAMESS, but not for Psi4 !)
                        #Rq : in GAMESS, SP shells are designated as 'L' instead of 'SP'
                        #***In Psi 4, with 6-31G(d) basis set, 'SP' contacted shells do not exist and are 
                        #actually separated in one contrated S shell and one contracted P shell
                        
                        ##Hence the *2 factor for Gaussian
                        if (input_type=='Gaussian' or input_type=='GAMESS'):
                            
                            contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+2*number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+2*number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                        
                            compteur+=2*number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                            #We have constructed one more contracted shell  (SP shell)
                            #(by contracting 'S' and 'P' shells of same "zeta" exponents)
                            compteur_nb_encountered_contracted_SHELLS+=1
                            compteur_nb_encountered_contracted_SHELLS_per_atom+=1
                            
                            #The two lines 'S' and 'P' (in gbasis_output_cclib[nb]) are treated => we advance of two lines
                            mu+=2
                            
                        elif (input_type=='Psi4'):
                            
                            ###########
                            #We treat the contraction coefficients of the 'S' shell :
                            contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                            compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                            
                            #and then the contraction coefficients of the 'P' shell :
                            contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS+1][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1])]
                            compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1]
                            
                            compteur_nb_encountered_contracted_SHELLS+=2
                            compteur_nb_encountered_contracted_SHELLS_per_atom+=2
                            
                            mu+=2
                                
                            """
                            contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                        
                            compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                            
                            #We have constructed one more contracted shell  (S shell)
                            #(the following 'P' shell wharing the same "zeta" exponents will be constructed
                            #in the next iteration of the loop)
                            compteur_nb_encountered_contracted_SHELLS+=1
                            compteur_nb_encountered_contracted_SHELLS_per_atom+=1
                            
                            #Only the line 'S' (in gbasis_output_cclib[nb]), and not yet 'P', is treated => we advance of one line only
                            mu+=1
                            """
                            
                    else:
                        #If the exponents of the first 'S' and 'P' shell of the 'SP' shell are not equal
                        #=> this is not an 'SP' shell : this is a regular S shell AND a regular P shell

                        ###########
                        #We thus treat the contraction coefficients of the 'S' shell :
                        contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                        compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                        
                        #and then the contraction coefficients of the 'P' shell :
                        contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS+1][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1])]
                        compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom+1]
                        
                        compteur_nb_encountered_contracted_SHELLS+=2
                        compteur_nb_encountered_contracted_SHELLS_per_atom+=2
                        
                        #We have treated two lines of 'gbasis_output_cclib[nb]' i.e. two contracted shells
                        mu+=2
                        ###########

                else: 
                    contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                    compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                    compteur_nb_encountered_contracted_SHELLS+=1
                    compteur_nb_encountered_contracted_SHELLS_per_atom+=1
                    mu+=1
                    
            else:
               
                contraction_matrix_cSHELLS_basis_pSHELLS[compteur_nb_encountered_contracted_SHELLS][compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]=contraction_coefficients_one_array[compteur:(compteur+number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom])]
                compteur+=number_primitive_shells_per_contracted_shell[nb][compteur_nb_encountered_contracted_SHELLS_per_atom]
                compteur_nb_encountered_contracted_SHELLS+=1
                compteur_nb_encountered_contracted_SHELLS_per_atom+=1
                mu+=1
            
        # "compteur_nb_encountered_contracted_SHELLS_per_atom" should be equal to "nb_contracted_shells[nb]" finally
        
    return contraction_matrix_cSHELLS_basis_pSHELLS
##############################################################################
    
##############################################################################
#Function (table) which associates :
#alpha (contracted GTO) --> index of 'mother' contracted shell (i.e. full shell with all quantum number m=-l..l)
#A cGTO = \chi_{l_{alpha},m_{alpha}}^{cGTO}(r) = sum with contraction coefficients (primitive--> contracted SHELLS)
#of 'pure' solid harmonics (of angular momentum numbers (l_{alpha},m_{alpha}) times gaussians of different 
#zeta exponents (as many as the degree of contraction i.e. the number of contraction coefficients 
#FOR THE CONTRACTED SHELL TO WHICH THIS cGTO belongs)
##########################################
#Convention for indexing contracted GTOs :
#by increasing order of atom index, and for SP shells : firts S subshell then P subshell
def correspondence_index_contracted_GTOs_contracted_shells(data,output_type,nb_contracted_GTOs,boolean_cartesian_D_shells,boolean_cartesian_F_shells):
   
    tab_correspondence_index_contracted_GTO_contracted_shell=np.zeros(nb_contracted_GTOs)
    
    nb_atoms=data.natom
            
    list_contracted_shells=import_types_contracted_shells(data,output_type)
    
    #Number of encountered contracted GTOs :
    compteur_nb_encountered_contracted_GTOs=0
    
    #Number of encountered contracted SHELLS :
    compteur_nb_encountered_contracted_SHELLS=0
    
    for k in range(nb_atoms):
        
        #"nb_contracted_GTOs[k]"=sum_{mu contracted Shell on atom k} (sum_{l \in L_{mu}^{cShell}} (2*l+1) )
        #where L_{mu}^{cShell} = all angular momentum numbers of the primitive shells entering the cShell
        #={0} for a S ccontracted Shell
        #={0,1} for an SP contracted Shell
        #={2} for a D contracted Shell
        #etc...
        for l in range(len(list_contracted_shells[k])):
        
            if (list_contracted_shells[k][l]=='S'):
                
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs]=int(compteur_nb_encountered_contracted_SHELLS)
                compteur_nb_encountered_contracted_GTOs+=1
                compteur_nb_encountered_contracted_SHELLS+=1
            
            elif (list_contracted_shells[k][l]=='P'):
                
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+3)]=int(compteur_nb_encountered_contracted_SHELLS)
                compteur_nb_encountered_contracted_GTOs+=3
                compteur_nb_encountered_contracted_SHELLS+=1
            
            
            elif (list_contracted_shells[k][l]=='SP'):
                #one primitive GTO the the 'S' shell part, 3 primitive GTOs for the 'P' shell part
                #Index order of the associated cGTOs : S, PX, PY, PZ 
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+4)]=int(compteur_nb_encountered_contracted_SHELLS)

                compteur_nb_encountered_contracted_GTOs+=4
                compteur_nb_encountered_contracted_SHELLS+=1
                
            
            elif (list_contracted_shells[k][l]=='D'):
                
                
                #A 'D' shell in Gaussian is by convention either written in terms of 
                #cartesian coordinates (6 components : by index order on XX, YY, ZZ, XY, XZ, YZ)
                #or in terms of 5 spherical harmonics (l=2,m=0), (l=2,m=1), (l=2,m=-1), (l=2,m=2), (l=2,m=-2)
                #(cf. 'aonames' cclib attribute)
                
                if (boolean_cartesian_D_shells==True):
                    
                    tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+6)]=int(compteur_nb_encountered_contracted_SHELLS)
                    compteur_nb_encountered_contracted_GTOs+=6
                    compteur_nb_encountered_contracted_SHELLS+=1
                    
                elif (boolean_cartesian_D_shells==False):
                    
                    tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+5)]=int(compteur_nb_encountered_contracted_SHELLS)
                    compteur_nb_encountered_contracted_GTOs+=5
                    compteur_nb_encountered_contracted_SHELLS+=1
                
            
            elif (list_contracted_shells[k][l]=='F'):
                #An 'F' shell in Gaussian is by convention written in terms of 
                #spherical f-functions (m=-3..3) i.e. 7 components
                #Index order of the cGTOs : m=0, m=1, m=-1,m=2,m=-2, m=3,m=-3
                #in spherical form, 10 otherwise (cartesian representation) e.g. 
                #for GAMESS / Psi4 calculations, by default
                #(cf. 'aonames' cclib attribute)
                if (boolean_cartesian_F_shells==True):
                    
                    tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+10)]=int(compteur_nb_encountered_contracted_SHELLS)
                    compteur_nb_encountered_contracted_GTOs+=10
                    compteur_nb_encountered_contracted_SHELLS+=1  
                    
                elif (boolean_cartesian_F_shells==False):
                    
                    tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+7)]=int(compteur_nb_encountered_contracted_SHELLS)
                    compteur_nb_encountered_contracted_GTOs+=7
                    compteur_nb_encountered_contracted_SHELLS+=1  
            
            elif (list_contracted_shells[k][l]=='G'):  
            
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+9)]=int(compteur_nb_encountered_contracted_SHELLS)
                compteur_nb_encountered_contracted_GTOs+=9
                compteur_nb_encountered_contracted_SHELLS+=1
                
            elif (list_contracted_shells[k][l]=='H'):  
            
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+11)]=int(compteur_nb_encountered_contracted_SHELLS)
                compteur_nb_encountered_contracted_GTOs+=11
                compteur_nb_encountered_contracted_SHELLS+=1
             
            elif (list_contracted_shells[k][l]=='I'):  
            
                tab_correspondence_index_contracted_GTO_contracted_shell[compteur_nb_encountered_contracted_GTOs:(compteur_nb_encountered_contracted_GTOs+13)]=int(compteur_nb_encountered_contracted_SHELLS)
                compteur_nb_encountered_contracted_GTOs+=13
                compteur_nb_encountered_contracted_SHELLS+=1
                
    return tab_correspondence_index_contracted_GTO_contracted_shell
##############################################################################   

##############################################################################
#Function which associates :
#i (primitive GTO) --> index of the 'mother' primitive shell (i.e. full shell with all quantum number m=-l..l)
#############################################################################
#'tab_correspondence_index_primitive_GTOs_primitive_shells' will be used
#to grab the appropriate contraction coefficients (from primitive shells to contracted shells)
#in the function 'compute_density_matrix_coefficient_pGTOs()' 
#[computation of the density matrix coefficients relative to the expression of the total
#density with products of pGTOs, rather than with cGTOs]
def correspondence_index_primitive_GTOs_primitive_shells(data,output_type,nb_primitive_GTOs,boolean_cartesian_D_shells,boolean_cartesian_F_shells):
    
    tab_correspondence_index_primitive_GTOs_primitive_shells=np.zeros(nb_primitive_GTOs)
    
    nb_atoms=data.natom
            
    list_contracted_shells=import_types_contracted_shells(data,output_type)
    
    number_primitive_shells_per_contracted_shell=import_number_primitive_shells_per_contracted_shell(data,output_type)
    
    #Number of encountered primitive GTOs :
    compteur_nb_encountered_primitive_GTOs=0
    
    #Number of encountered primitive SHELLS :
    compteur_nb_encountered_primitive_SHELLS=0
    
    #Loop on the atoms
    for k in range(nb_atoms):
        
        #Loop on the contracted shells associated to each atom of index 'k'
        for l in range(len(list_contracted_shells[k])):
        
            if (list_contracted_shells[k][l]=='S'):
                
                for h in range(number_primitive_shells_per_contracted_shell[k][l]):

                    tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs]=compteur_nb_encountered_primitive_SHELLS
                    compteur_nb_encountered_primitive_GTOs+=1
                    compteur_nb_encountered_primitive_SHELLS+=1
               
            elif (list_contracted_shells[k][l]=='P'):
                
                ##############
                #Treating PX, PY and PZ primitive GTOs  (of different zeta exponents) :
                ##############
                for i in range(3):
                    
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                 
                #We count the 'P' primitive shells of this 'P' contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]


            elif (list_contracted_shells[k][l]=='SP'):
                #one primitive GTO the the 'S' shell part, 3 primitive GTOs for the 'P' shell part
                #Index order of the associated cGTOs : S, PX, PY, PZ 
                
                ##BEWARE : 
                #in 'number_primitive_shells_per_contracted_shells' : one 'SP' primitive shell
                #is considered as one primitive shell only (no separation between S and P as the )
                    
                #Beware : here we count the 2 S and P shells as in the contraction matrix
                #'contraction_coefficients_pSHELLS_to_cSHELLS', SP shells are separated in
                #S and P shells (as they have different contraction coefficients)
                #and we will use 'tab_correspondence_index_primitive_GTOs_primitive_shells'
                #to grab the correct primitive shell associated to a primitive GTO
                ##############
                #First : treating S primitive GTOs (of different zeta exponents) 
                ##############
                for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                    tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                
                #In the case of 'S' primitive shells : number of primitive shells = number of primitive GTOs
                compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                 
                #We count the 'S' primitive shells of this 'SP' contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]
                    
                ##############
                #Then : treating PX, PY and PZ primitive GTOs  (of different zeta exponents) :
                ##############
                for i in range(3):

                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                   
                #We count the 'P' primitive shells of this 'SP' contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]

            elif (list_contracted_shells[k][l]=='D'):
                #A 'D' shell in Gaussian is by convention either written in terms of 
                #cartesian coordinates (6 components : by index order on XX, YY, ZZ, XY, XZ, YZ)
                #or in terms of 5 spherical harmonics
                #(cf. 'aonames' cclib attribute)
                
                if (boolean_cartesian_D_shells==True):
                    nb_d_functions_shell=6
                else:
                    nb_d_functions_shell=5
                
                ##############
                #Treating XX, YY, ZZ, XY, XZ, YZ d-type [case of spherical functions]
                #or spherical harmonics (l=2,m=0), (l=2,m=1), (l=2,m=-1), (l=2,m=2), (l=2,m=-2) 
                #primitive GTOs  (of different zeta exponents) :
                ##############
                    
                for i in range(nb_d_functions_shell):
                        
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                    
                #We count the 'D' primitive shells of this 'D' contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]
                
            
            elif (list_contracted_shells[k][l]=='F'):
                #An 'F' shell in Gaussian is by convention written in terms of 
                #spherical f-functions (m=-3..3) i.e. 7 components
                #Index order of the cGTOs : m=0, m=1, m=-1,m=2,m=-2, m=3,m=-3
                #(cf. 'aonames' cclib attribute)
                
                ##############
                #Treating F 0, F+1, F-1, F+2, F-2, F+3, F-3 f-type [case of spherical f-functions]
                #or XXX, YYY, ZZZ; XXY, XXZ, YYX, YYZ, ZZX, ZZY, XYZ f-type functions [case of cartesian f-functions]
                #primitive GTOs  (of different zeta exponents) :
                ##############
                
                if (boolean_cartesian_F_shells==True):
                    nb_f_functions_shell=10
                else:
                    nb_f_functions_shell=7
                    
                for i in range(nb_f_functions_shell):
                
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                    
                #We count the 'F' primitive shells of this 'F' contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]
               
                
            elif (list_contracted_shells[k][l]=='G'):
                
                for i in range(9):
                    
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                   
                #We count the primitive shells of this contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]

            elif (list_contracted_shells[k][l]=='H'):
                
                for i in range(11):
                    
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                   
                #We count the primitive shells of this contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]

            elif (list_contracted_shells[k][l]=='I'):
                
                for i in range(13):
                    
                    for h in range(number_primitive_shells_per_contracted_shell[k][l]):
                        
                        tab_correspondence_index_primitive_GTOs_primitive_shells[compteur_nb_encountered_primitive_GTOs+h]=int(compteur_nb_encountered_primitive_SHELLS)+h
                    
                    compteur_nb_encountered_primitive_GTOs+=number_primitive_shells_per_contracted_shell[k][l]
                   
                #We count the primitive shells of this contracted shell    
                compteur_nb_encountered_primitive_SHELLS+=number_primitive_shells_per_contracted_shell[k][l]


    #'compteur_nb_encountered_primitive_GTOs' should be equal to 'nb_primitive_GTOs' at the end of the loop
    return tab_correspondence_index_primitive_GTOs_primitive_shells
############################################################################## 
    

##############################################################################
##############################################################################
#Computes density matrix coefficients in terms of PRIMITIVE GTOs 
#from density matrix coefficients in terms of CONTRACTED GTOs representation 
#(matrix of size nb_contracted_GTOs*nb_contracted_GTOs)
#(the latter being directly computed from .fchk file :
# readable from "Alpha/Beta MO coefficients" : what does import_density_matrix_contracted_GTOs() function.
#Goal of this function : going from a nb_contracted_GTOs*nb_contracted_GTOs matrix 
#to a nb_primitive_GTOs*nb_primitive_GTOs matrix
#using the contraction coefficients between contracted SHELLS and primitive SHELLS
##D_{i,j}^{pGTO} = 2 * sum_{\alpha \in I_i,\beta \in I_j} c_{s(i)}^{mu_{alpha}} c_{s(j)}^{mu_{beta}} * N_{alpha} * N_{\beta} * D_{alpha,beta}^{cGTO}
##                   + (sum_{\alpha \in I_i,\beta \in I_i} |c_{s(i)}^{mu_{alpha}}|^2 *  N_{alpha}^2 * D_{alpha,alpha}^{cGTO})*delta_{i,i}
#where delta_{i,i} is the Kronecker symbol ; I_i is the set of contracted GTOs where the primitive GTO
#of index 'i' enters the definition, s(i) is the index of the primitive shell 'mother' of the pGTO 'i',
#mu_{alpha} is the index of the contracted shell 'mother' of the contracted GTO of index 'alpha'
#D_{alpha,beta}^{cGTO} are density matrix coefficients in terms of cGTOs, N_{alpha} are normalization coefficients for GTOs
#and c_{s(i)}^{mu_{alpha}} is the contraction coefficient from the primitive shelle of insex 's(i)'
#to the contracted shell of index 'mu_{alpha}'.
###############################
#Density matrix coefficients in terms of PRIMITIVE GTOs are indeed needed 
#for Distributed Multipole Moments analysis as the elementary multipole moments 
#computed are those of products of pGTOs.
################################
#'tab_correspondence_index_contracted_GTOs_contracted_shells' and 'tab_correspondence_index_primitive_GTOs_primitive_shells'
#computed previously (before calling this function) as :
#tab_correspondence_index_contracted_GTOs_contracted_shells=correspondence_index_contracted_GTOs_contracted_shells(data,nb_contracted_GTOs)
#tab_correspondence_index_primitive_GTOs_primitive_shells=correspondence_index_primitive_GTOs_primitive_shells(data,total_nb_primitive_GTOs)
def compute_density_matrix_coefficient_pGTOs(data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,density_matrix_coefficient_contracted_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,correspondence_index_contracted_GTOs_primitive_GTOs,correspondence_index_primitive_GTOs_contracted_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,position_nuclei_associated_basis_function_pGTO,total_nb_primitive_GTOs,nb_contracted_GTOs,nb_primitive_shells,nb_contracted_shells,logfile_DMA):
    
    t_begin=time.clock()
    
    density_matrix_coefficient_pGTOs=np.zeros([total_nb_primitive_GTOs,total_nb_primitive_GTOs])
    
    ##list_indexes_pGTOs_associated_to_cSHELLs=indexes_pGTOs_associated_to_cSHELLs(data)
    
    #Normalization coeffs. of contracted GTOs : computed once and for all here :
    normalization_coeffs_cGTOs=[compute_normalization_coefficient_contracted_GTO(i,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO) for i in range(nb_contracted_GTOs)]

    ###############################
    #METHOD 1
    
    ###################    
    for i in range(total_nb_primitive_GTOs):
        
        #################################################################
        #We treat first the case i=j (i.e. the coefficient D_(i,i)^{pGTO}
        ###########
        #Index of the contracted (full) shell to which the primitive GTO (l_i,m_i,zeta_i) 
        #[of well defined first and second angular momentum numbers l_i and m_i, as well as
        #well defined Gaussian zeta exponent zeta_i]
        primitive_shell_i=tab_correspondence_index_primitive_GTOs_primitive_shells[i]
       
        #Contracted GTOs associated to the primitive GTO of index i :
        #(only one in the case of a SEGMENTED contraction ; or several in the case
        #of a more general contraction scheme [but sum at most on a few terms])
        for l in range(len(correspondence_index_primitive_GTOs_contracted_GTOs[i])):
                 
            alpha=int(correspondence_index_primitive_GTOs_contracted_GTOs[i][l])
           
            #Normalization constant of the contracted GTO of index 'alpha'
            N_alpha=normalization_coeffs_cGTOs[alpha]

            #Index of the contracted (full) shell to which the contracted GTO (l_alpha,m_alpha) 
            #[and with no gaussian zeta exponent properly defined ; sum of several (as much as 
            #the contraction degree for the 'mother' contracted shell)] belongs
            contracted_shell_alpha=tab_correspondence_index_contracted_GTOs_contracted_shells[alpha]
            
            #Case of spin polarised calculations : UHF or ROHF
            #(i.e. alpha and beta orbitals / density matrices) 
            #==> len(density_matrix_coefficient_contracted_GTOs)=2 
            #(alpha and beta density matrices, whose contributions have to be summed up)
            #Otherwise : regular Restricted calculation (e.g. RHF) : len(density_matrix_coefficient_contracted_GTOs)=1
            for sigma in range(len(density_matrix_coefficient_contracted_GTOs)):
                               
                density_matrix_coefficient_pGTOs[i][i]+=(N_alpha**2)*(contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_alpha)][int(primitive_shell_i)]**2)*density_matrix_coefficient_contracted_GTOs[sigma][alpha][alpha]
         
        #Terms 2*sum_{alpha < beta \in I_{i_0}^2} c_{s(i_0)}^{\mu_{\alpha}} * c_{s(i_0)}^{\mu_{\beta}} * D_{alpha,beta}^{cGTOs}
        #where s(i_0) is the index of the primitive shell associated to the pGTO i_O
        #and \mu_{\alpha} (resp. \mu_{\beta}) is the index of the contracted shell associated to the cGTO alpha (resp. beta)
        #I_{i_0} is the set on indexes of cGTOs where the pGTO of index 'i_0' plays in (enters) the definition 
        #(it is simply limited to one single index in the case of a segmented contraction scheme).
        for index_cGTO1 in range(len(correspondence_index_primitive_GTOs_contracted_GTOs[i])):
            
            for index_cGTO2 in range(len(correspondence_index_primitive_GTOs_contracted_GTOs[i])):
                
                if (index_cGTO1<index_cGTO2):
                    
                    alpha=index_cGTO1
                    beta=index_cGTO2
                    N_alpha=normalization_coeffs_cGTOs[alpha]
                    N_beta=normalization_coeffs_cGTOs[beta]
                    contracted_shell_alpha=tab_correspondence_index_contracted_GTOs_contracted_shells[alpha]
                    contracted_shell_beta=tab_correspondence_index_contracted_GTOs_contracted_shells[beta]
                
                    density_matrix_coefficient_pGTOs[i][i]+=2*N_alpha*N_beta*contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_alpha)][int(primitive_shell_i)]*contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_beta)][int(primitive_shell_i)]*density_matrix_coefficient_contracted_GTOs[0][alpha][beta]
        ###############################################################

        ###############################################################
        #We now treat the case i<j (i.e. the coefficients D_(i,j)^{pGTOs} for i<j):
        #############################
        for j in range(i+1,total_nb_primitive_GTOs):
 
            #Index of the contracted (full) shell to which the primitive GTO (l_j,m_j,zeta_j)
            primitive_shell_j=tab_correspondence_index_primitive_GTOs_primitive_shells[j]
            
            #Contracted GTOs associated to the primitive GTO of index i :
            #(only one in the case of a SEGMENTED contraction ; or several in the case
            #of a more general contraction scheme [but sum at most on a few terms])        
            for l in range(len(correspondence_index_primitive_GTOs_contracted_GTOs[i])):
           
                alpha=int(correspondence_index_primitive_GTOs_contracted_GTOs[i][l])
           
                #Normalization constant of the contracted GTO of index 'alpha'
                N_alpha=normalization_coeffs_cGTOs[alpha]

                #Index of the contracted (full) shell to which the contracted GTO (l_alpha,m_alpha) 
                #[and with no gaussian zeta exponent properly defined ; sum of several (as much as 
                #the contraction degree for the 'mother' contracted shell)] belongs
                contracted_shell_alpha=tab_correspondence_index_contracted_GTOs_contracted_shells[alpha]    
           
                #Contracted GTOs associated to the primitive GTO of index j :
                #(only one in the case of a SEGMENTED contraction ; or several in the case
                #of a more general contraction scheme [but sum at most on a few terms])
                for h in range(len(correspondence_index_primitive_GTOs_contracted_GTOs[j])):
                    
                    beta=int(correspondence_index_primitive_GTOs_contracted_GTOs[j][h])
                   
                    N_beta=normalization_coeffs_cGTOs[beta]
                   
                    contracted_shell_beta=tab_correspondence_index_contracted_GTOs_contracted_shells[beta]
                    
                    #Case of spin polarised calculations : UHF or ROHF
                    #(i.e. alpha and beta orbitals / density matrices) 
                    #==> len(density_matrix_coefficient_contracted_GTOs)=2 
                    #(alpha and beta density matrices, whose contributions have to be summed up)
                    #Otherwise : regular Restricted calculation (e.g. RHF) : len(density_matrix_coefficient_contracted_GTOs)=1
                    for sigma in range(len(density_matrix_coefficient_contracted_GTOs)):
                                                
                        #Factor "2" by symmetry of the expression 
                        #n(\vec(r)) = \sum_{alpha,beta cGTOs} D_{alpha,beta}^{cGTOs} \chi_{alpha}^{cGTO}(r) * \chi_{beta}^{cGTO}(r)
                        
                        density_matrix_coefficient_pGTOs[i][j]+=2*N_alpha*N_beta*contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_alpha)][int(primitive_shell_i)]*contraction_coefficients_pSHELLS_to_cSHELLS[int(contracted_shell_beta)][int(primitive_shell_j)]*density_matrix_coefficient_contracted_GTOs[sigma][alpha][beta]
                    
            ##By symmetry of the density matrix coefficients (case of REAL valued basis functions) :
            density_matrix_coefficient_pGTOs[j][i]=density_matrix_coefficient_pGTOs[i][j]
    
    ###################
    
    t_end=time.clock()
    logfile_DMA.write('Time to compute density matrix pGTOs from density matrix cGTOs : '+str(t_end-t_begin))
    logfile_DMA.write('\n')
    logfile_DMA.write('\n')
    
    return density_matrix_coefficient_pGTOs
##############################################################################
##############################################################################

##############################################################################
#Returns <\chi^{pGTO}_{alpha} | \chi^{pGTO}_{beta} > (for NON-normalized pGTOs i.e.
#of the form constant*|r|^{l_alpha} * Y_{l_alpha}^{m_alpha}(\vec{r}/|r|) * exp(-zeta_{alpha}*r^2)
#where  alpha and beta are respectively the indexes of the two primitives
#l_alpha,m_alpha,zeta_1 (gaussian exponent) characterize the first primitive GTO : r^{l_alpha}*Y_{l_alpha}^{m_alpha}(theta,phi)*exp(-zeta_1*r^2)
#l_beta,m_beta,zeta_2 (gaussian exponent) characterize the second primitive GTO : r^{l_beta}*Y_{l_beta}^{m_beta}(theta,phi)*exp(-zeta_2*r^2)
#'angular_momentum_numbers_primitive_GTOs' and 'basis_set_exponents_primitive_GTOs' assumed to 
#have been already computed before calling this function
#'correspondence_basis_pGTOs_atom_index' : assumed to have been computed previously as :
#correspondence_basis_pGTOs_atom_index=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,nb_atoms)
def compute_scalar_product_two_primitive_GTOs(alpha,beta,
                                              boolean_cartesian_D_shells,
                                              boolean_cartesian_F_shells,
                                              angular_momentum_numbers_primitive_GTOs,
                                              basis_set_exponents_primitive_GTOs,
                                              correspondence_basis_pGTOs_atom_index,
                                              position_nuclei_associated_basis_function_pGTO):
    
    zeta_alpha=basis_set_exponents_primitive_GTOs[alpha]
    zeta_beta=basis_set_exponents_primitive_GTOs[beta]
    
    l_alpha=angular_momentum_numbers_primitive_GTOs[alpha][0]
    m_alpha=angular_momentum_numbers_primitive_GTOs[alpha][1]
    l_beta=angular_momentum_numbers_primitive_GTOs[beta][0]
    m_beta=angular_momentum_numbers_primitive_GTOs[beta][1]

    ##Information on the coefficients for conversion of r^l * Y_l^m(\vec{r}/|\vec{r}|)
    # "conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l,m)" 
    #returns [p(l,m),[[n_1(l,m),m_1(l,m),t_1(l,m)],..[n_{p(l,m)}(l,m),m_{p(l,m)}(l,m),t_{p(l,m)}(l,m)]],[w_1^(l,m),w_2^(l,m),..,w_{p(l,m)}(l,m)]]
    
    ##BEWARE : case of a 'd-type' primitive GTO in Gaussian ==> x² * exp(-zeta*r^2) or y² * exp(-zeta*r^2) or
    #z² * exp(-zeta*r^2) or xy * exp(-zeta*r^2) of xz * exp(-zeta*r^2) or yz * exp(-zeta*r^2)
    #==> OK  : case included in the function 'conversion_spherical_harmonics_cartesian_homogeneous_polynoms_with_cartesian_D_shell_exception()'
    #(case when l_alpha==2 or l_beta==2)
    if (boolean_cartesian_D_shells==True):
        
        if (boolean_cartesian_F_shells==False):
            if (l_alpha<=4):
                coeff_harmonics_cartesian_alpha=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells(l_alpha,m_alpha)
            else:
                coeff_harmonics_cartesian_alpha=computes_coefficient_solid_harmonic_l_m_list_format(l_alpha,m_alpha)
            
            if (l_beta<=4):
                coeff_harmonics_cartesian_beta=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells(l_beta,m_beta)
            else:
                coeff_harmonics_cartesian_beta=computes_coefficient_solid_harmonic_l_m_list_format(l_beta,m_beta)
                
        elif (boolean_cartesian_F_shells==True):
            if (l_alpha<=4):
                coeff_harmonics_cartesian_alpha=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells(l_alpha,m_alpha)
            else:
                coeff_harmonics_cartesian_alpha=computes_coefficient_solid_harmonic_l_m_list_format(l_alpha,m_alpha)
                
            if (l_beta<=4):
                coeff_harmonics_cartesian_beta=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_D_shells_cartesian_F_shells(l_beta,m_beta)
            else:
                coeff_harmonics_cartesian_beta=computes_coefficient_solid_harmonic_l_m_list_format(l_beta,m_beta)
            
    elif (boolean_cartesian_D_shells==False):
        
        if (boolean_cartesian_F_shells==False):
            
            if (l_alpha<=4):
                coeff_harmonics_cartesian_alpha=conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l_alpha,m_alpha)
            else:
                coeff_harmonics_cartesian_alpha=computes_coefficient_solid_harmonic_l_m_list_format(l_alpha,m_alpha)
            
            if (l_beta<=4):
                coeff_harmonics_cartesian_beta=conversion_spherical_harmonics_cartesian_homogeneous_polynoms(l_beta,m_beta)
            else:
                coeff_harmonics_cartesian_beta=computes_coefficient_solid_harmonic_l_m_list_format(l_beta,m_beta)
            
        elif (boolean_cartesian_F_shells==True):
            if (l_alpha<=4):
                coeff_harmonics_cartesian_alpha=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells(l_alpha,m_alpha)
            else:
                coeff_harmonics_cartesian_alpha=computes_coefficient_solid_harmonic_l_m_list_format(l_alpha,m_alpha)
                
            if (l_beta<=4):
                coeff_harmonics_cartesian_beta=conversion_spherical_harmonics_cartesian_homogeneous_polynoms_cartesian_F_shells(l_beta,m_beta)
            else:
                coeff_harmonics_cartesian_beta=computes_coefficient_solid_harmonic_l_m_list_format(l_beta,m_beta)
            
            
    #Number p(l_alpha,m_alpha) of homogeneous cartesian polynoms of degree l_alpha entering the definition of \chi_{\alpha}(.)^{pGTO}:
    #(=1 in the case of Gaussian (cartesian) d-shells)
    p_alpha=coeff_harmonics_cartesian_alpha[0]
    p_beta=coeff_harmonics_cartesian_beta[0]
    
    #[[n_1(l_alpha,m_alpha),m_1(l_alpha,m_alpha),t_1(l_alpha,m_alpha)],..[n_{p(l_alpha,m_alpha)}(l_alpha,m_alpha),m_{p(l_alpha,m_alpha)}(l_alpha,m_alpha),t_{p(l_alpha,m_alpha)}(l_alpha,m_alpha)]] :
    #tab_indexes_alpha[k][0] = n_k(l_alpha,m_alpha)
    #tab_indexes_alpha[k][1] = m_k(l_alpha,m_alpha)
    #tab_indexes_alpha[k][2] = t_k(l_alpha,m_alpha)
    tab_indexes_alpha=coeff_harmonics_cartesian_alpha[1]
    
    #Similarly :
    #tab_indexes_beta[q][0] = n_q(l_beta,m_beta)
    #tab_indexes_beta[q][1] = m_q(l_beta,m_beta)
    #tab_indexes_beta[q][2] = t_q(l_beta,m_beta)
    tab_indexes_beta=coeff_harmonics_cartesian_beta[1]
    
    #[w_1^(l_alpha,m_alpha),w_2^(l_alpha,m_alpha),..,w_{p(l_alpha,m_alpha)}(l_alpha,m_alpha)]
    #weights_spherical_harmonics_to_cartesian_alpha[k]= w_k^{l_alpha,m_alpha}
    weights_spherical_harmonics_to_cartesian_alpha=coeff_harmonics_cartesian_alpha[2]

    weights_spherical_harmonics_to_cartesian_beta=coeff_harmonics_cartesian_beta[2]
    
    ##############################################
    #Case where alpha==beta (we are computing the scalar product of a primitive GTO with itself
    #i.e. its norm,
    #or case when the two pGTOs are centered on the same atom :
    #==> the scalar product can be rewritten after translation change of variables :
    #int (r**(l_alpha+l_beta)*Y_{l_alpha}^{m_alpha}(\vec{r}/|r|)*Y_{l_beta}^{m_beta}(\vec{r}/|r|) * exp(-[zeta_alpha+zeta_beta]*r^2) d\vec{r})
    #i.e. integrating a sum of HOMOGENEOUS polynoms of degree (l_alpha+l_beta) times a Gaussian with 'zeta' exponent (zeta_alpha+zeta_beta)
    #It is the case naturally in particular in the case of the two same pGTOs
    if ((alpha==beta) or (correspondence_basis_pGTOs_atom_index[alpha]==correspondence_basis_pGTOs_atom_index[beta])):
        
        ###################
        #METHOD 1 (vectorized version)
        #Products of coefficients (arising in the conversion of spherical harmonics to their cartesian form)
        #times integral values ==> non-dimensional coefficients
        product_integrals_x_y_z=[[weights_spherical_harmonics_to_cartesian_alpha[k]*weights_spherical_harmonics_to_cartesian_beta[q]*gaussian_polynom_integral_all_R(tab_indexes_alpha[k][0]+tab_indexes_beta[q][0])*gaussian_polynom_integral_all_R(tab_indexes_alpha[k][1]+tab_indexes_beta[q][1])*gaussian_polynom_integral_all_R(tab_indexes_alpha[k][2]+tab_indexes_beta[q][2]) for q in range(p_beta)] for k in range(p_alpha)]
        
        #'np.sum(A)' of a matrix A sums all the coefficients of the matrix A 
        
        #==>  Result in Bohr^{3+l_alpha+l_beta}
        return (1/(math.sqrt(zeta_alpha+zeta_beta))**(3+l_alpha+l_beta))*np.sum(product_integrals_x_y_z)
        #'np.sum(A)' of a matrix A sums all the coefficients of the matrix A 
                

    #Case where the two pGTOs are NOT centered on the same atom :
    #Similar expression as for the multipole moments with respect to 'natural' overlap centers
    #implemented in multipoles2020_v1.compute_DMA_multipole_moment_natural_expansion_center_primitive_GTOs
    else:
        res_scalar_product_pGTOs=0
        
        R_alpha=position_nuclei_associated_basis_function_pGTO[alpha]
        R_beta=position_nuclei_associated_basis_function_pGTO[beta]
    
        vector_between_two_centers_alpha_beta=[R_alpha[i]-R_beta[i] for i in range(3)]
        
        K_alpha_beta=math.exp(-(conversion_bohr_angstrom**(-2))*(zeta_alpha*zeta_beta/(zeta_alpha+zeta_beta))*(np.linalg.norm(vector_between_two_centers_alpha_beta)**2))
        
        for k in range(p_alpha):
        
            for q in range(p_beta):
            
                for i_1 in range(tab_indexes_alpha[k][0]+1):
                    
                    for i_2 in range(tab_indexes_beta[q][0]+1):
                        
                        for j_1 in range(tab_indexes_alpha[k][1]+1):
                            
                            for j_2 in range(tab_indexes_beta[q][1]+1):
                                
                                for h_1 in range(tab_indexes_alpha[k][2]+1):
                                    
                                    for h_2 in range(tab_indexes_beta[q][2]+1):
                                        
                                        product_integrals=gaussian_polynom_integral_all_R(i_1+i_2)*gaussian_polynom_integral_all_R(j_1+j_2)*gaussian_polynom_integral_all_R(h_1+h_2)
                                      
                                        #Final dimension : length**3 * length**(l_{\alpha}+l_{\beta}-i_1-i_2-j_1-j_2-h_1-h_2) * length**(i_1+i_2+j_1+j_2+h_1+h_2)
                                        #=length **(3+l_{\alpha}+l_{\beta})
                                        
                                        #'coeff_W_binomial_expansion_polynoms' coefficient :
                                        #in Bohr**(l_alpha+l_beta-i_1-i_2-j_1-j_2-h_1-h_2)
                                        #zeta exponent already in bohr^{-2}
                                        res_scalar_product_pGTOs+=(K_alpha_beta/(math.sqrt((zeta_alpha+zeta_beta))**3))*weights_spherical_harmonics_to_cartesian_alpha[k]*weights_spherical_harmonics_to_cartesian_beta[q]*product_integrals*coeff_W_binomial_expansion_polynoms(l_alpha,l_beta,tab_indexes_alpha[k],tab_indexes_beta[q],i_1,j_1,i_2,j_2,h_1,h_2,R_alpha,R_beta,zeta_alpha,zeta_beta)/(math.sqrt(zeta_alpha+zeta_beta))**(i_1+i_2+j_1+j_2+h_1+h_2)
                                        #==> result in Bohr**(l_alpha+l_beta+3)
        return res_scalar_product_pGTOs          
##############################################################################
    
##############################################################################
def compute_normalization_coefficient_contracted_GTO(i,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO):
    
    #List of primitive GTOs playing in the definition of the contracted GTO of index "i"
    list_primitive_GTOs=correspondence_index_contracted_GTOs_primitive_GTOs[i]
    
    #Index of the 'mother' contracted shell of the cGTO of index "i"
    index_contracted_shell=tab_correspondence_index_contracted_GTOs_contracted_shells[i]
    
    #Information used by 'compute_scalar_product_two_primitive_GTOs()'
    correspondence_basis_pGTOs_atom_index=import_correspondence_pGTOs_atom_index(nb_primitive_GTOs,data.natom)

    ######################
    #Vectorized version :
    tab_contributions_denominator=[[compute_normalization_coefficient_primitive_GTO(alpha,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)*compute_normalization_coefficient_primitive_GTO(beta,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)*contraction_coefficients_pSHELLS_to_cSHELLS[int(index_contracted_shell)][int(tab_correspondence_index_primitive_GTOs_primitive_shells[alpha])]*contraction_coefficients_pSHELLS_to_cSHELLS[int(index_contracted_shell)][int(tab_correspondence_index_primitive_GTOs_primitive_shells[beta])]*compute_scalar_product_two_primitive_GTOs(alpha,beta,boolean_cartesian_D_shells,boolean_cartesian_F_shells,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO) for beta in list_primitive_GTOs] for alpha in list_primitive_GTOs]    
    
    #np.sum(X) : sum of all the coefficients of a matrix X
    return 1/math.sqrt(np.sum(tab_contributions_denominator))
##############################################################################
    
##############################################################################
def compute_scalar_product_two_contracted_GTOs(i,j,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO):
       
    #OLD (error, corrected 19/06/20) : error for <\chi_i^{cGTO} | \chi_i^{cGTO} >
    #==> same formula applied also for the case i=j
    #if (i==j):
    #    return compute_normalization_coefficient_contracted_GTO(i,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO)
    #else:
    
    list_primitive_GTOs_i=correspondence_index_contracted_GTOs_primitive_GTOs[i]

    list_primitive_GTOs_j=correspondence_index_contracted_GTOs_primitive_GTOs[j]

    #Index of the 'mother' contracted shell of the cGTO of index "i"
    index_contracted_shell_i=tab_correspondence_index_contracted_GTOs_contracted_shells[i]
    
    #Index of the 'mother' contracted shell of the cGTO of index "j"
    index_contracted_shell_j=tab_correspondence_index_contracted_GTOs_contracted_shells[j]
    
    normalization_coeff_cGTO_i=compute_normalization_coefficient_contracted_GTO(i,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO)
    normalization_coeff_cGTO_j=compute_normalization_coefficient_contracted_GTO(j,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,nb_primitive_GTOs,correspondence_index_contracted_GTOs_primitive_GTOs,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,contraction_coefficients_pSHELLS_to_cSHELLS,tab_correspondence_index_contracted_GTOs_contracted_shells,tab_correspondence_index_primitive_GTOs_primitive_shells,position_nuclei_associated_basis_function_pGTO)
    
    contributions_scalar_product=[[compute_normalization_coefficient_primitive_GTO(alpha,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)*compute_normalization_coefficient_primitive_GTO(beta,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs)*contraction_coefficients_pSHELLS_to_cSHELLS[int(index_contracted_shell_i)][int(tab_correspondence_index_primitive_GTOs_primitive_shells[alpha])]*contraction_coefficients_pSHELLS_to_cSHELLS[int(index_contracted_shell_j)][int(tab_correspondence_index_primitive_GTOs_primitive_shells[beta])]*compute_scalar_product_two_primitive_GTOs(alpha,beta,boolean_cartesian_D_shells,boolean_cartesian_F_shells,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO) for beta in list_primitive_GTOs_j] for alpha in list_primitive_GTOs_i]
    
    return normalization_coeff_cGTO_i*normalization_coeff_cGTO_j*np.sum(contributions_scalar_product)
##############################################################################
    
##############################################################################
#Computes the normalization constant "N" of a pGTO such that
#N * |r|^l * Y_l^m(\vec{r}/|r|) * exp(-zeta*r^2) is normalized
def compute_normalization_coefficient_primitive_GTO(alpha,data,boolean_cartesian_D_shells,boolean_cartesian_F_shells,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs):
            
    #'compute_scalar_product_two_primitive_GTOs()' : between two NON-normalized pGTOs
    #(i.e. simple |r|^l * Y_l^m(theta,phi) * exp(-zeta*r^2) )
     
    #result of 'compute_scalar_product_two_primitive_GTOs' is a length**(3+l_{\alpha}+l_{\beta})
    #(Bohr**(l_alpha+l_beta+3) in the chosen atomic units convention)
    return 1/math.sqrt(compute_scalar_product_two_primitive_GTOs(alpha,alpha,boolean_cartesian_D_shells,boolean_cartesian_F_shells,angular_momentum_numbers_primitive_GTOs,basis_set_exponents_primitive_GTOs,correspondence_basis_pGTOs_atom_index,position_nuclei_associated_basis_function_pGTO))
    #==>result of 'compute_normalization_coefficient_primitive_GTO()' is in Bohr^{-3/2-l_alpha}
##############################################################################
    
#End of functions to import data from output QM file 
################################################################################################
################################################################################################

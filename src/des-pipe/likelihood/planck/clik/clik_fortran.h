/*
 *  clik.h
 *  lowly_project
 *
 *  Created by Karim Benabed on 16/03/11.
 *  Copyright 2011 Institut d'Astrophysique de Paris. All rights reserved.
 *
 */

#include "clik.h"
#ifndef _clik_FORTRAN_
#define _clik_FORTRAN_

#ifdef ADD0US
void fortran_clik_init(long* pself,char* hdffilepath,int* fpathlen);
void fortran_clik_get_has_cl(long* pself, int* has_cl);
void fortran_clik_get_extra_parameter_names(long* pself, char* names, int* numnames);
void fortran_get_lmax(long *pself, int* lmax);
void fortran_clik_compute(long* pself, double* cl_and_pars, double* lkl);
void fortran_clik_cleanup(long* pself);
#elseif ADD2US
void fortran_clik_init__(long* pself,char* hdffilepath,int* fpathlen);
void fortran_clik_get_has_cl__(long* pself, int* has_cl);
void fortran_clik_get_extra_parameter_names__(long* pself, char* names, int* numnames);
void fortran_get_lmax__(long *pself, int* lmax);
void fortran_clik_compute__(long* pself, double* cl_and_pars, double* lkl);
void fortran_clik_cleanup__(long* pself);
#else
void fortran_clik_init_(long* pself,char* hdffilepath,int* fpathlen);
void fortran_clik_get_has_cl_(long* pself, int* has_cl);
void fortran_clik_get_extra_parameter_names_(long* pself, char* names, int* numnames);
void fortran_get_lmax_(long *pself, int* lmax);
void fortran_clik_compute_(long* pself, double* cl_and_pars, double* lkl);
void fortran_clik_cleanup_(long* pself);
#endif


#endif

/*
 *  aplowly.h
 *  lowly_project
 *
 *  Created by Karim Benabed on 11/03/09.
 *  Copyright 2009 Institut d'Astrophysique de Paris. All rights reserved.
 *
 */

#include "lowly_common.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_gamma.h>

#ifndef __APLOWLY__
#define __APLOWLY__

typedef struct {
  double *data;
  double *alpha,*beta,*wl,*C,*gl;
  double Nl,logdets2;
  int is_zero;
  size_t ndim;
  gsl_matrix_view  C_view_container;
  gsl_matrix  *C_view;
  gsl_vector_view  gl_view_container;
  gsl_vector *gl_view;
  
} approx_lowly;


approx_lowly* approx_lowly_init(size_t ndim,int* ell, int lmax,double *clhat, double *fsky, double *wl, double *V,double noisevar, long nside, error **err);
void free_approx_lowly(approx_lowly **self);
double approx_lowly_log_pdf(void *self, double *Cl, error **err);
double approx_zero_lowly_log_pdf(void* pelf,double* pars, error **err);


typedef struct {
  double *data;
  double *al,*bl,*Cm,*gl,*nl;
  int *ell;
  double logdets2;
  int isdiag;
  int nell;
} tease;

typedef struct {
  double *data;
  double *cl,*vl,*nl,*Cm,*gl;
  int *ell;
  double logdets2;
  int isdiag;
  int nell;
} gausslkl;

tease* tease_init(size_t nell, int* ell,double *al, double *bl, double *nl, int nlcst, double *Cm, int Cmisdiag, error **err);
void tease_free(void **ping);
double tease_log_pdf(void* ing, double* pars, error **err);

gausslkl* gausslkl_init(size_t nell, int* ell,double *al, double *bl, double *nl, int nlcst, double *Cm, int Cmisdiag, error **err);
void gausslkl_free(void **ping);
double gausslkl_log_pdf(void* ing, double* pars, error **err);

typedef struct {
  double *data;
  double *ldlh,*sig2,*wl;
  double Nl;
  size_t ndim;  
} log_normal;

log_normal* log_normal_init(size_t ndim,int* ell, double *clhat, double *fsky, double *wl, double noisevar, long nside, error **err);
void free_log_normal(log_normal **self);
double log_normal_log_pdf(void *self, double *Cl, error **err);


typedef struct {
  int np;
  int *nb;
  int *obe;
  int *obv;
  double *be;
  double *bv;
} binlkl;

binlkl* init_binlkl(int np, int* nb, double* be, double* bv, error **err);
double binlkl_lkl(void *vbon, double *pars, error **err);
void free_binlkl(void** pvbon);

#define ZEROLKL -1e30

#define lowly_no_correlation -20 + clowly_base

#endif

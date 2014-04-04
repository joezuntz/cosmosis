/*
 *  smica.h
 *  lowly_project
 *
 *  Created by Karim Benabed on 30/10/09.
 *  Copyright 2009 Institut d'Astrophysique de Paris. All rights reserved.
 *
 */


// some includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "pmc.h"
#include "lowly_common.h"

#ifndef _SMICA_
#define _SMICA_

/* typedef struct {
  int nq; // number of bins
  int nell; // lmax+1
  int m;  // number of channels
  double *wq  ; // number of modes
  double *bin ; // bins
  double *rq_hat; // empirical covariance matrices

  double *rq; // if not null contains sum_c R_q^c

  double *A;  // if not null contains mixing matrix 
  double *P;  // if not null contains power spectra
  int    *C;  // if not null contains component dimensions
  int    nc;  // number of component 
  

} smica;

smica* smica_init(int nq,int nell,int m, double *wq,double *bin,double *rq_hat, double *rq, double *A, double *P, int *C,int nc, error **err);

double smica_lkl(void* smica, double *pars, error **err);

void free_smica(smica** psmica);

//void compute_rq(smica* smic, double* cl,double *nuis,double *rq, error **err);
void bin_cl(smica* smic, double* cl , error **err);
void compute_rq(smica* smic, error **err);
double compute_smica_lkl(smica* smic, error **err);

*/
// Newer better interface

typedef void (update_rq)(void* data,double* locpars, double* rq, error **err);
typedef char _smicanames[256];
typedef double (smica_crit)(void* smic, error **err);

typedef struct {
  int ndim,m,nq;
  void* data;
  update_rq *update;
  posterior_log_free *free;
  _smicanames *names;
} SmicaComp;

typedef struct {
  int nq; 
  double *wq;
  int m;
  double *rq_hat, *rq_0, *rq;
  double *z_buf; // buffer
  int nc;
  SmicaComp **SC;
  int *offset_nc;
  smica_crit *crit;
  int crit_classic_init;
  double *gvec;
  double *crit_cor;
  double *eig_buf;
  int eig_lwork;
  double* eig_nrm;
  int *quad_mask;
  int quad_sn;
} Smica;

Smica* Smica_init(int nq, double *wq, int m, double *rq_hat, double* rq_0, int nc, SmicaComp **SC,error **err);
double smica_crit_classic(void *vsmic,error **err);
double smica_crit_gauss(void *vsmic, error **err);
double smica_crit_eig(void *vsmic, error **err);
double smica_crit_quad(void *vsmic,error **err);
double smica_crit_quadfid(void *vsmic,error **err);
double smica_crit_quad_mask(void *vsmic,error **err);

void smica_set_crit_gauss(Smica *smic, double *crit_cor,error **err);
void smica_set_crit_eig(Smica *smic, double *nrm, error **err);
void smica_set_crit_quad(Smica *smic,double *fid, int *mask, error **err);

void quad_mask_me(Smica *smic,double *delta);

double Smica_lkl(void* smic, double* pars, error **err);

void free_Smica(void **smic);

double kld(int n, double* rq_hat, double* rq, error **err);

// components
SmicaComp* alloc_SC(int ndim,int nq,int m,void* data, update_rq* update, posterior_log_free* pfree, error **err);
void SC_setnames(SmicaComp *Sc, char** names, error **err);

typedef struct {
  int Acst;
  double *AAt;
} SC_1D_data;
void comp_1D_AAt(int m, double *A, double *AAt, error **err);
SmicaComp* comp_1D_init(int q, int m, double *A, error **err);
void free_comp_1D(void** data);
void comp_1D_update(void* data,double* locpars, double* rq, error **err);

typedef struct {
  int nd,Acst;
  double *A,*Ab,*P;
} SC_nD_data;
SmicaComp* comp_nD_init(int q, int m, int nd, double *A, error **err);
void free_comp_nD(void** data);
void comp_nD_update(void* data,double* locpars, double* rq, error **err);

SmicaComp* comp_diag_init(int q, int m, error **err);
void comp_diag_update(void* data,double* locpars, double* rq, error **err);

typedef struct {
  double* locpars;
  SmicaComp *SCnD;
  int has_cl[6];
  int jmp_cl[6];
  int trois;
} SC_CMB_data;

SmicaComp * comp_CMB_init(int nbins, int mt,int mp, int *has_cl, double* Acprs, error **err);
void free_comp_CMB(void** data);
void comp_CMB_update(void* data,double* locpars, double* rq, error **err);

void printMat(double* A, int n, int m);

typedef struct {
  int ntpl;
  int *ngcal;
  double *gcaltpl;
  double *tpl;
  double *tpll;
  double *bins;
  int nell;
} SC_gcal;


SmicaComp* comp_gcal_log_init(int q,int m, int *ngcal, double* galtpl, int nell, double *bins,error **err);
SmicaComp* comp_gcal_lin_init(int q,int m, int *ngcal, double* galtpl, int nell, double *bins,error **err);
void comp_gcal_log_update(void* data,double* locpars, double* rq, error **err);
void comp_gcal_lin_update(void* data,double* locpars, double* rq, error **err);
void comp_gcal_free(void** data);

SmicaComp* comp_cst_init(int nq, int m, double *rq_0, error **err);
void comp_cst_update(void* data,double* locpars, double* rq, error **err);
void free_comp_cst(void** data);

void amp_diag_free(void** data);
void amp_diag_update(void* data,double* locpars, double* rq, error **err);
SmicaComp* amp_diag_init(int nq, int m, double* tmpl, error **err);

#define smica_uncomp           -102 + clowly_base

#endif

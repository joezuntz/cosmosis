/*
 *  lowly.h
 *  low-ell likelihood in the pixel domain 
 *
 *  JF Cardoso and Simon Prunet.  January 2008.
 *
 */

#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_gamma.h>

#ifdef HAS_PMC
#include "invgamma.h"
#endif
#include "lowly_common.h"
#include "chealpix.h"


#ifndef __LOWLY__
#define __LOWLY__

// llll = low ell log-likelihood
typedef struct  { 
  long npix_seen ;
  long npix,nside ;
  double * X_seen  ;
  double * F_seen  ;
  double * NoiseVar ;
  double * CosMat ;
  long lmax ;
  double * Wl  ;
  double * q_buf;
  double * X_seen_buf;
  double * F_seen_buf;
  double * covmat_buf;
  double * buffer;
  int isLog;
  int partial_nindex;
  int* partial_index;
  double *partial_cl;
  double v2;
  STRUCT_TIMER_DEFS;
} llll_struct ;


typedef llll_struct llll_partial_struct;


// create and initialize the structure holding the material to compute the likelihood
llll_struct* init_lowly(
                        double *X,          // map size npix 
                        unsigned char *Mask,       // mask size npix
                        double* Foreground, // if !=NULL a foreground map
                        long npix,          
                        char* ordering,     // ring or nested
                        double noisevar,    // microK/pixel
                        double v2,          // sigma^2 for foreground marginalization
                        double *Wl,         // wl Cl unit lmax+1 values
                        double lmax,        
                        error **err ); 

void free_lowly(llll_struct** self);

// return the low-ell log-likelihood for spectrum C_l
double lowly_lkl(
    void * llll_str ,
    double * C_l ,
    error **err
    ) ;

// return the low-ell log-likelihood for log(Cl)
double lowly_lkl_logcl(
    void * llll_str,
    double * logCl,
    error **err
    );


llll_struct * init_lowly_log(llll_struct* self,error **err);

// These are wrappers on top of lowly_lkl and lowly_lkl_logcl
// in case only a subset of Cls are varied as parameters
double lowly_partial_lkl(void *llll_str, double *cl, error **err);
double lowly_partial_lkl_logcl(void *llll_str, double *logcl, error **err);

void free_lowly_partial(llll_partial_struct **elf);
llll_partial_struct* init_lowly_partial( llll_struct *orig,int n,int* index, double* cl,error **err) ;
llll_partial_struct* init_lowly_partial_logcl( llll_struct *orig,int n,int* index, double* cl,error **err) ;
double lowly_lkl_with_noise(void *llll_str, double *cl, error **err);

// ------ internals.  Used for computation ---------------------

// assign and compute the npix_seen * npix_seen matrix of the cosines between seen pixels 
double * build_cosine_matrix (
    const long nside,
    const long * pixel_indices ,
    const long npix_seen ,
    const int ordering,
    error **err
    ) ;

// assign and compute \sum_l W_l C_l P_l(cos) (2l+1)/4pi
double * build_cov_matrix (
    double *orig,
    const double * CosMat ,
    const double * NoiseVar,
    const long npix_seen ,
    const long lmax ,
    const double * q,
   error **err
    ) ;

// assign and compute inverse covariance matrix, same inputs as above
double * build_invcov_matrix (
    double *orig,
    const double * cosmat,
    const double * noisevar,
    const long npix_seen,
    const long lmax,
    const double *q,
    error **err
    ) ;

// assign and compute W_l P_l(cos) (2l+1)/4pi
double * build_dcov_matrix (
    double *orig,
    const double *cosmat,
    const long npix_seen,
    const long lmax,
    const double wl,
    error **err
    );

// compute signal covariance for one value of z=cos(theta_ij)
double scalCov (
    const double z,
    const long lmax,
    const double *q
    ) ;

// compute signal covariance derivative (with respect to C_l) for one value of z=cos(theta_ij)
double scaldCov (
    const double z,
    const double wl, // = Wl[ell]
    const long ell
    ) ;

// tentative computation of signal covariance matrix using matrix (blas) algebra
double * matCov(
    const double * cosmat, 
    const long npix_seen, 
    const long lmax, 
    const double * q);

// Computes gradient of likelihood with respect to C_l
double * grad_lowly_lkl (
    double * orig,
    llll_struct * self,
    const double * covmat,
    const double * invcovmat,
    error **err
    ) ;

// Computes diagonal of fisher matrix, with respect to C_l
double * diag_fisher_lowly_lkl (
    double *orig,
    llll_struct * self,
    //const double * invcovmat,
    const double *q,
    error **err) ;

// Computes peak of posterior by pseudo-Newton iteration
double * pseudo_newton(
    llll_struct * self,
    double * cl,
    long niter,
    error **err
    ) ;

#ifdef HAS_PMC
typedef struct {
  invgamma * _ivg;
  double *data,*wl,*N;
  size_t ndim;
} invgamma_prop;

invgamma_prop* invgamma_prop_init(size_t ndim,int* ell,double* clhat, double* fsky, double* wl,double noisevar,long nside,error **err);
void free_invgamma_prop(invgamma_prop **pelf);
double invgamma_prop_log_pdf(invgamma_prop *self, double *x, error **err);
long simulate_invgamma_prop(pmc_simu *psim,invgamma_prop *proposal, gsl_rng * r,parabox *pb,error **err);
size_t get_importance_weight_invgamma_prop(pmc_simu *psim, const invgamma_prop *m,
                                           posterior_log_pdf_func *posterior_log_pdf, 
                                           void *extra, error **err);
void update_invgamma_prop(invgamma_prop *g,pmc_simu *psim,error **err);
invgamma_prop* invgamma_prop_init_new(size_t ndim,int* ell,double* clhat, double* fsky, double* wl,double* N,error **err);
#endif

void computeAlphaBeta(double *alpha, double *beta,int ell, double clhat, double fsky, double wl, double Nl);
double cl2xl(double cl, double wl, double Nl);
  
#define lowly_base           -10 + clowly_base
#define lowly_outofrange     -2 + lowly_base
#define lowly_negcl          -4 + lowly_base
#define lowly_fish_negative  -5 + lowly_base
#define lowly_spec_negative  -6 + lowly_base


#endif

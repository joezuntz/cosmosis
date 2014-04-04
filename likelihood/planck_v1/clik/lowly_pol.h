/*
 *  lowly.h
 *  low-ell likelihood in the pixel domain 
 *
 *  JF Cardoso and Simon Prunet.  January 2008.
 *
 */
#include "lowly_common.h"
#include "lowly.h"
#include "chealpix.h"

/*
#include "mkl_lapack.h"
#include "chealpix.h"
*/
#ifndef __LOWLY_POL__
#define __LOWLY_POL__


typedef struct {
  int npix,nside,npix_t,npix_p;
  int lmax;
  double *X,*trig_mat,*S,*X_temp;
  double *Ndiag;
  double *N;
  double *buffer;
  int *ell;
  int nell;
  double *Cl,*tCl;
  int offset_cl[6];
  STRUCT_TIMER_DEFS;
} plowly;

// ------ internals.  Used for computation ---------------------

// T and P mask are identical
void scalCovPol (double *covmat, 
                 const double cb, 
                 const double c2a, 
                 const double s2a, 
                 const double c2g,
                 const double s2g,
                 const long lmax,
                 const double *qtt,
                 const double *qee,
                 const double *qbb,
                 const double *qte,
                 const double *qtb,
                 const double *qeb);
double* build_trigo_matrices(const long npix_seen,
                             const double * xyz,
                             error **err);
double* build_trig_mat_plist(const long nside,
                             const long * pixel_indices,
                             const long npix_seen,
                             const int ordering,
                             error **err);
double* build_cov_matrix_pol (double *orig,
                      const double *trig_mat,
                      const double *noisevar,
                      const long npix_seen,
                      const long lmax,
                      const double * qtt,
                      const double * qee,
                      const double * qbb,
                      const double * qte,
                      const double * qtb,
                      const double * qeb,
                      error **err);

// T and P mask are different
void scalCovCross(double *covcross,
                  const double cb,
                  const double c2g,
                  const double s2g,
                  const long lmax,
                  const double *qte,
                  const double *qtb);
void scalCovQU(double *covQU,
               const double cb,
               const double c2a,
               const double s2a,
               const double c2g,
               const double s2g,
               const long lmax,
               const double *qee,
               const double *qbb,
               const double *qeb);
double* build_trigo_matrices_general (const long npix_temp,
                                      const long npix_pol,
                                      const double * xyz_temp,
                                      const double * xyz_pol,
                                      error **err);
double* build_trig_mat_general_plist(const long nside,
                                     const long * pixel_temp,
                                     const long npix_temp,
                                     const long * pixel_pol,
                                     const long npix_pol,
                                     const int ordering,
                                     error **err);
double* build_cov_matrix_pol_general (double *orig,
                                      const double *trig_mat,
                                      const double *noisevar,
                                      const long npix_temp,
                                      const long npix_pol,
                                      const long lmax,
                                      const double * qtt,
                                      const double * qee,
                                      const double * qbb,
                                      const double * qte,
                                      const double * qtb,
                                      const double * qeb,
                                      error **err);


// both cases
double lowly_XtRX_lkl (double *S,double *X, double* buffer,long npix_tot,
                    error **err);

plowly *init_plowly(int nside,char* ordering, 
                    unsigned char *mask_T, unsigned char *mask_P, 
                    double *mapT,double *mapQ, double *mapU,
                    double *Ndiag, double* N,int reduced,
                    long lmax,int nell, int *ell,
                    double *Cl,int *has_cl,
                    error **err);

double* plowly_build_S(double* orig, plowly *self, double* pars,error **err);
double plowly_lkl(void* pself, double* pars, error **err);
void free_plowly(void **pself);
#endif

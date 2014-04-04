#include "pmc.h"
#ifndef _EGFS_
#define _EGFS_


typedef char egfs_parstr[256];
typedef struct {
  int instance_id;
  int nkv;
  int ndf;
  int nfr;
  int np;
  int nell,lmin,lmax;
  double *buf,*brq, *bdrq;
  egfs_parstr *keys,*values;
  int nmodels;
  int model[10];
  int cid[30];
  } egfs;
  
egfs *egfs_init(int nvar, char **keyvars, int ndefaults, char** keys, char** values, 
                int lmin, int lmax, double* cib_clustering,double *patchy_ksz, 
                double *homogenous_ksz,double *tsz,
                double *cib_decor_clust, double * cib_decor_poisson, error **err);
                
void egfs_compute(egfs *self, double *pars, double *rq, double *drq, error **err);
void egfs_free(void **pelf);

#endif

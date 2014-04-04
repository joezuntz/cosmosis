/*
 *  clik_fortran.h
 *  lowly_project
 *
 *  Created by Karim Benabed on 16/03/11.
 *  Copyright 2011 Institut d'Astrophysique de Paris. All rights reserved.
 *
 */

#include "clik.h"
#ifndef _clik_FORTRAN_
#define _clik_FORTRAN_


// in each of the following functions, if err is set to NULL the code exit as soon as it encounters an error (with a call to exit, bad for you if you're using mpi...)

// initialize the planck likelihood from an hdf file
#ifdef ADD0US
void fortran_clik_init(long* pself,char* hdffilepath,int* fpathlen) {
#elseif ADD2US
void fortran_clik_init__(long* pself,char* hdffilepath,int* fpathlen) {
#else
void fortran_clik_init_(long* pself,char* hdffilepath,int* fpathlen) {
#endif
  clik_object* self;
  char *tmpchain;
  int i,ln;
  ln = *fpathlen;
  ln ++;

  tmpchain = malloc(sizeof(char)*(ln));
  for(i=0;i<ln-1;i++) {
    tmpchain[i] = hdffilepath[i];
  }
  tmpchain[ln-1]='\0';
  self = clik_init(tmpchain,NULL);
  *pself = (long) self; 
  free(tmpchain);
}


// retrieve the list of cls as a list of flags 
// order is  TT EE BB TE TB EB
// for example for a likelihood acting on TT 
// has_cl = 1 0 0 0 0 0
#ifdef ADD0US
void fortran_clik_get_has_cl(long* pself, int* has_cl) {
#elseif ADD2US
void fortran_clik_get_has_cl__(long* pself, int* has_cl) {
#else
void fortran_clik_get_has_cl_(long* pself, int* has_cl) {
#endif
  clik_object* self;
  self = (clik_object*) *pself;
  clik_get_has_cl(self,has_cl,NULL);

}

// retrieve the number of extra parameters (needed to allocate
// on fortran side)

#ifdef ADD0US
void fortran_clik_get_extra_parameter_number(long* pself, int* numnames) {
#elseif ADD2US
void fortran_clik_get_extra_parameter_number__(long* pself, int* numnames) {
#else
void fortran_clik_get_extra_parameter_number_(long* pself, int* numnames) {
#endif

  clik_object* self;
  parname* names;
  self = (clik_object*) *pself;
  *numnames = clik_get_extra_parameter_names(self,&names,NULL);

}

// retrieve the names of extra parameters
#ifdef ADD0US
void fortran_clik_get_extra_parameter_names(long* pself, char* names) {
#elseif ADD2US
void fortran_clik_get_extra_parameter_names__(long* pself, char* names) {
#else
void fortran_clik_get_extra_parameter_names_(long* pself, char* names) {
#endif  
  clik_object* self;
  int i,ii;
  int numnames;
  parname *pnames;
  self = (clik_object*) *pself;
  numnames = clik_get_extra_parameter_names(self,&pnames,NULL);

  // Copy parameter names in fortran character array
  for (i=0;i<numnames;i++) {
    memset(&names[i*_pn_size],' ',sizeof(char)*256);
    sprintf(&names[i*_pn_size],"%s",pnames[i]);
  }
  // Get rid of pnames
  free(pnames);

}

#ifdef ADD0US
void fortran_clik_get_version(long* pself, char* names) {
#elseif ADD2US
void fortran_clik_get_version__(long* pself, char* names) {
#else
void fortran_clik_get_version_(long* pself, char* names) {
#endif  
  clik_object* self;
  char *version;
  
  self = (clik_object*) *pself;
  
  version = clik_get_version(self,NULL);
  sprintf(names,"%s",version);
  names[strlen(version)]=' ';
  free(version);

}

// retrieve the lmax for each power spectrum
// -1 --> no cl
// order is TT EE BB TE TB EB
// for example for a likelihood acting ont TT EE TE with same lmax 2000 
// lmax = 2000 2000 -1 2000 -1 -1 -1
#ifdef ADD0US
void fortran_get_lmax(long *pself, int* lmax) {
#elseif ADD2US
void fortran_get_lmax__(long *pself, int* lmax) {
#else
void fortran_get_lmax_(long *pself, int* lmax) {
#endif
  clik_object* self;
  self = (clik_object *)*pself;
  clik_get_lmax(self,lmax,NULL);

}

// compute a log likelyhood value
// cl_and_pars is order this way
// first the powerspectra from l=0 to l=lmax[cli] (included) in the order 
// TT EE BB TE TB EB. Only the one where lmax[cli]!=-1 have to be included
// then the extra parameters in the order given by clik_get_extra_parameters.
// The power spectra are in microK^2
// for example, for a likelihood acting on TT, EE and TE with 3 extra parameters 
// will expect an array ordered this way
// C_0^TT ... C_lmax[0]^TT C_0^EE ... C_lmax[1]^EE C_0^TE ... C_lmax[3]^T3 extrapar1 extrapar2 extrapar3


#ifdef ADD0US
void fortran_clik_compute(long* pself, double* cl_and_pars, double* lkl) {
#elseif ADD2US
void fortran_clik_compute__(long* pself, double* cl_and_pars, double* lkl) {
#else
void fortran_clik_compute_(long* pself, double* cl_and_pars, double* lkl) {
#endif
  clik_object* self;
  self = (clik_object*) *pself;
  error *_err;
  error **err;
  _err = NULL;
  err = &_err;
  *lkl=clik_compute(self,cl_and_pars,err);
  if (isError(*err)) {
    printError(stderr,*err);
    purgeError(err);
    *lkl = nan("");
  }
}

#ifdef ADD0US
void fortran_clik_compute_with_error(long* pself, double* cl_and_pars, double* lkl,int *ier) {
#elseif ADD2US
void fortran_clik_compute_with_error__(long* pself, double* cl_and_pars, double* lkl,int *ier) {
#else
void fortran_clik_compute_with_error_(long* pself, double* cl_and_pars, double* lkl,int *ier) {
#endif
  clik_object* self;
  self = (clik_object*) *pself;
  error *_err;
  error **err;
  _err = NULL;
  err = &_err;
  *ier = 1;
  *lkl=clik_compute(self,cl_and_pars,err);
  if (isError(*err)) {
    printError(stderr,*err);
    purgeError(err);
    *lkl = nan("");
    *ier = 1;
  }
  *ier = 0;
}

// cleanup
#ifdef ADD0US
void fortran_clik_cleanup(long* pself) {
#elseif ADD2US
void fortran_clik_cleanup__(long* pself) {
#else
void fortran_clik_cleanup_(long* pself) {
#endif
  clik_object* self;
  self = (clik_object*) *pself;
  clik_cleanup(&self);
  *pself = (long) self;

}

#ifdef CLIK_LENSING

#ifdef ADD0US
void fortran_clik_lensing_init(long *pself,char *fpath, int* fpathlen) {
#elseif ADD2US
void fortran_clik_lensing_init__(long *pself,char *fpath, int* fpathlen) {
#else
void fortran_clik_lensing_init_(long *pself,char *fpath, int* fpathlen) {
#endif
  clik_lensing_object* self;
  fpath[*fpathlen]='\0';
  self = clik_lensing_init(fpath,NULL);
  *pself = (long) self; 
}

#ifdef ADD0US
void fortran_clik_try_lensing(int *isl,char *fpath, int* fpathlen) {
#elseif ADD2US
void fortran_clik_try_lensing__(int *isl,char *fpath, int* fpathlen) {
#else
void fortran_clik_try_lensing_(int *isl,char *fpath, int* fpathlen) {
#endif
  fpath[*fpathlen]='\0';
  *isl = clik_try_lensing(fpath,NULL);
  
}

#ifdef ADD0US
void fortran_clik_lensing_get_lmax(int* lmax,long* *pself) {
#elseif ADD2US
void fortran_clik_lensing_get_lmax__(int* lmax,long* *pself) {
#else
void fortran_clik_lensing_get_lmax_(int* lmax,long* *pself) {
#endif
  clik_lensing_object* self;
  self = (clik_lensing_object*) *pself;
  *lmax = clik_lensing_get_lmax(self,NULL);
}

#ifdef ADD0US
void fortran_clik_lensing_compute(double* res, long *pself, double *pars) {
#elseif ADD2US
void fortran_clik_lensing_compute__(double* res, long *pself, double *pars) {
#else
void fortran_clik_lensing_compute_(double* res, long *pself, double *pars) {
#endif
  clik_lensing_object* self;
    
  self = (clik_lensing_object*) *pself;
  error *_err;
  error **err;
  _err = NULL;
  err = &_err;
  
  *res=clik_lensing_compute(self,pars,err);
  
  if (isError(*err)) {
    printError(stderr,*err);
    purgeError(err);
    *res = nan("");
  }
  
}

// retrieve the names of extra parameters
#ifdef ADD0US
void fortran_clik_lensing_get_extra_parameter_names(long* pself, char* names) {
#elseif ADD2US
void fortran_clik_lensing_get_extra_parameter_names__(long* pself, char* names) {
#else
void fortran_clik_lensing_get_extra_parameter_names_(long* pself, char* names) {
#endif  
  clik_lensing_object* self;
  int i,ii;
  int numnames;
  parname *pnames;
  self = (clik_lensing_object*) *pself;
  numnames = clik_lensing_get_extra_parameter_names(self,&pnames,NULL);

  // Copy parameter names in fortran character array
  for (i=0;i<numnames;i++) {
    memset(&names[i*_pn_size],' ',sizeof(char)*256);
    sprintf(&names[i*_pn_size],"%s",pnames[i]);
  }
  // Get rid of pnames
  free(pnames);
}

#ifdef ADD0US
void fortran_clik_lensing_get_extra_parameter_number(long* pself, int* numnames) {
#elseif ADD2US
void fortran_clik_lensing_get_extra_parameter_number__(long* pself, int* numnames) {
#else
void fortran_clik_lensing_get_extra_parameter_number_(long* pself, int* numnames) {
#endif

  clik_object* self;
  parname* names;
  self = (clik_lensing_object*) *pself;
  *numnames = clik_lensing_get_extra_parameter_names(self,&names,NULL);

}

#ifdef ADD0US
void fortran_clik_lensing_cleanup(long* pself) {
#elseif ADD2US
void fortran_clik_lensing_cleanup__(long* pself) {
#else
void fortran_clik_lensing_cleanup_(long* pself) {
#endif
  clik_lensing_object* self;
  self = (clik_lensing_object*) *pself;
  clik_lensing_cleanup(&self);
  *pself = (long) self;
}

#ifdef ADD0US
void fortran_clik_lensing_cltt_fid(long* pself, double *cltt) {
#elseif ADD2US
void fortran_clik_lensing_cltt_fid__(long* pself, double *cltt) {
#else
void fortran_clik_lensing_cltt_fid_(long* pself, double *cltt) {
#endif
  double *tmp;
  int lmax;
  clik_lensing_object* self;
  self = (clik_lensing_object*) *pself;
  tmp = clik_lensing_cltt_fid(self, NULL);
  lmax = clik_lensing_get_lmax(self,NULL);
  memcpy(cltt,tmp,sizeof(double)*(lmax+1));
  free(tmp);
}

#ifdef ADD0US
void fortran_clik_lensing_clpp_fid(long* pself, double *cltt) {
#elseif ADD2US
void fortran_clik_lensing_clpp_fid__(long* pself, double *cltt) {
#else
void fortran_clik_lensing_clpp_fid_(long* pself, double *cltt) {
#endif
  double *tmp;
  int lmax;
  clik_lensing_object* self;
  self = (clik_lensing_object*) *pself;
  tmp = clik_lensing_clpp_fid(self, NULL);
  lmax = clik_lensing_get_lmax(self,NULL);
  memcpy(cltt,tmp,sizeof(double)*(lmax+1));
  free(tmp);
}

#endif

#endif

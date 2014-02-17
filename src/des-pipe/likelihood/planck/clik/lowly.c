#ifdef __PLANCK__
#include "HL2_likely/target/lowly.h"
#else
#include "lowly.h"
#endif

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "scalCov"
/* Pixel covariance for a given separation z=cos(\theta_{ij}) */

double scalCov (const double z, const long lmax, const double *q) {
  
  double cov, plm2, plm1, pl;
  long l;
  double dl;
  // Initialize recurrence, compute l=0 and l=1 contrib
  plm2=1.0;
  plm1=z;
  cov=1.0*plm2*q[0] + 3.0*plm1*q[1];
  
  for (l=2;l<=lmax;l++) {
    dl = (double)l;
    pl = 2.0*z*plm1 - plm2 - (z*plm1-plm2) / dl ; 
    cov += (2.0*dl+1.0)*pl*q[l];
    plm2=plm1;
    plm1=pl;
  }
  cov /= 4.0*M_PI;
  return(cov);
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "scaldCov"
/* Derivative of pixel covariance matrix for a given separation z=cos(\theta_{ij}) */

double scaldCov (const double z, const double wl, const long ell) {

    double dcov, plm2, plm1, pl;
    long l;
    double dl;
    // Same recurrence as before, but up to ell
    plm2=1.0;
    plm1=z;
    if (ell==0) return(wl/4.0/M_PI);
    if (ell==1) return(3.0*wl*z/4.0/M_PI);
    for (l=2;l<=ell;l++) {
	dl = (double)l;
	pl = 2.0*z*plm1 - plm2 - (z*plm1-plm2)/dl;
	plm2=plm1;
	plm1=pl;
    }
    dcov = (2.0*dl+1.0)*wl*pl/4.0/M_PI;
    return(dcov);

}


/*****************************************************************************************/

#undef __FUNC__
#define __FUNC__ "build_cosine_matrix"
double * build_cosine_matrix(const long nside,
                             const long * pixel_indices,
                             const long npix_seen, 
                             const int ordering,error **err) {
  
  long i,j,d;
  double * posvec;
  double * cosmat;
  double z;
  
  cosmat = (double*) malloc_err(npix_seen*npix_seen*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);

  posvec = lowly_get_posvec(nside,pixel_indices,npix_seen,ordering,err);
  forwardError(*err,__LINE__,NULL);
	
  // Fill the cosines matrix
  for (i=0;i<npix_seen;i++) {
    for (j=0;j<=i;j++) {
	    // Compute cosine by dot product
	    z=0.0;
	    for (d=0;d<3;d++) z += posvec[3*i+d]*posvec[3*j+d];
	    cosmat[i*npix_seen+j] = z;
	    cosmat[j*npix_seen+i] = z;
    }
  }
  free(posvec);
  return(cosmat);
  
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "build_cov_matrix"

double * build_cov_matrix (
                           double *orig,
                           const double *cosmat,
                           const double *noisevar,
                           const long npix_seen,
                           const long lmax,
                           const double * q,
                           error **err) {
  
  long i,j;
  double * covmat;
  double z, cov;
  
  
  MALLOC_IF_NEEDED(covmat,orig,npix_seen*npix_seen*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);

  for (i=0;i<npix_seen;i++) {
    for (j=0;j<=i;j++) {
	    z = cosmat[i*npix_seen+j];
	    cov = scalCov(z,lmax,q);
	    if (i==j) cov += noisevar[i]; // add noise
	    covmat[i*npix_seen+j]=cov;
	    covmat[j*npix_seen+i]=cov;
    }
  } 
  
  return(covmat);
  
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "build_dcov_matrix"

double * build_dcov_matrix (
    double *orig,
    const double *cosmat,
    const long npix_seen,
    const long l,
    const double wl, // = Wl[l]
    error **err) {

  long i,j;
  double *dcovmat;
  double z, dcov;
  
  
  MALLOC_IF_NEEDED(dcovmat,orig,npix_seen*npix_seen*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);

    for (i=0;i<npix_seen;i++) {
	for (j=0;j<=i;j++) {
	    z = cosmat[i*npix_seen+j];
	    dcov = scaldCov(z,wl,l);
	    dcovmat[i*npix_seen+j]=dcov;
	    dcovmat[j*npix_seen+i]=dcov;
	    }
	}
    
    return(dcovmat);

}

#undef __FUNC__
#define __FUNC__ "build_invcov_matrix"

/*****************************************************************************************/
double * build_invcov_matrix (
    double *orig,
    const double *cosmat,
    const double *noisevar,
    const long npix_seen,
    const long lmax,
    const double *q,
    error **err) {

  int int_npix_seen, info;
  double *invcovmat,*buffer,*result;
  double sum, diff, res;
  char uplo, trans;
  const double one=1.0, zero=0.0;
  long i,j;
  
  invcovmat = build_cov_matrix(orig,cosmat,noisevar,npix_seen,lmax,q,err);
  forwardError(*err,__LINE__,NULL);

  
  // Now invert this bouzin
  uplo='L';
  int_npix_seen = (int)npix_seen;
  dpotrf(&uplo,&int_npix_seen,invcovmat,&int_npix_seen,&info);
  testErrorRetVA(info!=0,lowly_chol,"Could not cholesky decompose using dpotrf (status %d)",*err,__LINE__,NULL,info);

  dpotri(&uplo,&int_npix_seen,invcovmat,&int_npix_seen,&info);
  testErrorRetVA(info!=0,lowly_chol,"Could not invert using dpotri (status %d)",*err,__LINE__,NULL,info);
  
  // Now need to symmetrize this (only one triangle is valid inverse, which one ????)
  for (i=0;i<npix_seen;i++) {
    for (j=0;j<i;j++) {
	    invcovmat[i*npix_seen+j]=invcovmat[j*npix_seen+i];
    }
  }
  
  return(invcovmat);
}
    


/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "grad_lowly_lkl"

double * grad_lowly_lkl (
    double *orig,
    llll_struct * self,
    const double * covmat,
    const double * invcovmat,
    error **err) {

    long l,i,j;
    double zero,one,half;
    char uplo,side;
    int int_one, int_npix_seen;
    double * grad, *cmx, *deriv, *dcmx;
    double dprod, trace;


  MALLOC_IF_NEEDED(grad,orig,(self->lmax+1)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  memset((void*)grad,0,(self->lmax+1)*sizeof(double));
  
  cmx=(double*)calloc_err(self->npix_seen,sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  
  dcmx=(double*)calloc_err(self->npix_seen,sizeof(double),err);
  forwardError(*err,__LINE__,NULL);

  // Compute C^-1.x
  uplo='L';
  int_one=1;
  one=1.0;
  zero=0.0;
  int_npix_seen=(int)self->npix_seen;
  
  dsymv(&uplo,&int_npix_seen,&one,invcovmat,&int_npix_seen,self->X_seen,&int_one,&zero,cmx,&int_one);
  
  deriv=NULL;
  for (l=0;l<=self->lmax;l++) {
    //fprintf(stderr,"l = %ld\n",l);
    // Compute dC/dC_l
    deriv=build_dcov_matrix(deriv,self->CosMat,self->npix_seen,l,self->Wl[l],err);
    forwardError(*err,__LINE__,NULL);
    // Compute dC/dC_l.(C^-1.x)
    dsymv(&uplo,&int_npix_seen,&one,deriv,&int_npix_seen,cmx,&int_one,&zero,dcmx,&int_one);
    // Compute (C^-1.x)^T.dC/dC_l.(C^-1.x)
    dprod=0.0;
    for (i=0;i<self->npix_seen;i++) {
	    dprod += cmx[i]*dcmx[i];
    }
    // Compute Tr[C^-1.dC/dC_l]
    trace=0.0;
    for (i=0;i<self->npix_seen*self->npix_seen;i++) {
	    trace += invcovmat[i]*deriv[i];
    }
    grad[l]=0.5*(dprod-trace);
  }
  
  free(deriv);
  free(cmx);
  free(dcmx);
  
  return(grad);
}


/*****************************************************************************************/
// This function computes the diagonal part of the Fisher matrix
// stored in a vector
#undef __FUNC__
#define __FUNC__ "diag_fisher_lowly_lkl"

double * diag_fisher_lowly_lkl (
    double *orig,
    llll_struct * self,
    //const double * invcovmat,
    const double *q,
    error **err) {

  long l,i,j;
  double zero,one,half;
  char uplo,side;
  int int_one, int_npix_seen;
  double * diag_fisher, *deriv, *buff, *buff2;
  double fsky, mean_noisevar, N_l;
  
  MALLOC_IF_NEEDED(diag_fisher,orig,(self->lmax+1)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  memset((void*)diag_fisher,0,(self->lmax+1)*sizeof(double));
  
  
  // Compute diagonal approximation of inverse fisher
  
  fsky=(double)self->npix_seen/(double)self->npix;
  mean_noisevar=0.0;
  for (i=0;i<self->npix_seen;i++) {
    mean_noisevar += self->NoiseVar[i];
  }
  mean_noisevar /= (double)self->npix_seen;
  N_l = lowly_computeNl(mean_noisevar,self->nside);
  
  for (l=0;l<=self->lmax;l++) {
    testErrorRetVA(q[l] + N_l <= 0.0,lowly_spec_negative,"Negative power spectrum: cannot compute approximate Fisher (%d : %g %g)",*err,__LINE__,NULL,l,q[l],N_l);
    diag_fisher[l] = ((double)l+0.5)*fsky*(self->Wl[l]*self->Wl[l])/((q[l]+N_l)*(q[l]+N_l));
  } 
	
  return(diag_fisher);
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "pseudo_newton"

double * pseudo_newton(llll_struct * self,
                       double * cl,
                       long niter,
                       error **err) {
  
  double *q, *covmat, *invcovmat;
  double *grad, *fish;
  long l, iter;
  double cl_old;

  q = malloc_err((self->lmax+1)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  
  
  covmat=NULL;
  invcovmat=NULL;
  grad=NULL,
  fish=NULL;
  fprintf(stdout,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
  for (iter=0;iter<niter;iter++) {    
    // fprintf(stderr,"In newton, iteration = %ld\n",iter);
    // Initialize q with updated value of Cls
    for (l=0;l<=self->lmax;l++) {
	    q[l] = cl[l]*self->Wl[l];
      //fprintf(stderr,"%d->%g %g %g\n",l,cl[l],self->Wl[l],q[l]);
    }
    // Compute covariance and inverse covariance
    //fprintf(stderr,"In newton, computing covariance\n");
    covmat = build_cov_matrix(covmat,self->CosMat,self->NoiseVar,self->npix_seen,self->lmax,q,err);
    forwardError(*err,__LINE__,NULL);
    //fprintf(stderr,"In newton, computing inverse covariance\n");
    invcovmat = build_invcov_matrix(invcovmat,self->CosMat,self->NoiseVar,self->npix_seen,self->lmax,q,err);
    forwardError(*err,__LINE__,NULL);
    //fprintf(stderr,"In newton, computing gradient\n");
    grad = grad_lowly_lkl(grad,self,covmat,invcovmat,err);
    forwardError(*err,__LINE__,NULL);
    //fprintf(stderr,"In newton, computing Fisher\n");
    //fish = diag_fisher_lowly_lkl(fish,self,invcovmat,err);
    fish = diag_fisher_lowly_lkl(fish,self,q,err);
    forwardError(*err,__LINE__,NULL);
    
    for (l=0;l<=self->lmax;l++) {
      testErrorRetVA(fish[l] <= 0.0,lowly_fish_negative,"Fisher matrix is negative or zero at %d : %g",*err,__LINE__,NULL,l,fish[l]);
      cl_old = cl[l];
      cl[l] += grad[l]/fish[l];
      if (cl[l] < 0.0) {
        cl[l]=cl_old/100.0; // Make sure it does not get negative
      }
      //fprintf(stderr,"%d->%g %g %g\n",l,cl[l],grad[l],fish[l]);
      fprintf(stdout,"l = %ld -> cl = %g\n",l,cl[l]);
    }
  }
  fprintf(stdout,"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n");
  free(grad);
  free(fish);
  free(covmat);
  free(invcovmat);
  return(cl);
  
}
#undef __FUNC__

#define __FUNC__ "init_lowly"

llll_struct* init_lowly(double *X, unsigned char *Mask, double* Foreground,  long npix, char* ordering,double noisevar, double v2,  double *Wl, double lmax,error **err )  {
  // Create and initialize the structure holding the material to compute the likelihood
  
  long * pixel_indices;
  long nside,npix_seen;
  long ipix, jpix ; // indices to pixels
  int ordering_flag;

  ordering_flag = lowly_which_order(ordering,err);
  forwardError(*err,__LINE__,NULL);
  
  // Create the structure
  llll_struct * self;
  self = malloc_err(sizeof(llll_struct),err);
  forwardError(*err,__LINE__,NULL);
  SET_PRINT_STAT(self);
  
  // 
  self->lmax = lmax ;
  
  self->npix = npix;
  npix_seen = npix;
  pixel_indices = lowly_build_pixel_list(Mask, &npix_seen, err);
  forwardError(*err,__LINE__,NULL);
  self->npix_seen = npix_seen;
    
  // Copy X build noisevar 
  self->X_seen=malloc_err(sizeof(double)*npix_seen,err);
  forwardError(*err,__LINE__,NULL);
  
  self->NoiseVar=malloc_err(sizeof(double)*npix_seen,err);
  forwardError(*err,__LINE__,NULL);
  
  for(ipix = 0; ipix < npix_seen; ipix++) {
    long lpix;
    lpix = pixel_indices[ipix];
    self->X_seen[ipix]   = X[lpix];
	  self->NoiseVar[ipix] = noisevar;
  }
  
  // Deal with foreground
  self->v2 = v2;
  self->F_seen = NULL;
  if (Foreground!=NULL) {
    self->F_seen = malloc_err(sizeof(double)*self->npix_seen,err);
    forwardError(*err,__LINE__,NULL);
    for(ipix = 0; ipix < npix_seen; ipix++) {
      long lpix;
      lpix = pixel_indices[ipix];
      self->X_seen[ipix] = Foreground[lpix];
    }
  }
  
  // Fill in CosMat
  self->nside = (long)sqrt(npix/12);
  self->CosMat = build_cosine_matrix(self->nside,pixel_indices,
                                     self->npix_seen,ordering_flag,err);
  forwardError(*err,__LINE__,NULL);
  
  
  // Now, for the spectral window
  self->Wl = malloc_err(sizeof(double)*(self->lmax+1),err);
  forwardError(*err,__LINE__,NULL);
  
  memcpy((void*)self->Wl,(void*) Wl, sizeof(double)*(self->lmax+1) );
 
  //allocate buffer for computations
  self->buffer = malloc_err(sizeof(double)*(self->lmax+1+self->npix_seen*(2+self->npix_seen)),err);
  forwardError(*err,__LINE__,NULL);
  self->q_buf=self->buffer;
  self->X_seen_buf=self->q_buf+self->lmax+1;
  self->F_seen_buf=self->X_seen_buf+self->npix_seen;  
  self->covmat_buf=self->F_seen_buf+self->npix_seen;  
  
  self->partial_nindex=0;
  self->partial_index=NULL;
  self->partial_cl=NULL;
  
  self->isLog=0;
  
  self->time_build = 0;
  self->time_tot = 0;
  self->time_chol = 0;
  self->n_tests = 0;
  
  return self;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "init_lowly_logcl"
llll_struct * init_lowly_log(llll_struct* self,error **err) {
  self->isLog=1;
  return self;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "init_lowly_partial"

llll_partial_struct* init_lowly_partial( llll_struct *orig,int n,int* index, double* cl,error **err)  {
  
  llll_partial_struct *self;
  int i;
  self=orig;
  self->partial_nindex=n;
  for(i=0;i<n;i++) {
    testErrorRetVA(index[i]>orig->lmax+1,lowly_outofrange,"Index Out of Range for partial lowly (%d : %d > %d)",*err,__LINE__,NULL,i,index[i],orig->lmax+1);
  }
  self->partial_index=malloc_err(sizeof(int)*n,err);
  forwardError(*err,__LINE__,NULL);
  memcpy(self->partial_index,index,n*sizeof(int));
  self->partial_cl=malloc_err(sizeof(double)*(orig->lmax+1),err);
  forwardError(*err,__LINE__,NULL);
  memcpy(self->partial_cl,cl,(orig->lmax+1)*sizeof(double));
  
  return self;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "init_lowly_partial_logcl"
/* receives cl as input, stores internally logcl in the structure */

llll_partial_struct* init_lowly_partial_logcl( llll_struct *orig,int n,int* index, double* cl,error **err)  {
  
  llll_partial_struct *self;
  int i;
  double *tmpbuf;
  
  tmpbuf = (double*)malloc_err((orig->lmax+1)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  /* Test for positivity of Cls */
  for (i=0;i<orig->lmax+1;i++) {
    testErrorRetVA(cl[i] <= 0.0,lowly_negcl,"Negative Cls (%d : %g)",*err,__LINE__,NULL,i,cl[i]);
    tmpbuf[i]=log(cl[i]);
  }
  
  self=init_lowly_partial(orig,n,index,tmpbuf,err);
  if (self == NULL ) {
    free(tmpbuf);
    forwardError(*err,__LINE__,NULL);
  }
  free(tmpbuf);
  self->isLog=1;
  
  return self;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "lowly_lkl"

double lowly_lkl(void *llll_str, double *cl, error **err) {
  
  llll_struct * self;
  double * covmat, *q, *X_seen,*F_seen;
  double llkl, logdet,chi2,logdet_corr,chi2_corr,FRF,XRF,lkl;
  long l,i;
  int int_npix_seen, info, xinc;
  char uplo,trans,diag;
  FILE*fp;

  double *mcl;
  TIMER_DEFS;
  int tot_time;
  
  TIMER_IN;
  self = llll_str;
  
  // Fill in q=Cl*Wl
  q = self->q_buf;
  if (self->partial_nindex>0) {
    for(l=0;l<self->partial_nindex;l++) {
      self->partial_cl[self->partial_index[l]]=cl[l];
    }
    mcl=self->partial_cl;
  } else {
    mcl=cl;
  }
  
  if (self->isLog) {
    for (l=0;l<=self->lmax;l++) {
      q[l]=self->Wl[l]*exp(mcl[l]);
    }
  } else {
    for (l=0;l<=self->lmax;l++) {
      q[l]=self->Wl[l]*mcl[l];
      //fprintf(stderr,"%d %g %g\n",l,mcl[l],q[l]);
    }
  }
  
  // Build covariance matrix for given Cl
  covmat = build_cov_matrix(self->covmat_buf,self->CosMat,self->NoiseVar,self->npix_seen,self->lmax,q,err);
  forwardError(*err,__LINE__,0);
  TIMER_OUT
  tot_time+=TIMER_MSEC;
  self->time_build += TIMER_MSEC;
  
  TIMER_IN;
  lkl = lowly_XtRX_lkl (covmat,self->X_seen,self->X_seen_buf,self->npix_seen,err);
  forwardError(*err,__LINE__,0);
  
  TIMER_OUT;
  tot_time+=TIMER_MSEC;
  self->time_chol += TIMER_MSEC;

  chi2_corr   = 0;
  logdet_corr = 0;
  
  // should I fit a forground to that ?
  if (self->F_seen!=0) {
    // Yes !
    F_seen=self->F_seen_buf;
    X_seen=self->X_seen_buf;
    memcpy((void*)F_seen,(void*)self->F_seen,self->npix_seen*sizeof(double));
    trans='N'; diag='N';xinc=1;
    dtrsv(&uplo,&trans,&diag,&int_npix_seen,covmat,&int_npix_seen,F_seen,&xinc);
    FRF = 0;
    XRF = 0;
    for (i=0;i<self->npix_seen;i++) {
#ifdef LOWLY_PARANOID_NAN_TEST
      testErrorRetVA((isnan(F_seen[i])!=0 || isinf(F_seen[i])!=0),pmc_infinite,"F_seen is invalid (%g) at indice %d",*err,__LINE__,F_seen[i],i,0);
#endif
      FRF += F_seen[i]*F_seen[i];
      XRF += X_seen[i]*F_seen[i];
    }
    chi2_corr   = - XRF*XRF / ( 1./self->v2 + FRF);
#ifdef LOWLY_PARANOID_NAN_TEST
    testErrorRetVA((isnan(chi2_corr)!=0 || isinf(chi2_corr)!=0),pmc_infinite,"chi2_corr is invalid (%g)",*err,__LINE__,0,chi2_corr);
#endif
    logdet_corr = log( 1./self->v2 + FRF );
#ifdef LOWLY_PARANOID_NAN_TEST
    testErrorRetVA((isnan(logdet_corr)!=0 || isinf(logdet_corr)!=0),pmc_infinite,"logdet_corr is invalid (%g)",*err,__LINE__,0,logdet_corr);
#endif
  }
    
    
  //printf("in lkl_lowly, logdet = %lg\n",logdet);
  
  lkl   += -(chi2_corr + logdet_corr)/2.;
  logdet += logdet_corr;
  llkl = -( self->npix_seen*log(ERF_SQRT2PI) ) + lkl;
  
  if (self->isLog) {
    /* Add jacobian term for the change of variable Cl -> log(Cl) */
    for (l=0;l<=self->lmax;l++) {
      llkl += mcl[l];
    }    
  }
  TIMER_OUT;
  tot_time+=TIMER_MSEC;
  self->time_tot += tot_time;
  self->n_tests++;

  //fprintf(stderr,"total time %d msec\n",TIMER_MSEC);
  
  return(llkl);
  
  // Solve R^1/2 y = X and compute y^t.y
  // Compute determinant from R^1/2 diag elements
  // Return -2*loglike = X^T R^-1 X + log det R
  
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "lowly_lkl_logcl"

/* 
   This function allows sampling on log(Cl) instead of Cl
   It assumes that the noise/regul term is negligible...
   Corrects for the Jacobian factor to keep the prior flat in Cl
*/
double lowly_lkl_logcl(void *llll_str, double *logcl, error **err) {
  double res;
  
  res=lowly_lkl(llll_str,logcl,err);
  forwardError(*err,__LINE__,0)
  return res;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "lowly_partial_lkl"
double lowly_partial_lkl(void *llll_str, double *cl, error **err) {
  double res;
  
  res=lowly_lkl(llll_str,cl,err);
  forwardError(*err,__LINE__,0)
  //fprintf(stdout,"%g %g\n",cl[0],res);
  return res;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "lowly_partial_lkl_logcl"

/* Takes as input logcl, assumes logcl is stored in llll_str */
double lowly_partial_lkl_logcl(void *llll_str, double *cl, error **err) {
  double res;
  
  res=lowly_lkl(llll_str,cl,err);
  forwardError(*err,__LINE__,0)
  return res;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "lowly_lkl_with_noise"
double lowly_lkl_with_noise(void *llll_str, double *cl, error **err) {
  int np,last;
  llll_struct *self;
  double llkl;
  
  self=llll_str;
  
  if (self->partial_nindex>0) {
    last=self->partial_nindex;
  } else {
    last=self->lmax+2;
  }
  for(np=0;np<=self->npix_seen;np++) {
    self->NoiseVar[np]=cl[last];
  }
  llkl=lowly_lkl(llll_str,cl,err);
  forwardError(*err,__LINE__,0);
  return llkl;
}

/*****************************************************************************************/
#undef __FUNC__ 
#define __FUNC__ "free_lowly"
void free_lowly(llll_struct **self) {
  
  llll_struct *tmp;
  tmp = *self;
  
  DO_PRINT_STAT(tmp);
  
  free(tmp->buffer);
  free(tmp->X_seen);
  free(tmp->NoiseVar);
  free(tmp->CosMat);
  free(tmp->Wl);
  if (tmp->partial_nindex!=0) {
    free(tmp->partial_index);
    free(tmp->partial_cl);
  }
  free(tmp);
  tmp=NULL;
  
  return;
}

/*****************************************************************************************/
#undef __FUNC__
#define __FUNC__ "free_lowly_partial"
void free_lowly_partial(llll_partial_struct **elf) {
  free_lowly(elf);
}

/*****************************************************************************************/
#undef __FUNC__ 

#ifdef HAS_PMC
#define __FUNC__ "invgamma_prop_init"
invgamma_prop* invgamma_prop_init(size_t ndim,int* ell,double* clhat, double* fsky, double* wl,double noisevar,long nside,error **err) {
  long i;
  invgamma_prop *g;
  double *N;
  int mell;
  double Ncst;
  
  mell=ell[0];
  for(i=1;i<ndim;i++) {
    if (mell<ell[i])
      mell=ell[i];
  }
  N = malloc_err(sizeof(double)*(mell+1), err);
  forwardError(*err,__LINE__,NULL);
  Ncst = lowly_computeNl(noisevar,nside);
  for(i=0;i<mell+1;i++) 
    N[i] = Ncst;
  
  g = invgamma_prop_init_new(ndim, ell, clhat, fsky, wl, N, err);
  forwardError(*err,__LINE__,NULL);
  
  free(N);
  
  return g;  
}

#undef __FUNC__ 
#define __FUNC__ "invgamma_prop_init_new"
invgamma_prop* invgamma_prop_init_new(size_t ndim,int* ell,double* clhat, double* fsky, double* wl,double* N,error **err) {
  double *shape,*scale;
  long i;
  invgamma_prop *g;
  
  g = (invgamma_prop *) malloc_err(sizeof(invgamma_prop),err);
  forwardError(*err,__LINE__,NULL);
  
  shape=malloc_err(sizeof(double)*ndim*2,err);
  forwardError(*err,__LINE__,NULL);
  
  scale=shape+ndim;
  g->data=malloc_err(ndim*(sizeof(double)*3),err);
  forwardError(*err,__LINE__,NULL);
  g->wl=g->data+ndim;
  g->N=g->wl+ndim;
    
  for(i=0;i<ndim;i++){
    size_t ml;
    //fprintf(stderr,"-> %d ",i);
    ml = ell[i];
    //fprintf(stderr," %d (%p)",ml,wl);
    //fprintf(stderr," %g ",wl[ml]);
    if (wl==NULL) {
      g->wl[i]=1;
    } else {
      g->wl[i]=wl[(int)ml];
    }
    g->N[i]=N[(int)ml];
    //fprintf(stderr," %g ",g->wl[i]);
    computeAlphaBeta(&(shape[i]), &(scale[i]),ell[i], clhat[ml],fsky[ml], g->wl[i],g->N[i]);
    //fprintf(stderr," %g %g\n",shape[i],scale[i]);
  }
  
  g->_ivg=invgamma_init(ndim,scale,shape,err);
  forwardError(*err,__LINE__,NULL);
  free(shape);
  
  g->ndim=ndim;
  return g;
}

/*****************************************************************************************/
#undef __FUNC__ 
#define __FUNC__ "free_invgamma_prop"
void free_invgamma_prop(invgamma_prop **pelf) {
  invgamma_prop *self;
  
  self=*pelf;
  
  free(self->data);
  invgamma_free(&(self->_ivg));
  free(self);
  *pelf=NULL;
}

/*****************************************************************************************/
#undef __FUNC__ 
#define __FUNC__ "invgamma_prop_log_pdf"

double invgamma_prop_log_pdf(invgamma_prop *self, double *x, error **err) {
  long i;
  double res;
  
  for(i=0;i<self->ndim;i++) {
    self->data[i]=cl2xl(x[i],self->wl[i],self->N[i]);
  }
  
  
  res=invgamma_log_pdf(self->_ivg, self->data, err);
  forwardError(*err,__LINE__,0);
  return res;
}
/*****************************************************************************************/
#undef __FUNC__ 
#define __FUNC__ "simulate_invgamma_prop"

long simulate_invgamma_prop(pmc_simu *psim,invgamma_prop *proposal, gsl_rng * r,parabox *pb,error **err) {
  long i,j,wi;
  int ok;
  size_t ind;
  i=0;
  while(i<psim->nsamples) {
    wi=i*psim->ndim;
    invgamma_ran(psim->X+wi,proposal->_ivg,r,err);
    forwardError(*err,__LINE__,0);
    //fprintf(stderr,"%d->",i);
    for(j=0;j<proposal->ndim;j++) {
      //fprintf(stderr,"%d: %g (%g %g)-",j,psim->X[wi+j],proposal->Nl,proposal->wl[j]);
      psim->X[wi+j]-=proposal->N[j];
      psim->X[wi+j]*=1./proposal->wl[j];
      //fprintf(stderr,"%g ",psim->X[wi+j]);
    }
    //fprintf(stderr,"\n");
    
    ok=1;
    if (pb!=NULL) {
      if (isinBox(pb,psim->X+wi,err)==0) 
        ok=0;
      forwardError(*err,__LINE__,0);
    }
    if (ok) {	
      psim->indices[ i]=1;
      i++;
      psim->flg[i]=1;
    }
  }
  return i;
}
/*****************************************************************************************/
#undef __FUNC__ 

#define __FUNC__ "get_importance_weight_invgamma_prop"
size_t get_importance_weight_invgamma_prop(pmc_simu *psim, const invgamma_prop *m,
                                      posterior_log_pdf_func *posterior_log_pdf, 
                                      void *extra, error **err) {
  
  size_t rr;
  
  rr=generic_get_importance_weight(psim, m, &invgamma_prop_log_pdf, posterior_log_pdf, extra, err);
  forwardError(*err,__LINE__,0);
  return rr;
} 
#undef __FUNC__

/*****************************************************************************************/
#undef __FUNC__ 

#define __FUNC__ "update_invgamma_prop"
void update_invgamma_prop(invgamma_prop *g,pmc_simu *psim,error **err) {
  double *var,*mean;
  double tw,cw,cv;
  size_t l,i,wi;
  
  /* update the inverse gamma proposal using the moment method to evaluate the new parameters*/
  normalize_importance_weight(psim,err);
  forwardError(*err,__LINE__,);
  
  mean=malloc_err(g->ndim*2*sizeof(double),err);
  forwardError(*err,__LINE__,);
  var=mean+g->ndim;
  for(i=0;i<g->ndim*2;i++)
    mean[i]=0;
  tw=0;
  // compute <1/Cl>
  for(i=0;i<psim->nsamples;i++) {
    if (psim->flg[i]!=1)
      continue;
    cw=psim->weights[i];
    tw+=tw;
    wi=i*g->ndim;
    for(l=0;l<g->ndim;l++) {
      mean[l]+=(1/psim->X[wi+l])*cw;
    }
  }
  for(i=0;i<g->ndim;i++) {
    mean[i]/=tw;
  }
  // compute <1/Cl^2>-<1/Cl>^2
  for(i=0;i<psim->nsamples;i++) {
    if (psim->flg[i]!=1)
      continue;
    cw=psim->weights[i];
    wi=i*g->ndim;
    for(l=0;l<g->ndim;l++) {
      cv=(1/psim->X[wi+l]);
      cv=cv*cv;
      cv-=mean[l]*mean[l];
      var[i]+=cv*cw;
    }
  }
  for(i=0;i<g->ndim;i++) {
    var[i]/=tw;
  }
  
  for(l=0;l<g->ndim;l++) {
    g->_ivg->scale[l]=var[l]/mean[l];    
    g->_ivg->shape[l]=mean[l]/g->_ivg->scale[l];
  }
  free(mean);
}

#undef __FUNC__ 

#endif

#define __FUNC__ "computeAlphaBeta"
void computeAlphaBeta(double *alpha, double *beta,int ell, double clhat, double fsky, double wl, double Nl) {
  double l2p1s2;
  size_t ml;
  
  ml = ell;
  l2p1s2 = (2*ml+1.)/2.;
    
  (*alpha) = l2p1s2 * fsky;
  (*beta)  = 1./(*alpha) / (wl*clhat+Nl);
  (*alpha) -= 1;
  
  return;
}
  
  
#undef __FUNC__ 

#define __FUNC__ "cl2xl"
double cl2xl(double cl, double wl, double Nl) {
  double xl;
  xl=cl*wl+Nl;
  //fprintf(stderr,"%g -> %g %g %g\n",cl,wl,Nl,xl
  return xl;
}


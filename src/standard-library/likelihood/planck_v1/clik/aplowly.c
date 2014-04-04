/*
 *  aplowly.c
 *  lowly_project
 *
 *  Created by Karim Benabed on 11/03/09.
 *  Copyright 2009 Institut d'Astrophysique de Paris. All rights reserved.
 *
 */

#ifdef __PLANCK__
#include "HL2_likely/target/aplowly.h"
#else
#include "aplowly.h"
#endif

#define __FUNC__ "gausslkl_init"
gausslkl* gausslkl_init(size_t nell, int* ell,double *cl, double *vl, double *nl, int nlcst, double *Cm, int Cmisdiag, error **err) {
  gausslkl *ing;
  int plus;
  int info;
  double det;
  char uplo;
  int i;
  
  ing = malloc_err(sizeof(gausslkl), err);
  forwardError(*err,__LINE__,NULL);
  
  ing->nell = nell;
  plus=1;
  
  if (Cm==NULL) {
    plus = 0;
  }
  ing->data = malloc_err(sizeof(double)*(nell*(4 + plus*nell)), err);
  forwardError(*err,__LINE__,NULL);
  
  ing->ell = malloc_err(sizeof(int)*nell,err);
  forwardError(*err,__LINE__,NULL);
  if (ell==NULL) {
    for (i=0;i<ing->nell;i++) {
      ing->ell[i]=i;
    }
  } else {
    memcpy(ing->ell, ell, sizeof(int)*nell);    
  }
  
  
  ing->cl = ing->data;
  ing->vl = ing->cl + nell;
  ing->nl = ing->vl + nell;
  ing->gl = ing->nl + nell;
  
  if (plus==1) {
    ing->Cm = ing->gl + nell;
  } else {
    ing->Cm = NULL;
  } 
  
  memcpy(ing->cl, cl, sizeof(double)*nell);
  memcpy(ing->vl, vl, sizeof(double)*nell);
  
  if (nlcst==1) {
    for(i=0;i<nell;i++) {
      ing->nl[i]=nl[0];
      //_DEBUGHERE_("%d :: %g %g",i,ing->nl[i],nl[0]);
    }
  } else{
    memcpy(ing->nl, nl, sizeof(double)*nell);
  }
  
  if (plus==1) {
    if (Cmisdiag<0) {
      // diagonale constante
      memset(ing->Cm, 0, sizeof(double)*nell*nell);
      for(i=0;i<nell;i++) {
        ing->Cm[i*nell+i]=sqrt(Cm[0]);
      }
      ing->isdiag=1;
    } else if (Cmisdiag>0) {
      int i;
      memset(ing->Cm, 0, sizeof(double)*nell*nell);
      for(i=0;i<nell;i++) {
        ing->Cm[i*nell+i]=sqrt(Cm[i]);
      }
      ing->isdiag=1;
    } else {
      ing->isdiag=0;
      memcpy(ing->Cm, Cm, sizeof(double)*nell*nell);
      uplo = 'L';
      dpotrf(&uplo,&ing->nell,ing->Cm,&ing->nell,&info);
      testErrorRetVA(info!=0,lowly_chol,"Could not cholesky decompose using dpotrf (%d)",*err,__LINE__,NULL,info);
    }
    det = 1;
    for (i = 0; i < ing->nell*ing->nell; i+=(ing->nell+1)) {
      det *= ing->Cm[i];
    }
    ing->logdets2=log(det);
  } 
  return ing;
}
# undef __FUNC__

#define __FUNC__ "gausslkl_free"
void gausslkl_free(void **ping) {
  gausslkl * ing;
  ing = *ping;
  free(ing->data);
  free(ing->ell);
  free(ing);
  *ping = NULL;
}
# undef __FUNC__

#define __FUNC__ "gausslkl_log_pdf"
double gausslkl_log_pdf(void* ping, double* pars, error **err) {
  gausslkl *ing;
  int i;
  double log_CN;
  char uplo,trans,diag;
  int xinc;
  
  ing = ping;
  log_CN=0;
  
  for(i=0;i<ing->nell;i++) {
    // add nl
    ing->gl[i] = (ing->nl[i] + pars[i] - ing->cl[i])/ing->vl[i];
    //_DEBUGHERE_("i %d ell %d nl %g cl %g vl %g pl %g gl %g",i,ing->ell[i],ing->nl[i],ing->cl[i],ing->vl[i],pars[i],ing->gl[i]);
  }
  
  if (ing->Cm !=NULL) {
    if (ing->isdiag==1) {
      for(i=0;i<ing->nell;i++) {
        ing->gl[i] /= ing->Cm[i+ing->nell*i];
      }
    } else {
      uplo  = 'L';
      trans = 'N';
      diag = 'N';
      xinc = 1;
      dtrsv(&uplo,&trans,&diag,&ing->nell,ing->Cm,&ing->nell,ing->gl,&xinc);
    }    
  }
  
  for(i=0;i<ing->nell;i++) {
    log_CN += ing->gl[i]*ing->gl[i];     
    //_DEBUGHERE_("i %d ell %d lkl %g",i,ing->ell[i],log_CN);
  }
  
  return - 0.5 * (log_CN) - ing->logdets2;
}
# undef __FUNC__

#define __FUNC__ "tease_init"
tease* tease_init(size_t nell, int* ell,double *al, double *bl, double *nl, int nlcst, double *Cm, int Cmisdiag, error **err) {
  tease *ing;
  int plus;
  int info;
  double det;
  char uplo;
  int i;
  
  ing = malloc_err(sizeof(tease), err);
  forwardError(*err,__LINE__,NULL);
  
  ing->nell = nell;
  plus=1;
  
  if (Cm==NULL) {
    plus = 0;
  }
  ing->data = malloc_err(sizeof(double)*(nell*(4 + plus*nell)), err);
  forwardError(*err,__LINE__,NULL);

  ing->ell = malloc_err(sizeof(int)*nell,err);
  forwardError(*err,__LINE__,NULL);
  
  if (ell==NULL) {
    for (i=0;i<ing->nell;i++) {
      ing->ell[i]=i;
    }
  } else {
    memcpy(ing->ell, ell, sizeof(int)*nell);    
  }
  
  ing->al = ing->data;
  ing->bl = ing->al + nell;
  ing->nl = ing->bl + nell;
  ing->gl = ing->nl + nell;
  if (plus==1) {
    ing->Cm = ing->gl + nell;
  } else {
    ing->Cm = NULL;
  } 
  
  memcpy(ing->al, al, sizeof(double)*nell);
  memcpy(ing->bl, bl, sizeof(double)*nell);
  
  if (nlcst==1) {
    for(i=0;i<nell;i++) {
      ing->nl[i]=nl[0];
      //_DEBUGHERE_("%d :: %g %g",i,ing->nl[i],nl[0]);
    }
  } else{
    memcpy(ing->nl, nl, sizeof(double)*nell);
  }
  
  if (plus==1) {
    if (Cmisdiag<0) {
      // diagonale constante
      memset(ing->Cm, 0, sizeof(double)*nell*nell);
      for(i=0;i<nell;i++) {
        ing->Cm[i*nell+i]=sqrt(Cm[0]);
      }
      ing->isdiag=1;
    } else if (Cmisdiag>0) {
      int i;
      memset(ing->Cm, 0, sizeof(double)*nell*nell);
      for(i=0;i<nell;i++) {
        ing->Cm[i*nell+i]=sqrt(Cm[i]);
      }
      ing->isdiag=1;
    } else {
      ing->isdiag=0;
      memcpy(ing->Cm, Cm, sizeof(double)*nell*nell);
      uplo = 'L';
      dpotrf(&uplo,&ing->nell,ing->Cm,&ing->nell,&info);
      testErrorRetVA(info!=0,lowly_chol,"Could not cholesky decompose using dpotrf (%d)",*err,__LINE__,NULL,info);
    }
    det = 1;
    for (i = 0; i < ing->nell*ing->nell; i+=(ing->nell+1)) {
      det *= ing->Cm[i];
    }
    ing->logdets2=log(det);
  } 
  return ing;
}
# undef __FUNC__

#define __FUNC__ "tease_free"
void tease_free(void **ping) {
  tease * ing;
  ing = *ping;
  free(ing->data);
  free(ing->ell);
  free(ing);
  *ping = NULL;
}
# undef __FUNC__

#define __FUNC__ "tease_log_pdf"
double tease_log_pdf(void* ping, double* pars, error **err) {
  tease *ing;
  int i;
  double xl,bl_gsl;
  double log_ivg, log_N, log_CN,ul;
  char uplo,trans,diag;
  int xinc;
  double livgl;
  
  ing = ping;
  log_ivg = 0;
  log_N=0;
  log_CN=0;
  
  for(i=0;i<ing->nell;i++) {
    // add nl
    xl = ing->nl[i] + pars[i];
    
    //_DEBUGHERE_("i %d ell %d al %g bl %g nl %g cl %g pl %g xl %g",i,ing->ell[i],ing->al[i],ing->bl[i],ing->nl[i],ing->bl[i]/(ing->al[i]+1)-ing->nl[i],pars[i],xl);
    // compute uncorrelated part 
    //log_ivg += log(gsl_ran_gamma_pdf (1./xl, ing->al[i], bl_gsl)) - 2*log(xl);
    //_DEBUGHERE_("livg %g",log_ivg);
    livgl = (-ing->al[i]-1.) * log(xl) - ing->bl[i]/xl;
    testErrorRetVA((isnan(livgl)!=0 || isinf(livgl)!=0), pmc_infinite,
                   "1 : Infinite log pdf %g i=%d ell=%d al=%g bl=%g nl=%g pl=%g", *err, __LINE__,0, 
                   livgl, i,ing->ell[i],ing->al[i],ing->bl[i],ing->nl[i],pars[i]);
    livgl += -gsl_sf_lngamma(ing->al[i]) + log(ing->bl[i]) * ing->al[i];
    testErrorRetVA((isnan(livgl)!=0 || isinf(livgl)!=0), pmc_infinite,
                   "2 : Infinite log pdf %g i=%d ell=%d al=%g bl=%g nl=%g pl=%g", *err, __LINE__,0, 
                   livgl, i,ing->ell[i],ing->al[i],ing->bl[i],ing->nl[i],pars[i]);
    log_ivg += livgl;
    //_DEBUGHERE_("-> livgl %g sum %g",livgl,log_ivg);
    if (ing->Cm!=NULL) { // we are correlated, a bit more work...
      // uniformize
      ul = gsl_sf_gamma_inc_Q(ing->al[i],ing->bl[i]/xl);
      // gaussianize
      ing->gl[i] = ERF_SQRT2 * erfinv(2*ul-1,err);
      log_N += ing->gl[i]*ing->gl[i];
    }
  }
  
  if (ing->Cm==NULL) {
    //nothing else to do, we are go to go
    return log_ivg;
  }
  
  if (ing->isdiag==1) {
    for(i=0;i<ing->nell;i++) {
      ing->gl[i] /= ing->Cm[i+ing->nell*i];
    }
  } else {
    uplo  = 'L';
    trans = 'N';
    diag = 'N';
    xinc = 1;
    dtrsv(&uplo,&trans,&diag,&ing->nell,ing->Cm,&ing->nell,ing->gl,&xinc);
  }
  
  for(i=0;i<ing->nell;i++) {
    log_CN += ing->gl[i]*ing->gl[i]; 
  }
  
  return - 0.5 * (log_CN - log_N) - ing->logdets2 + log_ivg;
}
# undef __FUNC__

#define __FUNC__ "log_normal_init"
log_normal* log_normal_init(size_t ndim,int* ell, double *clhat, double *fsky, double *wl, double noisevar, long nside, error **err) {
  log_normal *self;
  size_t i,j,mi,mj;
  double det;
  
  
  //fprintf(stderr,"A\n");
  self = (log_normal *) malloc_err(sizeof(log_normal),err);
  forwardError(*err,__LINE__,NULL);
  
  //fprintf(stderr,"B\n");
  self->ndim=ndim;
  
  self->data = (double*) malloc_err(sizeof(double)*ndim * 3,err);
  forwardError(*err,__LINE__,NULL);
  
  //fprintf(stderr,"C\n");
  
  self->ldlh =  self->data;
  self->sig2 = self->ldlh + ndim;
  self->wl =   self->sig2 + ndim;
  
  self->Nl=lowly_computeNl(noisevar, nside);
  //fprintf(stderr,"D\n");
  
  for(i=0;i<ndim;i++) {
    long l;
    //fprintf(stderr,"E %d\n",i);
    l=ell[i];
    //fprintf(stderr,"%d\n",l);
    self->ldlh[i] = log(clhat[l]*wl[l]+self->Nl);
    //fprintf(stderr,"%g ",self->ldlh[i]);
    self->sig2[i] = 2./((2.*l+1)*fsky[l]);
    //fprintf(stderr,"%g ",self->sig2[i]);
    self->wl[i]   = wl[l];
    //fprintf(stderr,"%g \n",self->wl[i]);
  }
  //fprintf(stderr,"F\n");
  
  return self;
}
#undef __FUNC__ 

#define __FUNC__ "free_log_normal"
void free_log_normal(log_normal **self) {
  free((*self)->data);
  free(*self);
  self=NULL;
}
#undef __FUNC__ 

#define __FUNC__ "log_normal_log_pdf"
double log_normal_log_pdf(void *pelf, double *Cl, error **err) {
  double res,ldl;
  log_normal *self;
  long i;
  
  //fprintf(stderr,"pdf: 0\n");
  
  self = (log_normal*) pelf;
  
  res=0;
  for(i=0;i<self->ndim;i++) {
    ldl = log(self->wl[i]*Cl[i]+self->Nl);
    ldl = ldl-self->ldlh[i];
    res += -ldl*ldl/2./self->sig2[i];
    //fprintf(stderr,"%g ",res);
  }
  //fprintf(stderr," -> %g\n",res);  
  return res;
}
#undef __FUNC__ 

#define __FUNC__ "approx_lowly_init"
approx_lowly* approx_lowly_init(size_t ndim,int* ell,int lmax, double *clhat, double *fsky, double *wl, double *V,double noisevar, long nside, error **err) {
  approx_lowly *self;
  size_t i,j,mi,mj;
  double det;
  
  //fprintf(stderr,"A\n");
  
  self = (approx_lowly *) malloc_err(sizeof(approx_lowly),err);
  forwardError(*err,__LINE__,NULL);
  
  self->ndim=ndim;
  //fprintf(stderr,"B\n");
  
  self->data = (double*) malloc_err(sizeof(double) * ndim * (4 + ndim),err);
  forwardError(*err,__LINE__,NULL);
  
  self->alpha = self->data;
  self->beta  = self->alpha + ndim;
  self->wl    = self->beta  + ndim;
  self->gl    = self->wl    + ndim;
  self->C     = self->gl    + ndim;  
  //fprintf(stderr,"C\n");
  
  self->Nl=lowly_computeNl(noisevar, nside);
  
  for(i=0;i<ndim;i++) {
    size_t ml;
    ml=ell[i];
    self->wl[i]=wl[ml];
    computeAlphaBeta(&(self->alpha[i]),&(self->beta[i]),ml, clhat[ml],fsky[ml],self->wl[i],self->Nl);    
  }
  //fprintf(stderr,"D\n");
  
  if (V==NULL) {
    self->is_zero=1;
    return self;
  }
  //fprintf(stderr,"E\n");
  
  for(i=0;i<ndim;i++) {
    mi=ell[i]*(lmax+1);
    for(j=0;j<ndim;j++) {
      mj=ell[j];
      self->C[i*ndim+j]=V[mi+mj];
      //fprintf(stderr,"%g ",self->C[i*ndim+j]);
    }
    //fprintf(stderr,"\n");
  }
  self->C_view_container=gsl_matrix_view_array(self->C,ndim,ndim); 
  self->C_view=&(self->C_view_container.matrix);
  //fprintf(stderr,"F\n");
  
  gsl_set_error_handler_off();
  testErrorRet(gsl_linalg_cholesky_decomp(self->C_view) ==  GSL_EDOM,mv_cholesky,"Cholesky decomposition failed",*err,__LINE__,);
  //fprintf(stderr,"G\n");
  
  det = 1;
  for (i = 0; i < ndim*ndim; i+=(ndim+1)) {
    det *= self->C[i];
  }
  self->logdets2=log(det);
  //fprintf(stderr,"H\n");
  
  self->gl_view_container=gsl_vector_view_array(self->gl,self->ndim);
  self->gl_view = &(self->gl_view_container.vector);
  
  self->is_zero=0;
  return self;
  
}
#undef __FUNC__ 

#define __FUNC__ "approx_lowly_log_pdf"
double approx_lowly_log_pdf(void* pelf,double* pars, error **err) {
  approx_lowly* self;
  double xl,ul,gl,vul;
  double log_ivg, log_N, log_CN;
  size_t i;
  
  self=pelf;
  log_ivg=0;
  log_N=0;
  log_CN=0;
  
  //testErrorRet(self->is_zero==1,lowly_no_correlation,"no correlation defined !",*err,__LINE__,-1);
  
  if (self->is_zero==1) {
    xl=approx_zero_lowly_log_pdf(pelf, pars, err);
    forwardError(*err,__LINE__,0);
    return xl;
  }
  for(i=0;i<self->ndim;i++) {
    xl = cl2xl(pars[i],self->wl[i],self->Nl);
    //fprintf(stderr,"%d-> %g %g %g ",i,xl,self->alpha[i],self->beta[i]);
    ul = gsl_sf_gamma_inc_Q(self->alpha[i],1./xl/self->beta[i]);
    //fprintf(stderr,"%g ",ul);
    log_ivg += log(gsl_ran_gamma_pdf(1./xl, self->alpha[i], self->beta[i]))
    - 2*log(xl);
    self->gl[i] = ERF_SQRT2 * erfinv(2*ul-1,err);
    forwardError(*err,__LINE__,-1);
    //fprintf(stderr,"%g ",self->gl[i]);
    
    log_N += self->gl[i]*self->gl[i];
    //fprintf(stderr,"%g\n",log_N);
  }
  
  gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, self->C_view, self->gl_view);
  
  for(i=0;i<self->ndim;i++) {
    log_CN += self->gl[i]*self->gl[i]; 
  }
  //fprintf(stderr,"%g %g %g %g\n",log_CN,log_N,self->logdets2,log_ivg);
  //fprintf(stderr,"%g\n",- 0.5 * (log_CN - log_N) - self->logdets2 + log_ivg);
  return - 0.5 * (log_CN - log_N) - self->logdets2 + log_ivg;
}
#undef __FUNC__ 

#define __FUNC__ "approx_zero_lowly_log_pdf"
double approx_zero_lowly_log_pdf(void* pelf,double* pars, error **err) {
  approx_lowly* self;
  double xl,ul,gl;
  double log_ivg, log_N, log_CN;
  size_t i;
  
  self=pelf;
  log_ivg=0;
  
  for(i=0;i<self->ndim;i++) {
    xl = cl2xl(pars[i],self->wl[i],self->Nl);
    log_ivg += log(gsl_ran_gamma_pdf (1./xl, self->alpha[i], self->beta[i]))
    - 2*log(xl);
  }
  
  return log_ivg;
}
#undef __FUNC__ 

#define __FUNC__ "free_approx_lowly"
void free_approx_lowly(approx_lowly **self) {
  free((*self)->data);
  free(*self);
  self=NULL;
}
#undef __FUNC__ 


binlkl *init_binlkl(int np, int* nb, double* be, double* bv, error **err) {
  binlkl *bon;
  int i;
  int obe,obv;
  
  bon = malloc_err(sizeof(binlkl),err);
  forwardError(*err,__LINE__,NULL);
  
  bon->np = np;
  bon->nb = malloc_err(sizeof(int)*np,err);
  forwardError(*err,__LINE__,NULL);
  bon->obe = malloc_err(sizeof(int)*np,err);
  forwardError(*err,__LINE__,NULL);
  bon->obv = malloc_err(sizeof(int)*np,err);
  forwardError(*err,__LINE__,NULL);
  
  obe=0;obv=0;
  for(i=0;i<np;i++) {
    bon->nb[i] = nb[i];
    bon->obe[i] = obe;
    bon->obv[i] = obv;
    obe += nb[i]+1;
    obv += nb[i];
  }
  
  bon->be = malloc_err(sizeof(double)*obe,err);
  forwardError(*err,__LINE__,NULL);
  bon->bv = malloc_err(sizeof(double)*obv,err);
  forwardError(*err,__LINE__,NULL);
  
  memcpy(bon->be,be,sizeof(double)*obe);
  memcpy(bon->bv,bv,sizeof(double)*obv);
  
  return bon;
}

double binlkl_lkl(void *vbon, double *pars, error **err) {
  int i;
  double res;
  binlkl *bon;
  
  bon = vbon;
  res = 0;
  for(i=0;i<bon->np;i++) {
    int j;
    double pv;
    double *be,*bv;
    int nb;
    nb = bon->nb[i];
    be = &(bon->be[bon->obe[i]]);
    bv = &(bon->bv[bon->obv[i]]);
    if ((pv<be[0])||(be[nb-1]<pv)) {
      res = ZEROLKL;
      return;
    }
    for(j=1;j<nb;j++) {
      if (bon->obe[j]<pv) {
        break;
      }
    }
    res += bv[j-1];
  }
  return res;
}

void free_binlkl(void** pvbon) {
  binlkl *bon;
  
  bon = *pvbon;
  free(bon->nb);
  free(bon->obe);
  free(bon->obv);
  free(bon->be);
  free(bon->bv);
  free(bon);
  *pvbon = NULL;
}
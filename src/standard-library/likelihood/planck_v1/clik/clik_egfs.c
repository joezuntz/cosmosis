#include "clik_egfs.h"

egfs *egfs_init(int nvar, char **keyvars, int ndefaults, char** keys, char** values, int lmin, int lmax, double* cib_clustering,double *patchy_ksz, double *homogenous_ksz,double *tsz, double *cib_decor_clust, double * cib_decor_poisson,error **err) {
  egfs * self;
  int i,npm;
  
  self = malloc_err(sizeof(egfs),err);
  forwardError(*err,__LINE__,NULL);
  
  self->nkv = nvar+ndefaults;
  self->ndf = ndefaults;
  self->np = nvar;
  
  if (self->nkv!=0) {
    self->keys = malloc_err(sizeof(egfs_parstr)*self->nkv,err);
    forwardError(*err,__LINE__,NULL);
    self->values = malloc_err(sizeof(egfs_parstr)*self->nkv,err);
    forwardError(*err,__LINE__,NULL);  
  } else{
    self->keys = malloc_err(sizeof(egfs_parstr)*1,err);
    forwardError(*err,__LINE__,NULL);
    self->values = malloc_err(sizeof(egfs_parstr)*1,err);
    forwardError(*err,__LINE__,NULL);
  }
  memset(self->keys,' ',sizeof(char)*256*self->nkv);
  memset(self->values,' ',sizeof(char)*256*self->nkv);
  for(i=0;i<ndefaults;i++) {
    sprintf((self->keys[i]),"%s",keys[i]);
    sprintf((self->values[i]),"%s",values[i]);
    //_DEBUGHERE_("%s = %s",self->keys[i],self->values[i]);
  }
  for(i=self->ndf;i<self->nkv;i++) {
    sprintf((self->keys[i]),"%s",keyvars[i-self->ndf]);
    //_DEBUGHERE_("%s = ?",self->keys[i]);
  }  
  
  self->nell = lmax-lmin+1;
  self->lmin = lmin;
  self->lmax = lmax;
  if (cib_clustering!=NULL) {
    clik_egfs_init_frommem_(&(self->instance_id),self->keys,self->values,&self->ndf,cib_clustering, patchy_ksz, homogenous_ksz,tsz);
  } else {
    clik_egfs_init_(&(self->instance_id),self->keys,self->values,&self->ndf);
  }
  
  
  testErrorRet(self->instance_id==0,-1010101001,"problem in egfs init",*err,__LINE__,NULL);
  
  clik_get_nfr_(&self->instance_id,  &self->nfr);

  self->buf = malloc_err(sizeof(double)*(self->nfr*self->nfr*self->nell+self->nfr*self->nfr*self->nell*(20)),err);
  forwardError(*err,__LINE__,NULL);

  self->brq = self->buf;
  self->bdrq = self->brq + self->nfr*self->nfr*self->nell;
  
  clik_egfs_order_(&nvar,&(self->keys[ndefaults]),self->cid);
  self->nmodels = 0;
  for(i=0;i<nvar;i++) {
    int imod,j;
    //_DEBUGHERE_("%s %d",self->keys[ndefaults+i],self->cid[i]);
    testErrorRetVA(self->cid[i]==0,-1010101001,"problem in egfs init (unknown parameter '%s')",*err,__LINE__,NULL,self->keys[ndefaults+i]);
    imod = ((int) self->cid[i]/10) + 1;
    for(j=0;j<self->nmodels;j++) {
      if (self->model[j]==imod) {
        imod = 0;
        break;
      }
    }
    if (imod!=0) {      
      self->model[self->nmodels] = imod;
      self->nmodels++;
    }
  }
  // add constant models
  for(i=0;i<ndefaults;i++) {
    int ione,lid;
    int imod,j;
    ione = 1;
    clik_egfs_order_(&ione,&(self->keys[i]),&lid);
    //_DEBUGHERE_("%s %d",self->keys[ndefaults+i],self->cid[i]);
    //testErrorRet(lid==0,-1010101001,"problem in egfs init",*err,__LINE__,NULL);
    imod = ((int) lid/10) + 1;
    for(j=0;j<self->nmodels;j++) {
      if (self->model[j]==imod) {
        imod = 0;
        break;
      }
    }
    if (imod!=0) {      
      self->model[self->nmodels] = imod;
      self->nmodels++;
    }
  }

  if (cib_decor_clust!=NULL) {
    clik_set_cib_decor_clust_(&(self->instance_id),cib_decor_clust,&self->nfr);
  }
  if (cib_decor_poisson!=NULL) {
    clik_set_cib_decor_poisson_(&(self->instance_id),cib_decor_poisson,&self->nfr);
  }
  return self;
}

void egfs_compute(egfs *self, double *pars, double *rq, double *drq, error **err) {
  int im,np;
  int errid;
  int ip,fr1,fr2,q,imod,nip;
  
  for(ip=0;ip<self->np;ip++) {
    memset(self->values[ip+self->ndf],' ',256);
    sprintf(self->values[ip+self->ndf],"%lg",pars[ip]);
  }
  if (rq!=NULL) {
    memset(rq,0,sizeof(double)*self->nfr*self->nfr*self->nell);
  }
  for (im=0;im<self->nmodels;im++) {
    clik_get_nder_(&(self->model[im]), &np);
    errid=0;
    clik_get_egfs_component_(&(self->instance_id),&(self->model[im]),self->keys,self->values,&(self->nkv),self->brq,self->bdrq,&(self->nfr),&(self->lmin),&(self->lmax),&np,&errid);
    testErrorRetVA(errid!=0,-1010101001,"problem in egfs component %d",*err,__LINE__,,self->model[im]);
    // do rq
    if (rq!=NULL) {
      for(fr1=0;fr1<self->nfr;fr1++) {
        for(fr2=0;fr2<self->nfr;fr2++) {
          for(q=0;q<self->nell;q++) {
            rq[fr1*self->nfr*self->nell+fr2*self->nell+q] += self->brq[fr1+fr2*self->nfr+q*self->nfr*self->nfr];
            /*if (fr1==0 && q==1000)
              _DEBUGHERE_("%d %d %d %d %g %g",fr1*self->nfr*self->nell+fr2*self->nell+q,q,fr1,fr2,rq[fr1*self->nfr*self->nell+fr2*self->nell+q],self->brq[fr1+fr2*self->nfr+q*self->nfr*self->nfr]);*/
          }
        }
      }
    }
    
    // do drq
    if (drq!=NULL) {
      for(ip=0;ip<self->np;ip++) {
        imod = ((int) self->cid[ip]/10) + 1;
        //_DEBUGHERE_("%d %d %d %d",im,ip,self->cid[ip],imod);
        if (imod==self->model[im]) {
          nip = self->cid[ip]%10;
          //_DEBUGHERE_("-->> %d %d %d",self->model[im],self->cid[ip],nip);
          for(fr1=0;fr1<self->nfr;fr1++) {
            for(fr2=0;fr2<self->nfr;fr2++) {
              for(q=0;q<self->nell;q++) {
                drq[fr1 * (self->nfr*self->nell*self->np) + 
                          fr2 * (self->nell*self->np) +
                          q * (self->np) +
                          ip] = self->brq[fr1 + 
                                          fr2 * (self->nfr) + 
                                          q * (self->nfr*self->nfr) + 
                                          nip * (self->nell*self->nfr*self->nfr)];
              }
            }
          }
        }
      }
    }
  }  
}

void egfs_free(void **pelf) {
  egfs *self;
  self=*pelf;
  clik_free_instance_(&(self->instance_id));
  free(self->buf);
  free(self->keys);
  free(self->values);
  free(self);
  *pelf = NULL;
}
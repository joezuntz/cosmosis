#include "clik.h"
#include "clik_helper.h"
#include <errno.h>
#include <string.h>

void free_CAMspec(void **none) {
  camspec_extra_free_();
}

double CAMspec_lkl(void* none, double* pars, error **err) {
  double lkl;
  
  camspec_extra_lkl_(&lkl,pars);
  return lkl;
}

void camspec_extra_init_(int*, int*,int*,int*,int*,int*, double*,double*,int*,double*,double*,double*,int*,int*,int*,int*,double*,double*,int*,int*,int*,double*,int*,int*,double*);
//SUBROUTINE CAMSPEC_EXTRA_INIT(iNspec, inX,ilminX,ilmaxX,inp,inpt, ic_inv,iX,ilmax_sz,isz_143_temp,iksz_temp,itszxcib_temp,ibeam_Nspec,inum_modes_per_beam,ibeam_lmax,icov_dim,ibeam_cov_inv,ibeam_modes,ihas_dust,ihas_calib,imarge_flag, imarge_mode,imarge_num, ikeep_num)

cmblkl* clik_CAMspec_init(cldf *df, int nell, int* ell, int* has_cl, double unit,double* wl, double *bins, int nbins, error **err) {
  int bok;
  cmblkl *cing;
  char likefile[2048],szfile[2048];
  int xcase;
  double xdim;
  char *xnames_def[] = {"A_ps_100","A_ps_143", "A_ps_217", "A_cib_143", "A_cib_217", "A_sz",  
                      "r_ps", "r_cib","n_Dl_cib","cal_100", "cal_143", "cal_217","xi_sz_cib","A_ksz","A_dust"};
  
  char *xnames_1[] = {"A_ps_100","A_ps_143", "A_ps_217", "A_cib_143", "A_cib_217", "A_sz",  
                      "r_ps", "r_cib","cal0", "cal1", "cal2"};
  char *xnames_0[] = {"A_ps_143", "A_cib_143", "A_sz",  
                      "cal1", "cal2"};

  char *xnames_tot[300];

  int Nspec, nX, lmax_sz;
  int *lmaxX, *lminX, *np, *npt;
  double *X,*c_inv,*tsz,*ksz,*tszXcib;
  int X_sz,c_inv_sz,sz_temp_sz;
  int beam_Nspec,num_modes_per_beam,beam_lmax,cov_dim;
  double *beam_cov_inv,*beam_modes;
  int beam_cov_inv_sz,beam_modes_sz;
  int i,j,cnt;
  int has_dust, has_calib_prior;
  int hk;
  int marge_num,keep_num;
  int *marge_flag;
  double *marge_mode;
  int sz;
  double bs_factor;

  //int frq[] = {100,143,217};
  
  camspec_extra_only_one_(&bok);
  testErrorRet(bok!=0,-100,"CAMspec already initialized",*err,__LINE__,NULL);
  
  Nspec = cldf_readint(df,"Nspec",err);
  forwardError(*err,__LINE__,NULL);
  
  lminX = cldf_readintarray(df,"lminX",&Nspec,err);
  forwardError(*err,__LINE__,NULL);
  lmaxX = cldf_readintarray(df,"lmaxX",&Nspec,err);
  forwardError(*err,__LINE__,NULL);

  np = cldf_readintarray(df,"np",&Nspec,err);
  forwardError(*err,__LINE__,NULL);
  npt = cldf_readintarray(df,"npt",&Nspec,err);
  forwardError(*err,__LINE__,NULL);
 
  nX = cldf_readint(df,"nX",err);
  forwardError(*err,__LINE__,NULL);

  lmax_sz = cldf_readint(df,"lmax_sz",err);
  forwardError(*err,__LINE__,NULL);

  X_sz = -1;
  X = cldf_readfloatarray(df,"X",&X_sz, err);
  forwardError(*err,__LINE__,NULL);
  
  c_inv_sz = -1;
  c_inv = cldf_readfloatarray(df,"c_inv",&c_inv_sz, err);
  forwardError(*err,__LINE__,NULL);

  sz_temp_sz = -1;
  tsz = cldf_readfloatarray(df,"tsz",&sz_temp_sz, err);
  forwardError(*err,__LINE__,NULL);
  sz_temp_sz = -1;
  ksz = cldf_readfloatarray(df,"ksz",&sz_temp_sz, err);
  forwardError(*err,__LINE__,NULL);
  sz_temp_sz = -1;
  tszXcib = cldf_readfloatarray(df,"tszXcib",&sz_temp_sz, err);
  forwardError(*err,__LINE__,NULL);


  beam_Nspec = cldf_readint(df,"beam_Nspec",err);
  forwardError(*err,__LINE__,NULL);

  if (beam_Nspec==0) {
    num_modes_per_beam = 0;
    beam_lmax = 0;
    cov_dim   = 0;
    beam_cov_inv = malloc_err(sizeof(double)*1,err);
    forwardError(*err,__LINE__,NULL);
    beam_modes = malloc_err(sizeof(double)*1,err);
    forwardError(*err,__LINE__,NULL);
  } else {
    beam_lmax = cldf_readint(df,"beam_lmax",err);
    forwardError(*err,__LINE__,NULL);
    num_modes_per_beam = cldf_readint(df,"num_modes_per_beam",err);
    forwardError(*err,__LINE__,NULL);
    cov_dim = cldf_readint(df,"cov_dim",err);
    forwardError(*err,__LINE__,NULL);
  
    beam_cov_inv_sz = -1;
    beam_cov_inv = cldf_readfloatarray(df,"beam_cov_inv",&beam_cov_inv_sz, err);
    forwardError(*err,__LINE__,NULL);
    beam_modes_sz = -1;
    beam_modes = cldf_readfloatarray(df,"beam_modes",&beam_modes_sz, err);
    forwardError(*err,__LINE__,NULL);
  }


  has_dust =  cldf_readint_default(df, "has_dust",0,err);
  forwardError(*err,__LINE__,NULL);

  has_calib_prior = cldf_readint_default(df, "has_calib_prior",1,err);

  hk = cldf_haskey(df,"marge_flag",err);
  forwardError(*err,__LINE__,NULL);
  marge_num = 0;
  if (hk==0) {
    marge_flag = malloc_err(sizeof(int)*cov_dim,err);
    forwardError(*err,__LINE__,NULL);
    for(i=0;i<cov_dim;i++) {
      marge_flag[i] = 0;
    }  
  } else {
    sz = cov_dim;
    marge_flag = cldf_readintarray(df,"marge_flag",&sz,err);
    forwardError(*err,__LINE__,NULL);
    for(i=0;i<cov_dim;i++) {
      marge_num += marge_flag[i];  
    }
  }
  keep_num = cov_dim - marge_num;

  if (marge_num>0) {
    sz = keep_num * marge_num;
    marge_mode = cldf_readfloatarray(df,"marge_mode",&sz,err);
    forwardError(*err,__LINE__,NULL);
  } else {
    marge_num = 1;
    marge_mode = malloc_err(sizeof(double)*keep_num,err);
    forwardError(*err,__LINE__,NULL);
  }

  bs_factor = cldf_readfloat_default(df,"bs_factor",1,err);
  forwardError(*err,__LINE__,NULL);

  camspec_extra_init_(&Nspec, &nX,lminX,lmaxX,np,npt,c_inv,X,&lmax_sz, tsz,ksz,tszXcib,&beam_Nspec,&num_modes_per_beam,&beam_lmax,&cov_dim,beam_cov_inv,beam_modes,&has_dust,&has_calib_prior,marge_flag,marge_mode,&marge_num,&keep_num,&bs_factor);    
  
  //camspec_extra_getcase_(&xcase);
  xdim = 14 + has_dust + keep_num;
  
  for(i=0;i<14+ has_dust;i++) {
    xnames_tot[i] = xnames_def[i];
  }

  cnt = 14+ has_dust;
  
  for(i=0;i<beam_Nspec;i++) {
    for(j=0;j<num_modes_per_beam;j++) {
      if (marge_flag[cnt-14-has_dust]==0) {
      xnames_tot[cnt] = malloc_err(sizeof(char)*50,err);
      forwardError(*err,__LINE__,NULL);

      sprintf(xnames_tot[cnt],"Bm_%d_%d",i+1,j+1);
      cnt++;        
      }
    }
  }

  cing = init_cmblkl(NULL, &CAMspec_lkl, 
                     &free_CAMspec,
                     nell,ell,
                     has_cl,ell[nell-1],unit,wl,0,bins,nbins,xdim,err);
  forwardError(*err,__LINE__,NULL);
  /*if (xcase==0) {
    cmblkl_set_names(cing, xnames_0,err);
  forwardError(*err,__LINE__,NULL);  
  } else {*/
  cmblkl_set_names(cing, xnames_tot,err);
  forwardError(*err,__LINE__,NULL);
  //}
  
  free(X);
  free(c_inv);
  free(ksz);
  free(tsz);
  free(tszXcib);
  free(beam_modes);
  free(beam_cov_inv);
  free(marge_flag);
  free(marge_mode);
  for(i=14 + has_dust;i<xdim;i++) {
    free(xnames_tot[i]);
  }

  return cing;
}

double* camspec_get_fg(void* camclik,double *par,int lmax,error **err) {
  double *res;

  res = malloc_err(sizeof(double)*(lmax+1)*4,err);
  forwardError(*err,__LINE__,NULL);

  camspec_extra_fg_(res,par,&lmax);

  return res;
}

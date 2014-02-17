#include "clik_helper.h"

// NO USER SERVICEABLE PART HERE.
// THINGS CAN CHANGE WITHOUT NOTICE. 



// ARE YOU STILL READING ?

// YOU HAVE BEEN WARNED !


#define _dealwitherr error *lerr,**err; if(_err==NULL) {lerr=NULL;err=&lerr;} else {err=_err;}

#define _forwardError(A,B,C) if(_err!=NULL) {forwardError(A,B,C);} else {quitOnError(A,B,stderr);}
#define _testErrorRetVA(A,B,C,D,E,F,...) if(_err!=NULL) {testErrorRetVA(A,B,C,D,E,F,__VA_ARGS__);} else {testErrorExitVA(A,B,C,D,E,__VA_ARGS__);}

#ifdef HDF5_COMPAT_MODE
double* hdf5_double_datarray(hid_t group_id,char*  cur_lkl,char* name,int* sz, error **err) {
  hsize_t ndum;
  H5T_class_t dum;
  size_t ddum;
  herr_t hstat;
  double *res;
  
  hstat = H5LTget_dataset_info( group_id, name, &ndum, &dum, &ddum);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  testErrorRetVA((ndum!=*sz && *sz>0),hdf5_base,"Bad size for %s in %s (got %d expected %d)",*err,__LINE__,NULL,name,cur_lkl,ndum,*sz);
  res = malloc_err(sizeof(double)*ndum,err);
  forwardError(*err,__LINE__,NULL);
  hstat = H5LTread_dataset_double(group_id, name,res);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  if (*sz<0) {
    *sz = ndum;
  }
  return res;
}

int* hdf5_int_datarray(hid_t group_id,char*  cur_lkl,char* name,int* sz, error **err) {
  hsize_t ndum;
  H5T_class_t dum;
  size_t ddum;
  herr_t hstat;
  int *res;
  
  hstat = H5LTget_dataset_info( group_id, name, &ndum, &dum, &ddum);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  testErrorRetVA((ndum!=*sz && *sz>0),hdf5_base,"Bad size for %s in %s (got %d expected %d)",*err,__LINE__,NULL,name,cur_lkl,ndum,*sz);
  res = malloc_err(sizeof(int)*ndum,err);
  forwardError(*err,__LINE__,NULL);
  hstat = H5LTread_dataset_int(group_id, name,res);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  if (*sz<0) {
    *sz = ndum;
  }
  return res;
  //_DEBUGHERE_("%d -> %d",res[0],res[ndum-1]);
}

double* hdf5_double_attarray(hid_t group_id,char*  cur_lkl,char* name,int* sz, error **err) {
  hsize_t ndum;
  H5T_class_t dum;
  size_t ddum;
  herr_t hstat;
  double *res;
  
  hstat = H5LTget_attribute_info( group_id, ".",name, &ndum, &dum, &ddum);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  testErrorRetVA((ndum!=*sz && *sz>0),hdf5_base,"Bad size for %s in %s (got %d expected %d)",*err,__LINE__,NULL,name,cur_lkl,ndum,*sz);
  res = malloc_err(sizeof(double)*ndum,err);
  forwardError(*err,__LINE__,NULL);
  hstat = H5LTget_attribute_double(group_id, ".",name,res);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  if (*sz<0) {
    *sz = ndum;
  }
  return res;
}

int* hdf5_int_attarray(hid_t group_id,char*  cur_lkl,char* name,int* sz, error **err) {
  hsize_t ndum;
  H5T_class_t dum;
  size_t ddum;
  herr_t hstat;
  int *res;
  
  hstat = H5LTget_attribute_info( group_id, ".",name, &ndum, &dum, &ddum);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  testErrorRetVA((ndum!=*sz && *sz>0),hdf5_base,"Bad size for %s in %s (got %d expected %d)",*err,__LINE__,NULL,name,cur_lkl,ndum,*sz);
  res = malloc_err(sizeof(double)*ndum,err);
  forwardError(*err,__LINE__,NULL);
  hstat = H5LTget_attribute_int(group_id, ".",name,res);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  if (*sz<0) {
    *sz = ndum;
  }
  return res;
}

char* hdf5_char_attarray(hid_t group_id,char*  cur_lkl,char* name,int* sz, error **err) {
  hsize_t ndum;
  H5T_class_t dum;
  size_t ddum;
  herr_t hstat;
  char *res;
  
  ndum = 1020;
  //_DEBUGHERE_("%d %s %s",group_id, ".",name);
  hstat = H5LTget_attribute_info( group_id, ".",name, &ddum, &dum, &ndum);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  testErrorRetVA((ndum!=*sz && *sz>0),hdf5_base,"Bad size for %s in %s (got %d expected %d)",*err,__LINE__,NULL,name,cur_lkl,ndum,*sz);
  //_DEBUGHERE_("%s size %ld hst %d ff %d",name,ndum,hstat,ddum);
  res = malloc_err(sizeof(char)*ndum,err);
  forwardError(*err,__LINE__,NULL);
  hstat = H5LTget_attribute_string(group_id, ".",name,res);
  testErrorRetVA(hstat<0,hdf5_base,"cannot read %s in %s (got %d)",*err,__LINE__,NULL,name,cur_lkl,hstat);
  if (*sz<0) {
    *sz = ndum;
  }
  return res;
}
#endif
int clik_getenviron_integer(char* name, int sfg, error **err) {
  int res;
  char *cres;
  int flg;
  
  cres = getenv(name);
  if (cres!=NULL) {
    flg = sscanf(cres,"%d",&res);
    if (flg==1) {
      return res;
    }
  }
  return sfg;
}

double clik_getenviron_real(char* name, double sfg, error **err) {
  double res;
  char *cres;
  int flg;

  cres = getenv(name);
  if (cres!=NULL) {
    flg = sscanf(cres,"%lg",&res);
    if (flg==1) {
      return res;
    }
  }
  return sfg;
}


char* clik_getenviron_string(char* name, char* sfg, error **err) {
  char *cres;
  
  cres = getenv(name);
  if (cres!=NULL) {
    return cres;
  }
  return sfg;
}

int clik_getenviron_numthread(char* name, int sfg, error **err) {
  int np;
  char fullname[2048];
  int i;
  
  sprintf(fullname,"%s_NUMTHREADS",name);
  for(i=0;i<strlen(name);i++) {
    fullname[i] = toupper(fullname[i]);
  }
  
  np = clik_getenviron_integer(fullname,sfg,err);
  forwardError(*err,__LINE__,sfg);
  
  testErrorRetVA(np<0 && np!=sfg, -100,"%s env variable meaningless",*err,__LINE__,sfg,fullname);
  //_DEBUGHERE_("%s = %d",fullname,np);
  return np;
}

cmblkl * clik_lklobject_init(cldf *df,error **err) {
  cmblkl *clkl;
  parname lkl_type;
  char *version;
  int has_cl[6];
  int nell, *ell,nbins,i,cli;
  double *wl,*bins;
  double unit;
  int lmin, lmax;
  char init_func_name[2048];
  clik_lkl_init_func *clik_dl_init;
  //clik_addon_init_func *clik_addondl_init;
  void* dlhandle;   
  char cur_addon[256];
  char *addon_type;
  int i_add,n_addons;
  int sz;
  char *dm;
  int *dmi;
  int hk,j;

#ifdef HAS_RTLD_DEFAULT 
  dlhandle = RTLD_DEFAULT;
#else
  dlhandle = NULL;
#endif

  // get the lkl type
  memset(lkl_type,0,_pn_size*sizeof(char));
  dm = cldf_readstr(df,"lkl_type",NULL,err);
  forwardError(*err,__LINE__,NULL);
  sprintf(lkl_type,"%s",dm);
  free(dm);

  // get unit
  unit = cldf_readfloat(df,"unit",err);
  forwardError(*err,__LINE__,NULL);
  
  sz = 6;
  dmi = cldf_readintarray(df,"has_cl",&sz,err);
  forwardError(*err,__LINE__,NULL);
  for(j=0;j<6;j++) {
    has_cl[j] = dmi[j];
  }
  free(dmi);

  // get ells
  hk = cldf_haskey(df,"lmax",err);
  forwardError(*err,__LINE__,NULL);
  if (hk==1) {
    // has lmax !
    lmax = cldf_readint(df,"lmax",err);
    forwardError(*err,__LINE__,NULL);
    
    lmin = cldf_readint_default(df,"lmin",0,err);
    forwardError(*err,__LINE__,NULL);
    
    nell = lmax+1-lmin;
    ell = malloc_err(sizeof(int)*(nell),err);
    forwardError(*err,__LINE__,NULL);
    for(i=lmin;i<lmax+1;i++) {
      ell[i-lmin] = i;
    }
  } else {
    nell = -1;
    ell = cldf_readintarray(df,"ell",&nell,err);
    forwardError(*err,__LINE__,NULL);
  }  
  
  lmax = ell[nell-1];
  
  // get wl
  wl = NULL;
  hk = cldf_haskey(df,"wl",err);
  forwardError(*err,__LINE__,NULL);
  if (hk==1) {
    int nwl;
    nwl = lmax+1;
    wl = cldf_readfloatarray(df,"wl",&nwl,err);
    forwardError(*err,__LINE__,NULL);
  }
  
  // deals with bins
  nbins = 0;
  bins = NULL;
  nbins = cldf_readint_default(df,"nbins",0,err);
  forwardError(*err,__LINE__,NULL);  
  
  if (nbins!=0) {
    int nd;
    int ncl;
    
    nd = 0;
    ncl = 0;
    for(cli=0;cli<6;cli++) {
      if (has_cl[cli]==1) {
        nd += nell;
        ncl++;
      }
    }

    hk = cldf_haskey(df,"bins",err);
    forwardError(*err,__LINE__,NULL);
    if (hk==1) { // full binning matrix
      int nbn;
      nbn = nbins*nd;

      bins = cldf_readfloatarray(df,"bins",&nbn,err);
      forwardError(*err,__LINE__,NULL);  
    } else { //packed binning matrix
      double *bin_ws;
      int *ellmin,*ellmax;
      int nw; 
      int ib,jb,il;
      int wsz;
      
      nw=-1;
      bin_ws = cldf_readfloatarray(df,"bin_ws",&nw,err);
      forwardError(*err,__LINE__,NULL);  
      
      ellmin = cldf_readintarray(df,"bin_lmin",&nbins,err);
      forwardError(*err,__LINE__,NULL);  

      ellmax = cldf_readintarray(df,"bin_lmax",&nbins,err);
      forwardError(*err,__LINE__,NULL);  
      
      bins = malloc_err(sizeof(double)*nd*nbins,err);
      forwardError(*err,__LINE__,NULL);  
      memset(bins,0,sizeof(double)*nd*nbins);
      
      jb=0;
      for(ib=0;ib<nbins;ib++) {        
        wsz = ellmax[ib] - ellmin[ib] + 1;
        testErrorRetVA(jb+wsz>nw,-11111,"argl bins",*err,__LINE__,NULL,"");
        memcpy(&(bins[ib*nd+ellmin[ib]]),&(bin_ws[jb]),wsz*sizeof(double));
        jb+=wsz;
      }
      free(ellmin);
      free(ellmax);
      free(bin_ws);
    }
  }
  
  clkl = NULL;

  sprintf(init_func_name,"clik_%s_init",lkl_type);
  clik_dl_init = dlsym(dlhandle,init_func_name);
  testErrorRetVA(clik_dl_init==NULL,-1111,"Cannot initialize lkl type %s from %s dl error : %s",*err,__LINE__,NULL,lkl_type,df->root,dlerror()); 

  clkl = clik_dl_init(df,nell,ell,has_cl,unit,wl,bins,nbins,err);
  forwardError(*err,__LINE__,NULL); 

  hk = cldf_haskey(df,"pipeid",err);
  forwardError(*err,__LINE__,NULL);
  if (hk==1) { 
    char vv[1000];
    version = cldf_readstr(df,"pipeid",NULL,err);
    forwardError(*err,__LINE__,NULL);  
      
    sprintf(vv,"%s %s",lkl_type,version);
    cmblkl_set_version(clkl,vv);
    free(version);
  } else {
    cmblkl_set_version(clkl,lkl_type);
  }

  hk = cldf_haskey(df,"free_calib",err);
  forwardError(*err,__LINE__,NULL);
  if (hk==1) {
    char *free_cal_name;
    char **xnames;
    parname *xnames_buf;
    int xdim;

    free_cal_name = cldf_readstr(df,"free_calib",NULL,err);
    forwardError(*err,__LINE__,NULL);
    
    xdim = clkl->xdim;
    xdim +=1;

    xnames = malloc_err(sizeof(char*)*xdim,err);
    forwardError(*err,__LINE__,NULL);
  
    xnames_buf = malloc_err(sizeof(parname)*xdim,err);
    forwardError(*err,__LINE__,NULL);
    
    for(i=0;i<xdim-1;i++) {
      sprintf(xnames_buf[i],"%s",clkl->xnames[i]);
      xnames[i] = (char*) &(xnames_buf[i]);
    }
    xnames[xdim-1] = free_cal_name;

    clkl->xdim = xdim;
    cmblkl_set_names(clkl, xnames,err);
    forwardError(*err,__LINE__,NULL);

    free(xnames);
    free(xnames_buf);
    free(free_cal_name);
    clkl->free_calib_id = clkl->xdim-1;

  }

  // cleanups
  if(wl!=NULL) {
    free(wl);    
  }
  if(bins!=NULL) {
    free(bins);    
  }
  free(ell);
  
  // look for addons
  //n_addons = cldf_readint_default(df,"n_addons",0,err);
  //forwardError(*err,__LINE__,NULL);
  //
  //for (i_add=0;i_add<n_addons;i_add++) {
  //  cldf *cdf;
  //  sprintf(cur_addon,"addon_%d",i_add);
  //  cdf  = cldf_openchild(df,cur_addon,err);
  //  forwardError(*err,__LINE__,NULL);
  //  addon_type = cldf_readstr(cdf,"addon_type",NULL,err);
  //  forwardError(*err,__LINE__,NULL);
  //
  //  sprintf(init_func_name,"clik_addon_%s_init",addon_type);
  //  clik_addondl_init = dlsym(dlhandle,init_func_name);
  //  testErrorRetVA(clik_addondl_init==NULL,-1111,"Cannot initialize addon type %s from %s/%s dl error : %s",*err,__LINE__,NULL,addon_type,df->root,cur_addon,dlerror());
  //  
  //  // pretty print purpose
  //  sprintf(cur_addon,"%s",cdf->root);
  //  
  //  clkl = clik_addondl_init(clkl,add_group_id,cur_addon,err);
  //  forwardError(*err,__LINE__,NULL);  
  //
  //  cldf_close(&cdf);
  //  free(addon_type);
  //}
  
  return clkl;
}




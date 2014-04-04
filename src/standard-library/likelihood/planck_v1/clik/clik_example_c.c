#include "clik.h"


int main_lensing(int argc, char **argv) {
  error *_err,**err;
  clik_lensing_object* clikid;
  int i,cli;
  int lmax;
  parname clnames[6];
  parname *names;
  int nextra;
  int ndim;
  double *cl_and_pars;
  double log_clikid;
  
  _err = initError();
  err = &_err;
  
  //testErrorExitVA(argc<3,-1,"Bad number of command line args!\nusage : %s clikidfile clfile [clfile ...]",*err,__LINE__,argv[0]);
  
  clikid = clik_lensing_init(argv[1],err);
  quitOnError(*err,__LINE__,stderr);
  
  nextra = clik_lensing_get_extra_parameter_names(clikid,&names,err);
  quitOnError(*err,__LINE__,stderr);
  
  if (nextra!=0) {
    fprintf(stdout,"With %d extra parameters\n",nextra);
    for(i=0;i<nextra;i++) {
      fprintf(stdout,"  %s\n",names[i]);
    }    
  }
  free(names);
  
  lmax = clik_lensing_get_lmax(clikid,err);
  quitOnError(*err,__LINE__,stderr);
  
  ndim = nextra + 2*(lmax+1);
  
  for(i=2;i<argc;i++) {
    // read cl as ascii file
    cl_and_pars = read_double_vector(argv[i],ndim,err);
    quitOnError(*err,__LINE__,stderr);

    log_clikid = clik_lensing_compute(clikid,cl_and_pars,err);
    quitOnError(*err,__LINE__,stderr);
    
    fprintf(stdout,"Log likelihood for file %s : %g\n",argv[i],log_clikid);
    
    free(cl_and_pars);
  }
  
  clik_lensing_cleanup(&clikid);
}

int main(int argc, char **argv) {
  error *_err,**err;
  clik_object* clikid;
  int i,cli;
  int has_cl[6],lmax[6];
  parname clnames[6];
  parname *names;
  int nextra;
  int ndim;
  double *cl_and_pars;
  double log_clikid;
  int isl;

  _err = initError();
  err = &_err;
  
  if (argc<2) {
    fprintf(stderr, "usage: %s clikfile [clfile ...]\n  Prints some info on the likelihood clikfile \n  and compute log likelihoods for each clfile\n",argv[0]);
    return -1;
  }

  //testErrorExitVA(argc<3,-1,"Bad number of command line args!\nusage : %s clikidfile clfile [clfile ...]",*err,__LINE__,argv[0]);
  
  isl  = clik_try_lensing(argv[1],err);
  quitOnError(*err,__LINE__,stderr);
  if (isl==1) {
    return main_lensing(argc,argv);
  }

  clikid = clik_init(argv[1],err);
  quitOnError(*err,__LINE__,stderr);
  
  // retrieve has_cl and lmax
  clik_get_has_cl(clikid,has_cl,err);
  quitOnError(*err,__LINE__,stderr);
  clik_get_lmax(clikid,lmax,err);
  quitOnError(*err,__LINE__,stderr);
  
  sprintf(clnames[0],"TT");
  sprintf(clnames[1],"EE");
  sprintf(clnames[2],"BB");
  sprintf(clnames[3],"TE");
  sprintf(clnames[4],"TB");
  sprintf(clnames[5],"EB");
  
  fprintf(stdout,"Likelihood use Cl\n");
  for(cli=0;cli<6;cli++) {
    if (has_cl[cli]==1) {
      fprintf(stdout,"  %s from l=0 to l=%d (incl.)\n",clnames[cli],lmax[cli]);
    }    
  }
  
  nextra = clik_get_extra_parameter_names(clikid,&names,err);
  quitOnError(*err,__LINE__,stderr);
  
  if (nextra!=0) {
    fprintf(stdout,"With %d extra parameters\n",nextra);
    for(i=0;i<nextra;i++) {
      fprintf(stdout,"  %s\n",names[i]);
    }    
  }
  free(names);
  
  // compute size of the parameter vector
  
  ndim = nextra;
  for(cli=0;cli<6;cli++) {
    ndim += lmax[cli] + 1;
  }
  
  fprintf(stdout,"parameter vector has %d elements\n",ndim);
  
  for(i=2;i<argc;i++) {
    // read cl as ascii file
    cl_and_pars = read_double_vector(argv[i],ndim,err);
    quitOnError(*err,__LINE__,stderr);

    log_clikid = clik_compute(clikid,cl_and_pars,err);
    quitOnError(*err,__LINE__,stderr);
    
    fprintf(stdout,"Log likelihood for file %s : %g\n",argv[i],log_clikid);
    
    free(cl_and_pars);
  }
  
  clik_cleanup(&clikid);
}
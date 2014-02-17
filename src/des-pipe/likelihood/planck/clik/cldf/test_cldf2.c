#include "cldf.h"

int main(int argc, char **argv) {
	int fitserr;
  fitsfile *fitsptr;
  cldf *df;
  char tt[1000];
  error **err,*_err;

  _err = NULL;

  err = &_err;

  sprintf(tt,"%s/%s",argv[1],argv[2]);
  _DEBUGHERE_("%s",tt);
  fits_open_data(&fitsptr, tt, READONLY, &fitserr);
  
  fits_close_file(fitsptr, &fitserr);
_DEBUGHERE_("%s",tt);

	df = cldf_open(argv[1],err);
	quitOnError(*err,__LINE__,stderr);
_DEBUGHERE_("%s",tt);
cldf_close(&df);
_DEBUGHERE_("%s",tt);
	

  
}
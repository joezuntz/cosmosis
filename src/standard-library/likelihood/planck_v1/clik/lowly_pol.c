#ifdef __PLANCK__
#include "HL2_likely/target/lowly_pol.h"
#else
#include "lowly_pol.h"
#endif



/*******************************************************************************

 This part assumes that T and Pol mask are identical

 ******************************************************************************/

#undef __FUNC__
#define __FUNC__ "scalCovPol"
/* Polarized pixel covariance for a given rotation */
/* This is a 3x3 matrix, expressed as a linear array */
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
                 const double *qeb) {

  double c2apg, s2apg, c2amg, s2amg;  
  double xitt, xip, rxim, ixim, rxic, ixic;
  double rxic2,ixic2,rxim2,ixim2,rxip2,ixip2;
  double dl00, dl22, dl2m2, dl20;
  double plm2, plm1;
  long l;
  double dl;
  
  /* Construct cos2(alpha \pm gamma), sin2(alpha \pm gamma) */
  c2apg = c2a*c2g-s2a*s2g;
  s2apg = s2a*c2g+c2a*s2g;
  c2amg = c2a*c2g+s2a*s2g;
  s2amg = s2a*c2g-c2a*s2g;
  
  /* Recurrence loop to construct rotation invariant correlations */
  
  /* Temperature auto-correlation */
  plm2=1.0;
  plm1=cb;
  xitt = 1.0*plm2*qtt[0] + 3.0*plm1*qtt[1];
  for (l=2;l<=lmax;l++) {
    dl = (double) l;
    dl00 = 2.0*cb*plm1 - plm2 - (cb*plm1-plm2)/dl;
    xitt += (2.0*dl+1)*dl00*qtt[l];
    plm2 = plm1;
    plm1 = dl00;
  }
  xitt /= 4.0*M_PI;
  
  /* Temperature-polarization cross-correlation */
  plm2 = sqrt(6.0)*(1.0+cb)*(1.0-cb)/4.0; /* d^2_{20} */
  plm1 = sqrt(5.0)*cb*plm2; /* d^3_{20} */
  rxic = 5.0*plm2*qte[2] + 7.0*plm1*qte[3];
  ixic = -(5.0*plm2*qtb[2] + 7.0*plm1*qtb[3]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl20 = -(2.0*dl-1)/sqrt((dl-2.0)*(dl+2.0))*
      ( -cb*plm1 + sqrt((dl-3.0)*(dl+1.0))/(2.0*dl-1.0)*plm2 );
    rxic += (2.0*dl+1.0)*dl20*qte[l];
    ixic -= (2.0*dl+1.0)*dl20*qtb[l];
    plm2 = plm1;
    plm1 = dl20;
  }
  rxic /= 4.0*M_PI;
  ixic /= 4.0*M_PI;
  
  /* Polarization '+' correlation, based on d^l_{22} */
  plm2 = (1.0+cb)*(1.0+cb)/4.; /* d^2_{22} */
  plm1 = (3.0*cb-2.0)*plm2; //!!!
  xip = 5.0*plm2*(qee[2]+qbb[2]) + 7.0*plm1*(qee[3]+qbb[3]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl22 = -dl*(2.0*dl-1.0)/((dl-2.0)*(dl+2.0))*
      ( (4.0/dl/(dl-1.0)-cb)*plm1 +(dl-3.0)*(dl+1.0)/((dl-1.0)*(2.0*dl-1.0))*plm2 );
    xip += (2.0*dl+1.0)*dl22*(qee[l]+qbb[l]);
    plm2 = plm1;
    plm1 = dl22;
  }
  xip /= 4.0*M_PI;
  
  /* Polarization '-' correlation, based on d^l_{2-2} */
  plm2 = (1.0-cb)*(1.0-cb)/4.0; /* d^2_{2-2} */
  plm1 = (3.0*cb+2.0)*plm2; ///!!!
  rxim = 5.0*plm2*(qee[2]-qbb[2]) + 7.0*plm1*(qee[3]-qbb[3]);
  ixim = -2.0*(5.0*plm2*qeb[2] + 7.0*plm1*qeb[3]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl2m2 = -dl*(2.0*dl-1.0)/((dl-2.0)*(dl+2.0))*
      ( (-4.0/dl/(dl-1.0)-cb)*plm1 +(dl-3.0)*(dl+1.0)/((dl-1.0)*(2.0*dl-1.0))*plm2 );
    rxim += (2.0*dl+1)*dl2m2*(qee[l]-qbb[l]);
    ixim -= 2.0*(2.0*dl+1.0)*dl2m2*qeb[l];
    plm2 = plm1;
    plm1 = dl2m2;
    }
  rxim /= 4.0*M_PI;
  ixim /= 4.0*M_PI;
  
  /* Now put frame dependence (alpha, gamma), suffix '2'*/
  rxic2 = rxic*c2g - ixic*s2g;
  ixic2 = rxic*s2g + ixic*c2g;
  rxip2 = xip*c2amg;
  ixip2 = xip*s2amg;
  rxim2 = rxim*c2apg - ixim*s2apg;
  ixim2 = rxim*s2apg + ixim*c2apg;
  
  /* Now fill covariance matrix */
  //covmat = (double*) malloc_err(9*sizeof(double),err);
  
  covmat[0] = xitt;
  covmat[1] = rxic2; /* TQ */
  covmat[2] = ixic2; /* TU */
  covmat[3] = covmat[1];
  covmat[4] = (rxip2+rxim2)/2.0; /* QQ */
  covmat[5] = (ixip2+ixim2)/2.0; /* QU */
  covmat[6] = covmat[2];
  covmat[7] = covmat[5];
  covmat[8] = (rxip2-rxim2)/2.0; /* UU */

  return;    
}


#undef __FUNC__
#define __FUNC__ "build_trigo_matrices"
double* build_trigo_matrices(const long npix_seen,
                             const double * xyz,
                             error **err) {
  
  long i,j,d;
  double e_r_i[3], e_r_j[3], e_theta_i[3], e_phi_i[3];
  double cb, norm, den, x1, x2;
  double *res,*cb_ij, *c2psi_ij, *s2psi_ij;
  
  res = (double*) malloc_err(3*_SZT_(npix_seen)*_SZT_(npix_seen)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  
  cb_ij    = res;  
  c2psi_ij = res + npix_seen*npix_seen;
  s2psi_ij = res + 2*npix_seen*npix_seen;
  
  for (i=0;i<npix_seen;i++) {
    for (j=0;j<npix_seen;j++) {
      if (i==j) { // diagonal case: same pixel 'pair'
        cb_ij[i*npix_seen+i]=1.0;
        c2psi_ij[i*npix_seen+i]=1.0;
        s2psi_ij[i*npix_seen+i]=0.0;
      } else {
        // Get local values of e_r for this pixel pair
        for (d=0;d<3;d++) {
          e_r_i[d] = xyz[3*i+d];
          e_r_j[d] = xyz[3*j+d];
        }
        // e_phi = (-y,x,0)/sqrt(x^2+y^2)
        e_phi_i[0] = -e_r_i[1];
        e_phi_i[1] = e_r_i[0];
        e_phi_i[2] = 0.0;
        norm = sqrt(e_phi_i[0]*e_phi_i[0]+e_phi_i[1]*e_phi_i[1]);
        e_phi_i[0] /= norm;
        e_phi_i[1] /= norm;
        // e_theta = e_phi x e_r
        e_theta_i[0] =  e_phi_i[1]*e_r_i[2] - e_phi_i[2]*e_r_i[1];
        e_theta_i[1] = -e_phi_i[0]*e_r_i[2] + e_phi_i[2]*e_r_i[0];
        e_theta_i[2] =  e_phi_i[0]*e_r_i[1] - e_phi_i[1]*e_r_i[0];
        
        x1=0.0;
        x2=0.0;
        cb=0.0;
        // Compute dot products
        for (d=0;d<3;d++) {
          x1 += e_theta_i[d]* e_r_j[d];
          x2 += e_phi_i[d]  * e_r_j[d];
          cb += e_r_i[d]    * e_r_j[d];
        }
        //if (cb > 1.0) cb=1.0;
        //if (cb < -1.0) cb=-1.0;
        // Store cb
        cb_ij[i*npix_seen+j] = cb;
        // Compute c2psi, s2psi
        den = x1*x1+x2*x2;
        if (fabs(den) < 1e-10) {// Paire antipodale
          c2psi_ij[i*npix_seen+j]=1.0;
          s2psi_ij[i*npix_seen+j]=0.0;
        } else {
          c2psi_ij[i*npix_seen+j] = (x1*x1-x2*x2)/den;
          s2psi_ij[i*npix_seen+j] = 2.0*x1*x2/den;
        }
      }
    }
  }
  return res;
}

#undef __FUNC__
#define __FUNC__ "build_trig_mat_plist"
double* build_trig_mat_plist(const long nside,
                             const long * pixel_indices,
                             const long npix_seen,
                             const int ordering,
                             error **err) {
  double *posvec;
  double *res;
  
  posvec = lowly_get_posvec(nside,pixel_indices,npix_seen,ordering,err);
  forwardError(*err,__LINE__,NULL);

  res = build_trigo_matrices(npix_seen,posvec,err);
  forwardError(*err,__LINE__,NULL);

  free(posvec);
  
  return res;
}
#undef __FUNC__
#define __FUNC__ "build_cov_matrix_pol"

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
                              error **err) {
  
  long i,j;
  double cb, c2a, c2g, s2a, s2g;
  double cov[9];
  double *covmat;
  double *cb_ij,*c2psi_ij,*s2psi_ij;
  
  MALLOC_IF_NEEDED(covmat,orig,9*_SZT_(npix_seen)*_SZT_(npix_seen)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  memset((void*)covmat,0,9*npix_seen*npix_seen*sizeof(double));

  cb_ij    = trig_mat;  
  c2psi_ij = trig_mat + npix_seen*npix_seen;
  s2psi_ij = trig_mat + 2*npix_seen*npix_seen;
  
  for (i=0;i<npix_seen;i++) {
    //for (j=0;j<=i;j++) {
    for (j=0;j<npix_seen;j++) {
	    cb = cb_ij[i*npix_seen+j];
	    c2a = c2psi_ij[i*npix_seen+j];
	    c2g = c2psi_ij[j*npix_seen+i];
	    s2a = s2psi_ij[i*npix_seen+j];
	    s2g = s2psi_ij[j*npix_seen+i];
	    scalCovPol(cov,cb,c2a,s2a,c2g,s2g,lmax,qtt,qee,qbb,qte,qtb,qeb); 
	    // Add noise
	    if (i==j) {
	      cov[0] += noisevar[i];
	      cov[4] += noisevar[npix_seen+i];
	      cov[8] += noisevar[2*npix_seen+i];
	    }
	    // Now store in big monster matrix
	    covmat[3*i*npix_seen+j]=cov[0]; //TT lower
	    //covmat[3*j*npix_seen+i]=cov[0]; //TT upper
      
	    // PATCH: TAKE -transpose(TQ,TU): needs to find why !!! //
	    covmat[(j+npix_seen)*3*npix_seen+i]=-cov[1]; //TQ lower
	    covmat[i*3*npix_seen+npix_seen+j]=-cov[1]; //TQ upper
	    covmat[(j+2*npix_seen)*3*npix_seen+i]=-cov[2]; //TU lower 
	    covmat[i*3*npix_seen+2*npix_seen+j]=-cov[2]; //TU upper
	    //
      
	    covmat[(i+npix_seen)*3*npix_seen+npix_seen+j]=cov[4]; //QQ lower
	    //covmat[(j+npix_seen)*3*npix_seen+npix_seen+i]=cov[4]; //QQ upper
	    covmat[(i+2*npix_seen)*3*npix_seen+npix_seen+j]=cov[5]; //QU lower
	    covmat[(j+npix_seen)*3*npix_seen+2*npix_seen+i]=cov[5]; //QU upper
	    covmat[(i+2*npix_seen)*3*npix_seen+2*npix_seen+j]=cov[8]; //UU lower
	    //covmat[(j+2*npix_seen)*3*npix_seen+2*npix_seen+i]=cov[8]; //UU upper	    
    }
  } 
  return covmat;
}


/*******************************************************************************
 
 This part allows T and Pol mask to be different 
 
 ******************************************************************************/


#undef __FUNC__
#define __FUNC__ "scalCovCross"
/* Pixel TQ, TU covariance for a given separation, and a given third euler angle */
void scalCovCross(double *covcross,
                  const double cb,
                  const double c2g,
                  const double s2g,
                  const long lmax,
                  const double *qte,
                  const double *qtb) {
  
  double rxic, ixic;
  double rxic2,ixic2;
  double dl20;
  double plm2, plm1;
  long l;
  double dl;

  /* Temperature-polarization cross-correlation */
  plm2 = sqrt(6.0)*(1.0+cb)*(1.0-cb)/4.0; /* d^2_{20} */
  plm1 = sqrt(5.0)*cb*plm2; /* d^3_{20} */
  rxic = 5.0*plm2*qte[2] + 7.0*plm1*qte[3];
  ixic = -(5.0*plm2*qtb[2] + 7.0*plm1*qtb[3]);
  //_DEBUGHERE_("%g %g %g %g",rxic,ixic,qte[2],qtb[2]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl20 = -(2.0*dl-1)/sqrt((dl-2.0)*(dl+2.0))*
      ( -cb*plm1 + sqrt((dl-3.0)*(dl+1.0))/(2.0*dl-1.0)*plm2 );
    rxic += (2.0*dl+1.0)*dl20*qte[l];
    ixic -= (2.0*dl+1.0)*dl20*qtb[l];
    plm2 = plm1;
    plm1 = dl20;
  }
  rxic /= 4.0*M_PI;
  ixic /= 4.0*M_PI;
  /* Now rotate into e_theta, e_phi frame */
  rxic2 = rxic*c2g - ixic*s2g;
  ixic2 = rxic*s2g + ixic*c2g;
  // Store result
  covcross[0]=rxic2;
  covcross[1]=ixic2;
  
}

#undef __FUNC__
#define __FUNC__ "scalCovQU"
/* Pixel QQ,UU,QU covariance for a given separation and euler angles */
// Result stored as [QQ,UU,QU]
void scalCovQU(double *covQU,
               const double cb,
               const double c2a,
               const double s2a,
               const double c2g,
               const double s2g,
               const long lmax,
               const double *qee,
               const double *qbb,
               const double *qeb) {

  double c2apg, s2apg, c2amg, s2amg;  
  double xip, rxim, ixim;
  double rxim2,ixim2,rxip2,ixip2;
  double dl22, dl2m2;
  double plm2, plm1;
  long l;
  double dl;
  
  /* Construct cos2(alpha \pm gamma), sin2(alpha \pm gamma) */
  c2apg = c2a*c2g-s2a*s2g;
  s2apg = s2a*c2g+c2a*s2g;
  c2amg = c2a*c2g+s2a*s2g;
  s2amg = s2a*c2g-c2a*s2g;
  
  /* Polarization '+' correlation, based on d^l_{22} */
  plm2 = (1.0+cb)*(1.0+cb)/4.; /* d^2_{22} */
  plm1 = (3.0*cb-2.0)*plm2; //!!!
  xip = 5.0*plm2*(qee[2]+qbb[2]) + 7.0*plm1*(qee[3]+qbb[3]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl22 = -dl*(2.0*dl-1.0)/((dl-2.0)*(dl+2.0))*
    ( (4.0/dl/(dl-1.0)-cb)*plm1 +(dl-3.0)*(dl+1.0)/((dl-1.0)*(2.0*dl-1.0))*plm2 );
    xip += (2.0*dl+1.0)*dl22*(qee[l]+qbb[l]);
    plm2 = plm1;
    plm1 = dl22;
  }
  xip /= 4.0*M_PI;
  
  /* Polarization '-' correlation, based on d^l_{2-2} */
  plm2 = (1.0-cb)*(1.0-cb)/4.0; /* d^2_{2-2} */
  plm1 = (3.0*cb+2.0)*plm2; ///!!!
  rxim = 5.0*plm2*(qee[2]-qbb[2]) + 7.0*plm1*(qee[3]-qbb[3]);
  ixim = -2.0*(5.0*plm2*qeb[2] + 7.0*plm1*qeb[3]);
  for (l=4;l<=lmax;l++) {
    dl = (double)l;
    dl2m2 = -dl*(2.0*dl-1.0)/((dl-2.0)*(dl+2.0))*
      ( (-4.0/dl/(dl-1.0)-cb)*plm1 +(dl-3.0)*(dl+1.0)/((dl-1.0)*(2.0*dl-1.0))*plm2 );
    rxim += (2.0*dl+1)*dl2m2*(qee[l]-qbb[l]);
    ixim -= 2.0*(2.0*dl+1.0)*dl2m2*qeb[l];
    plm2 = plm1;
    plm1 = dl2m2;
  }
  rxim /= 4.0*M_PI;
  ixim /= 4.0*M_PI;
  
  /* Now put frame dependence (alpha, gamma), suffix '2'*/
  rxip2 = xip*c2amg;
  ixip2 = xip*s2amg;
  rxim2 = rxim*c2apg - ixim*s2apg;
  ixim2 = rxim*s2apg + ixim*c2apg;
  
  covQU[0] = (rxip2+rxim2)/2.0; /* QQ */
  covQU[1] = (ixip2+ixim2)/2.0; /* QU */
  covQU[2] = (rxip2-rxim2)/2.0; /* UU */
}

#undef __FUNC__
#define __FUNC__ "build_trigo_matrices_general"
double* build_trigo_matrices_general (const long npix_temp,
                                      const long npix_pol,
                                      const double * xyz_temp,
                                      const double * xyz_pol,
                                      error **err) {

  long i,j,d,TROIS;
  double *e_r_i, *e_r_j, e_theta_i[3], e_phi_i[3];
  double cb, norm, den, x1, x2;
  double *res,*cb_ij, *c2psi_ij, *s2psi_ij;
  long npix_sum;
  double * xyz;

  npix_sum = npix_temp + npix_pol;

  xyz = (double*) malloc_err(3*npix_sum*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);

  memcpy(xyz,xyz_temp,3*npix_temp*sizeof(double));
  memcpy(xyz+3*npix_temp,xyz_pol,3*npix_pol*sizeof(double));
  
  TROIS = 3;
  if (npix_pol==0) {
    TROIS = 1;
  }
  
  res = (double*) malloc_err(TROIS*_SZT_(npix_sum)*_SZT_(npix_sum)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  
  cb_ij    = res;  
  c2psi_ij = res + npix_sum*npix_sum;
  s2psi_ij = res + 2*npix_sum*npix_sum;
    
  //Everybody 
#pragma omp parallel for default (shared) private (i,j,d,e_r_i,e_r_j,e_phi_i,norm,e_theta_i,x1,x2,cb,den)
  for (i=0;i<npix_sum;i++) {
    for (j=0;j<npix_sum;j++) {
      if (i==j) { // diagonal case: same pixel 'pair'
        cb_ij[i*npix_sum+i]=1.0;
      } else {
        // Get local values of e_r for this pixel pair
        e_r_i = &xyz[3*i];
        e_r_j = &xyz[3*j];
        
        cb=0.0;
        // Compute dot products
        for (d=0;d<3;d++) {
          cb += e_r_i[d]    * e_r_j[d];
        }
        cb_ij[i*npix_sum+j] = cb;
      }
    }
  }
  
  if (npix_pol==0) {
    free(xyz);
    return res;
  }
  //Polar only
#pragma omp parallel for default (shared) private (i,j,d,e_r_i,e_r_j,e_phi_i,norm,e_theta_i,x1,x2,cb,den)
  for (i=0;i<npix_sum;i++) {
    for (j=0;j<npix_sum;j++) {
      if (i==j) { // diagonal case: same pixel 'pair'
        c2psi_ij[i*npix_sum+i]=1.0;
        s2psi_ij[i*npix_sum+i]=0.0;
      } else {
        // Get local values of e_r for this pixel pair
        e_r_i = &xyz[3*i];
        e_r_j = &xyz[3*j];

        // e_phi = (-y,x,0)/sqrt(x^2+y^2)
        e_phi_i[0] = -e_r_i[1];
        e_phi_i[1] = e_r_i[0];
        e_phi_i[2] = 0.0;
        norm = sqrt(e_phi_i[0]*e_phi_i[0]+e_phi_i[1]*e_phi_i[1]);
        e_phi_i[0] /= norm;
        e_phi_i[1] /= norm;
        // e_theta = e_phi x e_r
        e_theta_i[0] =  e_phi_i[1]*e_r_i[2] - e_phi_i[2]*e_r_i[1];
        e_theta_i[1] = -e_phi_i[0]*e_r_i[2] + e_phi_i[2]*e_r_i[0];
        e_theta_i[2] =  e_phi_i[0]*e_r_i[1] - e_phi_i[1]*e_r_i[0];
        
        x1=0.0;
        x2=0.0;
        cb=0.0;
        // Compute dot products
        for (d=0;d<3;d++) {
          x1 += e_theta_i[d]* e_r_j[d];
          x2 += e_phi_i[d]  * e_r_j[d];
        }

        // Compute c2psi, s2psi
        den = x1*x1+x2*x2;
        if (fabs(den) < 1e-10) {// Antipodal pair, or duplicated pixel
          c2psi_ij[i*npix_sum+j]=1.0;
          s2psi_ij[i*npix_sum+j]=0.0;
        } else {
          c2psi_ij[i*npix_sum+j] = (x1*x1-x2*x2)/den;
          s2psi_ij[i*npix_sum+j] = 2.0*x1*x2/den;
        }
      }
    }
  }
  free(xyz);
  return res;
}

#undef __FUNC__
#define __FUNC__ "build_trig_mat_general_plist"
double* build_trig_mat_general_plist(const long nside,
                                     const long * pixel_temp,
                                     const long npix_temp,
                                     const long * pixel_pol,
                                     const long npix_pol,
                                     const int ordering,
                                     error **err) {
  double *pos_temp,*pos_pol;
  double *res;
  
  pos_temp = lowly_get_posvec(nside,pixel_temp,npix_temp,ordering,err);
  forwardError(*err,__LINE__,NULL);
  pos_pol = lowly_get_posvec(nside,pixel_pol,npix_pol,ordering,err);
  forwardError(*err,__LINE__,NULL);
  
  res = build_trigo_matrices_general(npix_temp,npix_pol,pos_temp,pos_pol,err);
  forwardError(*err,__LINE__,NULL);
  free(pos_temp);
  if (pos_pol != NULL)
    free(pos_pol);
  
  return res;
}


#undef __FUNC__
#define __FUNC__ "build_cov_matrix_pol_general"
// Beware: needs to be used in conjonction with build_trigo_matrices_general()
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
                                      error **err) {
  
  long i,j;
  double cb, c2a, c2g, s2a, s2g;
  double covTT;
  double covcross[2];
  double covQU[3];
  double *covmat;
  long npix_tot, npix_sum;
  double *cb_ij,*c2psi_ij,*s2psi_ij; 
  
  npix_tot = npix_temp+2*npix_pol; // Linear size of covariance matrix, size of noisevar array
  npix_sum = npix_temp+npix_pol; // Linear size of angle matrices
  
  cb_ij    = trig_mat;  
  c2psi_ij = trig_mat + npix_sum*npix_sum;
  s2psi_ij = trig_mat + 2*npix_sum*npix_sum;
  
  MALLOC_IF_NEEDED(covmat,orig,_SZT_(npix_tot)*_SZT_(npix_tot)*sizeof(double),err);
  forwardError(*err,__LINE__,NULL);
  memset((void*)covmat,0,npix_tot*npix_tot*sizeof(double));
  
  /*_DEBUGHERE_("TT %g %g %g %g %g %g",qtt[0],qtt[1],qtt[2],qtt[3],qtt[4],qtt[5]);
  _DEBUGHERE_("EE %g %g %g %g %g %g",qee[0],qee[1],qee[2],qee[3],qee[4],qee[5])
  _DEBUGHERE_("BB %g %g %g %g %g %g",qbb[0],qbb[1],qbb[2],qbb[3],qbb[4],qbb[5])
  _DEBUGHERE_("TE %g %g %g %g %g %g",qtt[0],qte[1],qte[2],qte[3],qte[4],qte[5])
  _DEBUGHERE_("TB %g %g %g %g %g %g",qtt[0],qtb[1],qtb[2],qtb[3],qtb[4],qtb[5])
  _DEBUGHERE_("EB %g %g %g %g %g %g",qeb[0],qeb[1],qeb[2],qeb[3],qeb[4],qeb[5])*/
  // Compute blocks separately now
  // Temperature block
  //_DEBUGHERE_("omp T","");
#pragma omp parallel for default (shared) private (i,j,cb,covTT)
  
  for (i=0;i<npix_temp;i++) {
    for (j=0;j<npix_temp;j++) {
      cb = cb_ij[i*npix_sum+j];
      covTT = scalCov(cb,lmax,qtt);
      if ((i==j) && (noisevar!=NULL)) {
        covTT += noisevar[i];
      }
      covmat[i*npix_tot+j] = covTT;
    }
  }
  
  // TQ and TU blocks
  //_DEBUGHERE_("omp PT","");
#pragma omp parallel for default (shared) private (i,j,cb,c2g,s2g,covcross)
  for (i=0;i<npix_temp;i++) {
    for (j=0;j<npix_pol;j++) {
      cb = cb_ij[i*npix_sum+npix_temp+j];
      c2g = c2psi_ij[(j+npix_temp)*npix_sum+i];
      s2g = s2psi_ij[(j+npix_temp)*npix_sum+i];
      //_DEBUGHERE_("%g %g %g",cb,c2g,s2g);
      scalCovCross(covcross,cb,c2g,s2g,lmax,qte,qtb);
      // PATCH: TAKE -transpose(TQ,TU): needs to find why !!! //
      //_DEBUGHERE_("%g %g",covcross[0],covcross[1]);
      
      covmat[(j+npix_temp)*npix_tot+i]=-covcross[0]; //TQ lower
      covmat[i*npix_tot+npix_temp+j]=-covcross[0]; //TQ upper
      covmat[(j+npix_temp+npix_pol)*npix_tot+i]=-covcross[1]; //TU lower 
      covmat[i*npix_tot+npix_temp+npix_pol+j]=-covcross[1]; //TU upper
      
    }      
  }
  
  // QQ, UU and QU blocks
  //_DEBUGHERE_("omp P","");
#pragma omp parallel for default (shared) private (i,j,cb,c2a,s2a,c2g,s2g,covQU)
  for (i=0;i<npix_pol;i++) {
    for (j=0;j<npix_pol; j++) {
      cb = cb_ij[(i+npix_temp)*npix_sum+npix_temp+j];
      c2a = c2psi_ij[(i+npix_temp)*npix_sum+npix_temp+j];
      s2a = s2psi_ij[(i+npix_temp)*npix_sum+npix_temp+j];
      c2g = c2psi_ij[(j+npix_temp)*npix_sum+npix_temp+i];
      s2g = s2psi_ij[(j+npix_temp)*npix_sum+npix_temp+i];
      scalCovQU(covQU,cb,c2a,s2a,c2g,s2g,lmax,qee,qbb,qeb);
      if ((i==j) &&(noisevar!=NULL)) {//add noise, assumes diag noise maps for TT, QQ, UU
        covQU[0] += noisevar[i+npix_temp];
        covQU[2] += noisevar[i+npix_temp+npix_pol];
      }
      covmat[(i+npix_temp)*npix_tot+npix_temp+j]=covQU[0]; // QQ
      covmat[(i+npix_temp+npix_pol)*npix_tot+npix_temp+npix_pol+j] = covQU[2]; // UU
      covmat[(i+npix_temp+npix_pol)*npix_tot+npix_temp+j] = covQU[1]; // QU lower
      covmat[(j+npix_temp)*npix_tot+npix_temp+npix_pol+i] = covQU[1]; // QU lower
    }
  }
  
  return covmat;
}

#undef __FUNC__
#define __FUNC__ "init_plowly"
plowly *init_plowly(int nside,char* ordering, 
                    unsigned char *mask_T, unsigned char *mask_P, 
                    double *mapT,double *mapQ, double *mapU,
                    double *Ndiag, double* N,int reduced,
                    long lmax,int nell, int *ell,
                    double *Cl,int *has_cl,
                    error **err) {
  plowly *self;
  int order;
  long *pix_T,*pix_P;
  long npix_t,npix_p;
  int i,j,ox,oy,offset;
  
  self = malloc_err(sizeof(plowly), err);
  forwardError(*err,__LINE__,NULL);
  SET_PRINT_STAT(self);
  
  self->nside=nside;

  order = lowly_which_order(ordering,err);
  forwardError(*err,__LINE__,NULL);

  
  npix_t = 12*nside*nside;
  pix_T = lowly_build_pixel_list(mask_T, &npix_t, err);
  forwardError(*err,__LINE__,NULL);

  if (mask_P==NULL || mapQ==NULL || mapU==NULL) {
    pix_P = NULL;
    npix_p = 0;
  } else {
    npix_p = 12*nside*nside;
    pix_P = lowly_build_pixel_list(mask_P, &npix_p, err);
    forwardError(*err,__LINE__,NULL);
  }
  
  self->npix_p = npix_p;
  self->npix_t = npix_t;
  
  self->npix = 2*npix_p + npix_t;
  _DEBUGHERE_("%d %d",npix_t,npix_p);
  self->trig_mat = build_trig_mat_general_plist(self->nside,pix_T,npix_t,pix_P,npix_p,order,err);
  forwardError(*err,__LINE__,NULL);
  
  self->buffer = malloc_err(sizeof(double)*(_SZT_(self->npix) * _SZT_(self->npix) + self->npix*2),err);
  forwardError(*err,__LINE__,NULL);
  
  self->S      = self->buffer;
  self->X      = self->S + self->npix*self->npix;
  self->X_temp = self->X + self->npix;
  
  _DEBUGHERE_("%g",mapT[0]);
  for(i=0;i<self->npix_t;i++) {
    self->X[i] = mapT[pix_T[i]];
  }
  
  if (mapQ!=NULL)
  _DEBUGHERE_("%g",mapQ[0]);  
  for(i=0;i<self->npix_p;i++) {
    self->X[i+self->npix_t] = mapQ[pix_P[i]];
  }

  if (mapU!=NULL)
    _DEBUGHERE_("%g",mapU[0]);  
  for(i=0;i<self->npix_p;i++) {
    self->X[i+self->npix_t+self->npix_p] = mapU[pix_P[i]];
  }
  
  self->Cl = malloc_err(sizeof(double)*(lmax+1)*2*6,err);
  forwardError(*err,__LINE__,NULL);
  memcpy(self->Cl,Cl,sizeof(double)*(lmax+1)*6);
  self->tCl = self->Cl + (lmax+1)*6;
  
  self->nell = nell;
  self->ell = lowly_get_ell(&(self->nell),ell,lmax,err);
  forwardError(*err,__LINE__,NULL);
  
  self->lmax= lmax;
  
  lowly_get_offset_cl(has_cl,self->offset_cl,self->nell);  
  /*_DEBUGHERE_("%d %d %d %d %d %d %d",self->nell,self->offset_cl[0],self->offset_cl[1],self->offset_cl[2],self->offset_cl[3],self->offset_cl[4],self->offset_cl[5]);
  
  _DEBUGHERE_("TT %g %g %g %g %g %g",self->Cl[0+self->offset_cl[0]],self->Cl[1+self->offset_cl[0]],self->Cl[2+self->offset_cl[0]],self->Cl[3+self->offset_cl[0]],self->Cl[4+self->offset_cl[0]],self->Cl[5+self->offset_cl[0]]); 
  _DEBUGHERE_("EE %g %g %g %g %g %g",self->Cl[0+self->offset_cl[1]],self->Cl[1+self->offset_cl[1]],self->Cl[2+self->offset_cl[1]],self->Cl[3+self->offset_cl[1]],self->Cl[4+self->offset_cl[1]],self->Cl[5+self->offset_cl[1]]); 
  _DEBUGHERE_("BB %g %g %g %g %g %g",self->Cl[0+self->offset_cl[2]],self->Cl[1+self->offset_cl[2]],self->Cl[2+self->offset_cl[2]],self->Cl[3+self->offset_cl[2]],self->Cl[4+self->offset_cl[2]],self->Cl[5+self->offset_cl[2]]); 
  _DEBUGHERE_("TE %g %g %g %g %g %g",self->Cl[0+self->offset_cl[3]],self->Cl[1+self->offset_cl[3]],self->Cl[2+self->offset_cl[3]],self->Cl[3+self->offset_cl[3]],self->Cl[4+self->offset_cl[3]],self->Cl[5+self->offset_cl[3]]); 
  _DEBUGHERE_("TB %g %g %g %g %g %g",self->Cl[0+self->offset_cl[4]],self->Cl[1+self->offset_cl[4]],self->Cl[2+self->offset_cl[4]],self->Cl[3+self->offset_cl[4]],self->Cl[4+self->offset_cl[4]],self->Cl[5+self->offset_cl[4]]); 
  _DEBUGHERE_("EB %g %g %g %g %g %g",self->Cl[0+self->offset_cl[5]],self->Cl[1+self->offset_cl[5]],self->Cl[2+self->offset_cl[5]],self->Cl[3+self->offset_cl[5]],self->Cl[4+self->offset_cl[5]],self->Cl[5+self->offset_cl[5]]); 
  */
  if (Ndiag!=NULL) {
    self->Ndiag = malloc_err(sizeof(double)*self->npix,err);
    forwardError(*err,__LINE__,NULL);
    self->N=NULL;
    if(reduced==1) {
      memcpy(self->Ndiag, Ndiag, sizeof(double)*self->npix);
    } else {
      for(i=0;i<npix_t;i++) {
        self->Ndiag[i]=Ndiag[pix_T[i]];
      }
      for(i=0;i<npix_p;i++) {
        self->Ndiag[npix_t+i]=Ndiag[12*nside*nside+pix_P[i]];
      }
      for(i=0;i<npix_p;i++) {
        self->Ndiag[npix_p+npix_t+i]=Ndiag[12*nside*nside*2+pix_P[i]];
      }
    }
  } else {
    self->N = malloc_err(sizeof(double)*_SZT_(self->npix)*_SZT_(self->npix),err);
    forwardError(*err,__LINE__,NULL);
    self->Ndiag=NULL;
    if(reduced==1) {
      memcpy(self->N, N, sizeof(double)*_SZT_(self->npix)*_SZT_(self->npix));
    } else {
      for(i=0;i<self->npix;i++) {
        if(i<self->npix_t) {
          ox = pix_T[i];
        } else if(i<self->npix_t+self->npix_p) {
          ox = pix_P[i-self->npix_t] + 12*nside*nside;
        } else {
          ox = pix_P[i-self->npix_t-self->npix_p] + 12*nside*nside*2;
        }
        for(j=0;j<self->npix;j++) {
          if(j<self->npix_t) {
            oy = pix_T[j];
          } else if(j<self->npix_t+self->npix_p) {
            oy = pix_P[j-self->npix_t] + 12*nside*nside;
          } else {
            oy = pix_P[j-self->npix_t-self->npix_p] + 12*nside*nside*2;
          }
          self->N[i*self->npix+j] = N[ox*12*nside*nside*3+oy];
        }
      }
    }
  }      
  
  free(pix_T);
  if (mask_P==NULL) {
    free(pix_P);
  }
  self->time_build = 0;
  self->time_tot = 0;
  self->time_chol = 0;
  self->n_tests = 0;
  
  return self;
}
#undef __FUNC__

#define __FUNC__ "plowly_build_S"
double* plowly_build_S(double* orig,plowly *self,double* pars,error **err) {
  int i,j,ioff,ilm,jls,lmp1;
  double *res;
  
  lmp1 = (self->lmax+1);
  
  
  // prepare Cls
  memcpy(self->tCl, self->Cl, lmp1*6);
  for(i=0;i<6;i++) {
    if (self->offset_cl[i]!=-1) {
      ilm = i*lmp1;
      ioff = self->offset_cl[i];
      for(j=0;j<self->nell;j++) {
        jls = self->ell[j];
        self->tCl[ilm+jls]=pars[ioff+j];
        
      }
    }
  }
  
  
  //build cov matrice
  res=build_cov_matrix_pol_general(orig,self->trig_mat,self->Ndiag, 
                               self->npix_t, self->npix_p,self->lmax,
                               self->tCl+ 0*lmp1,self->tCl + 1*lmp1,
                               self->tCl+ 2*lmp1,self->tCl + 3*lmp1,
                               self->tCl+ 4*lmp1,self->tCl + 5*lmp1,
                               err);
  forwardError(*err,__LINE__,NULL);
  
  // add noise if needed
  if (self->N!=NULL) {
    int sz,one;
    double fone;
    
    sz=self->npix*self->npix;
    one=1;
    fone=1;
    daxpy(&sz, &fone, self->N, &one, res, &one);
  }
  
  return res;
}
#undef __FUNC__

#define __FUNC__ "plowly_lkl"
double plowly_lkl(void* pself, double* pars, error **err) {
  plowly* self;
  int tot_time,i,j;
  double res;
  TIMER_DEFS;
  
  self=pself;
  tot_time=0;
  
  // build S 
  TIMER_IN;
  plowly_build_S(self->S,self,pars,err);
  forwardError(*err,__LINE__,0);
  
  TIMER_OUT;
  tot_time+=TIMER_MSEC;
  self->time_build += TIMER_MSEC;
  
  // compute lkl;
  TIMER_IN;
  res = lowly_XtRX_lkl(self->S,self->X,self->X_temp,self->npix,err);
  forwardError(*err,__LINE__,0);
  TIMER_OUT;
  tot_time+=TIMER_MSEC;
  self->time_chol += TIMER_MSEC;
  
  tot_time+=TIMER_MSEC;
  self->time_tot += tot_time;
  
  self->n_tests++;
  
  return res;
          
}

#undef __FUNC__
#define __FUNC__ "free_plowly"
void free_plowly(void **pself) {
  plowly *self;
  self=*pself;
  
  DO_PRINT_STAT(self);
  free(self->trig_mat);
  free(self->buffer);
  free(self->ell);
  free(self->Cl);
  if (self->N!=NULL) {
    free(self->N);
  } else {
    free(self->Ndiag);
  }
  free(self);
  *pself=NULL;
}
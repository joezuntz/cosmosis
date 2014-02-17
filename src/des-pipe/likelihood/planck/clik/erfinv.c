/*
 *  erfinv.c
 *  likely
 *
 *  Created by Karim Benabed on 03/06/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *  Based on http://home.online.no/~pjacklam/notes/invnorm/ for erfinv_1
 *  Based on Cephes for erfinv_2
 */

#ifdef __PLANCK__
#include "HL2_likely/target/erfinv.h"
#else
#include "erfinv.h"
#endif

#define __FUNC__ "erfinv_1"
double erfinv_1(double v,error **err) {
  double x;
  double p,q, r, u, e;
  
  p=(v+1.)/2.;
  
  /* compute first approx */
  if ((0 < p )  && (p < ERF_P_LOW)) {
  
    q = sqrt(-2*log(p));
    x = (((((ERF_C1*q+ERF_C2)*q+ERF_C3)*q+ERF_C4)*q+ERF_C5)*q+ERF_C6) / ((((ERF_D1*q+ERF_D2)*q+ERF_D3)*q+ERF_D4)*q+1);
  
  } else if ((ERF_P_LOW <= p) && (p <= ERF_P_HIGH)) {
    
    q = p - 0.5;
    r = q*q;
    x = (((((ERF_A1*r+ERF_A2)*r+ERF_A3)*r+ERF_A4)*r+ERF_A5)*r+ERF_A6)*q /(((((ERF_B1*r+ERF_B2)*r+ERF_B3)*r+ERF_B4)*r+ERF_B5)*r+1);
  
  } else if ((ERF_P_HIGH < p) && (p < 1)) {
  
    q = sqrt(-2*log(1-p));
    x = -(((((ERF_C1*q+ERF_C2)*q+ERF_C3)*q+ERF_C4)*q+ERF_C5)*q+ERF_C6) / ((((ERF_D1*q+ERF_D2)*q+ERF_D3)*q+ERF_D4)*q+1);
  
  } else {
    testErrorRetVA(1,erfi_outofrange,"Argument out of range (got %g)",*err,__LINE__,-1,v);
  }
  
  /* extra refinement step */
  if( (0 < p) && (p < 1) ){
    e = 0.5 * erfc(-x/ERF_SQRT2) - p;
    u = e * ERF_SQRT2PI * exp(x*x/2);
    x = x - u/(1 + x*u/2);
  }
  
  return x/ERF_SQRT2;
}


/* from Cephes */

/* approximation for 0 <= |y - 0.5| <= 3/8 */
static double ERF_P0[5] = {
  -5.99633501014107895267E1,
  9.80010754185999661536E1,
  -5.66762857469070293439E1,
  1.39312609387279679503E1,
  -1.23916583867381258016E0,
};
static double ERF_Q0[8] = {
  /* 1.00000000000000000000E0,*/
  1.95448858338141759834E0,
  4.67627912898881538453E0,
  8.63602421390890590575E1,
  -2.25462687854119370527E2,
  2.00260212380060660359E2,
  -8.20372256168333339912E1,
  1.59056225126211695515E1,
  -1.18331621121330003142E0,
};

/* Approximation for interval z = sqrt(-2 log y ) between 2 and 8
 * i.e., y between exp(-2) = .135 and exp(-32) = 1.27e-14.
 */
static double ERF_P1[9] = {
  4.05544892305962419923E0,
  3.15251094599893866154E1,
  5.71628192246421288162E1,
  4.40805073893200834700E1,
  1.46849561928858024014E1,
  2.18663306850790267539E0,
  -1.40256079171354495875E-1,
  -3.50424626827848203418E-2,
  -8.57456785154685413611E-4,
};
static double ERF_Q1[8] = {
  /*  1.00000000000000000000E0,*/
  1.57799883256466749731E1,
  4.53907635128879210584E1,
  4.13172038254672030440E1,
  1.50425385692907503408E1,
  2.50464946208309415979E0,
  -1.42182922854787788574E-1,
  -3.80806407691578277194E-2,
  -9.33259480895457427372E-4,
};

/* Approximation for interval z = sqrt(-2 log y ) between 8 and 64
 * i.e., y between exp(-32) = 1.27e-14 and exp(-2048) = 3.67e-890.
 */
static double ERF_P2[9] = {
  3.23774891776946035970E0,
  6.91522889068984211695E0,
  3.93881025292474443415E0,
  1.33303460815807542389E0,
  2.01485389549179081538E-1,
  1.23716634817820021358E-2,
  3.01581553508235416007E-4,
  2.65806974686737550832E-6,
  6.23974539184983293730E-9,
};
static double ERF_Q2[8] = {
  /*  1.00000000000000000000E0,*/
  6.02427039364742014255E0,
  3.67983563856160859403E0,
  1.37702099489081330271E0,
  2.16236993594496635890E-1,
  1.34204006088543189037E-2,
  3.28014464682127739104E-4,
  2.89247864745380683936E-6,
  6.79019408009981274425E-9,
};

double polevl(double x, double *coef, int N ) {
  double ans;
  int i;
  double *p;
  
  p = coef;
  ans = *p++;
  i = N;
  
  do
    ans = ans * x  +  *p++;
  while( --i );
  
  return( ans );
}

double p1evl(double x, double *coef, int N ) {
  double ans;
  double *p;
  int i;
  
  p = coef;
  ans = x + *p++;
  i = N-1;
  
  do
    ans = ans * x  + *p++;
  while( --i );
  
  return( ans );
}

#undef __FUNC__
#define __FUNC__ "erfinv_2"
double erfinv_2(double y0,error **err) {
  double x, y, z, y2, x0, x1;
  int code;
  
  y=(y0+1.)/2.;
  if (y==0) {
    //clip to 0
    y=1e-16;
  }
  if (y==1) {
    //clip to 1
    y=1-1e-16;
  }
  testErrorRetVA((y <= 0.0)||(y >= 1. ),erfi_outofrange,"Argument out of range (%g is not in [-1,1])",*err,__LINE__,-1,y0);

  
  code = 1;

  
  if( y > (1.0 - 0.13533528323661269189) )  {/* 0.135... = exp(-2) */
    y = 1.0 - y;
    code = 0;
	}
  
  if( y > 0.13533528323661269189 ) {
    y = y - 0.5;
    y2 = y * y;
    x = y + y * (y2 * polevl( y2, ERF_P0, 4)/p1evl( y2, ERF_Q0, 8 ));
    x = x * ERF_SQRT2PI; 
    return x/ERF_SQRT2;
	}
  
  x = sqrt( -2.0 * log(y) );
  x0 = x - log(x)/x;
  
  z = 1.0/x;
  if( x < 8.0 ) /* y > exp(-32) = 1.2664165549e-14 */
    x1 = z * polevl( z, ERF_P1, 8 )/p1evl( z, ERF_Q1, 8 );
  else
    x1 = z * polevl( z, ERF_P2, 8 )/p1evl( z, ERF_Q2, 8 );
  x = x0 - x1;
  if( code != 0 )
    x = -x;
  return x/ERF_SQRT2;
}


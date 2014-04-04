/*
 *  erfinv.h
 *  likely
 *
 *  Created by Karim Benabed on 03/06/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *  Based on http://home.online.no/~pjacklam/notes/invnorm/
 *  Based on Cephes for erfinv_2
 */

#include "errorlist.h"

#include <math.h>

#ifndef __ERFINV__
#define __ERFINV__

/* definitions for  http://home.online.no/~pjacklam/notes/invnorm/ */

#define  ERF_SQRT2    1.414213562373095145474621858738828450441
#define  ERF_SQRT2PI  2.50662827463100050242E0 


#define  ERF_A1  (-3.969683028665376e+01)
#define  ERF_A2   2.209460984245205e+02
#define  ERF_A3  (-2.759285104469687e+02)
#define  ERF_A4   1.383577518672690e+02
#define  ERF_A5  (-3.066479806614716e+01)
#define  ERF_A6   2.506628277459239e+00

#define  ERF_B1  (-5.447609879822406e+01)
#define  ERF_B2   1.615858368580409e+02
#define  ERF_B3  (-1.556989798598866e+02)
#define  ERF_B4   6.680131188771972e+01
#define  ERF_B5  (-1.328068155288572e+01)

#define  ERF_C1  (-7.784894002430293e-03)
#define  ERF_C2  (-3.223964580411365e-01)
#define  ERF_C3  (-2.400758277161838e+00)
#define  ERF_C4  (-2.549732539343734e+00)
#define  ERF_C5   4.374664141464968e+00
#define  ERF_C6   2.938163982698783e+00

#define  ERF_D1   7.784695709041462e-03
#define  ERF_D2   3.224671290700398e-01
#define  ERF_D3   2.445134137142996e+00
#define  ERF_D4   3.754408661907416e+00

#define ERF_P_LOW   0.02425
/* P_high = 1 - p_low*/
#define ERF_P_HIGH  0.97575
double erfinv_1(double v,error **err);
  
  
/* definitions for cephes */
double erfinv_2(double v,error **err);

#define erfinv erfinv_2

#define erfi_outofrange 11100
#endif

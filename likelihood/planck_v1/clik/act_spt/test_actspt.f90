!===========================================================
program test_likelihood

! test high ell chi2

use highell_options
use highell_likelihood

implicit none

real(8), dimension(:), allocatable :: cl_tt
character(LEN=128) :: filename
real(8)            :: like_tot
real(8)            :: amp_tsz,amp_ksz,xi,aps148,aps217,aps95,aps150,aps220,acib150,acib220,rps0,rps1,rps,rcib,ags,age
real(8)            :: cas1,cas2,cae1,cae2,cal_1,cal_2,cal_3
integer            :: il, dummy 
!---------------------------------------------------

data_dir = "data_act_spt/data/"
ACT_data_dir = "data_act_spt/data/data_act/"
SPT_data_dir = "data_act_spt/data/data_spt/"
print *,""
print *,"High ell likelihood chi2 test"
print *,"==================================="

!---------------------------------------------------
! read in test Cls
!---------------------------------------------------
filename = 'actspt_cl1.par'
write(*,*)"Reading in Cls from: ",trim(filename)
open(unit=557,file=filename,action='read',status='old')
write(*,*) tt_lmax
allocate(cl_tt(2:tt_lmax))

cl_tt(2:tt_lmax)=0.d0

print *,1
read(557,*) dummy
print *,2
read(557,*) dummy

do il=2,tt_lmax_mc
   read(557,*) cl_tt(il)
   print *,il
enddo
read(557,*) amp_tsz 
read(557,*) amp_ksz 
read(557,*) xi      
read(557,*) aps148  
read(557,*) aps217  
read(557,*) aps95   
read(557,*) aps150  
read(557,*) aps220  
read(557,*) acib150 
read(557,*) acib220 
read(557,*) rps0    
read(557,*) rps1    
read(557,*) rps     
read(557,*) rcib    
read(557,*) ags    
read(557,*) age   
read(557,*) cas1    
read(557,*) cas2    
read(557,*) cae1    
read(557,*) cae2    
read(557,*) cal_1   
read(557,*) cal_2   
read(557,*) cal_3   

close(557)

call highell_likelihood_init

!do il=2,1999
!	cl_tt(il) = 0
!enddo
do il=2,tt_lmax_mc
   cl_tt(il) = cl_tt(il) *il *(il+1)/2./PI
enddo

print *,"amp_tsz" , amp_tsz
print *,"amp_ksz" , amp_ksz
print *,"xi" , xi
print *,"aps148" , aps148
print *,"aps217" , aps217
print *,"aps95" , aps95
print *,"aps150" , aps150
print *,"aps220" , aps220
print *,"acib150" , acib150
print *,"acib220" , acib220
print *,"rps0" , rps0
print *,"rps1" , rps1
print *,"rps" , rps
print *,"rcib" , rcib
print *,"ags" , ags
print *,"age" , age
print *,"cas1" , cas1
print *,"cas2" , cas2
print *,"cae1" , cae1
print *,"cae2" , cae2
print *,"cal_1" , cal_1
print *,"cal_2" , cal_2
print *,"cal_3" , cal_3

write(*,*) cl_tt(10),cl_tt(100),cl_tt(1000)

call highell_likelihood_compute(cl_tt,amp_tsz,amp_ksz,xi,aps148,aps217,aps95,aps150,aps220,acib150,acib220,rps0,rps1,rps,rcib,ags,age,cas1,cas2,cae1,cae2,cal_1,cal_2,cal_3,like_tot)
print *, "----------------------------------------" 
print *, 'Total High ell chi2  =', 2*like_tot
print *, "----------------------------------------"

end program test_likelihood

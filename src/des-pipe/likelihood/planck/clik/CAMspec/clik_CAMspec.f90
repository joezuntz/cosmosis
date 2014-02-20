MODULE CAMSPEC_EXTRA

    IMPLICIT NONE

    INTEGER:: BOK = 0
    real(8), dimension(:), allocatable :: cltt,nuisance
    INTEGER::lmin,lmax
    INTEGER::xcase,npar
END MODULE CAMSPEC_EXTRA



SUBROUTINE CAMSPEC_EXTRA_ONLY_ONE(MOK)
    USE CAMSPEC_EXTRA
    INTEGER,INTENT(OUT)::MOK
    MOK = BOK
    BOK = 1
END SUBROUTINE  CAMSPEC_EXTRA_ONLY_ONE

SUBROUTINE CAMSPEC_EXTRA_FREE()
    USE CAMSPEC_EXTRA
    DEALLOCATE(cltt)
    BOK =0
END SUBROUTINE  CAMSPEC_EXTRA_FREE

SUBROUTINE CAMSPEC_EXTRA_INIT(iNspec, inX,ilminX,ilmaxX,inp,inpt, ic_inv,iX,ilmax_sz,isz_143_temp,iksz_temp,itszxcib_temp,ibeam_Nspec,inum_modes_per_beam,ibeam_lmax,icov_dim,ibeam_cov_inv,ibeam_modes,ihas_dust,ihas_calib,imarge_flag, imarge_mode,imarge_num, ikeep_num,bs_factor)
    USE CAMSPEC_EXTRA
    USE temp_like
    implicit none
    integer,intent(in)::iNspec,inX,inum_modes_per_beam,ibeam_lmax,ibeam_Nspec,icov_dim,ilmax_sz,ihas_dust,ihas_calib
    integer,dimension(1:iNspec)::ilminX,ilmaxX,inp,inpt
    real*8, dimension(1:iNx) :: iX
    real*8,  dimension(1:iNx,1:iNx) ::ic_inv
    real*8,intent(in) :: iksz_temp(0:ilmax_sz), itszxcib_temp(0:ilmax_sz)
    real*8,intent(in) :: isz_143_temp(0:ilmax_sz)
    real*8, dimension(1:icov_dim,1:icov_dim) :: ibeam_cov_inv
    real*8, dimension(1:inum_modes_per_beam,0:ibeam_lmax,1:ibeam_Nspec) :: ibeam_modes ! mode#, l, spec#
    integer, dimension(1:icov_dim)::imarge_flag
    logical,dimension(1:icov_dim)::marge_flag
    integer::imarge_num,ikeep_num
    real*8,dimension(1:imarge_num, 1:ikeep_num)::imarge_mode
    real*8::bs_factor

    integer::i
    
    do i=1,icov_dim
        marge_flag(i) = (imarge_flag(i).eq.1)
    enddo

    call like_init_frommem(iNspec, inX,ilminX,ilmaxX,inp,inpt, ic_inv,iX,ilmax_sz,isz_143_temp,iksz_temp,itszxcib_temp,ibeam_Nspec,inum_modes_per_beam,ibeam_lmax,icov_dim,ibeam_cov_inv,ibeam_modes,ihas_dust,ihas_calib,marge_flag,imarge_mode)
    
    lmax = ilmaxX(1)
    DO i=2,iNspec
        IF (lmax<ilmaxX(i)) lmax = ilmaxX(i)
    ENDDO
    lmin = ilminX(1)
    DO i=2,iNspec
        IF (lmin>ilminX(i)) lmin = ilminX(i)
    ENDDO
    
    xcase = 1
    !IF (mlmax2.ne.0) xcase = 1

    ALLOCATE(cltt(0:lmax+1))

    allocate(nuisance(num_non_beam+ihas_dust+beam_Nspec*num_modes_per_beam - marge_num))
    npar = lmax+1-lmin+num_non_beam+ beam_Nspec * num_modes_per_beam +ihas_dust - marge_num
    beam_factor = bs_factor

END SUBROUTINE CAMSPEC_EXTRA_INIT

SUBROUTINE CAMSPEC_EXTRA_GETCASE(xc)
    USE CAMSPEC_EXTRA
    INTEGER::xc

    xc = xcase
END SUBROUTINE CAMSPEC_EXTRA_GETCASE

SUBROUTINE CAMSPEC_EXTRA_LKL(LKL,CL)
    USE CAMSPEC_EXTRA
    use temp_like
    implicit none

    REAL(8),INTENT(OUT)::LKL
    REAL(8),DIMENSION(0:npar-1)::CL
    real(8)::zlike, A_ps_100, A_ps_143, A_ps_217, A_cib_143, A_cib_217, A_sz, r_ps, r_cib, &
         cal0, cal1, cal2, xi, A_ksz
    real*8, dimension(1:beam_Nspec,1:num_modes_per_beam) :: beam_coeffs
    INTEGER::l,i,j,cnt
    real(8)::tlkl

    cltt(:lmin-1) = 0
    DO l=lmin,lmax
        ! camspec expects cl/2pi !!! argl !
        !cltt(l)=CL(l-lmin)/2./3.14159265358979323846264338328
        cltt(l)=CL(l-lmin)/2./3.141592653589793
    ENDDO

    do i=1,num_non_beam
        nuisance(i) = CL(lmax+1-lmin + i-1)
    enddo
    !A_ps_100  = CL(lmax+1-lmin + 0)
    !A_ps_143  = CL(lmax+1-lmin + 1)
    !A_ps_217  = CL(lmax+1-lmin + 2)
    !A_cib_143 = CL(lmax+1-lmin + 3)
    !A_cib_217 = CL(lmax+1-lmin + 4)
    !A_sz      = CL(lmax+1-lmin + 5)
    !r_ps      = CL(lmax+1-lmin + 6)
    !r_cib     = CL(lmax+1-lmin + 7) 
    !cal0      = CL(lmax+1-lmin + 8) 
    !cal1      = CL(lmax+1-lmin + 9) 
    !cal2      = CL(lmax+1-lmin + 10)     
    !xi        = CL(lmax+1-lmin + 11)     
    !A_ksz     = CL(lmax+1-lmin + 12)

    !print *,CL(lmax+1-lmin+14)

    cnt = 1
    DO i=1,keep_num
            nuisance(cnt+num_non_beam) = CL(lmax+1-lmin+num_non_beam+cnt-1)
            cnt = cnt + 1
    ENDDO


    call calc_like(tlkl,  cltt,nuisance)
    ! lkl is -2loglike clik returns loglik
    !print *,tlkl
    lkl = -tlkl/2.

END SUBROUTINE CAMSPEC_EXTRA_LKL

SUBROUTINE CAMSPEC_EXTRA_FG(rCL_FG,NUIS,lm)
    USE CAMSPEC_EXTRA
    use temp_like
    implicit none
    REAL(8),DIMENSION(1:num_non_beam+beam_Nspec*num_modes_per_beam)::NUIS
    REAL(8),dimension(4,0:lm)::rCL_FG
    integer::lm



    call COMPUTE_FG(rCL_FG,NUIS)
    
END SUBROUTINE



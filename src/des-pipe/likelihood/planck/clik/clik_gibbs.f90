MODULE GIBBS_EXTRA

	IMPLICIT NONE

	INTEGER,dimension(100):: CLIK_LMAX,CLIK_LMIN
	real(8),dimension(2:1000) :: cltt
	real(8),parameter:: PI    = 3.141592653589793238462643383279502884197

END MODULE GIBBS_EXTRA




SUBROUTINE GIBBS_EXTRA_FREE(handle)
	use comm_br_mod
	use GIBBS_EXTRA
	INTEGER,intent(in)::handle
	
	call comm_br_deallocate_object(handle)
	
	clik_lmin(handle) = -1
	clik_lmax(handle) = -1
	
END SUBROUTINE 	GIBBS_EXTRA_FREE

SUBROUTINE GIBBS_EXTRA_LKL(LKL,handle,CL)
	use comm_br_mod
	use GIBBS_EXTRA
	
	REAL(8),INTENT(OUT)::LKL
	INTEGER,intent(in)::handle
	REAL(8),INTENT(IN),DIMENSION(0:CLIK_LMAX(handle)-CLIK_LMIN(handle))::CL
	INTEGER::i,cur

	!TT
	cur = 0
	cltt = 0
	DO i = clik_lmin(handle),clik_lmax(handle)
		cltt(i)=CL(cur)*(i*(i+1.))/2./PI
		cur = cur + 1
	END DO	

	LKL = comm_br_compute_lnL(cltt(2:clik_lmax(handle)),handle)		
	
END SUBROUTINE 	GIBBS_EXTRA_LKL



SUBROUTINE GIBBS_EXTRA_PARAMETER_INIT(handle,datadir,l_datadir,lmin,lmax,firstchain,lastchain,firstsample,lastsample,step)
	use comm_br_mod
	use GIBBS_EXTRA
	
	INTEGER,INTENT(IN)::l_datadir,lmin,lmax,firstchain,lastchain,firstsample,lastsample,step
	character(len=l_datadir)::datadir
	INTEGER,INTENT(OUT)::handle
	character(len=1000)::sigma_file,cl_file,data_dir

		
	data_dir = TRIM(datadir)
	sigma_file = TRIM(data_dir)//"/sigma.fits"
	cl_file = TRIM(data_dir)//"/cl.dat"

	handle = 0

	call comm_br_initialize_object(sigma_file, cl_file, lmin, lmax, firstchain, lastchain, &
       & firstsample, lastsample, step, handle)
	
	clik_lmin(handle) = lmin
	clik_lmax(handle) = lmax
		
END SUBROUTINE 	GIBBS_EXTRA_PARAMETER_INIT

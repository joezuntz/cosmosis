program CAMtst
    use temp_like
    integer::i,j
    real*8,dimension(0:2500)::cltt
    real*8::dum,A_ps_100,  A_ps_143, A_ps_217, A_cib_143, A_cib_217, A_sz,  &
       r_ps, r_cib, xi, A_ksz, cal0, cal1, cal2,z,rz
    real*8,dimension(4,5)::bmf
    character*100 like_file, sz143_file, ksz_file, tszxcib_file, beam_file

    like_file = "like_v6.1F"
    tszxcib_file = "sz_x_cib_template.dat"
    sz143_file = "tsz_143_eps0.50.dat"
    ksz_file = "cl_ksz_148_trac.dat"
     
    beam_file = "beamfile_v61f.dat"
    
    call like_init(like_file, sz143_file, tszxcib_file, ksz_file, beam_file)

    print *,1
    open(50,file = "ns61F.1.dat",form='formatted',status='unknown')
    print *,2
    do i=0,2500
        read(50,*) dum
        cltt(i) = dum/i/(i+1.)
    enddo
    read(50,*) A_ps_100
    read(50,*) A_ps_143
    read(50,*) A_ps_217
    read(50,*) A_cib_143
    read(50,*) A_cib_217
    read(50,*) A_sz
    read(50,*) r_ps
    read(50,*) r_cib
    read(50,*) xi
    read(50,*) A_ksz
    read(50,*) cal0
    read(50,*) cal1
    read(50,*) cal2
    do i=1,4
        do j=1,5
            read(50,*) bmf(i,j)
        enddo
    enddo
    read(50,*) z

    !A_ps_100 = 0
    !A_ps_143 = 0
    !A_ps_217 = 0
    !A_cib_143 = 0
    !A_cib_217 = 0
    !A_sz = 0
    !r_ps = 0
    !r_cib = 0
    !xi = 0
    !A_ksz = 0
    !cal0 = 1
    !cal1 = 1
    !cal2 = 1
    !bmf = 0
    call calc_like(rz,  cltt, A_ps_100,  A_ps_143, A_ps_217, A_cib_143, A_cib_217, A_sz,  &
       r_ps, r_cib, xi, A_ksz, cal0, cal1, cal2, bmf)

    print *,z,rz,z-rz

end program
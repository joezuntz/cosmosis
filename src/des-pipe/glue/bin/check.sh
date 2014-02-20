#!/usr/bin/env bash
echo "--------------------------------------------------------------------"
echo "------ I am running some tests on the compilers and variables ------"
echo "-------    which should be specified in Makefile.mine        -------"
echo "--------------------------------------------------------------------"

if [ -e `which hg` ]
	then
	VERSION_ID=$(hg id -i 2>/dev/null)
	if [ "$VERSION_ID" == "" ] 
		then
		VERSION_ID="unknown"
	fi
	echo "Running repository version: $VERSION_ID"
	echo $VERSION_ID > version.txt
fi

if [ ! -e ../Makefile.mine ]
	then
	echo
	echo "I could not find a file called Makefile.mine"
	echo "You must make this file and put some variables in it"
	echo "to get anywhere.  There are example files in "
	exit 1
fi
echo -n .

if [ -z "$PYTHON" ]
	then
	echo
	echo You did not specify the variable PYTHON
	echo You need python 2.6 or 2.7
	exit 2
fi

echo -n .

if [ ! -e `which "$PYTHON"` ]
	then
	echo
	echo "I looked in '$'PYTHON=$PYTHON but could not find an executable file there."
	echo "'$'PYTHON should be set to the path to python 2.6 or 2.7"
	exit 3
fi

echo -n .

if ! $PYTHON -c 'import sys; assert sys.version_info[0]==2;assert sys.version_info[1]>=6' &> /dev/null
	then
	echo
	echo "The python version I found at $PYTHON was too old or new."
	echo "I need python 2.6 or 2.7 (3.x is too new for me, sorry about that)"
	echo "I found:"
	$PYTHON -c 'import sys; print sys.version'
	exit 4
fi

echo -n .

if ! file -L  `which $PYTHON` | grep -q 64
	then
	echo
	echo "The version of python I found was compiled as 32-bit"
	echo "We are going to need 64 bit python here, since many codes expect it"
	echo "When I checked using the 'file' command I got:"
	file -L `which $PYTHON`
	exit 5
fi

echo -n .

if [ -z "$CC" ]
	then
	echo
	echo The variable CC was not set in your Makefile.mine
	echo You need to set it to point to a C compiler.
	exit 6
fi

echo -n .

if [ ! -e `which "$CC"` ]
	then
	echo
	echo "I looked in '$'CC=$CC but could not find an executable file there."
	echo "'$'CC should be set to the path to a C compiler"
	exit 7
fi

echo -n .

echo 'int main(){return 0;}' > _check_test.c

if ! $CC $CFLAGS _check_test.c  -o _check_test
	then
	echo
	echo "I did not manage to compile a file using the CC and CFLAGS variables that I found"
	echo "The error should be printed above."
	exit 8
fi

echo -n .

 echo '#include "fitsio.h"
	int main(){return 0;}' > _check_test.c
if ! $CC $CFLAGS  _check_test.c  -o _check_test
	then
	echo
	echo "I did not manage to find the fitsio.h header that is part of cfitsio."
	echo "Your CFLAGS variable in Makefile.mine should include -I/path/to/fitsio.h"
	echo "The error should be printed above."
	exit 9
fi

echo -n .

echo '#include "fitsio.h"
int main(){return 0;}' > _check_test.c
if !  $CC $CFLAGS _check_test.c  -o _check_test $LDFLAGS -lcfitsio
	then
	echo
	echo "I did not manage to find the libcfitsio library that is part of cfitsio."
	echo "Your LDFLAGS variable in Makefile.mine should include -I/path/to/libcfitsio.a"
	echo "The error should be printed above."
	exit 9
fi

echo -n .

if [ -z "$F90" ]
	then
	echo
	echo "The variable F90 was not set in your Makefile.mine"
	echo "You need to set it to point to a fortran 90 compiler."
	echo "Currently only gfortran >= 4.5 is supported"
	exit 6
fi
echo -n .

if [ ! -e `which "$F90"` ]
	then
	echo
	echo "I looked in '$'F90=$F90 but could not find an executable file there."
	echo "'$'F90 should be set to the path to a fortran 90 compiler"
	echo "Currently only gfortran >= 4.5 is supported"
	exit 7
fi

echo -n .

echo 'program check_test
	write(*,*) 0
end program check_test' > _check_test.f90
if !  $F90 $FFLAGS  _check_test.f90  -o _check_test
	then
	echo
	echo "I did not manage to compile a file using the F90 and FFLAGS variables that I found"
	echo "The error should be printed above."
	exit 8
fi

echo -n .

echo 'program check_test
	USE ISO_C_BINDING
end program check_test
' > _check_test.f90
if !  $F90 $FFLAGS  _check_test.f90  -o _check_test
	then
	echo
	echo "I could not find the fortran built-in module ISO_C_BINDING in your fortran compiler"
	echo "This almost certainly means that you are using an older compiler that does not have this feature."
	echo "gfortran 4.5 and greater are the only compilers I know that completely work with this code"
	exit 9
fi

echo -n .


echo '
module mod1

    abstract interface
        function f(x)
            double precision f
            double precision, intent(in) :: x
        end function f
    end interface

contains

    subroutine printme(g)
        procedure(f), pointer, intent(in) :: g
        write(*,*) g(1d0), g(2d0), g(3d0)
    end subroutine printme

    subroutine printme2(g)
        procedure(f), pointer, intent(in) :: g
        call printme(g)
    end subroutine printme2

    function get_proc_pointer() result(my_func)
      procedure(f), pointer :: my_func
      my_func=>h
    end function get_proc_pointer

    function h(x)
      double precision :: h
      double precision, intent(in) :: x
      h = 3*x
    end function h

end module mod1


program test

    use mod1

    procedure(f), pointer :: pg

    pg => get_proc_pointer()
    call printme2(pg)

contains 

    function g(x)
        double precision g
        double precision, intent(in) :: x
        g = x**2
        return
    end function g

end program test' > _check_test.f90 


if ! $F90 $FFLAGS  _check_test.f90  -o _check_test &> _check_test.log
	then
	echo
	echo "The fortran compiler you sent, " $F90 ", cannot compile procedure pointers properly."
	echo "This probably means it is an older compiler without support for this feature."
	echo "Only newer versions of gfortran, 4.5 or later, can compile this code. I do not know if any ifort versions can."
	echo "When I do '$'F90 --version it says:"
	$F90 --version
	echo
	echo "I know this is a pain - sorry.  You can get precompiled binaries for most systems from the gfortran website or from package managers"
	echo "If you think your compiler should work look at _check_test.f90 and _check_test.log"
	exit 10
fi

echo -n .


echo '
program test

    write(*,*) 1,                                                                                                                                                                                                                          2

end program test' > _check_test.f90 



if ! $F90 $FFLAGS  _check_test.f90  -o _check_test &> _check_test.log
	then
	echo
	echo "Your fortran compiler cannot cope with very long lines. This is probably "
	echo "a 'feature' that the idiots who make gfortran added to ensure that they "
	echo "can compile code that was written to be fed into punched card machines"
	echo 
	echo "You can probably fix this by adding -ffree-line-length-none to your FFLAGS variable"
	echo "Also consider sending the gfortran devs some hate mail"
	exit 11
fi

echo -n .


if [ -z "$TARGETS" ]
	then
	echo
	echo "##############################################"
	echo "##### You did not the TARGETS variable   #####"
	echo "##############################################"
	echo "This will not actually break anything but if you want to be able to run any "
	echo "programs you will need to specify some by directory - e.g. boltzmann/camb, etc."
	echo "##############################################"

fi

 echo '
	int main(){return 0;}' > _check_test.c
	$CC $CFLAGS _check_test.c  -o _check_test
if ! file _check_test | grep -q 64
	then
	echo
	echo "Your C compiler with the specified flags seems to make 32 bit executables"
	echo "It should make 64 bit.  You probably need to add a flag like '-m64' to the CFLAGS" 
	exit 10
fi

echo -n .

 echo '
	int main(){dlopen(); return 0;}' > _check_test.c
	
if ! $CC $CFLAGS  _check_test.c  -o _check_test $LDFLAGS
	then
	echo "I could not find a particular function (called dlopen) in any library I used"
	echo "The error message should be above."
	echo "If you are on Linux, add the flag -ldl to the LDFLAGS variable"
	echo 
	exit 11
fi

echo -n .


 echo 'program check_test
	write(*,*) 0
end program check_test' > _check_test.f90
	$F90 $FFLAGS _check_test.f90  -o _check_test
if ! file _check_test | grep -q 64
	then
	echo
	echo "Your Fortran compiler ("'$'"F90 = $F90) with the specified flags ("'$'"FFLAGS = $FFLAGS) seems to make 32 bit executables"
	echo "It should make 64 bit.  You probably need to add a flag like '-m64' to the FFLAGS" 
	exit 10
fi

echo -n .

if ! $PYTHON -c 'import numpy' &> /dev/null
then
	echo
	echo "I could not find the required python package:  numpy"
	echo "If you think you have it somewhere you may need to add the directory to your PYTHONPATH"
	exit 11
fi

if ! $PYTHON -c 'import pyfits' &> /dev/null
then
	echo
	echo "I could not find the required python package:  pyfits"
	echo "If you think you have it somewhere you may need to add the directory to your PYTHONPATH"
	exit 12
fi

if ! $PYTHON -c 'import pyfits;pyfits.column' &> /dev/null
then
	echo
	echo "The version of pyfits you have is too old."
	echo "Please upgrade to a newer version."
	echo "If you think a newer one is already installed somewhere you may need to prepend the directory to your PYTHONPATH"
	exit 12
fi



rm -rf _check_test.f90 _check_test.log _check_test.o _check_test mod1.mod _check_test.dSYM _check_test.c

echo
echo
echo "--------------------------------------------------------------------"
echo "------ All my tests passed (but no guarantees!) --------------------"
echo "--------------------------------------------------------------------"


exit 0

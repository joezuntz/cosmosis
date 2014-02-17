!JAZ I found this online.
!It is a simple utility to convert a c string to a fortran string.
MODULE ISO_C_UTILITIES
   USE ISO_C_BINDING ! Intrinsic module
	implicit none

   CHARACTER(C_CHAR), DIMENSION(1), SAVE, TARGET, PRIVATE :: dummy_string="?"
   
CONTAINS   
   
   FUNCTION C_F_STRING(CPTR) RESULT(FPTR)
      ! Convert a null-terminated C string into a Fortran character array pointer
      TYPE(C_PTR), INTENT(IN) :: CPTR ! The C address
      CHARACTER(KIND=C_CHAR), DIMENSION(:), POINTER :: FPTR
      
      INTERFACE ! strlen is a standard C function from <string.h>
         ! int strlen(char *string)
         FUNCTION strlen(string) RESULT(len) BIND(C,NAME="strlen")
            USE ISO_C_BINDING
			integer(c_int) len
            TYPE(C_PTR), VALUE :: string ! A C pointer
         END FUNCTION
      END INTERFACE   
      
      IF(C_ASSOCIATED(CPTR)) THEN
         CALL C_F_POINTER(FPTR=FPTR, CPTR=CPTR, SHAPE=[strlen(CPTR)])
      ELSE
         ! To avoid segfaults, associate FPTR with a dummy target:
         FPTR=>dummy_string
      END IF
            
   END FUNCTION

END MODULE

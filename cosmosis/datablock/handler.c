#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

void (*default_handler) (int);


void cosmosis_segfault_handler(int sig) {
  void *array[32];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 32);

  // print out all the frames to stderr
  fprintf(stderr, "##################################################################################\n\n");
  fprintf(stderr, "Your program crashed with an error signal: %d\n\n", sig);
  fprintf(stderr, "This the trace of C functions being called\n(the first one or two may be part of the error handling):\n");
  fprintf(stderr, "##################################################################################\n\n");
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  // Now do the default handler
  fprintf(stderr, "##################################################################################\n\n");
  fprintf(stderr, "\nAnd here is the python faulthandler report and trace:\n\n");
  default_handler(sig);
  fprintf(stderr, "##################################################################################\n\n");

  //Finally quit
  exit(1);
}



void enable_cosmosis_segfault_handler(void){
  default_handler = signal(SIGSEGV, cosmosis_segfault_handler);   // install our handler
}

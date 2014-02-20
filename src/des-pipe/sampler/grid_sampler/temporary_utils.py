from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
from mpi4py import MPI

class _close_pool_message(object):
    def __repr__(self):
        return "<Close pool message>"

class _function_wrapper(object):
    def __init__(self, function):
        self.function = function

def _error_function(task):
    raise RuntimeError("Pool was sent tasks before being told what "
                       "function to apply.")



class MPIPool(object):
    """
    A pool that distributes tasks over a set of MPI processes. MPI is an
    API for distributed memory parallelism.  This pool will let you run
    emcee without shared memory, letting you use much larger machines
    with emcee.

    The pool only support the :func:`map` method at the moment because
    this is the only functionality that emcee needs. That being said,
    this pool is fairly general and it could be used for other purposes.

    Contributed by `Joe Zuntz <https://github.com/joezuntz>`_.

    :param comm: (optional)
        The ``mpi4py`` communicator.

    :param debug: (optional)
        If ``True``, print out a lot of status updates at each step.

    """
    def __init__(self, comm=MPI.COMM_WORLD, debug=False):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size() - 1
        self.debug = debug
        self.function = _error_function
        if self.size == 0:
            raise ValueError("Tried to create an MPI pool, but there "
                             "was only one MPI process available. "
                             "Need at least two.")

    def is_master(self):
        """
        Is the current process the master?

        """
        return self.rank == 0

    def wait(self):
        """
        If this isn't the master process, wait for instructions.

        """
        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        status = MPI.Status()

        while True:
            # Event loop.
            # Sit here and await instructions.
            if self.debug:
                print("Worker {0} waiting for task.".format(self.rank))

            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            if self.debug:
                print("Worker {0} got task {1} with tag {2}."
                                 .format(self.rank, task, status.tag))

            # Check if message is special sentinel signaling end.
            # If so, stop.
            if isinstance(task, _close_pool_message):
                if self.debug:
                    print("Worker {0} told to quit.".format(self.rank))
                break

            # Check if message is special type containing new function
            # to be applied
            if isinstance(task, _function_wrapper):
                self.function = task.function
                if self.debug:
                    print("Worker {0} replaced its task function: {1}."
                            .format(self.rank, self.function))
                continue

            # If not a special message, just run the known function on
            # the input and return it asynchronously.
            result = self.function(task)
            if self.debug:
                print("Worker {0} sending answer {1} with tag {2}."
                        .format(self.rank, result, status.tag))
            self.comm.send(result, dest=0, tag=status.tag)

    def map(self, function, tasks):
        """
        Like the built-in :func:`map` function, apply a function to all
        of the values in a list and return the list of results.

        :param function:
            The function to apply to the list.

        :param tasks:
            The list of elements.

        """
        ntask = len(tasks)

        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return

        if function is not self.function:
            if self.debug:

                print("Master replacing pool function with {0}."
                        .format(function))

            self.function = function
            F = _function_wrapper(function)

            # Tell all the workers what function to use.
            #requests = []
            for i in range(self.size):
                r = self.comm.send(F, dest=i + 1)
                #requests.append(r)

            # Wait until all of the workers have responded. See:
            #       https://gist.github.com/4176241
            # MPI.Request.waitall(requests)

        tasks_copy = tasks[:]
        def send_next_task_to_worker(worker):
            if not tasks_copy:
                return None
            task_index = len(tasks_copy)-1
            task = tasks_copy.pop()
            if self.debug:
                print("Sent task {0} to worker {1} with tag {2}."
                        .format(task, worker, task_index))
            r = self.comm.send(task, dest=worker, tag=task_index)
            return r


        #requests = []
        for worker in xrange(self.size):
            r = send_next_task_to_worker(worker+1)
            #requests.append(r)

        #MPI.Request.waitall(requests)

        nresult = 0
        results = [None for i in xrange(ntask)]

        while nresult < ntask:
            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.source
            task_index = status.tag
            results[task_index] = result
            send_next_task_to_worker(worker)
            nresult+=1

        return results

    def close(self):
        """
        Just send a message off to all the pool members which contains
        the special :class:`_close_pool_message` sentinel.

        """
        if self.is_master():
            for i in range(self.size):
                self.comm.send(_close_pool_message(), dest=i + 1)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

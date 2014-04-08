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
    def __init__(self, comm=MPI.COMM_WORLD, debug=False):
        try:
            from mpi4py import MPI
            self.MPI = MPI
        except ImportError:
            raise RuntimeError("MPI environment not found!")

        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.debug = debug

        self.function = _error_function

    def is_master(self):
        return self.rank == 0

    def wait(self):
        if self.is_master():
            raise RuntimeError("Master node told to await jobs")
        status = self.MPI.Status()
        while True:
            task = self.comm.recv(source=0, tag=self.MPI.ANY_TAG, status=status)

            if isinstance(task, _close_pool_message):
                break

            if isinstance(task, _function_wrapper):
                self.function = task.function
                continue

            results = map(self.function, task)
            self.comm.send(results, dest=0, tag=status.tag) 

    def map(self, function, tasks):
        # Should be called by the master only
        if not self.is_master():
            self.wait()
            return

        # send function if necessary
        if function is not self.function:
            self.function = function
            F = _function_wrapper(function)
            requests = [self.comm.isend(F, dest=i)
                        for i in range(1,self.size)]
            self.MPI.Request.waitall(requests)

        requests = []
        for i in range(1, self.size):
            req = self.comm.isend(tasks[i::self.size], dest=i)
            requests.append(req)

        results = [None]*len(tasks)
        results[::self.size] = map(self.function, tasks[0::self.size])
        status = self.MPI.Status()
        for i in range(self.size - 1):
            result = self.comm.recv(source=self.MPI.ANY_SOURCE, status=status)
            results[status.source::self.size] = result
        return results

    def gather(self, data, root=0):
        return self.comm.gather(data, root)

    def bcast(self, data, root=0):
        return self.comm.bcast(data, root)

    def send(self, data, dest=0, tag=0):
        self.comm.send(data, dest, tag)

    def recv(self, source=0, tag=0):
        return self.comm.recv(source, tag)

    def close(self):
        if self.is_master():
            for i in range(1, self.size):
                self.comm.isend(_close_pool_message(), dest=i)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

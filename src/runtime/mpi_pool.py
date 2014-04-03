import abc
from mpi4py import MPI

class ClosePoolMessage(object):
	def __repr__(self):
		return "<Close pool message>"
	pass

class MPIPool(object):
	__metaclass__ = abc.ABCMeta
	def __init__(self, comm=MPI.COMM_WORLD, debug=False):
		self.comm = comm
		self.rank = comm.Get_rank()
		self.size = comm.Get_size() 
		self.debug = debug
	
    def is_master(self):
		return self.rank == 0

	def wait(self):
		if self.is_master():
			raise RuntimeError("Master node told to await jobs")
		status = MPI.Status()
		while True:
			tasklist = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
			if isinstance(tasklist,ClosePoolMessage):
				break

            results = map(self.process_task, tasklist)
			self.comm.send(results, dest=0, tag=status.tag) 

	@abc.abstractmethod
	def process_task(self, task):
		""" Subclass must override this method with the task to be performed """
        raise NotImplementedError

	def map(self, dummy_function, tasks):
		# Should be called by the master only
		if not self.is_master():
			self.wait()
			return

		#Send all the tasks off.  Do not wait for them to be received, just continue.
        requests = []
        for i, tasks in enumerate(izip_longest(*[iter(tasklist)] * self.size, 
                                               fillvalue=self)):
            # reserve last group for master process since it is likely
            # shorter
            if i == self.size-1:
                # filter fillvalues out of resulting list
                local_tasks = [t for t in tasks if t != self]
            else:
                req = self.comm.isend( list(tasks), dest=(i-1) )
                requests.append(req)

        local_results = map(self.process_task, local_tasks)

		results = []
        for i in range(1,self.size):
            result = self.comm.recv(source=worker, tag=i)
            results.append(result)
        results.append(local_results)
        MPI.Request.Waitall(requests)
		return results

	def close(self):
		if self.is_master():
			for i in xrange(1,self.size):
				self.comm.isend(ClosePoolMessage(), dest=i+1)


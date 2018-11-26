
Options
-------

``charm4py.Options`` is a global object with the following attributes:

* **PROFILING** (default=False): if ``True``, charm4py will profile the program and
  collect timing and message statistics. To print these, the application must call
  ``charm.printStats()``. Note that this will affect performance of the application.

* **PICKLE_PROTOCOL** (default=-1): determines the pickle protocol used by Charm4py.
  A value of ``-1`` tells ``pickle`` to use the highest protocol number (recommended).
  Note that not every type of argument sent to a remote method is pickled.

* **LOCAL_MSG_OPTIM** (default=True): if ``True``, remote method arguments sent to a chare
  that is in the same PE as the caller will be passed by reference (instead of serialized).
  Best performance is obtained when this is enabled, but requires callers to relinquish
  ownership of any objects sent.

* **LOCAL_MSG_BUF_SIZE** (default=50): size of the pool used to store "local" messages
  (see above point).

* **AUTO_FLUSH_WAIT_QUEUES** (default=True): if ``True``, messages or threads waiting
  on a condition (see "when" and "wait" constructs in :ref:`chare-api-label` API) are checked and
  flushed automatically when the conditions are met.
  Otherwise, the application must explicitly call ``self.__flush_wait_queues__()``
  of the chare.

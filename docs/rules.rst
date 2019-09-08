===================
Rules and Semantics
===================

Remote method invocation
------------------------

Every method invocation *via a proxy* is asynchronous. The call returns immediately,
and a message is sent to the destination(s). Therefore, the method is not executed
immediately, not even if the destination is in the same process.

This is what happens to arguments passed via remote invocation:

1. If a call is point-to-point and the source and destination are in the same
   process, arguments are passed by reference in usual Python fashion. In other
   words, the objects become shared (no serialization or copying is involved,
   thus making the call efficient). The same is true for destinations of a section
   broadcast that are in the same process as the source.

2. In all other cases, arguments are serialized and copied into a message
   (see :doc:`serialization` for more information about serialization).

When a remote method is invoked, the recipient might not be the sole owner of
the arguments. This can happen if:

- The destination of a point-to-point send is in the same process as the sender
  (see case 1 above).

- Is the recipient of a broadcast message and there are multiple recipients
  on the same process. In this case, all of them receive references to the same
  Python objects.

The runtime adopts this behavior to guarantee best performance for all applications.
But with the above in mind, if an application needs chares to maintain separate
copies of specific arguments, it can simply copy them before sending or upon reception
as appropiate and required by the application.

Method execution
----------------

The methods of a chare execute on the process where the chare currently lives.
When a method starts executing, it runs until either the method completes or
suspends (this last case only if it is a coroutine). Note that coroutines
are non-preemtive and only yield when they are waiting for something
(typically occurs when they are blocked on a future or channel). When a method
stops executing, control returns to the scheduler which will start or resume
other methods on that process. Note that if a couroutine C of chare A suspends,
the scheduler is free to execute other methods of chare A, even other instances
of coroutine C.

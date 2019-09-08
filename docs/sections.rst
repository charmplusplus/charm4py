==============================================
Sections: Split, Slice and Combine Collections
==============================================

Sections are chare collections formed from subsets of other collections. You
can form sections by splitting, slicing and combining other collections
(Groups, Arrays, Sections) in arbitrary ways.

Creating sections
-----------------

.. caution::
    Creating sections involves messaging. When creating sections using any of
    the below mechanisms, store and reuse the resulting section proxies and do
    not repeatedly create the same sections.


* **charm.split(proxy, numsections, section_func=None, elems=None)**

    Split a chare collection into *numsections* number of sections. Returns a
    list of section proxies.

    The parameter *proxy* is a Group, Array or Section proxy, that refers to
    the collection to split.

    There are two methods of specifying sections. The first, and preferred
    method, is to use *section_func*, which is a function that takes a chare
    object and returns the list of sections to which the object belongs to
    (sections identified from 0 to numsections-1). It is evaluated at the PE
    where the object lives (the function must be defined on that PE). If the
    object is not part of any section, it must return a negative number
    (or empty list).

    The second method consists in using *elems*, which is a list of lists of
    chare indexes that are part of each section. For very large sections,
    this method can be much more expensive than using a *section_func*. Also,
    it is not recommended for collections where multiple elements have the
    same index, as it can't discriminate between them (such collections can
    be obtained using the combine operation -see below-).

    .. tip::
        Elements can be part of multiple sections if desired.

* **Proxy slicing:**

    This is a shorthand notation to obtain one section from a proxy, using
    slicing syntax instead of ``split``:

    * **proxy[[start]:[stop][:step]]**: for group proxies.

    * **proxy[start_0:stop_0[:step_0], ..., start_n-1:stop_n-1[:step_n-1]]**: for
      n-dimensional array proxies.

    See examples below.

* **charm.combine(*proxies)**

    Combines multiple collections into one. Returns a section proxy.

    The parameter *proxies* is a list of proxies that refer to the collections
    to combine. Collection and chare types don't have to match as long as the
    methods that will be called via the section proxy have the same signature.

    Note that sending one broadcast to a combined collection is more efficient
    than sending a separate broadcast to each component. Similarly for reductions.

    .. important::
        The proxy returned by combine can be used for broadcast and reductions
        on the combined collection, and can also be split. But it cannot be used for
        sending messages to individual elements in the combined collection.

Examples
--------

The following example first creates a group of chares and then creates a section
from elements on even-numbered PEs:

.. code-block:: python

    from charm4py import charm, Chare, Group

    class Test(Chare):
        def sayHi(self):
            print('Hello from', self.thisIndex)

    def sectionNo(obj):
        if obj.thisIndex % 2 == 0:
            return [0]
        else:
            return []

    def main(args):
        g = Group(Test)
        # creates one section of elements on even-numbered PEs
        secProxy = charm.split(g, 1, sectionNo)[0]
        # this does the same thing with slicing notation
        secProxy2 = g[::2]
        secProxy.sayHi(awaitable=True).get()
        exit()

    charm.start(main)


The following example creates a 4 x 4 array of chares, and splits it into
4 sections. It then sends the section proxies to the chares, and tells the first
section to perform a section reduction:

.. code-block:: python

    from charm4py import charm, Chare, Array, Future, Reducer

    class Test(Chare):
        def recvSecProxies(self, proxies):
            self.secProxy = proxies[sectionNo(self)]
        def doreduction(self, future):
            self.contribute(1, Reducer.sum, future, self.secProxy)

    def sectionNo(obj):
        return obj.thisIndex[0]  # first index determines the section number

    def main(args):
        a = Array(Test, (4, 4))  # create a 4 x 4 array
        # split array into 4 sections
        secProxies = charm.split(a, 4, sectionNo)
        a.recvSecProxies(secProxies, awaitable=True).get()  # blocks until proxies received
        f = Future()
        # tell section 0 to perform a reduction
        secProxies[0].doreduction(f)
        print(f.get())  # returns 4
        exit()

    charm.start(main)

This final example creates two 4 x 4 chare arrays, combines them into one
section, and broadcasts a message to this section. It then creates 4 sections,
each of which spans subsets of both arrays, and broadcasts a message to each
section:

.. code-block:: python

    from charm4py import charm, Chare, Array

    class Test(Chare):
        def sayHi(self):
            print('Hello from', self.thisIndex)

    def sectionNo(obj):
        return obj.thisIndex[0]  # first index determines the section number

    def main(args):
        a1 = Array(Test, (4, 4))  # create a 4 x 4 array
        a2 = Array(Test, (4, 4))  # create a 4 x 4 array
        combined = charm.combine(a1, a2)
        combined.sayHi()  # broadcast to all members of a1 and a2
        # make 4 cross-array sections involving the two arrays
        secProxies = charm.split(combined, 4, sectionNo)
        futures = []
        for proxy in secProxies:
            futures.append(proxy.sayHi(awaitable=True))
        charm.wait(futures)
        exit()

    charm.start(main)

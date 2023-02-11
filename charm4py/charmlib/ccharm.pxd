
# libcharm wrapper for Cython

cdef extern from "charm.h":

    cdef cppclass CkGroupID:
      int idx;
    cdef cppclass CkChareID:
      int onPE;
      void *objPtr;
    void StartCharmExt(int argc, char **argv);
    int CkMyPeHook();
    int CkNumPesHook();
    double CkWallTimerHook();
    void realCkExit(int exitcode);
    void CmiAbort(const char *, ...);
    void CmiPrintf(const char *, ...);
    void LBTurnInstrumentOn();
    void LBTurnInstrumentOff();

    int CmiPeOnSamePhysicalNode(int pe1, int pe2);
    int CmiNumPhysicalNodes();
    int CmiPhysicalNodeID(int pe);
    int CmiNumPesOnPhysicalNode(int node);
    void CmiGetPesOnPhysicalNode(int node, int **pelist, int *num);
    int CmiGetFirstPeOnPhysicalNode(int node);
    int CmiPhysicalRank(int pe);
    void CkSendMsg(int entryIndex, void *msg, const CkChareID *chare, int opts);

    void CkRegisterReadonlyExt(const char *name, const char *type, size_t msgSize, char *msg);
    void CkRegisterMainChareExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterGroupExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterSectionManagerExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayMapExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);

    int CkCreateGroupExt(int cIdx, int eIdx, int num_bufs, char **bufs, int *buf_sizes);
    int CkCreateArrayExt(int cIdx, int ndims, int *dims, int eIdx, int num_bufs, char **bufs, int *buf_sizes, int map_gid, char useAtSync);
    void CkInsertArrayExt(int aid, int ndims, int *index, int epIdx, int onPE, int num_bufs, char **bufs, int *buf_sizes, char useAtSync);
    void CkArrayDoneInsertingExt(int aid);
    void CkMigrateExt(int aid, int ndims, int *index, int toPe);

    void CkChareExtSend(int onPE, void *objPtr, int epIdx, char *msg, int msgSize);
    void CkChareExtSend_multi(int onPE, void *objPtr, int epIdx, int num_bufs, char **bufs, int *buf_sizes);
    void CkGroupExtSend(int gid, int npes, int *pes, int epIdx, char *msg, int msgSize);
    void CkGroupExtSend_multi(int gid, int npes, int *pes, int epIdx, int num_bufs, char **bufs, int *buf_sizes);
    void CkArrayExtSend(int aid, int *idx, int ndims, int epIdx, char *msg, int msgSize);
    void CkArrayExtSend_multi(int aid, int *idx, int ndims, int epIdx, int num_bufs, char **bufs, int *buf_sizes);
    void CkForwardMulticastMsg(int gid, int num_children, int *children);

    int CkGroupGetReductionNumber(int gid);
    int CkArrayGetReductionNumber(int aid, int ndims, int *index);
    void CkSetMigratable(int aid, int ndims, int *index, char migratable);

    void registerCkRegisterMainModuleCallback(void (*cb)());
    void registerMainchareCtorExtCallback(void (*cb)(int, void*, int, int, char **));
    void registerReadOnlyRecvExtCallback(void (*cb)(int, char*));
    void registerChareMsgRecvExtCallback(void (*cb)(int, void*, int, int, char*, int));
    void registerGroupMsgRecvExtCallback(void (*cb)(int, int, int, char *, int));
    void registerGroupMsgGPUDirectRecvExtCallback(void (*cb)(int, int, int, int *, void *, int, char *, int));
    void registerArrayMsgRecvExtCallback(void (*cb)(int, int, int *, int, int, char *, int));
    void registerArrayBcastRecvExtCallback(void (*cb)(int, int, int, int, int *, int, int, char *, int));
    void registerArrayElemLeaveExtCallback(int (*cb)(int, int, int *, char**, int));
    void registerArrayElemJoinExtCallback(void (*cb)(int, int, int *, int, char*, int));
    void registerArrayResumeFromSyncExtCallback(void (*cb)(int, int, int *));
    void registerCreateCallbackMsgExtCallback(void (*cb)(void*, int, int, int, int *, char**, int*));
    void registerPyReductionExtCallback(int (*cb)(char**, int*, int, char**));
    void registerArrayMapProcNumExtCallback(int (*cb)(int, int, const int *));

    void CkExtContributeToChare(void* contribute_params, int onPE, void* objPtr);
    void CkExtContributeToGroup(void* contribute_params, int gid, int pe);
    void CkExtContributeToArray(void* contribute_params, int aid, int* idx, int ndims);
    void CkExtContributeToSection(void* contribute_params, int sid_pe, int sid_cnt, int rootPE);
    void CkStartQDExt_ChareCallback(int onPE, void* objPtr, int epIdx, int fid);
    void CkStartQDExt_GroupCallback(int gid, int pe, int epIdx, int fid);
    void CkStartQDExt_ArrayCallback(int aid, int* idx, int ndims, int epIdx, int fid);
    void CkStartQDExt_SectionCallback(int sid_pe, int sid_cnt, int rootPE, int ep);
    void CcdCallFnAfter(void (*CcdVoidFn)(void *userParam,double curWallTime), void *arg, double msecs);

    # TODO: Organize these to place them near their related functions
    void CkArrayExtSendWithZCData(int aid, int *idx, int ndims,
                                  int epIdx, int num_bufs, char **bufs,
                                  int *buf_sizes,
                                  void *zcBufPtrs,
                                  int numZCBufs
                                  )

    # void CkArrayExtSendWithDeviceData(int aid, int *idx, int ndims,
    #                                   int epIdx, int num_bufs, char **bufs,
    #                                   int *buf_sizes,
    #                                   long *devBufPtrs,
    #                                   int *devBufSizesInBytes,
    #                                   long *streamPtrs, int numDevBufs
    #                                  );
    # void CkGroupExtSendWithDeviceData(int gid, int pe, int epIdx, int num_bufs, char **bufs,
                                      # int *buf_sizes, long *devBufPtrs,
                                      # int *devBufSizesInBytes,
                                      # long *streamPtrs, int numDevBufs
                                      # );

    struct CkReductionTypesExt:
      int nop
      int sum_char
      int sum_short
      int sum_int
      int sum_long
      int sum_long_long
      int sum_uchar
      int sum_ushort
      int sum_uint
      int sum_ulong
      int sum_ulong_long
      int sum_float
      int sum_double
      int product_char
      int product_short
      int product_int
      int product_long
      int product_long_long
      int product_uchar
      int product_ushort
      int product_uint
      int product_ulong
      int product_ulong_long
      int product_float
      int product_double
      int max_char
      int max_short
      int max_int
      int max_long
      int max_long_long
      int max_uchar
      int max_ushort
      int max_uint
      int max_ulong
      int max_ulong_long
      int max_float
      int max_double
      int min_char
      int min_short
      int min_int
      int min_long
      int min_long_long
      int min_uchar
      int min_ushort
      int min_uint
      int min_ulong
      int min_ulong_long
      int min_float
      int min_double
      int logical_and_bool
      int logical_or_bool
      int logical_xor_bool
      int external_py

    void registerArrayMsgZCRecvExtCallback(void (*cb)(int, int, int*, int, int, size_t*, void *, int, char*,int));
    void CkGetZCData(int numBuffers, void *recvBufPtrs, int *arrSizes,
                     void *remoteBufInfos, int futureId);
    int CkZCBufferSizeInBytes();

    void registerDepositFutureWithIdFn(void (*cb)(void*, void*));

cdef extern from "charm++.h":
  cdef cppclass CkEntryOptions:
    pass

cdef extern from "spanningTree.h":
    void getPETopoTreeEdges(int pe, int rootPE, int *pes, int numpes, unsigned int bfactor,
                            int *parent, int *child_count, int **children);

cdef extern from "ckarrayindex.h":
  cdef cppclass CkArrayID:
    CkGroupID _gid;
    CkArrayID(CkGroupID g);

  cdef cppclass CkArrayIndex:
    CkArrayIndex();
    CkArrayIndex(int ndims, int dims[]);

cdef extern from "ckarray.h":
  cdef cppclass CProxyElement_ArrayBase:
    @staticmethod
    void ckSendWrapper(CkArrayID _aid, CkArrayIndex _idx, void *m, int ep, int opts);
  void CkBroadcastMsgArray(int entryIndex, void* msg, CkArrayID aID, int opts);

cdef extern from "pup.h" namespace "PUP":
  cdef cppclass toMem:
    toMem();
    toMem(void *Nbuf, const unsigned int purpose);
    void operator|[T](T &a)
    void operator()(void *bytes, int size)

  cdef cppclass fromMem:
    fromMem();
    fromMem(void *Nbuf, const unsigned int purpose);
    void operator|[T](T &a)
    void operator()(void *bytes, int size)

  cdef cppclass sizer:
    sizer();
    sizer(const unsigned int purpose);
    void operator|[T](T &a)
    void operator()(void *bytes, int size)
    size_t size();

cdef extern from "ckmarshall.h":
  cdef cppclass CkMarshallMsg:
    char *msgBuf;
  CkMarshallMsg *CkAllocateMarshallMsg(int size, const CkEntryOptions *opts);

cdef extern from "cklocation.h":
  cdef cppclass CkArrayMessage:
    void array_setIfNotThere(unsigned int)

cdef extern from "envelope.h":
  cdef cppclass envelope:
    void setMsgType(const unsigned char m);
  envelope *UsrToEnv(const void *const msg);

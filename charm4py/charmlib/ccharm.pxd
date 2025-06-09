
# libcharm wrapper for Cython

cdef extern from "charm.h":

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

    void CkRegisterReadonlyExt(const char *name, const char *type, size_t msgSize, char *msg);
    void CkRegisterMainChareExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterGroupExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterSectionManagerExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayMapExt(const char *s, const char **emNames, int emNamesStart, int numEntryMethods, int *chareIdx, int *startEpIdx);

    int CkCreateGroupExt(int cIdx, int eIdx, int num_bufs, char **bufs, int *buf_sizes);
    int CkCreateArrayExt(int cIdx, int ndims, int *dims, int eIdx, int num_bufs, char **bufs, int *buf_sizes, int map_gid, char useAtSync, int otherIdx);
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

    void CkHapiAddCallback(long stream, void (*cb)(void*, void*), int fid);
    
    int CkTraceRegisterUserEvent(char *EventDesc, int eventID);
    void CkTraceBeginUserBracketEvent(int eventID);
    void CkTraceEndUserBracketEvent(int eventID);

cdef extern from "conv-header.h":
    ctypedef void (*CmiHandler)(void* )
    cdef const int CmiReservedHeaderSize

cdef extern from "sockRoutines.h":

    ctypedef struct skt_ip_t:
        int tag
    
    ctypedef struct ChMessageInt_t:
        unsigned char[4] data

cdef extern from "ccs-server.h":

    ctypedef struct CcsSecAttr:
        skt_ip_t ip
        ChMessageInt_t port
        ChMessageInt_t replySalt
        unsigned char auth
        unsigned char level
    
    ctypedef struct CcsImplHeader:
        CcsSecAttr attr
        char[32] handler
        ChMessageInt_t pe
        ChMessageInt_t replyFd
        ChMessageInt_t len

cdef extern from "conv-ccs.h":
    ctypedef struct CcsDelayedReply:
        CcsImplHeader *hdr;
    void CcsRegisterHandlerExt(const char *ccs_handlername, void *fn);
    int CcsIsRemoteRequest();
    void CcsSendReply(int replyLen, const void *replyData);
    void CcsSendDelayedReply(CcsDelayedReply d,int replyLen, const void *replyData)
    CcsDelayedReply CcsDelayReply()

cdef extern from "spanningTree.h":
    void getPETopoTreeEdges(int pe, int rootPE, int *pes, int numpes, unsigned int bfactor,
                            int *parent, int *child_count, int **children);


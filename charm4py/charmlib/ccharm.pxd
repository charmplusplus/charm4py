
# libcharm wrapper for Cython

cdef extern from "charm.h":

    void StartCharmExt(int argc, char **argv);
    int CkMyPeHook();
    int CkNumPesHook();
    double CkWallTimerHook();
    void realCkExit(int exitcode);
    void CmiAbort(const char *);
    void CmiPrintf(const char *);
    void LBTurnInstrumentOn();
    void LBTurnInstrumentOff();

    void CkRegisterReadonlyExt(const char *name, const char *type, size_t msgSize, char *msg);
    void CkRegisterMainChareExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterGroupExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);
    void CkRegisterArrayMapExt(const char *s, int numEntryMethods, int *chareIdx, int *startEpIdx);

    int CkCreateGroupExt(int cIdx, int eIdx, int num_bufs, char **bufs, int *buf_sizes);
    int CkCreateArrayExt(int cIdx, int ndims, int *dims, int eIdx, int num_bufs, char **bufs, int *buf_sizes, int map_gid, char useAtSync);
    void CkInsertArrayExt(int aid, int ndims, int *index, int epIdx, int onPE, int num_bufs, char **bufs, int *buf_sizes, char useAtSync);
    void CkArrayDoneInsertingExt(int aid);
    void CkMigrateExt(int aid, int ndims, int *index, int toPe);

    void CkChareExtSend(int onPE, void *objPtr, int epIdx, char *msg, int msgSize);
    void CkChareExtSend_multi(int onPE, void *objPtr, int epIdx, int num_bufs, char **bufs, int *buf_sizes);
    void CkGroupExtSend(int gid, int pe, int epIdx, char *msg, int msgSize);
    void CkGroupExtSend_multi(int gid, int pe, int epIdx, int num_bufs, char **bufs, int *buf_sizes);
    void CkArrayExtSend(int aid, int *idx, int ndims, int epIdx, char *msg, int msgSize);
    void CkArrayExtSend_multi(int aid, int *idx, int ndims, int epIdx, int num_bufs, char **bufs, int *buf_sizes);

    int CkGroupGetReductionNumber(int gid);
    int CkArrayGetReductionNumber(int aid, int ndims, int *index);

    void registerCkRegisterMainModuleCallback(void (*cb)());
    void registerMainchareCtorExtCallback(void (*cb)(int, void*, int, int, char **));
    void registerReadOnlyRecvExtCallback(void (*cb)(int, char*));
    void registerChareMsgRecvExtCallback(void (*cb)(int, void*, int, int, char*, int));
    void registerGroupMsgRecvExtCallback(void (*cb)(int, int, int, char *, int));
    void registerArrayMsgRecvExtCallback(void (*cb)(int, int, int *, int, int, char *, int));
    void registerArrayElemLeaveExtCallback(int (*cb)(int, int, int *, char**, int));
    void registerArrayElemJoinExtCallback(void (*cb)(int, int, int *, int, char*, int));
    void registerArrayResumeFromSyncExtCallback(void (*cb)(int, int, int *));
    void registerCreateReductionTargetMsgExtCallback(void (*cb)(void*, int, int, int, char**, int*));
    void registerPyReductionExtCallback(int (*cb)(char**, int*, int, char**));
    void registerArrayMapProcNumExtCallback(int (*cb)(int, int, const int *));

    void CkExtContributeToChare(void* contribute_params, int onPE, void* objPtr);
    void CkExtContributeToGroup(void* contribute_params, int gid, int pe);
    void CkExtContributeToArray(void* contribute_params, int aid, int* idx, int ndims);


cdef extern from "spanningTree.h":
    void getPETopoTreeEdges(int pe, int rootPE, int *pes, int numpes, unsigned int bfactor,
                            int *parent, int *child_count, int **children);


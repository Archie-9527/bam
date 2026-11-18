
/**
 * @file
 * @brief BaM 的 CPU 侧控制器初始化和 GPU 可访问的队列对描述。
 */
#ifndef __BENCHMARK_CTRL_H__
#define __BENCHMARK_CTRL_H__

// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <cstdint>
#include "buffer.h"
#include "nvm_types.h"
#include "nvm_ctrl.h"
#include "nvm_aq.h"
#include "nvm_admin.h"
#include "nvm_util.h"
#include "nvm_error.h"
#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <cstdlib>
#include <algorithm>
#include <cuda/atomic>
#include <cuda/std/atomic>

#include "queue.h"


#define MAX_QUEUES 65535


/// @brief 聚合 NVMe 控制器引用、管理队列、namespace 信息和 I/O 队列对。
struct Controller
{
    cuda::atomic<uint64_t, cuda::thread_scope_device> access_counter;
    nvm_ctrl_t*             ctrl; /**< libnvm 控制器句柄，持有 BAR 映射及控制器属性。 */
    nvm_aq_ref              aq_ref; /**< 提交主机侧管理命令的队列引用。 */
    DmaPtr                  aq_mem; /**< 管理队列及 Identify 数据缓冲区所在的主机 DMA 内存。 */
    struct nvm_ctrl_info    info;
    struct nvm_ns_info      ns;
    uint16_t                n_sqs;
    uint16_t                n_cqs;
    uint16_t                n_qps; /**< 实际创建并提供给 GPU 使用的 I/O 队列对数量。 */
    uint32_t                deviceId;
    QueuePair**             h_qps; /**< CPU 持有的队列对对象指针数组。 */
    QueuePair*              d_qps; /**< GPU 内存中的队列对描述数组。 */

    cuda::atomic<uint64_t, cuda::thread_scope_device> queue_counter; /**< GPU 选择 I/O 队列时使用的轮转计数器。 */

    uint32_t page_size;
    uint32_t blk_size; /**< namespace 逻辑块的字节数。 */
    uint32_t blk_size_log;


    void* d_ctrl_ptr; /**< GPU 侧 Controller 视图的地址。 */
    BufferPtr d_ctrl_buff;
#ifdef __DIS_CLUSTER__
    Controller(uint64_t controllerId, uint32_t nvmNamespace, uint32_t adapter, uint32_t segmentId);
#endif

    Controller(const char* path, uint32_t nvmNamespace, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues);

    void reserveQueues();

    void reserveQueues(uint16_t numSubmissionQueues);

    void reserveQueues(uint16_t numSubmissionQueues, uint16_t numCompletionQueues);

    void print_reset_stats(void);

    ~Controller();
};



using error = std::runtime_error;
using std::string;


inline void Controller::print_reset_stats(void) {
    cuda_err_chk(cudaMemcpy(&access_counter, d_ctrl_ptr, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>), cudaMemcpyDeviceToHost));
    std::cout << "------------------------------------" << std::endl;
    std::cout << std::dec << "#SSDAccesses:\t" << access_counter << std::endl;

    cuda_err_chk(cudaMemset(d_ctrl_ptr, 0, sizeof(cuda::atomic<uint64_t, cuda::thread_scope_device>)));
}

/**
 * @brief 建立管理队列，读取控制器和 namespace 属性及当前队列数。
 * @param[in,out] ctrl 正在初始化的控制器。
 * @param ns_id 要识别的 NVMe namespace ID。
 */
static void initializeController(struct Controller& ctrl, uint32_t ns_id)
{
    // Create admin queue reference
    int status = nvm_aq_create(&ctrl.aq_ref, ctrl.ctrl, ctrl.aq_mem.get());
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reset controller: ") + nvm_strerror(status));
    }

    // Identify controller
    status = nvm_admin_ctrl_info(ctrl.aq_ref, &ctrl.info, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Identify namespace
    status = nvm_admin_ns_info(ctrl.aq_ref, &ctrl.ns, ns_id, NVM_DMA_OFFSET(ctrl.aq_mem, 2), ctrl.aq_mem->ioaddrs[2]);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }

    // Get number of queues
    status = nvm_admin_get_num_queues(ctrl.aq_ref, &ctrl.n_cqs, &ctrl.n_sqs);
    if (!nvm_ok(status))
    {
        throw error(nvm_strerror(status));
    }
}



#ifdef __DIS_CLUSTER__
Controller::Controller(uint64_t ctrl_id, uint32_t ns_id, uint32_t)
    : ctrl(nullptr)
    , aq_ref(nullptr)
{
    // Get controller reference
    int status = nvm_dis_ctrl_init(&ctrl, ctrl_id);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Create admin queue memory
    aq_mem = createDma(ctrl, ctrl->page_size * 3, 0, 0);

    initializeController(*this, ns_id);
}
#endif



/**
 * @brief 打开 libnvm 字符设备，初始化 NVMe 控制器并向 GPU 提供 I/O 队列对。
 * @details CPU 先建立管理队列，再注册 BAR 映射并建立 GPU 上的 I/O SQ/CQ。
 * @param path libnvm 字符设备路径。
 * @param ns_id NVMe namespace ID。
 * @param cudaDevice 分配 I/O 队列的 CUDA 设备编号。
 * @param queueDepth 每条 I/O 队列的请求深度上限。
 * @param numQueues 请求使用的 I/O 队列对数量上限。
 */
inline Controller::Controller(const char* path, uint32_t ns_id, uint32_t cudaDevice, uint64_t queueDepth, uint64_t numQueues)
    : ctrl(nullptr)
    , aq_ref(nullptr)
    , deviceId(cudaDevice)
{
    int fd = open(path, O_RDWR);
    if (fd < 0)
    {
        throw error(string("Failed to open descriptor: ") + strerror(errno));
    }

    // Get controller reference
    size_t page_size = sysconf(_SC_PAGESIZE); /**< 用于对齐 BAR 映射长度的系统页大小。 */
    /// BAR 头部占 4096 B，每个队列对的两个 doorbell 共占 8 B，映射长度向上按页对齐。
    size_t mm_size = ((0x1000 + (numQueues + 1) * 8) + page_size - 1) & ~(page_size - 1);
    int status = nvm_ctrl_init(&ctrl, fd, mm_size);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to get controller reference: ") + nvm_strerror(status));
    }

    // Create admin queue memory
    aq_mem = createDma(ctrl, ctrl->page_size * 3);

    initializeController(*this, ns_id);
    cudaError_t err = cudaHostRegister((void*) ctrl->mm_ptr, mm_size, cudaHostRegisterIoMemory);
    if (err != cudaSuccess)
    {
        throw error(string("Unexpected error while mapping IO memory (cudaHostRegister): ") + cudaGetErrorString(err));
    }
    queue_counter = 0;
    page_size = ctrl->page_size;
    blk_size = this->ns.lba_data_size;
    blk_size_log = std::log2(blk_size);
    reserveQueues(MAX_QUEUES,MAX_QUEUES);
    n_qps = std::min(n_sqs, n_cqs);
    n_qps = std::min(n_qps, (uint16_t)numQueues);
    printf("SQs: %d\tCQs: %d\tn_qps: %d\n", n_sqs, n_cqs, n_qps);
    h_qps = (QueuePair**) malloc(sizeof(QueuePair)*n_qps);
    cuda_err_chk(cudaMalloc((void**)&d_qps, sizeof(QueuePair)*n_qps));
    for (size_t i = 0; i < n_qps; i++) {
        //printf("started creating qp\n");
        h_qps[i] = new QueuePair(ctrl, cudaDevice, ns, info, aq_ref, i+1, queueDepth);
        //printf("finished creating qp\n");
        cuda_err_chk(cudaMemcpy(d_qps+i, h_qps[i], sizeof(QueuePair), cudaMemcpyHostToDevice));
    }
    //printf("finished creating all qps\n");


    close(fd);

    d_ctrl_buff = createBuffer(sizeof(Controller), cudaDevice);
    d_ctrl_ptr = d_ctrl_buff.get();
    cuda_err_chk(cudaMemcpy(d_ctrl_ptr, this, sizeof(Controller), cudaMemcpyHostToDevice));
}



/// @brief 释放队列对的 GPU 和 CPU 描述，再销毁管理队列及控制器引用。
inline Controller::~Controller()
{
    cudaFree(d_qps);
    for (size_t i = 0; i < n_qps; i++) {
        delete h_qps[i];
    }
    free(h_qps);
    nvm_aq_destroy(aq_ref);
    nvm_ctrl_free(ctrl);

}



inline void Controller::reserveQueues()
{
    reserveQueues(n_sqs, n_cqs);
}



inline void Controller::reserveQueues(uint16_t numSubmissionQueues)
{
    reserveQueues(numSubmissionQueues, n_cqs);
}



/// @brief 通过管理命令与控制器协商 I/O SQ 和 CQ 数量。
inline void Controller::reserveQueues(uint16_t numSubs, uint16_t numCpls)
{
    int status = nvm_admin_request_num_queues(aq_ref, &numSubs, &numCpls);
    if (!nvm_ok(status))
    {
        throw error(string("Failed to reserve queues: ") + nvm_strerror(status));
    }

    n_sqs = numSubs;
    n_cqs = numCpls;

}



#endif

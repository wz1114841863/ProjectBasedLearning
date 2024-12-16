#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>
#ifdef USE_PTHREADS
#include <pthread.h>
#else
#include <omp.h>
#endif
#include <stdlib.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// SRAND48 and DRAND48 don't exist on windows, but these are the equivalent
// functions
void srand48(long seed) { srand((unsigned int)seed); }
double drand48() { return double(rand()) / RAND_MAX; }
#endif

const char *sSDKname = "UnifiedMemoryStreams";

template <typename T>
struct Task {
    unsigned int size, id;
    T *data;
    T *result;
    T *vector;

    Task() : size(0), id(0), data(NULL), result(NULL), vector(NULL) {};
    Task(unsigned int s) : size(s), id(0), data(NULL), result(NULL) {
        // allocate unified memory -- the operation performed in this example will be a DGEMV
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~Task() {
        // ensure all memory is dellocated
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(data));
        checkCudaErrors(cudaFree(result));
        checkCudaErrors(cudaFree(vector));
    }

    void allocate(const unsigned int s, const unsigned int unique_id) {
        // allocated unified memory outside of constructor
        id = unique_id;
        size = s;
        checkCudaErrors(cudaMallocManaged(&data, sizeof(T) * size * size));
        checkCudaErrors(cudaMallocManaged(&result, sizeof(T) * size));
        checkCudaErrors(cudaMallocManaged(&vector, sizeof(T) * size));
        checkCudaErrors(cudaDeviceSynchronize());

        for (unsigned int i = 0; i < size * size; ++i) {
            data[i] = drand48();
        }

        for (unsigned int i = 0; i < size; ++i) {
            result[i] = 0;
            vector[i] = drand48();
        }
    }
};

#ifdef USE_PTHREADS
struct threadData_t {
    int tid;
    Task<double> *TaskListPtr;
    cudaStream_t *streams;
    cublasHandle_t *handles;
    int taskSize;
}

typedef struct threadData_t threadData;
#endif


template <typename T>
void gemv (int m, int n, T alpha, T *A, T *x, T beta, T *result) {
    // simple host dgemv: assume data is in row-major format and square
    // 复杂度是O(n²)
    for (int i = 0; i < n; ++i) {
        result[i] *= beta;
        for (int j = 0; j < n; ++j) {
            result[i] += A[i * n + j] * x[j];
        }
    }
}

// execute a single task on a either host or device depending on size
#ifdef USE_PTHREADS
void *execute(void *inpArgs) {
    threadData *dataPtr = (threadData *)inpArgs;
    cudaStream_t *stream = dataPtr->streams;
    cublasHandle_t *handle = dataPtr->handles;
    int tid = dataPtr->tid;

    for (int i = 0; i < dataPtr->taskSize; ++i) {
        Task<double> &t = dataPtr->TaskListPtr[i];
        if (t.size < 100) {
            printf("Task %d, thread %d executing on host %d. \n", t.id. tid, t.size);
            // 将指定的内存区域与给定的 CUDA 流绑定.这种绑定可以优化数据在 CPU 和 GPU 之间的共享
            // stream[0] 是一个共享的 CUDA 流.
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
            // necessary to ensure Async cudaStreamAttachMemAsync calls have finished
            checkCudaErrors(cudaStreamSynchronize(stream[0]));

            gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
        }else {
            // perform on device
            printf("Task [%d], thread [%d] executing on device (%d)\n", t.id, tid, t.size);
            double one = 1.0;
            double zero = 0.0;
            // stream[tid + 1] 是分配给每个线程的独立流, 独立流允许每个线程在 GPU 上并行执行任务,从而实现任务的并发处理.
            checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0, cudaMemAttachSingle));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0, cudaMemAttachSingle));
            checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0, cudaMemAttachSingle));
            // call the device operation
            checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size,
                                        &one, t.data, t.size, t.vector, 1, &zero,
                                        t.result, 1));
        }
    }
    pthread_exit(NULL);
}
#else
template <typename T>
void execute(Task<T> &t, cublasHandle_t *handle, cudaStream_t *stream, int tid) {
    if (t.size < 100) {
        // perform on host
        printf("Task [%d], thread [%d] executing on host (%d)\n", t.id, tid, t.size);

        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.data, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.vector, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[0], t.result, 0, cudaMemAttachHost));
        checkCudaErrors(cudaStreamSynchronize(stream[0]));

        gemv(t.size, t.size, 1.0, t.data, t.vector, 0.0, t.result);
    }else {
        // perform on device
        printf("Task [%d], thread [%d] executing on device (%d)\n", t.id, tid, t.size);
        double one = 1.0;
        double zero = 0.0;

        checkCudaErrors(cublasSetStream(handle[tid + 1], stream[tid + 1]));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.data, 0, cudaMemAttachSingle));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.vector, 0, cudaMemAttachSingle));
        checkCudaErrors(cudaStreamAttachMemAsync(stream[tid + 1], t.result, 0, cudaMemAttachSingle));
        // call the device operation
        checkCudaErrors(cublasDgemv(handle[tid + 1], CUBLAS_OP_N, t.size, t.size,
                                    &one, t.data, t.size, t.vector, 1, &zero,
                                    t.result, 1));
    }
}
#endif

// populate a list of tasks with random sizes
template <typename T>
void initialise_tasks(std::vector<Task<T> > &TaskList) {
    for (unsigned int i = 0; i < TaskList.size(); i++) {
        // generate random size
        int size;
        size = std::max((int)(drand48() * 1000.0), 64);
        TaskList[i].allocate(size, i);
    }
}

int main(int argc, char **argv) {
    // set device
    cudaDeviceProp device_prop;
    int dev_id = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&device_prop, dev_id));

    if (!device_prop.managedMemory) {
        // This samples requires being run on a device that supports Unified Memory
        fprintf(stderr, "Unified Memory not supported on this device\n");
        exit(EXIT_WAIVED);
    }

    if (device_prop.computeMode == cudaComputeModeProhibited) {
        // This sample requires being run with a default or process exclusive mode
        fprintf(stderr,
                "This sample requires a device in either default or process "
                "exclusive mode\n");

        exit(EXIT_WAIVED);
    }

    // randomise task sizes
    int seed = (int)time(NULL);
    srand48(seed);

    // set number of threads
    const int nthreads = 4;

    // number of streams = number of threads
    cudaStream_t *streams = new cudaStream_t[nthreads + 1];
    cublasHandle_t *handles = new cublasHandle_t[nthreads + 1];

    for (int i = 0; i < nthreads + 1; ++i) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
        checkCudaErrors(cublasCreate(&handles[i]));
    }

    // create list of N tasks
    unsigned int N = 40;
    std::vector<Task<double> > TaskList(N);
    initialise_tasks(TaskList);

    printf("Executing tasks on host / device. \n");

// run through all tasks using threads and streams
#ifdef USE_PTHREADS
    pthread_t threads[nthreads];
    threadData *InputToThreads = new threadData[nthreads];

    for (int i = 0; i < nthreads; ++i) {
        checkCudaErrors(cudaSetDevice(dev_id));
        InputToThreads[i].tid = i;
        InputToThreads[i].streams = stream;
        InputToThreads[i].handles = handles;

        if ((TaskList.size() / nthreads) == 0) {
            InputToThreads[i].taskSize = (TaskList.size() / nthreads);
            InputToThreads[i].TaskListPtr = &TaskList[i * (TaskList.size() / nthreads)];
        }else {
            if (i == nthreads - 1) {
                InputToThreads[i].taskSize =
                    (TaskList.size() / nthreads) + (TaskList.size() % nthreads);
                InputToThreads[i].TaskListPtr =
                    &TaskList[i * (TaskList.size() / nthreads) +
                            (TaskList.size() % nthreads)];
            } else {
                InputToThreads[i].taskSize = (TaskList.size() / nthreads);
                InputToThreads[i].TaskListPtr =
                    &TaskList[i * (TaskList.size() / nthreads)];
            }
        }

        pthread_create(&threads[i], NULL, &execute, &InputToThreads[i]);
    }

    for (int i = 0; i < nthreads; ++i) {
        pthread_join(threads[i], NULL);
    }
#else
    // OpenMP 用于管理多线程并行,负责任务的动态分配
    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < TaskList.size(); ++i) {
        checkCudaErrors(cudaSetDevice(dev_id));
        int tid = omp_get_thread_num();
        execute(TaskList[i], handles, streams, tid);
    }
#endif

    cudaDeviceSynchronize();

    // Destroy CUDA Streams, cuBlas handles
    for (int i = 0; i < nthreads + 1; i++) {
        cudaStreamDestroy(streams[i]);
        cublasDestroy(handles[i]);
    }

    // Free TaskList
    std::vector<Task<double> >().swap(TaskList);

    printf("All Done!\n");
    exit(EXIT_SUCCESS);
}

# optimizing parallel reduction

Some notes from nvidia corp. Sources in references.

Reductions are a common and important data parallel primitive. We look at different methods.

## A tree based approach
Consider a contiguous block of memory, representing an array or buffer, where for pairwise addresses we are 
performing sums of values contained therein. Each 'valid' address is dictated by the length of the
array monomer datatype. The resulting array is then half of the length of the original. Continue until we have a
single value left and the vector reduction is completed.

Since for reductions the code is self-similar regardless of the depth reached, we can perform recursive kernel
invocations. Why this works? Cuda has no global synchronization -- the reasons for this are besides the point,
but surround hardware complexities and constraints wrt the programmer. Yet kernel launches serve as a global
synchronization point while kernel launch overhead is negligible in terms of hardware and is very modest in terms
of software. Of course, minimising overhead will be tied to masking queueing operations with ongoing executions,
but this is simply the asynchronous paradigm that is often seen in graphics programming.

My assumption: for this particular case, under no ongoing data transfers between the host and device, the effort is
still less. 

### Optimization objective (software)
We wish to maximize GPU performance. The immediate implication is that we should choose the appropriate metric. 
Typically the performance gauges are either of
* GFLOP/s for compute-bound kernels
* Bandwidth for memory-bound kernels

In the context of reductions, the arithmetic intensity is low, about 1 flop per element loaded and hence bandwidth 
optimal. Clearly, we then wish to maximise bandwidth utilization.

## Methods
From here out, we write in relation to the kernel CUDA code in ```reductions.cu```.

### Interleaved addressing with divergent branching
Recall that the ```__global__``` function execution space specifier defines it as a kernel. Particularly, one that can
be executed from either the host or device, and (inherently) executes on the device, and which can only
have the return type ```void```. These can not be member functions of a class, require specification of the execution
configuration, and are fully asynchronous (i.e., they return before the device has completed execution).

Consequently, the function takes in two parameters, ```int *u```, ```int *v```, which are the input and output
data arrays, respectively. Here we take advantage of the fact that arrays decay into pointers in the usual fashion.

Remark: in order to prevent such decay we must pass arrays by reference (e.g., ``` int (&u)[N]```)
implying array sizes are known at compile time. To clarify, decay here refers to the fact that there is a loss
of information upon pointer typecast. We will no longer have the ability to throw in
sizeof() operators for full array operations/traversal, and instead we would obtain the size of the first 
element in the array, as the recipient of the now-decayed pointer. This is why usually array pointers are
input as well as their size.

The first operation is a forward declaration of a shared memory block. The ```__shared__``` specifier tells us a
number of important things. It speaks of a variable that resides in the shared memory, which is considerably faster
than global memory. It resides in the shared space of a thread block (much higher locality). As a result this 
variable will have the same lifetime and scope as the block -- it will not be available to threads outside this block and
no persistence past it. Shared memory is also dynamic in that it has no constant address. The size of such an array is 
dynamically determined at launch-time (not compile-time) thorugh the execution configuration (introduces a third parameter).

We then move to defining the thread index state variables. We have two:
* ```idx```: identifies each thread within the block; a relative index of sorts
* ```i```: identifies the global ("true") index; calculates the threads' corresponding index in the global input array ```u```.

Note that threadIdx, blockIdx, and blockDim are all built-ins.

Right after we prompt each thread to load its corresponding element to the shared memory, and close the operation by syncing
all the threads (this ensures all threads finished before moving on).

Now we can look at the main logic block. We have a ```for``` loop which is block-wise in terms
of operation level, i.e., we are traversing blocks. For each block, we are reducing the number
of active threads while summing up adjacent elements in the shared memory array ```w[]```, until
only one remains, and the thread with index ```idx == 0``` stores the final result.
The appropriate interpretation here is that our stride ```s``` is the step size between elements
being added. It will start at 1 and double every iteration (with our update rule ```s *= 2```).
```idx```, the local thread index tracks the base values per se. 
On the first iteration, ```s = 1```, threads with indices ```idx``` = 0, 2, 4, ... add elements
elements that are 1 position appart, since ```idx % (2 * 1) == 0```; in the second iteration
only local thread indices ```idx = 0, 4, 8, ...``` since for them ```idx % (2 * 2) == 0``` and so on. 
After the right additions have taken place, we sync the threads. 

Once the loop finishes, we bring the result to global memory, by just reading from the
first index of the shared mem array.


## Comments
These notes are quite scrappy, I will transcribe them to latex and complete them a bit more
formally and better explained.

## References

[1.] Mark Harris. Nvidia Corp. *Optimizing Parallel Reduction in Cuda*. 
     https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

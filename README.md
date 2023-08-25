# zigTensor

zigTensor is a fast, flexible machine learning library written entirely in Zig;
the design is heavily inspired by the [Flashlight](https://github.com/flashlight/flashlight)
library.

### Status

The project is in the incredibly early stages of development and not enough of the core
functionality has been implemented for the library to be usable. Currently, development
is solely focused on OSx and leveraging ArrayFire for the backend.

### Roadmap

- [ ] autograd
- [ ] common
- [ ] contrib
- [ ] dataset
- [ ] distributed
- [ ] meter
- [ ] nn
- [ ] optim
- [ ] runtime
  - [ ] CUDADevice
  - [ ] CUDAStream
  - [ ] CUDAUtils (WIP)
  - [ ] Device (WIP)
  - [x] DeviceManager
  - [x] DeviceType
  - [ ] Stream
  - [ ] SynchronousStream
- [ ] tensor
  - [ ] backend
    - [ ] ArrayFire (current focus - WIP)
      - [ ] mem
        - [ ] CachingMemoryManager
        - [ ] DefaultMemoryManager
        - [ ] MemoryManagerAdapter
        - [ ] MemoryManagerAdapterDeviceInterface
        - [ ] MemoryManagerInstaller
      - [ ] AdvancedIndex
      - [ ] ArrayFireBLAS
      - [ ] ArrayFireBackend (WIP)
      - [ ] ArrayFireBinaryOps
      - [ ] ArrayFireCPUStream
      - [ ] ArrayFireReductions
      - [ ] ArrayFireShapeAndIndex
      - [ ] ArrayFireTensor (WIP)
      - [ ] ArrayFireUnaryOps
      - [ ] Utils (WIP)
    - [ ] JIT
    - [ ] oneDNN
    - [ ] Stub
  - [ ] CUDAProfile
  - [ ] Compute
  - [ ] DefaultTensorType (WIP)
  - [x] Index
  - [ ] Init (WIP)
  - [ ] Profile
  - [ ] Random
  - [x] Shape
  - [ ] TensorAdapter (WIP)
  - [ ] TensorBackend (WIP)
  - [ ] TensorBase (WIP)
  - [ ] TensorExtension
  - [x] Types

### Backend Roadmap

- [ ] ArrayFire (current focus - WIP)
  - [ ] CPU (priority 1 - supports OSx)
  - [ ] OpenCL
  - [ ] CUDA
- [ ] JIT
- [ ] oneDNN
- [ ] Stub

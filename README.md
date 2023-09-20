# zigTensor

zigTensor is a fast, flexible machine learning library written entirely in Zig;
the design is heavily inspired by the [Flashlight](https://github.com/flashlight/flashlight)
library.

## Minimum Requirements

| Requirement | Notes               |
| ----------- | ------------------- |
| Zig version | main                |
| ArrayFire   | latest via homebrew |
| OSx         | currently Mac only  |

Install with [zigup](https://github.com/marler8997/zigup):

```bash
zigup master
```

Install ArrayFire with [homebrew](https://formulae.brew.sh/formula/arrayfire#default):

```bash
brew install arrayfire
```

### Status

The project is in the incredibly early stages of development and not enough of the core
functionality has been implemented for the library to be usable. Currently, development
is solely focused on OSx and leveraging ArrayFire for the backend.

### Roadmap

- [ ] bindings
  - [x] ArrayFire
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
  - [x] Device
  - [x] DeviceManager
  - [x] DeviceType
  - [x] Stream
  - [x] SynchronousStream
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
      - [x] ArrayFireBLAS
      - [ ] ArrayFireBackend (WIP)
      - [x] ArrayFireBinaryOps
      - [ ] ArrayFireCPUStream (WIP)
      - [ ] ArrayFireReductions (WIP)
      - [ ] ArrayFireShapeAndIndex
      - [ ] ArrayFireTensor (WIP)
      - [ ] ArrayFireUnaryOps (WIP)
      - [x] Utils
    - [ ] JIT
    - [ ] oneDNN
    - [ ] Stub
  - [ ] CUDAProfile
  - [ ] Compute
  - [ ] DefaultTensorType (WIP)
  - [x] Index
  - [ ] Init (WIP)
  - [ ] Profile
  - [x] Random
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

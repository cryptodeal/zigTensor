# zigTensor

zigTensor is a fast, flexible machine learning library written entirely in Zig;
the design is heavily inspired by the [Flashlight](https://github.com/flashlight/flashlight)
library.

## Minimum Requirements

| Requirement | Notes                            |
| ----------- | -------------------------------- |
| Zig version | main                             |
| ArrayFire   | latest via homebrew              |
| OS          | OSx & Linux (verified on Ubuntu) |

Install with [zigup](https://github.com/marler8997/zigup):

```bash
zigup master
```

### OSx Setup

Install ArrayFire with [homebrew](https://formulae.brew.sh/formula/arrayfire#default):

```bash
brew install arrayfire
```

### Linux Setup

See [Github Action Workflow](/.github/workflows/fmt_test.yml) for example setup on Ubuntu.

### Status

The project is in the incredibly early stages of development and only the core Tensor/Tensor Ops
functionality is currently functional. Autograd is a WIP.

Currently, development is solely focused on OSx/Linux and leveraging ArrayFire for the backend.

### Basic Tensor Usage

```zig
const zt = @import("zigTensor");
const allocator = std.heap.c_allocator; // or your preferred allocator

const DType = zt.tensor.DType;
const deinit = zt.tensor.deinit;
const Tensor = zt.tensor.Tensor;

defer deinit(); // deinit global singletons (e.g. ArrayFire Backend/DeviceManager)

var a = try zt.tensor.rand(allocator, &.{5, 5}, DType.f32);
defer a.deinit();

var b = try zt.tensor.rand(allocator, &.{5, 5}, DType.f32);
defer b.deinit();

var c = try zt.tensor.add(allocator, Tensor, a, Tensor, b); // operator overloading pending (likely utilize comath library)
defer c.deinit();
```

### Roadmap

- [ ] bindings
  - [x] ArrayFire // TODO: optimize to remove unnecessary allocations
- [ ] autograd (WIP)
  - [ ] Functions (WIP)
  - [ ] Utils
  - [x] Variable TODO: optimize/allocate less
- [ ] common (WIP)
- [ ] contrib
- [ ] dataset
- [ ] distributed
- [ ] meter
- [ ] nn
- [ ] optim
- [ ] runtime (WIP)
  - [ ] CUDADevice
  - [ ] CUDAStream
  - [ ] CUDAUtils
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
      - [x] ArrayFireBackend
      - [x] ArrayFireBinaryOps
      - [x] ArrayFireCPUStream
      - [x] ArrayFireReductions
      - [x] ArrayFireShapeAndIndex
      - [x] ArrayFireTensor
      - [x] ArrayFireUnaryOps
      - [x] Utils
    - [ ] JIT
    - [ ] oneDNN
    - [ ] Stub
  - [ ] CUDAProfile
  - [x] Compute
  - [ ] DefaultTensorType (WIP)
  - [x] Index
  - [x] Init
  - [ ] Profile
  - [x] Random
  - [x] Shape
  - [x] TensorAdapter
  - [x] TensorBackend
  - [x] TensorBase
  - [ ] TensorExtension
  - [x] Types

### Backend Roadmap

- [ ] ArrayFire (current focus - WIP)
  - [x] CPU
  - [ ] OpenCL
  - [ ] CUDA
- [ ] JIT
- [ ] oneDNN
- [ ] Stub

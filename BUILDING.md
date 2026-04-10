# Building infer

## Quick start (CPU fallback)

No additional setup required. CPU inference uses the real llama.cpp runtime when
the `llama` feature is enabled:

```sh
cargo build --features llama
cargo test --features llama
```

Without any features the crate compiles to a stub backed by `MockBackend` — useful
for integration tests that do not require a model file.

---

## Vulkan (Windows / Linux — recommended GPU path)

Vulkan is the preferred GPU backend on Windows and Linux. It works across NVIDIA,
AMD, and Intel hardware without requiring vendor-specific toolkits.

### Requirements

| Platform | Requirement |
|----------|-------------|
| Windows  | [LunarG Vulkan SDK](https://vulkan.lunarg.com/sdk/home) installed **and** `VULKAN_SDK` environment variable set to the SDK root |
| Linux    | `libvulkan-dev` (or distro equivalent) installed; `VULKAN_SDK` is optional on most distributions |

> **Windows note:** The `llama-cpp-sys-2` build script checks for `VULKAN_SDK`
> at compile time and **panics with a clear error** if it is absent. Install the
> LunarG SDK and set both `VULKAN_SDK` and the SDK `Bin` directory on `PATH`
> before running `cargo build --features vulkan`.
>
> Additionally, the llama.cpp Vulkan shader sub-project (`vulkan-shaders-gen`)
> has ordering issues with the MSBuild Visual Studio generator. Use the **Ninja**
> generator to avoid the race condition:
>
> ```powershell
> $env:VULKAN_SDK   = "C:\VulkanSDK\<version>"           # e.g. 1.4.341.1
> $env:PATH         = "C:\VulkanSDK\<version>\Bin;$env:PATH"
> $env:CMAKE_GENERATOR = "Ninja"
> cargo build --features vulkan
> ```

### Build

```sh
# Windows (PowerShell)
$env:VULKAN_SDK      = "C:\VulkanSDK\<version>"
$env:PATH            = "C:\VulkanSDK\<version>\Bin;$env:PATH"
$env:CMAKE_GENERATOR = "Ninja"
cargo build --features vulkan

# Linux
cargo build --features vulkan
```

### Runtime detection

`BackendType::auto_detect()` returns `BackendType::Vulkan` when the `vulkan`
feature is compiled in **and** the Vulkan loader is present at runtime:

- **Windows:** `%SystemRoot%\System32\vulkan-1.dll`
- **Linux:** `/usr/lib/libvulkan.so.1` (or equivalent multiarch path)

---

## CUDA (opt-in legacy path)

CUDA is supported as an opt-in legacy path for environments where the CUDA
toolkit is already installed and Vulkan is not available.

```sh
cargo build --features cuda
```

Requires the NVIDIA CUDA toolkit. CUDA is **never** selected automatically when
the `vulkan` feature is also compiled in — Vulkan takes priority.

---

## Metal (macOS only)

```sh
cargo build --features metal
```

No additional setup beyond Xcode command-line tools.

---

## Feature priority at runtime

`BackendType::auto_detect()` selects backends in this order:

1. **Metal** — macOS, `metal` feature compiled in
2. **Vulkan** — Windows/Linux, `vulkan` feature compiled in + Vulkan loader present
3. **CUDA** — Windows/Linux, `cuda` feature compiled in + NVIDIA driver present
4. **CPU** — always available, final fallback

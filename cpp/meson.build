project(
    'golfcar_tracker_cpp',
    'cpp',
    version: '0.1',
    default_options: ['warning_level=2', 'cpp_std=c++20', 'buildtype=release'],
)

py_mod = import('python')
cpp_source_dir = meson.current_source_dir()
uv_python_path = join_paths(
    cpp_source_dir,
    '../python/.venv/bin/python',
)

# Pass the virtualenv Python path directly to find_installation.
py3 = py_mod.find_installation(uv_python_path)

# Print Python installation details for build verification.
message('================ 【Python】 ================')
message('👉 Path: @0@'.format(py3.full_path()))
message(
    '👉 dynamic lib suffix: @0@'.format(
        py3.get_variable('EXT_SUFFIX', 'no suffix found'),
    ),
)
message('=====================================================')

sys_name = host_machine.system()
extra_deps = []
extra_cpp_args = []
repo_root = join_paths(cpp_source_dir, '..')
extra_cpp_args += ['-DGOLFCAR_REPO_ROOT="@0@"'.format(repo_root)]

if sys_name == 'darwin'
    extra_deps += dependency(
        'appleframeworks',
        modules: ['Cocoa', 'CoreVideo', 'CoreMedia', 'AVFoundation'],
    )
    extra_cpp_args += ['-march=native']
endif


opencv_dep = dependency('opencv4')
onnxruntime_dep = dependency('onnxruntime')
pybind11_dep = dependency('pybind11')
yaml_cpp_dep = dependency('yaml-cpp')
bytetrack_dep = dependency('bytetrack')
yolos_cpp_dep = dependency('yolos_cpp')
libtorch_root = get_option('libtorch_root')
if libtorch_root == ''
    error('Set LibTorch path with -Dlibtorch_root=/path/to/libtorch')
endif

libtorch_dep = dependency(
    'Torch',
    method: 'cmake',
    modules: ['torch', 'torch_cuda'],
    cmake_module_path: [libtorch_root / 'share/cmake/Torch'],
    cmake_args: ['-DCMAKE_PREFIX_PATH=' + libtorch_root],
    required: true,
)




all_dependencies = [pybind11_dep, opencv_dep, onnxruntime_dep, bytetrack_dep] + extra_deps

video_processing_core_sources = files('src/video_processing.cpp')

onnx_core_sources = files('src/onnx.cpp')

tracker_sources = files('src/tracker.cpp')

include_dirs = include_directories('includes')
python_backend_install_dir = join_paths(cpp_source_dir, '../python/golfcar_backend')

subdir('python_bindings')
subdir('tests')
subdir('applications')

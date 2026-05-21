project('golfcar_tracker_cpp', 'cpp',
  version : '0.1',
  default_options : [
    'warning_level=2',
    'cpp_std=c++20',
    'buildtype=release'
  ]
)

py_mod = import('python')
uv_python_path = join_paths(meson.current_source_dir(), '../python/.venv/bin/python')

# Pass the virtualenv Python path directly to find_installation.
py3 = py_mod.find_installation(uv_python_path)

# Print Python installation details for build verification.
message('================ 【Python】 ================')
message('👉 Path: @0@'.format(py3.full_path()))
message('👉 dynamic lib suffix: @0@'.format(py3.get_variable('EXT_SUFFIX', 'no suffix found')))
message('=====================================================')

sys_name = host_machine.system()
extra_deps = []
extra_cpp_args = []

if sys_name == 'darwin'
  extra_deps += dependency('appleframeworks', modules : ['Cocoa', 'CoreVideo', 'CoreMedia', 'AVFoundation'])
  extra_cpp_args += ['-march=native']
endif


opencv_dep = dependency('opencv4')
onnxruntime_dep = dependency('onnxruntime')
pybind11_dep = dependency('pybind11')


all_dependencies = [pybind11_dep, opencv_dep, onnxruntime_dep] + extra_deps

core_sources = files(
  'src/video_processing.cpp',
)

binding_sources = files('bindings.cpp', 'src/onnx.cpp') + core_sources

include_dirs = include_directories('includes')


# ==========================================
# 4. Build output and install target.
# ==========================================
py3.extension_module('golf_backend',
  sources: binding_sources,
  include_directories : include_dirs,
  dependencies : all_dependencies,
  cpp_args : extra_cpp_args,
  install : true,
  install_dir : join_paths(meson.current_source_dir(), '../python')
)

test_extract_sources = files('test_extracted_frames.cpp') + core_sources

executable('test_extracted_frames',
           sources : test_extract_sources,
           include_directories : include_dirs,
           dependencies : [opencv_dep],
           cpp_args : extra_cpp_args,
           install : false
)

test_onnx_yolo_sources = files('test_onnx_yolo_inference.cpp', 'src/onnx.cpp') + core_sources

test_onnx_yolo_inference = executable('test_onnx_yolo_inference',
           sources : test_onnx_yolo_sources,
           include_directories : include_dirs,
           dependencies : [opencv_dep, onnxruntime_dep] + extra_deps,
           cpp_args : extra_cpp_args,
           install : false
)

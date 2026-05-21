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
yaml_cpp_dep = dependency('yaml-cpp')
bytetrack_dep = dependency('bytetrack')
yolos_cpp_dep = dependency('yolos_cpp')


all_dependencies = [pybind11_dep, opencv_dep, onnxruntime_dep] + extra_deps

video_processing_core_sources = files(
  'src/video_processing.cpp',
)

onnx_core_sources = files(
  'src/onnx.cpp',
)

video_processing_binding_sources = files(
  'python_bindings/video_processing.cpp',
  'python_bindings/mat_numpy_converter.cpp',
) + video_processing_core_sources

onnx_yolo_binding_sources = files(
  'python_bindings/onnx_yolo.cpp',
  'python_bindings/mat_numpy_converter.cpp',
) + onnx_core_sources

include_dirs = include_directories('includes')


# ==========================================
# 4. Build output and install target.
# ==========================================
py3.extension_module('video_processing',
  sources: video_processing_binding_sources,
  include_directories : include_dirs,
  dependencies : [pybind11_dep, opencv_dep] + extra_deps,
  cpp_args : extra_cpp_args,
  install : true,
  install_dir : join_paths(meson.current_source_dir(), '../python/golfcar_backend')
)

py3.extension_module('onnx_yolo',
  sources: onnx_yolo_binding_sources,
  include_directories : include_dirs,
  dependencies : all_dependencies,
  cpp_args : extra_cpp_args,
  install : true,
  install_dir : join_paths(meson.current_source_dir(), '../python/golfcar_backend')
)

test_extract_sources = files('test_extracted_frames.cpp') + video_processing_core_sources

executable('test_extracted_frames',
           sources : test_extract_sources,
           include_directories : include_dirs,
           dependencies : [opencv_dep],
           cpp_args : extra_cpp_args,
           install : false
)

test_onnx_yolo_sources = files('test_onnx_yolo_inference.cpp') + onnx_core_sources + video_processing_core_sources

test_onnx_yolo_inference = executable('test_onnx_yolo_inference',
           sources : test_onnx_yolo_sources,
           include_directories : include_dirs,
           dependencies : [opencv_dep, onnxruntime_dep] + extra_deps,
           cpp_args : extra_cpp_args,
           install : false
)

movement_track_sources = files('movement_track.cpp') + onnx_core_sources + video_processing_core_sources

movement_track = executable('movement_track',
           sources : movement_track_sources,
           include_directories : include_dirs,
           dependencies : [opencv_dep, onnxruntime_dep, yaml_cpp_dep] + extra_deps,
           cpp_args : extra_cpp_args,
           install : false
)

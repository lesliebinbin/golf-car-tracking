video_processing_binding_sources = files(
    'mat_numpy_converter.cpp',
    'video_processing.cpp',
) + video_processing_core_sources

onnx_yolo_binding_sources = files(
    'mat_numpy_converter.cpp',
    'onnx_yolo.cpp',
) + onnx_core_sources

py3.extension_module(
    'video_processing',
    sources: video_processing_binding_sources,
    include_directories: include_dirs,
    dependencies: [pybind11_dep, opencv_dep] + extra_deps,
    cpp_args: extra_cpp_args,
    install: true,
    install_dir: python_backend_install_dir,
)

py3.extension_module(
    'onnx_yolo',
    sources: onnx_yolo_binding_sources,
    include_directories: include_dirs,
    dependencies: all_dependencies,
    cpp_args: extra_cpp_args,
    install: true,
    install_dir: python_backend_install_dir,
)
test_extract_sources = files('extracted_frames.cpp') + video_processing_core_sources

executable(
    'test_extracted_frames',
    sources: test_extract_sources,
    include_directories: include_dirs,
    dependencies: [opencv_dep],
    cpp_args: extra_cpp_args,
    install: false,
)

test_onnx_yolo_sources = files('onnx_yolo_inference.cpp') + onnx_core_sources + video_processing_core_sources

test_onnx_yolo_inference = executable(
    'test_onnx_yolo_inference',
    sources: test_onnx_yolo_sources,
    include_directories: include_dirs,
    dependencies: [opencv_dep, onnxruntime_dep, bytetrack_dep] + extra_deps,
    cpp_args: extra_cpp_args,
    install: false,
)

smoke_test_yolo_torchscript = executable(
    'smoke_test_yolo_torchscript',
    sources: files('yolo_torchscript.cpp'),
    dependencies: [opencv_dep, libtorch_dep] + extra_deps,
    cpp_args: extra_cpp_args,
    install: false,
)
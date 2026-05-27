detect_with_yolos_cpp = executable(
    'detect_with_yolos_cpp',
    sources: files('detect_with_yolos_cpp.cpp'),
    include_directories: include_dirs,
    dependencies: [opencv_dep, onnxruntime_dep, yolos_cpp_dep] + extra_deps,
    cpp_args: extra_cpp_args,
    install: false,
)

movement_track_sources = files('movement_track.cpp') + onnx_core_sources + video_processing_core_sources + tracker_sources

movement_track = executable(
    'movement_track',
    sources: movement_track_sources,
    include_directories: include_dirs,
    dependencies: [opencv_dep, onnxruntime_dep, yaml_cpp_dep, bytetrack_dep, libtorch_dep] + extra_deps,
    cpp_args: extra_cpp_args,
    install: false,
)
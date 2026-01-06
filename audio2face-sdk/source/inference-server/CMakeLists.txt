add_executable(audio2face-inference-server
    main.cpp
    inference_sessions.cpp
    websocket_server.cpp
)

set_target_properties(audio2face-inference-server PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin;$<TARGET_FILE_DIR:audio2x>"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

target_link_libraries(audio2face-inference-server PRIVATE
    audio2x
    CUDA::cudart
    CXXOPTS::CXXOPTS
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(audio2face-inference-server PRIVATE ws2_32)
endif()

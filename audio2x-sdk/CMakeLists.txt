# Define the shared library target
add_library(audio2x SHARED)

# Set the output name to match the premake configuration
set_target_properties(audio2x PROPERTIES
    OUTPUT_NAME "audio2x"
)

set_target_properties(audio2x PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin
)

# Add source files from the three components
target_sources(audio2x PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../audio2face-sdk/source/audio2face/audio2face.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../audio2x-common/source/audio2x/audio2x.cpp
)

# Add compile definitions
target_compile_definitions(audio2x PRIVATE
    AUDIO2X_SDK_EXPORTS
)

# Platform-specific compile definitions
if(WIN32)
    target_compile_definitions(audio2x PRIVATE
        AUDIO2FACE_SDK_DLL_EXPORTS
    )
endif()

# Link against core libraries and dependencies
target_link_libraries(audio2x
    PRIVATE
        # Core libraries that need to be built first
        audio2face-core
        audio2x-common
)

# Expose include directories from the core libraries to consumers
target_include_directories(audio2x
    INTERFACE
        $<TARGET_PROPERTY:audio2face-core,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:audio2x-common,INTERFACE_INCLUDE_DIRECTORIES>
)

# Platform-specific linker flags
if(UNIX)
    # Those flags prevent symbol clashes with other binaries.
    target_link_options(audio2x PRIVATE
        "LINKER:-Bsymbolic"
        "LINKER:--exclude-libs,ALL"
    )
endif()

# Get header files (excluding internal directories)
file(GLOB AUDIO2FACE_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../audio2face-sdk/include/audio2face/*.h")
file(GLOB AUDIO2X_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../audio2x-common/include/audio2x/*.h")

# Create copy commands for all headers in one custom command
set(COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E echo "Copying headers to ${CMAKE_CURRENT_BINARY_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/include/audio2face"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/include/audio2x"
)

# Add copy commands for audio2face headers
foreach(header ${AUDIO2FACE_HEADERS})
    list(APPEND COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy "${header}" "${CMAKE_CURRENT_BINARY_DIR}/include/audio2face/")
endforeach()

# Add copy commands for audio2x headers
foreach(header ${AUDIO2X_HEADERS})
    list(APPEND COPY_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy "${header}" "${CMAKE_CURRENT_BINARY_DIR}/include/audio2x/")
endforeach()

# Copy VERSION.md to the binary directory
list(APPEND COPY_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/VERSION.md"
        "${CMAKE_CURRENT_BINARY_DIR}/"
)

# Execute all copy commands in one custom command
add_custom_command(TARGET audio2x POST_BUILD ${COPY_COMMANDS})

# Create an alias target for consistency
add_library(audio2x-sdk::audio2x ALIAS audio2x)

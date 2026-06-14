# linux specific dependencies

include("${CMAKE_MODULE_PATH}/dependencies/glad.cmake")

if(SUNSHINE_ENABLE_PYROWAVE)
    # Disable Granite features we don't need — only Vulkan compute core required.
    set(PYROWAVE_STANDALONE OFF CACHE BOOL "" FORCE)
    set(GRANITE_RENDERER OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_FOSSILIZE OFF CACHE BOOL "" FORCE)
    set(GRANITE_VULKAN_SHADER_MANAGER_RUNTIME_COMPILER OFF CACHE BOOL "" FORCE)
    set(GRANITE_TOOLS OFF CACHE BOOL "" FORCE)
    set(GRANITE_PLATFORM "headless" CACHE STRING "" FORCE)

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/Granite/CMakeLists.txt")
        message(FATAL_ERROR
            "SUNSHINE_ENABLE_PYROWAVE=ON but third-party/Granite is missing.\n"
            "Run: git submodule add https://github.com/Themaister/Granite third-party/Granite\n"
            "     git submodule update --init --recursive third-party/Granite")
    endif()
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third-party/pyrowave/CMakeLists.txt")
        message(FATAL_ERROR
            "SUNSHINE_ENABLE_PYROWAVE=ON but third-party/pyrowave is missing.\n"
            "Run: git submodule add <pyrowave-url> third-party/pyrowave")
    endif()

    add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/Granite" EXCLUDE_FROM_ALL)
    add_subdirectory("${CMAKE_SOURCE_DIR}/third-party/pyrowave" EXCLUDE_FROM_ALL)
endif()

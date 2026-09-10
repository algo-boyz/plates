# Compile

```zsh
mkdir build && cd build
cmake ..
cmake -DONNXRUNTIME_ROOT_DIR="~/work/libs/onnxruntime-osx-arm64-1.22" ..
cmake --build .

```
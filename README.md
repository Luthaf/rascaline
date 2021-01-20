# Rascaline

Start by getting a rust compiler: https://www.rust-lang.org/tools/install

Few thing that you can do:

1. build the code `cargo build`
2. run all tests `cargo test`
   - some tests requires the gnu scientific library (gsl) to be installed: `apt install libgsl0-dev` on Debian derivatives; `brew install gsl` on macOS.
3. install the python package `pip install .`
4. install the code in a CMake compatible way:

```bash
mkdir build && cd build
cmake ..
make install
```

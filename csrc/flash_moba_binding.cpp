#include <pybind11/pybind11.h>

void init_flash_moba(pybind11::module &);
void init_flash_topk(pybind11::module &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashMoBA";
    init_flash_moba(m);
    init_flash_topk(m);
}
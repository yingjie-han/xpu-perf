import os
import sys
import pathlib
import importlib.util
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.vector_norm_ops import SoftmaxOp


try:
    _OP_DIR = pathlib.Path(__file__).resolve().parent
    _SYCL_SO = _OP_DIR / "softmax_sycl.so"

    @ProviderRegistry.register_vendor_impl("softmax", "sycl_ext")
    class SyclExtSoftmaxOp(SoftmaxOp):
        _SO_MODULE = None

        def _get_int_option(self, key, env_key, default):
            value = self.args_dict.get(key, os.getenv(env_key, default))
            try:
                return int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid integer option for {key}/{env_key}: {value}"
                ) from e

        def _get_float_option(self, key, env_key, default):
            value = self.args_dict.get(key, os.getenv(env_key, default))
            try:
                return float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid float option for {key}/{env_key}: {value}"
                ) from e

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.arg_type != "default":
                raise ValueError("SyclExtSoftmaxOp only supports arg_type=default")

            self._so_cfg = self._parse_so_config()
            self._sycl_so = self._load_sycl_so()

            if self._so_cfg["verify"]:
                self._verify_so_compute_once()

            self._run_func = self._run_sycl_so_compute
            self._create_tensors_func = self._create_tensors_for_so

        @classmethod
        def _load_sycl_so(cls):
            if cls._SO_MODULE is not None:
                return cls._SO_MODULE

            if not _SYCL_SO.is_file():
                raise FileNotFoundError(
                    f"softmax sycl extension not found: {_SYCL_SO}. "
                    "Please run backends/INTEL/ops/sycl_ext/build.sh first."
                )

            spec = importlib.util.spec_from_file_location("softmax_sycl", str(_SYCL_SO))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._SO_MODULE = module
            return cls._SO_MODULE

        def _parse_so_config(self):
            iterations = self._get_int_option(
                key="sycl_ext_iterations",
                env_key="SYCL_EXT_SOFTMAX_ITERATIONS",
                default=100,
            )
            verify = self._get_int_option(
                key="sycl_ext_verify",
                env_key="SYCL_EXT_SOFTMAX_VERIFY",
                default=0,
            )
            warmup = self._get_int_option(
                key="sycl_ext_warmup",
                env_key="SYCL_EXT_SOFTMAX_WARMUP",
                default=0,
            )
            softmax_scale = self._get_float_option(
                key="sycl_ext_softmax_scale",
                env_key="SYCL_EXT_SOFTMAX_SCALE",
                default=1.0,
            )
            k_block = self._get_int_option(
                key="sycl_ext_k_block",
                env_key="SYCL_EXT_SOFTMAX_K_BLOCK",
                default=256,
            )
            wg_size = self._get_int_option(
                key="sycl_ext_wg_size",
                env_key="SYCL_EXT_SOFTMAX_WG_SIZE",
                default=0,
            )

            smalldim_mode = self.args_dict.get(
                "sycl_ext_smalldim_mode",
                os.getenv("SYCL_EXT_SOFTMAX_SMALLDIM_MODE", "baseline"),
            )
            smalldim_mode = str(smalldim_mode).strip().lower()
            if smalldim_mode not in {"baseline"}:
                raise ValueError(
                    "Invalid sycl_ext_smalldim_mode/SYCL_EXT_SOFTMAX_SMALLDIM_MODE. "
                    "Expected: baseline"
                )

            rows_per_group = self._get_int_option(
                key="sycl_ext_rows_per_group",
                env_key="SYCL_EXT_SOFTMAX_ROWS_PER_GROUP",
                default=1,
            )

            return {
                "iterations": iterations,
                "verify": verify,
                "warmup": warmup,
                "softmax_scale": softmax_scale,
                "k_block": k_block,
                "wg_size": wg_size,
                "smalldim_mode": smalldim_mode,
                "rows_per_group": rows_per_group,
                "so_variant": self.args_dict.get(
                    "sycl_ext_so_variant",
                    os.getenv("SYCL_EXT_SOFTMAX_SO_VARIANT", "so_v0"),
                ),
            }

        def _fill_src_tensor(self, src):
            idx = torch.arange(self.batch_size * self.dim_size, device=src.device, dtype=torch.float32)
            src.copy_(torch.sin(0.001 * idx).reshape(self.batch_size, self.dim_size).to(src.dtype))

        def _create_tensors_for_so(self, instance_num):
            all_tensor_list = self._create_in_out_tensors(
                instance_num,
                create_inputs=True,
                create_outputs=True,
            )
            for tensor_mapping in all_tensor_list:
                self._fill_src_tensor(tensor_mapping["src"])
            return all_tensor_list

        def _run_sycl_so_compute(self, tensor_mapping):
            self._sycl_so.softmax_compute_into(
                tensor_mapping["src"],
                tensor_mapping["dst"],
                float(self._so_cfg["softmax_scale"]),
                int(self._so_cfg["k_block"]),
                int(self._so_cfg["wg_size"]),
                str(self._so_cfg["smalldim_mode"]),
                int(self._so_cfg["rows_per_group"]),
            )
            return None

        def _verify_so_compute_once(self):
            tensor_mapping = self._create_tensors_for_so(1)[0]
            self._run_sycl_so_compute(tensor_mapping)
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()

            src = tensor_mapping["src"]
            got = tensor_mapping["dst"]
            ref = torch.nn.functional.softmax(src * float(self._so_cfg["softmax_scale"]), dim=-1)

            if self.dtype == "float32":
                atol, rtol = 1e-5, 1e-5
            elif self.dtype == "float16":
                atol, rtol = 2e-3, 2e-3
            else:
                atol, rtol = 5e-3, 5e-3

            if not torch.allclose(got, ref, atol=atol, rtol=rtol):
                max_abs = (got - ref).abs().max().item()
                raise RuntimeError(f"softmax_sycl.so verify failed, max_abs={max_abs}")

        def summary(self, latency_us, kernel_mapping={}):
            target_dict = super().summary(
                latency_us,
                [
                    "sycl_ext_softmax_sycl_so",
                ],
            )
            if target_dict:
                target_dict["sycl_ext_iterations"] = self._so_cfg.get("iterations")
                target_dict["sycl_ext_verify"] = self._so_cfg.get("verify")
                target_dict["sycl_ext_warmup"] = self._so_cfg.get("warmup")
                target_dict["sycl_ext_softmax_scale"] = self._so_cfg.get("softmax_scale")
                target_dict["sycl_ext_k_block"] = self._so_cfg.get("k_block")
                target_dict["sycl_ext_wg_size"] = self._so_cfg.get("wg_size")
                target_dict["sycl_ext_smalldim_mode"] = self._so_cfg.get("smalldim_mode")
                target_dict["sycl_ext_rows_per_group"] = self._so_cfg.get("rows_per_group")
                target_dict["sycl_ext_so_variant"] = self._so_cfg.get("so_variant")
            return target_dict

except Exception as e:
    print(f"[SyclExtSoftmaxOp] Failed to register: {e}")
    pass

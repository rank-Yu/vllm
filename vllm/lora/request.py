# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import msgspec


class LoRARequest(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    array_like=True,
):  # type: ignore[call-arg]
    """
    Request for a LoRA adapter.

    lora_int_id must be globally unique for a given adapter.
    This is currently not enforced in vLLM.

    load_inplace: If True, forces reloading the adapter even if one
        with the same lora_int_id already exists in the cache. This replaces
        the existing adapter in-place. If False (default), only loads if the
        adapter is not already loaded.
    """

    lora_name: str
    lora_int_id: int
    lora_path: str = ""
    base_model_name: str | None = msgspec.field(default=None)
    tensorizer_config_dict: dict | None = None
    load_inplace: bool = False
    peft_config: dict | None = None
    lora_tensors: dict | None = None

    def __post_init__(self):
        if self.lora_int_id < 1:
            raise ValueError(f"id must be > 0, got {self.lora_int_id}")

        has_path = bool(self.lora_path and self.lora_path.strip())
        has_peft = self.peft_config is not None
        has_tensors = self.lora_tensors is not None

        if has_peft != has_tensors:
            raise ValueError(
                "peft_config and lora_tensors must be provided together."
            )

        has_tensor_route = has_peft and has_tensors

        if has_path and has_tensor_route:
            raise ValueError(
                "Ambiguous LoRA source: provide either lora_path or "
                "(peft_config + lora_tensors), not both."
            )

        if not has_path and not has_tensor_route:
            raise ValueError(
                "Missing LoRA source: provide lora_path for file-based LoRA, or "
                "provide both peft_config and lora_tensors for in-memory LoRA."
            )

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path

    def __eq__(self, value: object) -> bool:
        """
        Overrides the equality method to compare LoRARequest
        instances based on lora_name. This allows for identification
        and comparison lora adapter across engines.
        """
        return isinstance(value, self.__class__) and self.lora_name == value.lora_name

    def __hash__(self) -> int:
        """
        Overrides the hash method to hash LoRARequest instances
        based on lora_name. This ensures that LoRARequest instances
        can be used in hash-based collections such as sets and dictionaries,
        identified by their names across engines.
        """
        return hash(self.lora_name)

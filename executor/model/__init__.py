def patch_attn_processor(module, attn_processor):
  for name, module in module.named_modules():
    if hasattr(module, "set_attn_processor"):
      module.set_attn_processor(attn_processor)

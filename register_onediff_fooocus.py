from onediff.infer_compiler import register
from onediff.infer_compiler.transform import proxy_class
from onediff.infer_compiler.utils import is_community_version

from pathlib import Path

#ldm_patched = Path(os.path.abspath(__file__)).parents[0] / "ldm_patched"
#register(package_names=[ldm_patched])

import ldm_patched
from ldm_patched.ldm.modules.attention import SpatialTransformer 

import oneflow as flow

class ops:
    class Linear(flow.nn.Linear):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return flow.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(flow.nn.Conv2d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(flow.nn.Conv3d):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(flow.nn.GroupNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return flow.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)


    class LayerNorm(flow.nn.LayerNorm):
        ldm_patched_cast_weights = False
        def reset_parameters(self):
            return None

        def forward_ldm_patched_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return flow.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.ldm_patched_cast_weights:
                return self.forward_ldm_patched_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

class SpatialTransformer1f(proxy_class(ldm_patched.ldm.modules.attention.SpatialTransformer)):

    def forward(self, x, context=None, transformer_options={}):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context] * len(self.transformer_blocks)
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.flatten(2, 3).permute(0, 2, 1)
        # x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = i
            x = block(x, context=context[i], transformer_options=transformer_options)
        if self.use_linear:
            x = self.proj_out(x)
        x = x.permute(0, 2, 1).reshape_as(x_in)
        # x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


torch2of_class_map = {
}

torch2of_class_map.update(
    {
        ldm_patched.ldm.modules.attention.SpatialTransformer: SpatialTransformer1f,
    }
)


register(torch2oflow_class_map=torch2of_class_map)

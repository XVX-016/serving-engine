with open('src/model/forward_gpu.cu', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace INT4 GEMV calls in decode_step_gpu and decode_step_gpu_graphable to check use_legacy_gemv
# 1. decode_step_gpu:
# QKV:
t1 = '''                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H));
                }'''
r1 = '''                } else if (use_legacy_gemv) {
                    cuda::gemv_int4_g128_bias_legacy(lw.qkv_proj_quant.qweight.data(),
                                                    lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                                    lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                                    qkv_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H));
                }'''

# O-proj:
t2 = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H));
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H));
            }'''
r2 = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H));
            } else if (use_legacy_gemv) {
                cuda::gemv_int4_g128_residual_legacy(lw.o_proj_quant.qweight.data(),
                                                    lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                                    buffers.hidden_states.data(), buffers.hidden_states.data(),
                                                    static_cast<int>(H), static_cast<int>(H));
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H));
            }'''

# GateUp:
t3 = '''                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                         lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                         buffers.gate_up_out.data(),
                                         gate_up_dim, static_cast<int>(H));
                }'''
r3 = '''                } else if (use_legacy_gemv) {
                    cuda::gemv_int4_g128_legacy(lw.gate_up_proj_quant.qweight.data(),
                                                lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                                buffers.gate_up_out.data(),
                                                gate_up_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                         lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                         buffers.gate_up_out.data(),
                                         gate_up_dim, static_cast<int>(H));
                }'''

# Down:
t4 = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim));
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim));
            }'''
r4 = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim));
            } else if (use_legacy_gemv) {
                cuda::gemv_int4_g128_residual_legacy(lw.down_proj_quant.qweight.data(),
                                                    lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                                    buffers.hidden_states.data(), buffers.hidden_states.data(),
                                                    static_cast<int>(H), static_cast<int>(inter_dim));
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim));
            }'''

text = text.replace(t1, r1, 1)
text = text.replace(t2, r2, 1)
text = text.replace(t3, r3, 1)
text = text.replace(t4, r4, 1)

# Now in decode_step_gpu_graphable (with stream argument)
t1_g = '''                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H), stream);
                }'''
r1_g = '''                } else if (use_legacy_gemv) {
                    cuda::gemv_int4_g128_bias_legacy(lw.qkv_proj_quant.qweight.data(),
                                                    lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                                    lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                                    qkv_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H), stream);
                }'''

t2_g = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
            }'''
r2_g = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H), stream);
            } else if (use_legacy_gemv) {
                cuda::gemv_int4_g128_residual_legacy(lw.o_proj_quant.qweight.data(),
                                                    lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                                    buffers.hidden_states.data(), buffers.hidden_states.data(),
                                                    static_cast<int>(H), static_cast<int>(H), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
            }'''

t3_g = '''                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                         lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                         buffers.gate_up_out.data(),
                                         gate_up_dim, static_cast<int>(H), stream);
                }'''
r3_g = '''                } else if (use_legacy_gemv) {
                    cuda::gemv_int4_g128_legacy(lw.gate_up_proj_quant.qweight.data(),
                                                lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                                buffers.gate_up_out.data(),
                                                gate_up_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                         lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                         buffers.gate_up_out.data(),
                                         gate_up_dim, static_cast<int>(H), stream);
                }'''

t4_g = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim), stream);
            }'''
r4_g = '''        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim), stream);
            } else if (use_legacy_gemv) {
                cuda::gemv_int4_g128_residual_legacy(lw.down_proj_quant.qweight.data(),
                                                    lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                                    buffers.hidden_states.data(), buffers.hidden_states.data(),
                                                    static_cast<int>(H), static_cast<int>(inter_dim), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.mlp_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim), stream);
            }'''

text = text.replace(t1_g, r1_g, 1)
text = text.replace(t2_g, r2_g, 1)
text = text.replace(t3_g, r3_g, 1)
text = text.replace(t4_g, r4_g, 1)

with open('src/model/forward_gpu.cu', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated forward_gpu.cu to support use_legacy_gemv for INT4.')

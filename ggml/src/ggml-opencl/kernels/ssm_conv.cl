kernel void kernel_ssm_conv_f32_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2,
    int           apply_silu
){
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc  = ne10;

    global float * s = (global float *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float * c = (global float *) (src1 + ir*nb11);
    global float * d = (global float *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    if (apply_silu) {
        sumf = sumf / (1.0f + exp(-sumf));
    }

    d[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2,
    int           apply_silu
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc = ne10;

    global float4 * s = (global float4 *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float4 * c = (global float4 *) (src1 + ir*nb11);
    global float  * d = (global float  *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    if (apply_silu) {
        sumf = sumf / (1.0f + exp(-sumf));
    }

    d[0] = sumf;
}

kernel void kernel_ssm_conv_split_f32_f32_4(
    global char * state,
    ulong         offset_state,
    global char * input,
    ulong         offset_input,
    global char * weights,
    ulong         offset_weights,
    global char * dst,
    ulong         offset_dst,
    ulong         state_nb0,
    ulong         state_nb1,
    ulong         state_nb2,
    ulong         input_nb0,
    ulong         input_nb1,
    ulong         input_nb2,
    int           state_tokens,
    ulong         weights_nb1,
    ulong         dst_nb0,
    ulong         dst_nb1,
    ulong         dst_nb2,
    int           apply_silu
) {
    state   += offset_state;
    input   += offset_input;
    weights += offset_weights;
    dst     += offset_dst;

    const int channel = get_global_id(0);
    const int token   = get_global_id(1);
    const int seq     = get_global_id(2);

    global float4 * c = (global float4 *) (weights + channel * weights_nb1);
    global float  * d = (global float  *) (dst + channel * dst_nb0 + token * dst_nb1 + seq * dst_nb2);

    float4 samples;
    if (token >= state_tokens) {
        global float * x = (global float *) (
            input + channel * input_nb1 + (token - state_tokens) * input_nb0 + seq * input_nb2);
        samples = (float4)(x[0], x[input_nb0 / sizeof(float)], x[2 * input_nb0 / sizeof(float)], x[3 * input_nb0 / sizeof(float)]);
    } else {
        float values[4];
        for (int tap = 0; tap < 4; ++tap) {
            const int pos = token + tap;
            if (pos < state_tokens) {
                global float * x = (global float *) (
                    state + channel * state_nb1 + pos * state_nb0 + seq * state_nb2);
                values[tap] = x[0];
            } else {
                global float * x = (global float *) (
                    input + channel * input_nb1 + (pos - state_tokens) * input_nb0 + seq * input_nb2);
                values[tap] = x[0];
            }
        }
        samples = (float4)(values[0], values[1], values[2], values[3]);
    }

    // Keep the same accumulation form as kernel_ssm_conv_f32_f32_4. This
    // avoids changing contraction/rounding decisions in recurrent decoding.
    float sumf = 0.0f;
    sumf += dot(samples, c[0]);
    if (apply_silu) {
        sumf = sumf / (1.0f + exp(-sumf));
    }

    d[0] = sumf;
}

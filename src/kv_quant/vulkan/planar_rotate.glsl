#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Vectors {
    float data[];
} vectors;

layout(set = 0, binding = 1) readonly buffer Params {
    float params[];
} rotation_params;

layout(push_constant) uniform PushConstants {
    uint num_vectors;
    uint dim;
} pc;

void main() {
    uint vec_idx = gl_GlobalInvocationID.x;
    if (vec_idx >= pc.num_vectors) {
        return;
    }

    uint base = vec_idx * pc.dim;
    uint num_pairs = pc.dim / 2;

    for (uint i = 0; i < num_pairs; i++) {
        float cos_t = rotation_params.params[i * 2];
        float sin_t = rotation_params.params[i * 2 + 1];

        uint idx = base + i * 2;
        float x = vectors.data[idx];
        float y = vectors.data[idx + 1];

        vectors.data[idx] = x * cos_t + y * sin_t;
        vectors.data[idx + 1] = -x * sin_t + y * cos_t;
    }
}

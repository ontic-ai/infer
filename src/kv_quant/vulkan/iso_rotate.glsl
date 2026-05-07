#version 450

layout(local_size_x = 32) in;

layout(set = 0, binding = 0) buffer Vectors {
    float data[];
} vectors;

layout(set = 0, binding = 1) readonly buffer Params {
    float params[];
} quat_params;

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
    uint num_quats = pc.dim / 4;

    for (uint i = 0; i < num_quats; i++) {
        float qw = quat_params.params[i * 4];
        float qx = quat_params.params[i * 4 + 1];
        float qy = quat_params.params[i * 4 + 2];
        float qz = quat_params.params[i * 4 + 3];

        uint idx = base + i * 4;
        float a = vectors.data[idx];
        float b = vectors.data[idx + 1];
        float c = vectors.data[idx + 2];
        float d = vectors.data[idx + 3];

        vectors.data[idx] = qw * a - qx * b - qy * c - qz * d;
        vectors.data[idx + 1] = qx * a + qw * b - qz * c + qy * d;
        vectors.data[idx + 2] = qy * a + qz * b + qw * c - qx * d;
        vectors.data[idx + 3] = qz * a - qy * b + qx * c + qw * d;
    }
}

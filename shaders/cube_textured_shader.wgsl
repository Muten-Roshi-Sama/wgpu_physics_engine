struct RenderParams {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

struct LightStruct {
    light: vec4<f32>,
}

// Uses utils/OrbitCamera logic, reserve group(0)  
@group(0) @binding(0) var<uniform> params: RenderParams;
// Texture + sampler
@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var diffuse_samp: sampler;
// light 
@group(2) @binding(0) var<uniform> light_uni : LightStruct;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip = params.proj * params.view * vec4<f32>(in.position, 1.0);
    out.position = in.position;
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(light_uni.light.xyz - in.position);
    let shading = clamp(dot(light_dir, normalize(in.normal)), 0.2, 1.0);
    let color = textureSample(diffuse_tex, diffuse_samp, in.uv);
    return vec4<f32>(color.xyz * shading*2.0, 1.0);
}
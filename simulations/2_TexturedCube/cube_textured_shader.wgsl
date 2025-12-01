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
    // ==============================================================================
    // Let an object(/Model) with coordinates p_model = (x_o, y_o, z_o, t_o=1.0)
    // World coordinates :  p_world = (x_w, ...)   =  p_model * Model
    // Camera space      :  p_view  = (x_cam, ..)  =  p_world * View Matrix (view)
    // clip-space        :  p_clip  = (x_clip, ..) =  p_view  * Projection matrix (proj)
    // -----
    // Here Model is the identity matrix (no per-object transform yet)
    out.clip = params.proj * params.view * vec4<f32>(in.position, 1.0);  //Clip = Proj * View * Model * (x_o, y_o, ...)
    out.position = in.position;       // original Object position (used for light)
    out.normal = in.normal;          // vertex normal unit-vector (not yet normalized)
    out.uv = in.uv;                 // texture coordinates to sample the given texture image
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    /*
        Apply diffuse lighting model with color clamping (Math.floor):
            - Intensity = (L * N) * C_d
            where :
                - L the direction vector from light-source to object.
                - N the normal vector to the surface.
                - C_d the (r, g, b) colors.
    */

    // Vectors
    let light_dir    = normalize(light_uni.light.xyz - in.position);            // 
    let N: vec3<f32> = normalize(in.normal);
    

    // Clamping parameter
    let ambient_min : f32 = 0.1;
    let ambient_max : f32 = 1.0;
    let luminosity: f32 = 2.4;  // shading multiplier

    // Intensity (shading)
    let shading = clamp(dot(light_dir, N), ambient_min, ambient_max);
    
    // Sample base color from sample
    let color = textureSample(diffuse_tex, diffuse_samp, in.uv);

    // return the diffuse intensity reflected.
    return vec4<f32>(color.xyz * shading * luminosity, 1.0);
}
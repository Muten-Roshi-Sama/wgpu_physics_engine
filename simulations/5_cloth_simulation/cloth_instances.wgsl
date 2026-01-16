// instances_shader.wgsl

struct RenderParams {
    // Buffer for Camera matrices 
    // group(0) binding(0)
    // ...
    view: mat4x4<f32>,    // View (Model pace -> View/Camera space)
    proj: mat4x4<f32>,   // Projection matrix (view-space -> clip-space)
};

// struct LightStruct {
//     light: vec4<f32>,          // light pos.xyz, pad
//     ks_shininess: vec2<f32>,  // [scalar specular strength, α-shininess exponent]
//     _pad: u32,               // padding to 16-byte alignment
//     compute_specular: u32,  // whether to use specular component
// }

// GROUP(0) : utils/OrbitCamera logic, reserve group(0)  
@group(0) @binding(0) var<uniform> params: RenderParams;
// GROUP(1) :  Texture + sampler
@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var diffuse_samp: sampler;
// GROUP(2) : light 
// @group(2) @binding(0) var<uniform> light_uni : LightStruct;



struct VertexInput {
    // attributes of each globe-instance (main globe and cloth particles)
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct Particles {
    // attributes of each moving-instance (modelMatrix, speed,...)
    @location(3) c0: vec4<f32>,
    @location(4) c1: vec4<f32>,
    @location(5) c2: vec4<f32>,
    @location(6) c3: vec4<f32>,
    // position
    // velocity
    // optional: accelleration
};


struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,  //TODO : make sure to normalize in FS into unit vector
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, particle: Particles) -> VertexOutput {
    /* ==============================================================================
        Let an object(/Model) with coordinates p_model = (x_o, y_o, z_o, t_o=1.0)
         World coordinates :  p_world = (x_w, ...)   =  p_model * Model
         Camera space      :  p_view  = (x_cam, ..)  =  p_world * View Matrix (view)
         clip-space        :  p_clip  = (x_clip, ..) =  p_view  * Projection matrix (proj)
    */ 
    var out: VertexOutput;
    let model = mat4x4<f32>(particle.c0, particle.c1, particle.c2, particle.c3);
    let rot = mat3x3<f32>(particle.c0.xyz, particle.c1.xyz, particle.c2.xyz);

    let world_pos = model * vec4<f32>(in.position, 1.0);  // local to world


    out.clip = params.proj * params.view * world_pos;
    out.position = world_pos.xyz / world_pos.w;
    out.normal = rot * in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    
    // let sampledColor = textureSample(diffuse_tex, diffuse_samp, in.uv);
    // return vec4<f32>(sampledColor.xyz, 1.0);
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);

}





// instances_shader.wgsl

struct RenderParams {
    view: mat4x4<f32>,    // View (Model pace -> View/Camera space)
    proj: mat4x4<f32>,   // Projection matrix (view-space -> clip-space)
};

struct LightStruct {
    light: vec4<f32>,          // light pos.xyz, pad
    ks_shininess: vec2<f32>,  // [scalar specular strength, α-shininess exponent]
    _pad: u32,               // padding to 16-byte alignment
    compute_specular: u32,  // whether to use specular component
}

// GROUP(0) : utils/OrbitCamera logic, reserve group(0)  
@group(0) @binding(0) var<uniform> params: RenderParams;
// GROUP(1) :  Texture + sampler
@group(1) @binding(0) var diffuse_tex: texture_2d<f32>;
@group(1) @binding(1) var diffuse_samp: sampler;
// GROUP(2) : light 
@group(2) @binding(0) var<uniform> light_uni : LightStruct;



struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct InstanceInput {
    @location(3) c0: vec4<f32>,
    @location(4) c1: vec4<f32>,
    @location(5) c2: vec4<f32>,
    @location(6) c3: vec4<f32>,
};


struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,  //TODO : make sure to normalize in FS into unit vector
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, instance: InstanceInput) -> VertexOutput {
    /* ==============================================================================
        Let an object(/Model) with coordinates p_model = (x_o, y_o, z_o, t_o=1.0)
         World coordinates :  p_world = (x_w, ...)   =  p_model * Model
         Camera space      :  p_view  = (x_cam, ..)  =  p_world * View Matrix (view)
         clip-space        :  p_clip  = (x_clip, ..) =  p_view  * Projection matrix (proj)
    */ 
    var out: VertexOutput;
    let model = mat4x4<f32>(instance.c0, instance.c1, instance.c2, instance.c3);
    let rot = mat3x3<f32>(instance.c0.xyz, instance.c1.xyz, instance.c2.xyz);

    let world_pos = model * vec4<f32>(in.position, 1.0);  // local to world


    out.clip = params.proj * params.view * world_pos;
    out.position = world_pos.xyz / world_pos.w;
    out.normal = rot * in.normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    

    // ========= Vectors ============
    let light_pos_view: vec3<f32> = (params.view * vec4<f32>(light_uni.light.xyz, 1.0)).xyz; // Light position in view space
    let l: vec3<f32> = normalize(light_pos_view - in.position);
    let n: vec3<f32> = normalize(in.normal);  // N: the normal vector to the surface.
    let v: vec3<f32> = normalize(-in.position);


    // --- Diffuse light model
    let ambient : f32 = 0.1;
    let luminosity: f32 = 2.4;
    let shading: f32 = clamp(dot(n, l), ambient, 1.0);
    let color =  textureSample(diffuse_tex, diffuse_samp, in.uv);
    let diffuse = color.xyz * shading * luminosity;

    // --- Early return if no specular component
    if (light_uni.compute_specular == 0u) {
        return vec4<f32>(diffuse, 1.0);
    }


    // --- Specular Light
        // Reflection (incident = -L)
    let r: vec3<f32> = normalize(reflect(-l, n));      // or 2.0 * dot(n, l) * n - l);   // -L is the incident vector !
    let alpha: f32 = light_uni.ks_shininess.y;        // shininess exponent
    let ks: f32 = light_uni.ks_shininess.x;          // spec strength
    let r_dot_v: f32 = max(dot(r, v), 0.0);         // spec angle (see graph)
    let light_color =  vec3<f32>(1.0, 1.0, 1.0);   // Reflected light color
    let spec: vec3<f32> = ks * pow(r_dot_v, alpha) * light_color;    // I_s = ks * (R·V)^α * C_L

    // 3. Combine Specular + diffuse
    let result: vec3<f32> = spec + diffuse;

    return vec4<f32>(result, 1.0);
}





// globe_shader.wgsl

struct RenderParams {
    view: mat4x4<f32>,    // View (Model pace -> View/Camera space)
    proj: mat4x4<f32>,   // Projection matrix (view-space -> clip-space)
};

struct LightStruct {
    light: vec4<f32>,       // light pos.xyz, pad
    ks: f32,               // scalar specular strength
    shininess: f32,        // α-shininess exponent
    _pad: vec2<f32>,        // padding to 16-byte alignment
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
struct VertexOutput {
    @builtin(position) clip: vec4<f32>,
    @location(0) position_view: vec3<f32>,
    @location(1) normal_view: vec3<f32>,  //TODO : make sure to normalize in FS into unit vector
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    /* ==============================================================================
        (see simulations/2_Textured_Cube/cube_textured_shader.wgsl)
            ...
        --> p_clip = proj * view * model * vec4(p_model, 1.0)

    */
    //Transform position into view space
    let view_space = params.view * vec4<f32>(in.position, 1.0);   // p_world * View Matrix (view), note that Model Matrix is Id
    
    // Clip space
    out.clip = params.proj * view_space;

    // Model Space
    // out.position = in.position;   // compute diffuse light in View space now
    // out.normal = in.normal;

    // View Space
    out.position_view = view_space.xyz;
    out.normal_view = (params.view * vec4<f32>(in.normal, 0.0)).xyz;

    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    /*
        Specular Lighting : The specular lighting corresponds to the reflection of the light source on the surface. It depends on the point of view.
            - I_s = (R * V)^α * C_L
            where :
                - R the reflection vector of the incident light about the surface normal.
                - V the view direction (from surface toward camera), normalized.
                - α the brightness of the material.
                - C_L the color/intensity of the light.

            - R = [2 * (N * L) * N] - L
            where :
                - N the normal vector to the surface.
                - L the vector directed towards the light, normalized.

    */

    // ========= Vectors ============
    let n: vec3<f32> = normalize(in.normal_view);  // N: the normal vector to the surface.
    let light_pos_view: vec3<f32> = (params.view * vec4<f32>(light_uni.light.xyz, 1.0)).xyz; // Light position in view space
    let l: vec3<f32> = normalize(light_pos_view - in.position_view);
    let v: vec3<f32> = normalize(-in.position_view);

    // Reflection (incident = -L)
    let r: vec3<f32> = normalize(reflect(-l, n)); // or 2.0 * dot(n, l) * n - l);   // -L is the incident vector !


    // --- Specular Light
    let alpha: f32 = light_uni.shininess;             // shininess exponent
    let ks: f32 = light_uni.ks;                      // spec strength
    let r_dot_v: f32 = max(dot(r, v), 0.0);         // spec angle (see graph)
    let light_color =  vec3<f32>(1.0, 1.0, 1.0);   // Reflected light color
    let spec: vec3<f32> = ks * pow(r_dot_v, alpha) * light_color;    // I_s = ks * (R·V)^α * C_L

    // --- Diffuse light model
    let ambient : f32 = 0.1;
    let luminosity: f32 = 2.4;
    let shading: f32 = clamp(dot(n, l), ambient, 1.0);
    let color =  textureSample(diffuse_tex, diffuse_samp, in.uv);
    let diffuse = color.xyz * shading * luminosity;

    // 3. Combine Specular + diffuse
    let result: vec3<f32> = spec + diffuse;

    return vec4<f32>(result, 1.0);
}
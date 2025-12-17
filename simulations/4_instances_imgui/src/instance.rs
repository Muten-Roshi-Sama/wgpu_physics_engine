// instance.rs
use std::path::Path;


use wgpu_bootstrap::{
    cgmath, egui,
    util::orbit_camera::{CameraUniform, OrbitCamera},
    wgpu::{self, util::DeviceExt},
    App, Context,
};

// =========== CONFIGURATIONS =============

const SHADER_FILE: &str = "instances_shader.wgsl";
const COMPUTE_SHADER_FILE: &str = "compute_movement.wgsl";

const TEXTURE_FILE: &str = "../../textures/grey.png";
// const TEXTURE_FILE: &str = "../../textures/texture.png";
// const TEXTURE_FILE: &str = "../../textures/earth2048.bmp";
// const TEXTURE_FILE: &str = "../../textures/moon1024.bmp";


// Specular light parameters
const LIGHT_POS: [f32; 4] = [2.0, 2.0, 2.0, 0.0];
const KS: f32 = 0.15;
const SHININESS: f32 = 128.0;
const _PAD: u32 = 0u32;

// Globe geometry
const RADIUS: f32 = 1.0;
const STACK_COUNT: usize = 64;
const SECTOR_COUNT: usize = 128;

// Physics
const NUM_PARTICLES: u32 = 5;
const PARTICLE_SCALE : f32 = 0.1;
const TIME_SCALE: f32 = 1.0;
const GRAVITY: [f32; 3] = [0.0, -9.81, 0.0];
const BOUNDS: f32 = 30.0;
const DAMPING: f32 = 0.8;


// =========== STRUCTS & IMPL ============
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2]
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    light: [f32; 4],          // maps to vec4<f32> in WGSL
    ks_shininess: [f32; 2],  // specular strength & shininess exponent
    _pad: u32,              // padding to 16-byte alignment
    compute_specular: u32, // whether to use specular component
}


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    model_matrix: [f32; 16],  // 4x4 matrix
    velocity: [f32; 4],       // velocity vector (x, y, z, w)
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SimulationUniform {
    dt: f32,
    gravity: [f32; 3],
    bounds: f32,
    damping: f32,
    _padding: [f32; 2],
}




impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // location 0: position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 1: normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // location 2: uv
                wgpu::VertexAttribute {
                    offset: (std::mem::size_of::<[f32; 3]>() * 2) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}





// ========== APP ==============
pub struct ParticleSimApp {
    // Rendering
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    num_indices: u32,

    // Bind gorups
    camera: OrbitCamera,
    texture_bind_group: wgpu::BindGroup,
    light_bind_group: wgpu::BindGroup,
    light_buffer: wgpu::Buffer,
    fps: f32,

    // Instancing
    instance_buffer: wgpu::Buffer,
    instance_count: u32,

    // Computing
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    sim_uniform_buffer: wgpu::Buffer,

    // Eframe
    light_pos: [f32; 3],
    checkbox_specular: bool,   // ui checkbox
    ks: f32,
    shininess: f32,

    // Physics
    time_scale:f32,
    gravity: [f32;3],
    bounds: f32,
    damping: f32,


    // geometry
    stack_count: usize,   // TODO: for now, display only....
    sector_count: usize,  // TODO: ... regenerating requires rebuilding buffers
}

impl ParticleSimApp {
    pub fn new(context: &Context) -> Self {

        // 1. Generate geometry
        let (vertices, indices, num_indices) = Self::create_sphere_geometry();


        // 2. gpu buff
        let vertex_buffer = Self::create_vertex_buffer(context, &vertices);
        let index_buffer = Self::create_index_buffer(context, &indices);

        // 2.x Instance buffer
        let instance_count: u32 = 5;
        let instances = Self::generate_instances(instance_count);
        let instance_buffer = Self::create_instance_buffer(context, &instances);  // then called in render pipeline !


        // 3. Bind Groups lyouts   
            /* 
            - GROUP(0) Camera
            - GROUP(1) Texture & Sampler
            - GROUP(2) Light Layout
            */ 
        let camera_bind_group_layout = context.device().create_bind_group_layout(&CameraUniform::desc());

        let texture_bind_group_layout = Self::create_texture_bind_group_layout(context);


        // 4. Light Bind Group, Buffer and Uniform
        let light_bind_group_layout = Self::create_light_bind_group_layout(context);
        
        let (light_buffer, light_bind_group) = Self::create_light_resources(
            context,
            &light_bind_group_layout,
        );

        // 5. Textures
        let texture_bind_group = Self::load_texture_and_create_bind_group(
            context,
            &texture_bind_group_layout,
            TEXTURE_FILE,
        );

        // 6. Render Pipeline
        let render_pipeline = Self::create_render_pipeline(
            context,
            &camera_bind_group_layout,
            &texture_bind_group_layout,
            &light_bind_group_layout,
        );

        // 7. Camera
        let camera = Self::setup_camera(context);

        // 8. Compute Setup
        let compute_bind_group_layout = Self::create_compute_bind_group_layout(context);
        let (sim_uniform_buffer, compute_bind_group) = Self::create_compute_resources(context, &compute_bind_group_layout, &instance_buffer);
        let compute_pipeline = Self::create_compute_pipeline(context, &compute_bind_group_layout);



        Self {
            // Rendering
            vertex_buffer,
            index_buffer,
            render_pipeline,
            num_indices,

            // Bind gorups
            camera,
            texture_bind_group,
            light_bind_group,
            light_buffer,
            fps: 0.0,

            // Instancing
            instance_buffer,
            instance_count,
            // Compute movement for instances
            compute_pipeline,
            compute_bind_group,
            sim_uniform_buffer,

            // Eframe
            light_pos: [LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2]],
            checkbox_specular:false,
            ks: KS,
            shininess: SHININESS,

            // Physics
            time_scale: TIME_SCALE,
            gravity: GRAVITY,
            bounds: BOUNDS,
            damping: DAMPING,

            // geometry
            stack_count: STACK_COUNT,
            sector_count: SECTOR_COUNT,
        }

    // --- end of new() ---
    }



    // =============================================
    //           Helpers
    // =============================================

    // 1. Geometry
    fn create_sphere_geometry() -> (Vec<Vertex>, Vec<u32>, u32) {
        let (raw_vertices, indices) =
            crate::sphere_vertices::generate_uv_sphere(RADIUS, STACK_COUNT, SECTOR_COUNT);
        
        let vertices: Vec<Vertex> = raw_vertices
            .into_iter()
            .map(|(pos, normal, uv)| Vertex { position: pos, normal, uv })
            .collect();
        
        let num_indices = indices.len() as u32;
        (vertices, indices, num_indices)
    }

    // 2. Buffers
    fn create_vertex_buffer(context: &Context, vertices: &[Vertex]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        })
    }
    
    fn create_index_buffer(context: &Context, indices: &[u32]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        })
    }

    // 2.x Instance buffer
    fn generate_instances(count: u32) -> Vec<Particle> {
        /* For now only create a simple impl of instances :
                - same speed, rot,..
                - spaced out along x axis
         */
        use cgmath::{Matrix4, Vector3, Deg};
        // use rand::Rng;
        // let mut rng = rand::rng();
        // let mut out = Vec::with_capacity(count as usize);

        let actual_radius = RADIUS * PARTICLE_SCALE;
        let diameter = 2.0 * actual_radius;
        let gap = 0.5 * diameter;
        let spacing = diameter + gap;
        let start_x = -0.5 * (count as f32 - 1.0) * spacing;
        let mut out = Vec::with_capacity(count as usize);

        for i in 0..count {
        let x = start_x + i as f32 * spacing;
        let y = actual_radius; // posé au-dessus du plan Y=0 si tu en as un
        let z = 0.0;

        let trans = Matrix4::from_translation(Vector3::new(x, y, z));
        let scale = Matrix4::from_scale(PARTICLE_SCALE);
        let model = trans * scale;

        let c0 = model.x;
        let c1 = model.y;
        let c2 = model.z;
        let c3 = model.w;

        out.push(Particle {
            model_matrix: [
                c0.x, c0.y, c0.z, c0.w,
                c1.x, c1.y, c1.z, c1.w,
                c2.x, c2.y, c2.z, c2.w,
                c3.x, c3.y, c3.z, c3.w,
            ],
            velocity: [0.0, 0.0, 0.0, 0.0], // pas de mouvement
        });
        }

        out
    }

    fn create_instance_buffer(context: &Context, instances: &[Particle]) -> wgpu::Buffer {
        context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(instances),
            usage: wgpu::BufferUsages::VERTEX 
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        })
    }

    fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Particle>() as wgpu::BufferAddress, // 64
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { offset: 0,  shader_location: 3, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 16, shader_location: 4, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 32, shader_location: 5, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 48, shader_location: 6, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }



    // 3. Bind Groups
    fn create_texture_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_bind_group_layout"),
            entries: &[
                // binding 0: texture view
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // binding 1: sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }


    // 4. Light Bind Group, Buffer and Uniform
    fn create_light_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    fn create_light_resources(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        ) -> (wgpu::Buffer, wgpu::BindGroup) {

        // Light init params
        let initial_light = LightUniform {
            light: LIGHT_POS,
            ks_shininess: [KS, SHININESS],
            _pad: _PAD,
            compute_specular: 0u32, // ? compute specular in shader ?
        };

        let light_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light Buffer"),
            contents: bytemuck::bytes_of(&initial_light),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group = context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("light_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        (light_buffer, light_bind_group)
    }


    // 5. Textures
    fn load_texture_and_create_bind_group(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        texture_path: &str,
        ) -> wgpu::BindGroup {
        
        // Load image
        let img_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(texture_path);
        let img = image::open(&img_path)
            .expect("failed to load texture")
            .to_rgba8();
        
        let (width, height) = img.dimensions();
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        
        let texture = context.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("diffuse_texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        context.queue().write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            texture_size,
        );
        
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = context.device().create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("texture_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&texture_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        })
    }

    // 6. Render Pipeline
    fn create_render_pipeline(
        context: &Context,
        camera_layout: &wgpu::BindGroupLayout,
        texture_layout: &wgpu::BindGroupLayout,
        light_layout: &wgpu::BindGroupLayout,
        ) -> wgpu::RenderPipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(SHADER_FILE);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("failed to read shader file");
        
        let shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("instances_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[camera_layout, texture_layout, light_layout],
            push_constant_ranges: &[],
        });
        
        context.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), Self::instance_buffer_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: context.depth_stencil_format(),
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
    }
    
    // 7. Camera
    fn setup_camera(context: &Context) -> OrbitCamera {
        let mut camera = OrbitCamera::new(
            context,
            45.0,
            context.size().x / context.size().y,
            0.1,
            100.0,
        );
        camera
            .set_target(cgmath::point3(0.0, 0.0, 0.0))
            .set_polar(cgmath::point3(2.0, 0.0, 0.0))
            .update(context);
        camera
    }



    // 8. Compute pipeline
    fn create_compute_bind_group_layout(context: &Context) -> wgpu::BindGroupLayout {
        context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compute_bind_group_layout"),
            entries: &[
                // @binding(0): SimulationData uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): Particle storage buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
    
    fn create_compute_resources(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        instance_buffer: &wgpu::Buffer,
        ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let sim_uniform = SimulationUniform {
            dt: TIME_SCALE,
            gravity: GRAVITY,
            bounds: BOUNDS,
            damping: DAMPING,
            _padding: [0.0, 0.0],
        };
    
        let sim_uniform_buffer = context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Simulation Uniform Buffer"),
                contents: bytemuck::bytes_of(&sim_uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
    
        let compute_bind_group = context.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });
    
        (sim_uniform_buffer, compute_bind_group)
    }
    
    fn create_compute_pipeline(
        context: &Context,
        layout: &wgpu::BindGroupLayout,
        ) -> wgpu::ComputePipeline {
        let shader_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(COMPUTE_SHADER_FILE);
        let shader_src = std::fs::read_to_string(&shader_path)
            .expect("Failed to read compute shader");
        
        let compute_shader = context.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
    
        let pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[layout],
            push_constant_ranges: &[],
        });
    
        context.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    }

    fn dispatch_compute(&self, context: &Context) {
        let mut encoder = context.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            }
        );
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            
            let workgroup_count = (self.instance_count + 63) / 64;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        
        context.queue().submit(Some(encoder.finish()));
    }

    // ============== Update =====================
    fn update_light_uniform(&self, context: &Context) {
        // check if need to update light uniform
        let compute_specular = if self.checkbox_specular { 1u32 } else { 0u32 };
        let updated_light = LightUniform {
            light: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
            ks_shininess: [self.ks, self.shininess],
            _pad: _PAD,
            compute_specular,
        };
        context.queue().write_buffer(
            &self.light_buffer,
            0,
            bytemuck::bytes_of(&updated_light),
        );
    }

}



impl App for ParticleSimApp {
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Set pipeline
        render_pass.set_pipeline(&self.render_pipeline);

        // Set buffers
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        //Bind Groups
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(2, &self.light_bind_group, &[]);

        // Draw
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.instance_count);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        egui::Window::new("Params").show(egui_ctx, |ui| {

            // Radius slider
            ui.heading("Camera");
            let mut radius = self.camera.radius();
            if ui.add(egui::Slider::new(&mut radius, 10.0..=2.0).text("Zoom")).changed() {  // ← swapped range
                self.camera.set_radius(radius).update(context);
            }

            // Light controls + SPECULAR
            // ui.heading("Light");
            // let mut light_changed = false;

            // light_changed |= ui.checkbox(&mut self.checkbox_specular, "Specular").changed();
            
            // // Light position + shininness
            // light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[0], -5.0..=5.0).text("Light X")).changed();
            // light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[1], -5.0..=5.0).text("Light Y")).changed();
            // light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[2], -5.0..=5.0).text("Light Z")).changed();
            // light_changed |= ui.add(egui::Slider::new(&mut self.ks, 0.0..=2.0).text("Specular (ks)")).changed();
            // light_changed |= ui.add(egui::Slider::new(&mut self.shininess, 1.0..=512.0).text("Shininess")).changed();

            // if light_changed {
            //     self.update_light_uniform(context);
            // }

            ui.add_space(5.0);
            
            
            ui.separator();

            // Physics
            ui.heading("Physics");
            ui.add(egui::Slider::new(&mut self.gravity[1], -20.0..=1.0).text("Gravity Y"));
            ui.add(egui::Slider::new(&mut self.time_scale, 0.0..=2.0).text("Time Scale"));
            ui.add(egui::Slider::new(&mut self.bounds, 1.0..=20.0).text("Bounds"));
            ui.add(egui::Slider::new(&mut self.damping, -1.0..=0.0).text("Damping"));
        


            // Geometry info (read-only for now)
            ui.heading("Geometry");
            ui.label(format!("Stacks: {}", self.stack_count));
            ui.label(format!("Sectors: {}", self.sector_count));
            ui.label(format!("Vertices: {}", (self.stack_count + 1) * (self.sector_count + 1)));
            
            ui.separator();
            


            // FPS
            ui.label(format!("FPS: {}", self.fps.round()));
            // Other

            // Debug
            ui.separator();
            ui.heading("Debug");
            ui.label(format!("Instance count: {}", self.instance_count));
            ui.label(format!("Bounds: {:.2}", self.bounds));



        });
    }


    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input, context);
    }

    fn update(&mut self, delta_time: f32, context: &Context) {
        self.fps = 1.0 / delta_time;


        // let sim_uniform = SimulationUniform {
        //     dt: self.time_scale * delta_time,
        //     gravity: self.gravity,
        //     bounds: self.bounds,
        //     damping: self.damping,
        //     _padding: [0.0, 0.0],
        // };


        // context.queue().write_buffer(
        //     &self.sim_uniform_buffer,
        //     0,
        //     bytemuck::bytes_of(&sim_uniform),
        // );

        // self.dispatch_compute(context);

    }

    fn resize(&mut self, new_width: u32, new_height: u32, context: &Context) {
        self.camera
            .set_aspect(new_width as f32 / new_height as f32)
            .update(context);
    }

}
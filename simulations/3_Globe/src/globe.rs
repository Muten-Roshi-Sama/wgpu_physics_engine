// render_cube_textured.rs
use std::path::Path;


use wgpu_bootstrap::{
    cgmath, egui,
    util::orbit_camera::{CameraUniform, OrbitCamera},
    wgpu::{self, util::DeviceExt},
    App, Context,
};

// =========== CONFIGURATIONS =============

const SHADER_FILE: &str = "globe_shader.wgsl";
// const TEXTURE_FILE: &str = "../../textures/texture.png";
// const TEXTURE_FILE: &str = "../../textures/earth2048.bmp";
// const TEXTURE_FILE: &str = "../../textures/moon1024.bmp";
const TEXTURE_FILE: &str = "../../textures/grey.png";

// Specular light parameters
const LIGHT_POS: [f32; 4] = [2.0, 2.0, 2.0, 0.0];
const KS: f32 = 0.15;
const SHININESS: f32 = 128.0;
const _PAD: [f32;2] = [0.0, 0.0];

// Globe geometry
const RADIUS: f32 = 1.0;
const STACK_COUNT: usize = 64;
const SECTOR_COUNT: usize = 128;



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
    light: [f32; 4],   // maps to vec4<f32> in WGSL
    ks: f32,           // specular strength
    shininess: f32,    // shininess exponent
    _pad: [f32; 2],    // padding to 16-byte alignment
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
pub struct SphereApp {
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

    // Eframe
    light_pos: [f32; 3],
    ks: f32,
    shininess: f32,
    stack_count: usize,   // TODO: for now, display only....
    sector_count: usize,  // TODO: ... regenerating requires rebuilding buffers
}

impl SphereApp {
    pub fn new(context: &Context) -> Self {

        // 1. Generate geometry
        let (vertices, indices, num_indices) = Self::create_sphere_geometry();


        // 2. gpu buff
        let vertex_buffer = Self::create_vertex_buffer(context, &vertices);
        let index_buffer = Self::create_index_buffer(context, &indices);


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



        Self {
            vertex_buffer,
            index_buffer,
            render_pipeline,
            num_indices,
            camera,
            texture_bind_group,
            light_bind_group,
            light_buffer,
            fps: 0.0,
            light_pos: [LIGHT_POS[0], LIGHT_POS[1], LIGHT_POS[2]],
            ks: KS,
            shininess: SHININESS,
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
            ks: KS,
            shininess: SHININESS,
            _pad: _PAD,
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
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
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
            label: Some("globe_shader"),
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
                buffers: &[Vertex::desc()],
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


    // ============== Update =====================
    fn update_light_uniform(&self, context: &Context) {
        let updated_light = LightUniform {
            light: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
            ks: self.ks,
            shininess: self.shininess,
            _pad: _PAD,
        };
        context.queue().write_buffer(
            &self.light_buffer,
            0,
            bytemuck::bytes_of(&updated_light),
        );
    }


    
}



impl App for SphereApp {
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Set pipeline, buffers
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        //Bind Groups
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);
        render_pass.set_bind_group(1, &self.texture_bind_group, &[]);
        render_pass.set_bind_group(2, &self.light_bind_group, &[]);

        // Draw
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        egui::Window::new("Params").show(egui_ctx, |ui| {

            // Radius slider
            ui.heading("Camera");
            let mut radius = self.camera.radius();
            if ui.add(egui::Slider::new(&mut radius, 2.0..=10.0).text("radius")).changed() {
                self.camera.set_radius(radius).update(context);
            }

            // Light controls
            ui.heading("Light");
            let mut light_changed = false;
            
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[0], -5.0..=5.0).text("Light X")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[1], -5.0..=5.0).text("Light Y")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.light_pos[2], -5.0..=5.0).text("Light Z")).changed();
            
            ui.add_space(5.0);
            light_changed |= ui.add(egui::Slider::new(&mut self.ks, 0.0..=2.0).text("Specular (ks)")).changed();
            light_changed |= ui.add(egui::Slider::new(&mut self.shininess, 1.0..=512.0).text("Shininess")).changed();
            // Update GPU buffer if any light param changed
            if light_changed {
                let updated_light = LightUniform {
                    light: [self.light_pos[0], self.light_pos[1], self.light_pos[2], 0.0],
                    ks: self.ks,
                    shininess: self.shininess,
                    _pad: _PAD,
                };
                context.queue().write_buffer(
                    &self.light_buffer,
                    0,
                    bytemuck::bytes_of(&updated_light)
                );
            }
            
            ui.separator();


            // Geometry info (read-only for now)
            ui.heading("Geometry");
            ui.label(format!("Stacks: {}", self.stack_count));
            ui.label(format!("Sectors: {}", self.sector_count));
            ui.label(format!("Vertices: {}", (self.stack_count + 1) * (self.sector_count + 1)));
            
            ui.separator();
            


            // FPS
            ui.label(format!("FPS: {}", self.fps.round()));
            // Other

        });
    }


    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input, context);
    }

    fn update(&mut self, delta_time: f32, _context: &Context) {
        self.fps = 1.0 / delta_time;
    }

    fn resize(&mut self, new_width: u32, new_height: u32, context: &Context) {
        self.camera
            .set_aspect(new_width as f32 / new_height as f32)
            .update(context);
    }

}
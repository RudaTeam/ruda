//! The air's tables of light, see `sky_tables.wgsl`: transmittance and
//! multiple scattering, worked out once; the sky from the camera and the
//! light at the camera, worked out every frame. The world pass reads them
//! to draw the sky and light the world.

use glam::Vec3;

/// `sky_tables.wgsl`, after the functions of the air it uses.
const SHADER: &str = concat!(
    include_str!("atmosphere.wgsl"),
    include_str!("sky_tables.wgsl")
);
/// Bytes of the `Tables` uniform.
const UNIFORMS_SIZE: u64 = 16;
const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
const MULTIPLE_SIZE: (u32, u32) = (32, 32);
const SKY_SIZE: (u32, u32) = (192, 108);

pub(crate) struct SkyTables {
    pub(crate) transmittance: wgpu::TextureView,
    multiple: wgpu::TextureView,
    pub(crate) sky: wgpu::TextureView,
    /// Two texels: the sky's light on a surface facing up, and the sun's or
    /// moon's light, both at the camera.
    pub(crate) ambient: wgpu::TextureView,
    uniforms: wgpu::Buffer,
    passes: [(wgpu::RenderPipeline, wgpu::BindGroup); 4],
    /// Whether the tables that never change are worked out yet.
    ready: bool,
}

impl SkyTables {
    pub(crate) fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky tables"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let texture = |label, (width, height)| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let transmittance = texture("transmittance", TRANSMITTANCE_SIZE);
        let multiple = texture("multiple scattering", MULTIPLE_SIZE);
        let sky = texture("sky", SKY_SIZE);
        let ambient = texture("ambient", (2, 1));
        let blank = texture("blank", (1, 1));

        let sampled = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky tables"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(UNIFORMS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                sampled(2),
                sampled(3),
                sampled(4),
            ],
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky tables"),
            size: UNIFORMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky tables"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let group = |textures: [&wgpu::TextureView; 3]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sky tables"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(textures[0]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(textures[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(textures[2]),
                    },
                ],
            })
        };
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky tables"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let passes = [
            (
                pipeline("transmittance_table"),
                group([&blank, &blank, &blank]),
            ),
            (
                pipeline("multiple_table"),
                group([&transmittance, &blank, &blank]),
            ),
            (
                pipeline("sky_table"),
                group([&transmittance, &multiple, &blank]),
            ),
            (
                pipeline("ambient_table"),
                group([&transmittance, &multiple, &sky]),
            ),
        ];
        Self {
            transmittance,
            multiple,
            sky,
            ambient,
            uniforms,
            passes,
            ready: false,
        }
    }

    /// Works out the sky for the sun towards `sun` and a camera `height`
    /// blocks above sea level.
    pub(crate) fn update(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        sun: Vec3,
        height: f32,
    ) {
        let floats = [sun.x, sun.y, sun.z, height];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        queue.write_buffer(&self.uniforms, 0, &bytes);
        let targets = [
            &self.transmittance,
            &self.multiple,
            &self.sky,
            &self.ambient,
        ];
        let first = if self.ready { 2 } else { 0 };
        self.ready = true;
        for ((pipeline, group), target) in self.passes.iter().zip(targets).skip(first) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky table"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn shader_is_valid_on_the_baseline_tier() {
        let module = naga::front::wgsl::parse_str(super::SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}

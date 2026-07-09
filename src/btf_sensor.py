# bowen's code

import mitsuba as mi
import drjit as dr

mi.set_variant('cuda_ad_rgb')

class BTFSensor(mi.Sensor):
    def __init__(self, props):
        super().__init__(props)
        self.camera_position = mi.Vector3f(props.get('camera_position', [2, 2, 3]))
        dr.enable_grad(self.camera_position) 
        self.surface_center = mi.Vector3f(props.get('surface_center', [0, 0, 0]))
        self.surface_size = props.get('surface_size', 1.0)
        self.m_film = props.get('film')
        
    def parameters_changed(self, keys):
        # Call parent implementation
        super().parameters_changed(keys)
        # Additional logic can be added here if needed

    def traverse(self, callback):
        # 将camera_position暴露给mitsuba参数系统
        callback.put_parameter('camera_position', self.camera_position, mi.ParamFlags.Differentiable)
        # 可选：也暴露surface_size和surface_center以便调试
        callback.put_parameter('surface_size', self.surface_size, mi.ParamFlags.Differentiable)
        callback.put_parameter('surface_center', self.surface_center, mi.ParamFlags.Differentiable)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        # position_sample在[0,1]^2范围内
        # 将position_sample从[0,1]^2映射到[-0.5,0.5]^2
        uv = position_sample - 0.5
        offset = uv * self.surface_size
        
        # 材质表面上的采样点
        surface_point = self.surface_center + mi.Vector3f(offset.x, offset.y, 0)
        
        # 从相机到表面的方向向量（未归一化）
        dir = surface_point - self.camera_position
        
        # 计算距离平方
        dist2 = dr.squared_norm(dir)
        
        # 归一化方向向量
        dir_normalized = dir * dr.rsqrt(dist2)
        
        # 创建从相机出发的光线
        ray = mi.Ray3f(
            o=self.camera_position,
            d=dir_normalized,
            time=time,
        )
        
        # 计算权重 - 这是关键修正！
        # 表面法线（假设朝上）
        surface_normal = mi.Vector3f(0, 0, 1)
        
        # 视角因子：相机到表面的方向与表面法线的夹角余弦
        # 注意：这里用-dir_normalized是因为我们需要的是"从表面指向相机"的方向
        cos_theta = dr.dot(-dir_normalized, surface_normal)
        
        # 采样面积
        area = self.surface_size * self.surface_size
        
        # 面积PDF = 1/A
        pdf_area = 1.0 / area
        
        # 面积PDF到方向PDF的转换：pdf_dir = pdf_area * (r² / |cosθ|)
        pdf_dir = pdf_area  / dr.abs(cos_theta)
        # pdf_dir = pdf_area / dr.abs(cos_theta)
        
        # 权重 = 1 / pdf_dir = A * |cosθ| / r²
        weight = dr.select(
            cos_theta > 0,
            1.0 / pdf_dir,  # = area * |cos_theta| / dist2
            0.0
        )
        
        return ray, mi.Spectrum(weight)
    
    def pdf_ray(self, ray, active=True):
        """计算给定光线的PDF（用于双向算法）"""
        # 计算光线与表面的交点
        # 假设表面在z=0平面上
        t = -ray.o.z / ray.d.z
        
        # 交点
        hit_point = ray.o + t * ray.d
        
        # 检查是否在表面范围内
        offset = hit_point - self.surface_center
        in_bounds = (dr.abs(offset.x) <= self.surface_size * 0.5) & \
                   (dr.abs(offset.y) <= self.surface_size * 0.5) & \
                   (t > 0)
        
        # 计算PDF
        # dist2 = dr.squared_norm(hit_point - ray.o)
        cos_theta = dr.abs(ray.d.z)  # 因为表面法线是(0,0,1)
        area = self.surface_size * self.surface_size
        pdf_area = 1.0 / area
        pdf_dir = pdf_area  / cos_theta

        
        return dr.select(in_bounds, pdf_dir, 0.0)

    def film(self):
        return self.m_film

    def bbox(self):
        # 返回包含相机和表面的边界框
        min_corner = dr.minimum(
            self.camera_position,
            self.surface_center - mi.Vector3f(self.surface_size * 0.5)
        )
        max_corner = dr.maximum(
            self.camera_position,
            self.surface_center + mi.Vector3f(self.surface_size * 0.5)
        )
        return mi.BoundingBox3f(min_corner, max_corner)

# 注册传感器
mi.register_sensor("btf_sensor", lambda props: BTFSensor(props))
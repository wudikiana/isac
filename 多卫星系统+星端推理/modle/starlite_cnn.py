import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return self.sigmoid(out).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class DynamicAttention(nn.Module):
    """动态注意力机制 - 根据特征重要性动态调整"""
    def __init__(self, in_channels):
        super().__init__()
        self.importance_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        self.cbam = CBAM(in_channels)
        
    def forward(self, x):
        # 计算特征重要性权重
        importance = self.importance_net(x)
        # 应用CBAM注意力
        attended = self.cbam(x)
        # 根据重要性动态调整
        return x + importance * attended

class FPNLayer(nn.Module):
    """FPN层 - 正确处理空间特征"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MultiScaleFPN(nn.Module):
    """多尺度特征融合 - 优化量化兼容性"""
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        # 优化：为不同kernel_size的卷积添加BatchNorm以提高量化兼容性
        self.scale_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, k, padding=k//2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for k in scales
        ])
        # 添加最终的融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = []
        for scale_layer in self.scale_layers:
            feat = scale_layer(x)
            multi_scale_features.append(feat)
        
        # 特征相加而非拼接
        fused_feature = multi_scale_features[0]
        for feat in multi_scale_features[1:]:
            fused_feature = fused_feature + feat
            
        # 最终融合
        return self.fusion_conv(fused_feature)

class EnhancedLandslideDetector(nn.Module):
    """增强的山体滑坡检测模型，支持QAT - 优化版本"""
    def __init__(self, num_classes=2, use_attention=True, use_fpn=True, use_dynamic_attention=False, use_multi_scale=False):
        super().__init__()
        # 使用预训练的MobileNetV3
        self.backbone = models.mobilenet_v3_small(pretrained=True, quantize=False)
        
        # 获取关键特征层
        self.feature_layers = []
        for i, layer in enumerate(self.backbone.features):
            if isinstance(layer, nn.Conv2d):
                self.feature_layers.append(i)
        
        # 注意力机制 - 支持动态注意力
        self.use_attention = use_attention
        self.use_dynamic_attention = use_dynamic_attention
        if use_attention:
            self.attention_layers = nn.ModuleDict()
            # 只在关键层添加注意力
            key_layers = self.feature_layers[-3:]  # 最后3个关键层
            for i in key_layers:
                layer = self.backbone.features[i]
                if hasattr(layer, 'out_channels'):
                    if use_dynamic_attention:
                        self.attention_layers[f'attn_{i}'] = DynamicAttention(layer.out_channels)
                    else:
                        self.attention_layers[f'attn_{i}'] = CBAM(layer.out_channels)
        
        # FPN特征金字塔网络 - 支持多尺度融合
        self.use_fpn = use_fpn
        self.use_multi_scale = use_multi_scale
        if use_fpn:
            self.fpn = nn.ModuleDict()
            fpn_channels = 128
            
            # 为最后3个特征层创建FPN层
            for i, layer_idx in enumerate(self.feature_layers[-3:]):
                layer = self.backbone.features[layer_idx]
                if hasattr(layer, 'out_channels'):
                    if use_multi_scale:
                        self.fpn[f'fpn_{i}'] = MultiScaleFPN(layer.out_channels, fpn_channels)
                    else:
                        self.fpn[f'fpn_{i}'] = FPNLayer(layer.out_channels, fpn_channels)
        
        # 获取分类器输入维度
        in_features = 576  # MobileNetV3-Small 默认值
        if hasattr(self.backbone.classifier, '__getitem__'):
            for m in self.backbone.classifier:
                if isinstance(m, nn.Linear):
                    in_features = m.in_features
                    break
        
        # 增强的分类头 - 使用LayerNorm替代BatchNorm以避免batch_size=1的问题
        if use_fpn and len(self.feature_layers) >= 3:
            # 使用FPN时，增加特征维度
            enhanced_features = in_features + fpn_channels * 3
        else:
            enhanced_features = in_features
            
        self.classifier = nn.Sequential(
            nn.Linear(enhanced_features, 512),
            nn.LayerNorm(512),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 使用LayerNorm替代BatchNorm
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # 量化兼容层
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        
        # 提取特征
        features = []
        attention_features = []
        
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
                # 在bottleneck后应用注意力
                if self.use_attention and f'attn_{i}' in self.attention_layers:
                    attn_x = self.attention_layers[f'attn_{i}'](x)
                    attention_features.append(attn_x)
        
        # 分类
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        if self.use_fpn and len(features) >= 3:
            # 使用FPN特征 - 优化版本：正确的上采样和特征融合
            fpn_features = []
            key_features = features[-3:]  # 最后3个特征层
            
            # 自顶向下的特征融合（正确的FPN实现）
            P5 = self.fpn['fpn_2'](key_features[2])  # 顶层特征
            P4 = self.fpn['fpn_1'](key_features[1]) + F.interpolate(P5, size=key_features[1].shape[2:], mode='bilinear', align_corners=False)
            P3 = self.fpn['fpn_0'](key_features[0]) + F.interpolate(P4, size=key_features[0].shape[2:], mode='bilinear', align_corners=False)
            
            # 全局平均池化
            P3_pooled = F.adaptive_avg_pool2d(P3, (1, 1))
            P4_pooled = F.adaptive_avg_pool2d(P4, (1, 1))
            P5_pooled = F.adaptive_avg_pool2d(P5, (1, 1))
            
            # 展平并拼接
            P3_flat = torch.flatten(P3_pooled, 1)
            P4_flat = torch.flatten(P4_pooled, 1)
            P5_flat = torch.flatten(P5_pooled, 1)
            
            # 特征相加
            fused_fpn = P3_flat + P4_flat + P5_flat
            
            # 拼接主特征和融合的FPN特征
            x = torch.cat([x, fused_fpn], dim=1)
        
        x = self.classifier(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """融合模型层以提高推理效率 - 优化版本"""
        # 融合MobileNetV3的所有可融合层
        for m in self.backbone.features:
            if type(m) == nn.Sequential:
                modules = list(m._modules.keys())
                # 先尝试融合三层（Conv+BN+ReLU）
                for i in range(len(modules) - 2):
                    try:
                        torch.quantization.fuse_modules(m, [modules[i], modules[i+1], modules[i+2]], inplace=True)
                    except Exception:
                        pass
                # 再尝试融合两层（Conv+BN）
                for i in range(len(modules) - 1):
                    try:
                        torch.quantization.fuse_modules(m, [modules[i], modules[i+1]], inplace=True)
                    except Exception:
                        pass
        
        # 融合FPN层
        if self.use_fpn:
            for fpn_layer in self.fpn.values():
                if isinstance(fpn_layer, FPNLayer):
                    try:
                        torch.quantization.fuse_modules(fpn_layer, ['conv', 'bn'], inplace=True)
                    except Exception:
                        pass
                elif isinstance(fpn_layer, MultiScaleFPN):
                    # 融合MultiScaleFPN中的各个层
                    for i, scale_layer in enumerate(fpn_layer.scale_layers):
                        try:
                            torch.quantization.fuse_modules(scale_layer, ['0', '1'], inplace=True)  # Conv+BN
                        except Exception:
                            pass
                    # 融合最终融合层
                    try:
                        torch.quantization.fuse_modules(fpn_layer.fusion_conv, ['0', '1'], inplace=True)
                    except Exception:
                        pass
        
        # 融合分类头中的LayerNorm层 - 优化版本
        def fuse_sequential(seq):
            """递归融合Sequential中的LayerNorm层"""
            for i in range(len(seq) - 1):
                if isinstance(seq[i], nn.Linear) and isinstance(seq[i+1], nn.LayerNorm):
                    try:
                        torch.quantization.fuse_modules(seq, [str(i), str(i+1)], inplace=True)
                    except Exception:
                        pass
        
        fuse_sequential(self.classifier)

class LandslideDetector(nn.Module):
    """山体滑坡检测模型，支持QAT - 保持向后兼容"""
    def __init__(self, num_classes=2, enhanced=True, use_attention=True, use_fpn=True, 
                 use_dynamic_attention=False, use_multi_scale=False):
        super().__init__()
        
        if enhanced:
            # 使用增强版本
            self.model = EnhancedLandslideDetector(
                num_classes=num_classes, 
                use_attention=use_attention, 
                use_fpn=use_fpn,
                use_dynamic_attention=use_dynamic_attention,
                use_multi_scale=use_multi_scale
            )
        else:
            # 使用原始版本保持兼容性
            self.backbone = models.mobilenet_v3_small(pretrained=True, quantize=False)
            # 修改分类头，确保 in_features 为 int 类型
            in_features = 576  # MobileNetV3-Small 默认值
            if hasattr(self.backbone.classifier, '__getitem__'):
                for m in self.backbone.classifier:
                    if isinstance(m, nn.Linear):
                        in_features = m.in_features
                        break
            self.backbone.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            # 量化兼容层
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        if hasattr(self, 'model'):
            return self.model(x)
        else:
            x = self.quant(x)
            x = self.backbone(x)
            x = self.dequant(x)
            return x

    def fuse_model(self):
        if hasattr(self, 'model'):
            return self.model.fuse_model()
        else:
            # 融合MobileNetV3的所有可融合层
            for m in self.backbone.features:
                if type(m) == nn.Sequential:
                    modules = list(m._modules.keys())
                    # 先尝试融合三层（Conv+BN+ReLU）
                    for i in range(len(modules) - 2):
                        try:
                            torch.quantization.fuse_modules(m, [modules[i], modules[i+1], modules[i+2]], inplace=True)
                        except Exception:
                            pass
                    # 再尝试融合两层（Conv+BN）
                    for i in range(len(modules) - 1):
                        try:
                            torch.quantization.fuse_modules(m, [modules[i], modules[i+1]], inplace=True)
                        except Exception:
                            pass

class SegmentationLandslideDetector(nn.Module):
    """分割版本的山体滑坡检测模型 - 将分类模型转换为分割模型"""
    def __init__(self, num_classes=1, use_attention=True, use_fpn=True, use_dynamic_attention=False, use_multi_scale=False):
        super().__init__()
        # 使用预训练的MobileNetV3作为backbone
        self.backbone = models.mobilenet_v3_small(pretrained=True, quantize=False)
        
        # 获取关键特征层
        self.feature_layers = []
        for i, layer in enumerate(self.backbone.features):
            if isinstance(layer, nn.Conv2d):
                self.feature_layers.append(i)
        
        # 注意力机制 - 支持动态注意力
        self.use_attention = use_attention
        self.use_dynamic_attention = use_dynamic_attention
        if use_attention:
            self.attention_layers = nn.ModuleDict()
            # 只在关键层添加注意力
            key_layers = self.feature_layers[-3:]  # 最后3个关键层
            for i in key_layers:
                layer = self.backbone.features[i]
                if hasattr(layer, 'out_channels'):
                    if use_dynamic_attention:
                        self.attention_layers[f'attn_{i}'] = DynamicAttention(layer.out_channels)
                    else:
                        self.attention_layers[f'attn_{i}'] = CBAM(layer.out_channels)
        
        # FPN特征金字塔网络 - 支持多尺度融合
        self.use_fpn = use_fpn
        self.use_multi_scale = use_multi_scale
        if use_fpn:
            self.fpn = nn.ModuleDict()
            fpn_channels = 128
            
            # 为最后3个特征层创建FPN层
            for i, layer_idx in enumerate(self.feature_layers[-3:]):
                layer = self.backbone.features[layer_idx]
                if hasattr(layer, 'out_channels'):
                    if use_multi_scale:
                        self.fpn[f'fpn_{i}'] = MultiScaleFPN(layer.out_channels, fpn_channels)
                    else:
                        self.fpn[f'fpn_{i}'] = FPNLayer(layer.out_channels, fpn_channels)
        
        # 分割头 - 将分类头改为分割头
        if use_fpn and len(self.feature_layers) >= 3:
            # 使用FPN时，增加特征维度
            enhanced_features = 576 + fpn_channels * 3  # MobileNetV3-Small输出576通道
        else:
            enhanced_features = 576
            
        # 分割解码器 - 修复：增加更多上采样层以恢复到原始尺寸
        self.segmentation_decoder = nn.Sequential(
            # 上采样层 - 从H/32, W/32恢复到H, W需要5次上采样
            nn.ConvTranspose2d(enhanced_features, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 最终分割头
            nn.Conv2d(16, num_classes, kernel_size=1)
        )
        
        # 量化兼容层
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # 保存原始输入尺寸
        original_size = x.shape[2:]
        x = self.quant(x)
        
        # 提取特征
        features = []
        attention_features = []
        
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
                # 在bottleneck后应用注意力
                if self.use_attention and f'attn_{i}' in self.attention_layers:
                    attn_x = self.attention_layers[f'attn_{i}'](x)
                    attention_features.append(attn_x)
        
        # 获取最终特征图
        final_feature = x  # [B, 576, H/32, W/32]
        
        if self.use_fpn and len(features) >= 3:
            # 使用FPN特征 - 正确的上采样和特征融合
            key_features = features[-3:]  # 最后3个特征层
            
            # 自顶向下的特征融合（正确的FPN实现）
            P5 = self.fpn['fpn_2'](key_features[2])  # 顶层特征
            P4 = self.fpn['fpn_1'](key_features[1]) + F.interpolate(P5, size=key_features[1].shape[2:], mode='bilinear', align_corners=False)
            P3 = self.fpn['fpn_0'](key_features[0]) + F.interpolate(P4, size=key_features[0].shape[2:], mode='bilinear', align_corners=False)
            
            # 将FPN特征上采样到与最终特征相同的尺寸
            fpn_feature = F.interpolate(P3, size=final_feature.shape[2:], mode='bilinear', align_corners=False)
            
            # 拼接主特征和FPN特征
            final_feature = torch.cat([final_feature, fpn_feature], dim=1)
        
        # 通过分割解码器
        output = self.segmentation_decoder(final_feature)
        
        # 确保输出尺寸与原始输入相同
        if output.shape[2:] != original_size:
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False)
        
        output = self.dequant(output)
        return output

    def fuse_model(self):
        """融合模型层以提高推理效率"""
        # 融合MobileNetV3的所有可融合层
        for m in self.backbone.features:
            if type(m) == nn.Sequential:
                modules = list(m._modules.keys())
                # 先尝试融合三层（Conv+BN+ReLU）
                for i in range(len(modules) - 2):
                    try:
                        torch.quantization.fuse_modules(m, [modules[i], modules[i+1], modules[i+2]], inplace=True)
                    except Exception:
                        pass
                # 再尝试融合两层（Conv+BN）
                for i in range(len(modules) - 1):
                    try:
                        torch.quantization.fuse_modules(m, [modules[i], modules[i+1]], inplace=True)
                    except Exception:
                        pass
        
        # 融合FPN层
        if self.use_fpn:
            for fpn_layer in self.fpn.values():
                if isinstance(fpn_layer, FPNLayer):
                    try:
                        torch.quantization.fuse_modules(fpn_layer, ['conv', 'bn'], inplace=True)
                    except Exception:
                        pass
                elif isinstance(fpn_layer, MultiScaleFPN):
                    # 融合MultiScaleFPN中的各个层
                    for i, scale_layer in enumerate(fpn_layer.scale_layers):
                        try:
                            torch.quantization.fuse_modules(scale_layer, ['0', '1'], inplace=True)  # Conv+BN
                        except Exception:
                            pass
                    # 融合最终融合层
                    try:
                        torch.quantization.fuse_modules(fpn_layer.fusion_conv, ['0', '1'], inplace=True)
                    except Exception:
                        pass
        
        # 融合分割解码器中的BatchNorm层
        for module in self.segmentation_decoder:
            if isinstance(module, nn.Sequential):
                for i in range(len(module) - 1):
                    if isinstance(module[i], (nn.Conv2d, nn.ConvTranspose2d)) and isinstance(module[i+1], nn.BatchNorm2d):
                        try:
                            torch.quantization.fuse_modules(module, [str(i), str(i+1)], inplace=True)
                        except Exception:
                            pass

def create_starlite_model(pretrained=False, num_classes=2, enhanced=True, use_attention=True, use_fpn=True,
                         use_dynamic_attention=False, use_multi_scale=False):
    """创建Starlite模型 - 优化权重加载逻辑"""
    model = LandslideDetector(
        num_classes=num_classes,
        enhanced=enhanced,
        use_attention=use_attention,
        use_fpn=use_fpn,
        use_dynamic_attention=use_dynamic_attention,
        use_multi_scale=use_multi_scale
    )
    
    if pretrained:
        # 加载预训练权重（如果有的话）- 优化版本
        try:
            checkpoint = torch.load('models/starlite_cnn.pth', map_location='cpu')
            
            # 智能处理不同前缀的权重键名
            new_state_dict = {}
            for key, value in checkpoint.items():
                # 处理增强模型的权重键名
                if enhanced and key.startswith('model.'):
                    new_key = key.replace('model.', '', 1)
                elif enhanced and key.startswith('backbone.'):
                    new_key = key.replace('backbone.', 'model.backbone.', 1)
                elif not enhanced and key.startswith('model.backbone.'):
                    new_key = key.replace('model.backbone.', 'backbone.', 1)
                else:
                    new_key = key
                new_state_dict[new_key] = value
            
            # 尝试加载权重
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            
            if missing_keys:
                print(f"警告：缺少以下权重键: {missing_keys[:5]}...")  # 只显示前5个
            if unexpected_keys:
                print(f"警告：意外的权重键: {unexpected_keys[:5]}...")  # 只显示前5个
                
            print("成功加载预训练权重")
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    
    return model

def create_enhanced_model(num_classes=2, use_attention=True, use_fpn=True, 
                         use_dynamic_attention=False, use_multi_scale=False):
    """创建增强模型的便捷函数"""
    return create_starlite_model(
        pretrained=False,
        num_classes=num_classes,
        enhanced=True,
        use_attention=use_attention,
        use_fpn=use_fpn,
        use_dynamic_attention=use_dynamic_attention,
        use_multi_scale=use_multi_scale
    )

def create_segmentation_landslide_model(num_classes=1, use_attention=True, use_fpn=True,
                                       use_dynamic_attention=False, use_multi_scale=False):
    """创建分割版本的LandslideDetector模型"""
    model = SegmentationLandslideDetector(
        num_classes=num_classes,
        use_attention=use_attention,
        use_fpn=use_fpn,
        use_dynamic_attention=use_dynamic_attention,
        use_multi_scale=use_multi_scale
    )
    return model
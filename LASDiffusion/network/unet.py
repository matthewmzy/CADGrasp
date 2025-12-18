import torch
import torch.nn as nn
from LASDiffusion.network.model_utils import *  # 保持必要的自定义模块
from LASDiffusion.utils.utils import default  # 移除不需要的常量引入

class UNetModel(nn.Module):
    def __init__(self,
                 model_mode: str = "diffusion",  # 模型模式
                 image_size: int = 64,        # 输入体素的分辨率
                 base_channels: int = 64,     # 基础通道数
                 dim_mults=(1, 2, 4, 8),      # 通道数倍增系数
                 dropout: float = 0.1,        # Dropout概率
                 num_heads: int = 1,          # 注意力头数
                 world_dims: int = 3,         # 数据维度(3D)
                 attention_resolutions=(4, 8),# 应用注意力的分辨率层级
                 with_attention: bool = False,# 是否启用自注意力
                 verbose: bool = False,       # 调试模式
                 ):
        super().__init__()
        
        # === 调试模式 ===
        self.verbose = verbose
        
        # === 通道数配置 ===
        channels = [base_channels, *map(lambda m: base_channels * m, dim_mults)]    # [32, 32, 64, 128, 256]
        in_out = list(zip(channels[:-1], channels[1:]))  # 生成(输入通道,输出通道)对    # [(32,32),(32,64),(64,128),(128,256)]

        # === 时间步嵌入 ===
        self.use_time_emb = True if model_mode == "diffusion" else False  # 是否使用时间嵌入
        emb_dim = base_channels * 4  # 嵌入维度
        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)  # 学习式正弦位置编码
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),  # 增加1维防止零初始化
            activation_function(),                  # SiLU激活
            nn.Linear(emb_dim, emb_dim)
        )

        # === 场景点云条件编码嵌入层 ===
        self.scene_emb = conv_nd(world_dims, 1, base_channels, 3, padding=1)  # 3D卷积，处理场景点云条件
        # === 场景点云条件编码下采样 ===
        self.scene_downs = nn.ModuleList([])
        scene_ds = 1  # 当前下采样倍数
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if is_last:
                break
            res = image_size // scene_ds # 当前分辨率
            self.scene_downs.append(nn.ModuleList([
                # ResNet块（移除文本条件相关参数）
                ResnetBlock(world_dims, dim_in, dim_out,
                           emb_dim=emb_dim, dropout=dropout,
                           use_text_condition=False, use_time_condition=False),  # 关闭文本条件
                # 自注意力模块
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(dim_out, num_heads=num_heads)
                ) if scene_ds in attention_resolutions and with_attention else our_Identity(),
                # 下采样层
                Downsample(dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                scene_ds *= 2

        # === 输入嵌入层 === 
        # 3D卷积，处理输入体素(通道数2=原始输入+自条件)
        if self.use_time_emb:
            self.input_emb = conv_nd(world_dims, 4, base_channels, 3, padding=1)
        else:
            self.input_emb = conv_nd(world_dims, 2, base_channels, 3, padding=1)
        
        # === 下采样路径 ===
        self.downs = nn.ModuleList([])
        ds = 1  # 当前下采样倍数
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            res = image_size // ds  # 当前分辨率
            
            self.downs.append(nn.ModuleList([
                # ResNet块（移除文本条件相关参数）
                ResnetBlock(world_dims, 2*dim_in, dim_out, 
                           emb_dim=emb_dim, dropout=dropout,
                           use_text_condition=False, use_time_condition=self.use_time_emb),  # 关闭文本条件
                
                # 自注意力模块
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(dim_out, num_heads=num_heads)
                ) if ds in attention_resolutions and with_attention else our_Identity(),
                
                # 下采样层
                Downsample(dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2  # 更新下采样倍数

        # === 中间层 ===
        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, 
            emb_dim=emb_dim, dropout=dropout,
            use_text_condition=False,  # 关闭文本条件
            use_time_condition=self.use_time_emb
        )
        
        # 移除中间跨模态注意力
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim,
            emb_dim=emb_dim, dropout=dropout,
            use_text_condition=False,  # 关闭文本条件
            use_time_condition=self.use_time_emb
        )

        # === 上采样路径 ===
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):          # [(128,256),(64,128),(32,64)]
            is_last = ind >= (len(in_out) - 1)
            res = image_size // ds
            
            self.ups.append(nn.ModuleList([
                # ResNet块：处理跳跃连接的特征拼接（移除文本条件）
                ResnetBlock(world_dims, int(dim_out*2.5), dim_in,
                           emb_dim=emb_dim, dropout=dropout,
                           use_text_condition=False, use_time_condition=self.use_time_emb),

                # 自注意力
                nn.Sequential(
                    normalization(dim_in),
                    activation_function(),
                    AttentionBlock(dim_in, num_heads=num_heads)
                ) if ds in attention_resolutions and with_attention else our_Identity(),
                
                # 上采样层
                Upsample(dim_in, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2  # 更新上采样倍数

        # === 输出处理 ===
        self.end = nn.Sequential(
            normalization(base_channels),
            activation_function()  # SiLU
        )
        self.out = conv_nd(world_dims, base_channels, 2, 3, padding=1)  # 输出1通道

    def forward(self, x, t, c, x_self_cond=None):
        """
        简化后的前向传播
        参数：
            x: 输入体素 (B,2,H,W,D)
            t: 时间步 (B,)
            c: 场景点云条件 (B,1,H,W,D)
            g: 抓取类型 (B,)
            x_self_cond: 自条件输入（可选）
        """
        if self.verbose:
            print(f"[DEBUG] input x size: {x.shape}, c size: {c.shape}, g size: {g}")
        # 处理自条件输入
        if self.use_time_emb:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, x_self_cond), dim=1)  # (B,4,H,W,D)
        # else x: (B,2,H,W,D)

        if self.verbose:
            print(f"[DEBUG] input x size: {x.shape}, c size: {c.shape}")

        # === 场景条件嵌入 ===
        c = self.scene_emb(c)   # (B,C,H,W,D)

        if self.verbose:
            print(f"[DEBUG] scene cond emb size {c.shape}")
        sc = [c]  # 保存场景条件

        # === 场景编码保存 ===
        for si, (resnet, self_attn, downsample) in enumerate(self.scene_downs):
            c = resnet(c, None, None)
            c = self_attn(c)
            c = downsample(c)
            sc.append(c)
            if self.verbose:
                print(f"[DEBUG] scene downsample {si} size {c.shape}")

        # === 初始嵌入 ===
        x = self.input_emb(x)  # (B,C,H,W,D)
        if self.use_time_emb:
            t = self.time_emb(self.time_pos_emb(t))  # (B,emb_dim)

        if self.verbose:
            print(f"[DEBUG] input emb size: {x.shape}")

        h = []  # 保存跳跃连接特征

        # === 下采样过程 ===
        for di, (resnet, self_attn, downsample) in enumerate(self.downs):
            # print(f"[DEBUG] downsample {di} size: {x.shape} {sc[di].shape}")
            x = torch.cat((x, sc[di]),dim=1)
            x = resnet(x, t, None)  # 移除了text_condition参数
            x = self_attn(x)        # 自注意力
            h.append(x)             # 保存特征
            x = downsample(x)       # 下采样
            
            if self.verbose:
                print(f"[DEBUG] downsample {di} size: {x.shape}")

        # === 中间处理 ===
        x = self.mid_block1(x, t, None)
        x = self.mid_self_attn(x)
        x = self.mid_block2(x, t, None)

        if self.verbose:
            print(f"[DEBUG] mid size: {x.shape}")

        # === 上采样过程 ===
        for ui, (resnet, self_attn, upsample) in enumerate(self.ups):
            if ui==0:
                x = torch.cat((x, h.pop(), sc[-1-ui]), dim=1)  # 拼接跳跃连接
            else:
                x = torch.cat((x, h.pop(), sc[-1-ui]), dim=1)  # 拼接跳跃连接
            x = resnet(x, t, None)     # 移除了text_condition参数
            x = self_attn(x)          # 自注意力
            x = upsample(x)           # 上采样
            
            if self.verbose:
                print(f"[DEBUG] upsample {ui} size: {x.shape}")

        # === 最终输出 ===
        x = self.end(x)
        x = self.out(x)
        if self.verbose:
            print(f"[DEBUG] output size: {x.shape}")
        return x  # (B,2,H,W,D)

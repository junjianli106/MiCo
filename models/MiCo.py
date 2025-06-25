import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    """Gated attention mechanism with sigmoid activation"""

    def __init__(self, input_dim=768, hidden_dim=256, num_classes=1, drop=0.25):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )

        # 注意力门控分支
        self.attention_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )

        self.attention_scorer = nn.Linear(hidden_dim, num_classes)

    def forward(self, features):
        transformed = self.feature_transform(features)
        gate = self.attention_gate(features)
        attention_weights = self.attention_scorer(transformed * gate)
        return attention_weights, features


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop != 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ClusterReducer(Mlp):

    def forward(self, x):
        """
        Args:
            x (Tensor): 大小为(num_clusters, embedding_dim)。
        """
        return super().forward(x.transpose(0, 1)).transpose(0, 1)


class MiCo(nn.Module):

    def __init__(self,
                 dim_in=768,
                 embedding_dim=512,
                 num_clusters=64,
                 num_classes=4,
                 survival=True,
                 cluster_init_path=None,
                 num_enhancers=3,
                 drop=0.25,
                 hard=False,
                 similarity_method='l2'):
        super().__init__()
        self.survival=survival
        self.hard_assignment = hard
        self.similarity_method = similarity_method
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters

        # -- 初始化聚类中心 --
        if cluster_init_path:
            # 从预计算的文件中加载K-means中心
            init_file = cluster_init_path
            initial_centers = torch.load(init_file)
            print('Initialize cluster centers with K-means, center shape:', initial_centers.shape)
            self.cluster_centers = nn.Parameter(torch.from_numpy(initial_centers), requires_grad=True)
        else:
            # 随机初始化
            self.cluster_centers = nn.Parameter(torch.randn(num_clusters, dim_in), requires_grad=True)

        self.patch_feature_projector = nn.Sequential(nn.Linear(dim_in, embedding_dim), nn.LeakyReLU(inplace=True))
        self.cluster_center_projector = nn.Sequential(nn.Linear(dim_in, embedding_dim), nn.LeakyReLU(inplace=True))

        # -- 多尺度上下文增强模块 --
        self.dynamic_num_clusters = [num_clusters // (2 ** i) for i in range(num_enhancers+1)] #[32, 16, 8]

        self.context_enhancers = nn.ModuleList([
            Mlp(embedding_dim, embedding_dim, embedding_dim, nn.ReLU, drop)  for _ in range(num_enhancers)
        ])
        self.cluster_reducers = nn.ModuleList([
            ClusterReducer(in_features=self.dynamic_num_clusters[i], hidden_features=self.dynamic_num_clusters[i], out_features=self.dynamic_num_clusters[i+1]) for i in range(num_enhancers)
        ])
        self.enhancer_norm_layers = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_enhancers)])

        # -- 特征处理与注意力池化 --
        self.feature_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        self.attention_network = GatedAttention(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_classes=1,
            drop=drop
        )
        self.aggregation_norm_layer = nn.LayerNorm(embedding_dim)

        # -- 分类器 --
        self.final_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)


        self.similarity_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _straight_through_softmax(self, logits, hard_assignment=True, dim=-1):
        y_soft = F.softmax(logits / self.similarity_scale, dim=1)
        if hard_assignment:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

    def _get_contextual_features(self, patch_embeddings, cluster_embeddings):
        """
        计算图块和聚类之间的相似度，并生成上下文感知的特征。

        Args:
            patch_embeddings (Tensor): 图块嵌入, 形状 (num_patches, embedding_dim).
            cluster_embeddings (Tensor): 聚类嵌入, 形状 (num_clusters, embedding_dim).

        Returns:
            Tensor: 上下文特征，通过加权聚类中心得到, 形状 (num_patches, embedding_dim).
        """
        if self.similarity_method == 'l2':
            # 基于L2距离计算相似度（负距离）
            similarity_scores = -torch.cdist(patch_embeddings, cluster_embeddings)
        else:  # 默认为点积相似度
            similarity_scores = patch_embeddings @ cluster_embeddings.transpose(-2, -1)

        # 获得图块到聚类的分配权重
        assignment_weights = self._straight_through_softmax(similarity_scores, self.hard_assignment, dim=1)

        # 根据分配权重对聚类中心进行加权求和，得到每个图块的上下文特征
        contextual_features = torch.matmul(assignment_weights, cluster_embeddings)
        return contextual_features

    def forward(self, **kwargs):
        # 如果是生存分析，输入是一个包含多个WSI特征的列表
        slide_features_list = kwargs['data'] if self.survival else [kwargs['data']]

        processed_slide_embeddings = []

        # 2. --- 多尺度上下文增强 (对每个WSI独立处理) ---
        for slide_patch_features in slide_features_list:
            # 移除批次维度，得到 (num_patches, feature_dim)
            patch_features = slide_patch_features.float().squeeze(0)

            # 将图块和聚类中心投影到嵌入空间
            patch_embeddings = self.patch_feature_projector(patch_features)
            cluster_embeddings = self.cluster_center_projector(self.cluster_centers)

            # 迭代增强上下文信息
            for i, enhancer_mlp in enumerate(self.context_enhancers):
                # a. 计算上下文特征
                contextual_features = self._get_contextual_features(patch_embeddings, cluster_embeddings)

                # b. 融合：将上下文信息添加到图块嵌入中 (残差连接)
                patch_embeddings = patch_embeddings + contextual_features

                # c. 归一化
                patch_embeddings = self.enhancer_norm_layers[i](patch_embeddings)

                # d. 增强：通过MLP进一步处理 (残差连接)
                patch_embeddings = patch_embeddings + enhancer_mlp(patch_embeddings)

                # e. 降维：减少下一阶段的聚类中心数量
                cluster_embeddings = self.cluster_reducers[i](cluster_embeddings)

            # 将最终的图块嵌入和剩余的聚类嵌入拼接，共同处理
            enhanced_embeddings = torch.cat([patch_embeddings, cluster_embeddings], dim=0)

            # 应用最终的特征处理器
            processed_embeddings = self.feature_processor(enhanced_embeddings)
            processed_slide_embeddings.append(processed_embeddings)

        # 3. --- 特征聚合 ---
        # 将处理过的所有WSI的特征拼接在一起
        aggregated_features = torch.cat(processed_slide_embeddings, dim=0)
        aggregated_features = self.aggregation_norm_layer(aggregated_features)

        # 4. --- 注意力池化 ---
        # 使用门控注意力网络计算每个特征的重要性分数
        attention_scores, _ = self.attention_network(aggregated_features)

        # 将注意力分数转换为权重 (softmax)
        attention_weights = F.softmax(attention_scores.transpose(0, 1), dim=1)

        # 使用注意力权重对特征进行加权求和，得到整个批次的WSI级表征
        slide_level_representation = torch.mm(attention_weights, aggregated_features)

        # 5. --- 分类 ---
        final_features = self.final_projector(slide_level_representation).squeeze()
        logits = self.classifier(final_features).unsqueeze(0) # 保证批次维度

        if self.survival:
            Y_hat = torch.argmax(logits, dim=1)
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)

            results_dict = {'hazards': hazards, 'S': S, 'Y_hat': Y_hat}
            return results_dict
        else:
            Y_prob = F.softmax(logits, dim=1)
            Y_hat = torch.topk(logits, 1, dim=1)[1]

            results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            return results_dict

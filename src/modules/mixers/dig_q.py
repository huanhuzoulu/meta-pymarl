import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DIGQ(nn.Module):
    def __init__(self, args):
        super(DIGQ, self).__init__()
        
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.digq_embed_dim
        self.num_heads = args.digq_num_heads  # 注意力头数
        
        # 节点特征映射
        self.node_proj = nn.Linear(1 + args.agent_obs_dim, self.embed_dim)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        # 多头图注意力层
        self.attention_layers = nn.ModuleList([
            nn.Linear(self.embed_dim * 3, self.num_heads)  # q_i, q_j, state
            for _ in range(args.digq_layers)
        ])
        
        # 节点更新网络
        self.node_update_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim * (1 + self.num_heads), self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            ) for _ in range(args.digq_layers)
        ])
        
        # 全局Q值预测器
        self.q_predictor = nn.Sequential(
            nn.Linear(self.embed_dim * self.n_agents, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )
    
    def forward(self, agent_qs, states, agent_obs):
        """
        :param agent_qs: (batch_size, seq_length, n_agents)
        :param states: (batch_size, seq_length, state_dim)
        :param agent_obs: (batch_size, seq_length, n_agents, agent_obs_dim)
        """
        bs, seq_len, _ = agent_qs.shape
        
        # 重塑为(batch_size * seq_length, ...)
        agent_qs = agent_qs.reshape(-1, self.n_agents, 1)
        states = states.reshape(-1, self.state_dim)
        agent_obs = agent_obs.reshape(-1, self.n_agents, agent_obs.shape[-1])
        
        # 创建初始节点特征 [q_i, obs_i]
        node_features = th.cat([agent_qs, agent_obs], dim=-1)  # (bs*seq, n_agents, 1+obs_dim)
        h = self.node_proj(node_features)  # (bs*seq, n_agents, embed_dim)
        
        # 编码全局状态
        state_emb = self.state_encoder(states)  # (bs*seq, embed_dim)
        
        # 多层图注意力传播
        for attn_layer, update_net in zip(self.attention_layers, self.node_update_nets):
            # 计算注意力权重
            attention_weights = self._compute_attention(h, state_emb, attn_layer)
            
            # 消息传递
            messages = self._message_passing(h, attention_weights)
            
            # 节点更新（带残差连接）
            h = h + update_net(th.cat([h, messages], dim=-1))
        
        # 聚合所有节点得到全局Q值
        h_flat = h.reshape(-1, self.n_agents * self.embed_dim)
        q_tot = self.q_predictor(h_flat)
        
        return q_tot.reshape(bs, seq_len, 1)
    
    def _compute_attention(self, h, state_emb, attn_layer):
        """
        计算多头注意力权重
        """
        n = h.shape[1]
        
        # 扩展状态向量
        state_emb_exp = state_emb.unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, embed_dim)
        state_emb_exp = state_emb_exp.expand(-1, n, n, -1)  # (bs, n, n, embed_dim)
        
        # 扩展节点特征用于计算所有对
        h_i = h.unsqueeze(2).expand(-1, n, n, -1)  # (bs, n, n, embed_dim)
        h_j = h.unsqueeze(1).expand(-1, n, n, -1)  # (bs, n, n, embed_dim)
        
        # 计算注意力分数
        attn_input = th.cat([h_i, h_j, state_emb_exp], dim=-1)  # (bs, n, n, 3*embed_dim)
        attn_scores = attn_layer(attn_input)  # (bs, n, n, num_heads)
        
        # 应用softmax获取注意力权重
        attn_weights = F.softmax(attn_scores, dim=2)  # (bs, n, n, num_heads)
        
        return attn_weights
    
    def _message_passing(self, h, attention_weights):
        """
        执行消息传递
        """
        # 转置以进行批矩阵乘法
        h_trans = h.transpose(1, 2)  # (bs, embed_dim, n_agents)
        
        # 计算加权消息 (bs, n, n, heads) -> (bs, heads, n, n)
        attn_weights = attention_weights.permute(0, 3, 1, 2)  # (bs, heads, n, n)
        
        # 消息聚合 (bs, heads, n, n) * (bs, embed_dim, n) -> (bs, heads, n, embed_dim)
        messages = th.matmul(attn_weights, h_trans.unsqueeze(1))
        
        # 重塑并连接所有头的消息
        messages = messages.permute(0, 2, 1, 3).reshape(
            h.shape[0], h.shape[1], -1
        )  # (bs, n_agents, heads*embed_dim)
        
        return messages
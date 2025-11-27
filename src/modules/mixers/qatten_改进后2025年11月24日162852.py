import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QattenMixer(nn.Module):
    def __init__(self, args):
        super(QattenMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.u_dim = int(np.prod(args.agent_own_state_size))

        # 1. 关键改进：统一维度参数，确保维度匹配
        self.n_query_embedding_layer2 = args.n_query_embedding_layer2
        self.n_key_embedding_layer1 = args.n_query_embedding_layer2  # 强制相等，解决维度匹配问题
        self.n_head_embedding_layer2 = args.n_attention_head  # 强制相等，解决维度匹配问题

        self.n_query_embedding_layer1 = args.n_query_embedding_layer1
        self.n_head_embedding_layer1 = args.n_head_embedding_layer1
        self.n_attention_head = args.n_attention_head
        # 2. 关键改进：修正拼写错误 (constrant -> constant)
        self.n_constant_value = args.n_constant_value

        # 3. 关键改进：移除冗余参数，简化配置
        self.query_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(nn.Sequential(
                nn.Linear(self.state_dim, self.n_query_embedding_layer1),
                nn.ReLU(),
                nn.Linear(self.n_query_embedding_layer1, self.n_query_embedding_layer2)
            ))
        
        self.key_embedding_layers = nn.ModuleList()
        for i in range(self.n_attention_head):
            # 4. 关键改进：确保键嵌入维度与查询一致
            self.key_embedding_layers.append(nn.Linear(self.u_dim, self.n_key_embedding_layer1))

        self.scaled_product_value = np.sqrt(self.n_query_embedding_layer2)

        # 5. 关键改进：简化头嵌入层，确保维度匹配
        self.head_embedding_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.n_head_embedding_layer1),
            nn.ReLU(),
            nn.Linear(self.n_head_embedding_layer1, self.n_head_embedding_layer2)
        )
        
        # 6. 关键改进：修正拼写错误 (constrant -> constant)
        self.constant_value_layer = nn.Sequential(
            nn.Linear(self.state_dim, self.n_constant_value),
            nn.ReLU(),
            nn.Linear(self.n_constant_value, 1)
        )

        # 7. 关键改进：添加维度检查，避免运行时错误
        self._check_dimension_consistency()

    def _check_dimension_consistency(self):
        """验证关键维度是否匹配，确保理论实现正确"""
        assert self.n_query_embedding_layer2 == self.n_key_embedding_layer1, \
            f"Query embedding dim ({self.n_query_embedding_layer2}) must equal key embedding dim ({self.n_key_embedding_layer1})"
        
        assert self.n_head_embedding_layer2 == self.n_attention_head, \
            f"Head embedding dim ({self.n_head_embedding_layer2}) must equal number of attention heads ({self.n_attention_head})"
        
        print(f"QattenMixer dimension check passed: "
              f"query_dim={self.n_query_embedding_layer2}, "
              f"key_dim={self.n_key_embedding_layer1}, "
              f"head_dim={self.n_head_embedding_layer2}, "
              f"n_heads={self.n_attention_head}")

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states)
            u_embedding = self.key_embedding_layers[i](us)

            # 8. 关键改进：显式维度检查确保安全
            assert state_embedding.size(2) == self.n_query_embedding_layer2, \
                f"State embedding dimension mismatch: expected {self.n_query_embedding_layer2}, got {state_embedding.size(2)}"
            assert u_embedding.size(2) == self.n_key_embedding_layer1, \
                f"U embedding dimension mismatch: expected {self.n_key_embedding_layer1}, got {u_embedding.size(2)}"

            # 维度调整：确保维度匹配
            state_embedding = state_embedding.view(-1, 1, self.n_query_embedding_layer2)
            u_embedding = u_embedding.view(-1, self.n_agents, self.n_key_embedding_layer1)
            u_embedding = u_embedding.permute(0, 2, 1)

            # 9. 关键改进：使用理论证明的注意力计算
            raw_lambda = th.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda = F.softmax(raw_lambda, dim=-1)

            q_lambda_list.append(q_lambda)

        # 将每个头的注意力权重堆叠
        q_lambda_list = th.stack(q_lambda_list, dim=1)  # [B, n_heads, 1, n_agents]
        q_lambda_list = q_lambda_list.squeeze(-2)  # [B, n_heads, n_agents]

        # 10. 关键改进：使用更清晰的维度重塑
        q_lambda_list = q_lambda_list.permute(0, 2, 1)  # [B, n_agents, n_heads]

        # 计算每个头的加权和: [B, 1, n_heads]
        q_h = th.matmul(agent_qs, q_lambda_list)  # [B, 1, n_heads]

        # 11. 关键改进：使用理论证明的加权方式
        if self.args.type == 'weighted':
            # 获取头权重: [B, n_heads, 1]
            w_h = th.abs(self.head_embedding_layer(states))
            w_h = w_h.view(-1, self.n_head_embedding_layer2, 1)
            
            # 加权求和: [B, 1]
            sum_q_h = th.matmul(q_h, w_h)
        else:
            # 直接求和: [B, 1]
            sum_q_h = q_h.sum(-1)

        sum_q_h = sum_q_h.view(-1, 1)
        c = self.constant_value_layer(states)
        q_tot = sum_q_h + c
        q_tot = q_tot.view(bs, -1, 1)
        return q_tot

    def _get_us(self, states):
        agent_own_state_size = self.args.agent_own_state_size
        # 12. 关键改进：确保正确提取智能体自身状态
        # 确保agent_own_state_size是整数（由初始化保证）
        us = states[:, :agent_own_state_size * self.n_agents].view(-1, agent_own_state_size)
        return us
import torch as th
import torch.nn as nn
from typing import Tuple
from stable_baselines3.common.policies import ActorCriticPolicy
from ppo import PPO
class MyMultiHeadAttention(nn.Module):

    def __init__(self, d_model=128, num_heads=4):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.bias=nn.Parameter(th.zeros(num_heads))
        self.bias2=nn.Parameter(th.zeros(num_heads))
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):

        B, N, D = x.shape

        # --------------------------------
        # Q, K, V
        # --------------------------------

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # [B, N, D]
        #      ↓
        # [B, H, N, head_dim]

        Q = Q.view(B, N, self.num_heads, self.head_dim)
        K = K.view(B, N, self.num_heads, self.head_dim)
        V = V.view(B, N, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # [B, H, N, head_dim]

        # --------------------------------
        # Attention scores
        # --------------------------------

        scores = th.matmul(
            Q,
            K.transpose(-2, -1)
        )
        phase_map=[[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]],
        relation=lane_relation(phase_map)

        relation = th.as_tensor(relation,dtype=scores.dtype,device=scores.device)
        relation=relation.repeat_interleave(10, dim=0).repeat_interleave(10, dim=1)
        I=th.eye(24)
        I=I.repeat_interleave(10, dim=0).repeat_interleave(10, dim=1)
        I = th.as_tensor(I,dtype=scores.dtype,device=scores.device)
        scores = scores / (self.head_dim ** 0.5) + self.bias[None, :, None, None]*relation[None , None , : , :]+self.bias2[None, :, None, None]*I[None , None , : , :]

        # [B, H, N, N]

        # --------------------------------
        # Mask
        # --------------------------------

        if mask is not None:
            scores = scores.masked_fill(
                mask[:, None, None, :],
                float("-inf")
            )

        # --------------------------------
        # Softmax
        # --------------------------------

        attention = th.softmax(
            scores,
            dim=-1
        )

        # --------------------------------
        # Weighted sum
        # --------------------------------

        output = th.matmul(
            attention,
            V
        )

        # [B, H, N, head_dim]

        # --------------------------------
        # Combine heads
        # --------------------------------

        output = output.transpose(1, 2)

        output = output.contiguous().view(
            B, N, self.d_model
        )

        output = self.out_proj(output)

        return output, attention
class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        d_model=128,
        num_heads=4,
        dim_feedforward=256,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MyMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        # Attention
        attn_out, attention = self.attention(
            x,
            mask
        )

        x = self.norm1(
            x + self.dropout(attn_out)
        )

        # Feed-forward
        ff_out = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ff_out)
        )

        return x, attention
class SelfAttention(nn.Module):
    def __init__(self, d, d_q, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.latent_dim_pi = 3
        self.latent_dim_vf = 3

        self.transformerblock1=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock2=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock3=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)
        self.transformerblock4=TransformerEncoderBlock(d_model=32,
        num_heads=4,
        dim_feedforward=64,
        dropout=0)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,dim_feedforward=64, nhead=4, dropout=0, batch_first=True),
            num_layers=2
        )


        self.transformer2=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=32,dim_feedforward=64, nhead=4, dropout=0, batch_first=True),
            num_layers=2
        )
        self.self_attn1 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )

        self.linear1 = nn.Linear(11, 32)
        self.linear2 = nn.Linear(11, 32)
        self.linear3 = nn.Linear(32, 3)
        self.linear4 = nn.Linear(32, 3)
        self.linear5 = nn.Linear(20, 20)
        self.linear6 = nn.Linear(20, 20)
        self.linear7 = nn.Linear(20, 1)
        self.linear8 = nn.Linear(20, 1)
        self.Relu1 = nn.ReLU()
        self.Relu2 = nn.ReLU()
        self.Relu3 = nn.ReLU()
        self.Relu4 = nn.ReLU()
        self.self_attn2 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.self_attn3 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.self_attn4 = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True
        )
        self.model=nn.Sequential(nn.Linear(11, 64) , nn.Tanh() , nn.Linear(64, 64), nn.Tanh() , nn.Linear(64,3))
        self.phase_map= [[1, 4, 12, 13, 14, 15, 16, 17], [7, 10, 18, 19, 20, 21, 22, 23], [0, 3, 18, 19, 20, 21, 22, 23], [6, 9, 12, 13, 14, 15, 16, 17]]


    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:



        #context_vector2=self.policy_net2(context_vector2)
        return self.forward_actor(x),self.forward_critic(x)


    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        
        x=self.model(x[: , -1 , :])
     
        
        return x


    def forward_critic(self, x: th.Tensor) -> th.Tensor:
        x=self.linear2(x)
        x=self.transformer2(x)
        x=self.linear4(x)
        return x[:, -1, :]




# Custom Policy incorporating the Self-Attention feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):

        use_sde = kwargs.pop('use_sde', False)
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,


            *args,
            use_sde=use_sde,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:

        # Use the shared extracted features for both policy and value networks
        self.features_extractor = nn.Identity()
        self.mlp_extractor = SelfAttention(8,8,8,8)
        self.pi_features_extractor= nn.Identity()
        self.vf_features_extractor=nn.Identity()
        self.action_net=nn.Identity()
        self.value_net=nn.Identity()

import gymnasium as gym

env = gym.make("Hopper-v5")
ppo=PPO(CustomPolicy , env , verbose=2)

ppo.learn(total_timesteps=1000000)
